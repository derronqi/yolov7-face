# reference code : https://github.com/Linaom1214/TensorRT-For-YOLO-Series

import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


import os
import sys

import numpy as np
from PIL import Image


class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(self, input, shape, dtype, max_num_images=None, exact_batches=False, preprocessor="fixed_shape_resizer"):
        """
        :param input: The input directory to read images from.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_num_images: The maximum number of images to read from the directory.
        :param exact_batches: This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        :param preprocessor: Set the preprocessor to use, depending on which network is being used.
        """
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
            self.images.sort()
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0:self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

        self.preprocessor = preprocessor

    def preprocess_image(self, image_path):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes padding,
        resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * Resizes and pads the image to fit the input size.
        :param image_path: The path to the image on disk to load.
        :return: Two values: A numpy array holding the image sample, ready to be contacatenated into the rest of the
        batch, and the resize scale used, if any.
        """

        def resize_pad(image, pad_color=(0, 0, 0)):
            """
            A subroutine to implement padding and resizing. This will resize the image to fit fully within the input
            size, and pads the remaining bottom-right portions with the value provided.
            :param image: The PIL image object
            :pad_color: The RGB values to use for the padded area. Default: Black/Zeros.
            :return: Two values: The PIL image object already padded and cropped, and the resize scale used.
            """

            # Get characteristics.
            width, height = image.size
            width_scale = width / self.width
            height_scale = height / self.height

            # Depending on preprocessor, box scaling will be slightly different.
            if self.preprocessor == "fixed_shape_resizer":
                scale = [self.width / width, self.height / height]
                image = image.resize((self.width, self.height), resample=Image.BILINEAR)
                return image, scale
            elif self.preprocessor == "keep_aspect_ratio_resizer":
                scale = 1.0 / max(width_scale, height_scale)
                image = image.resize((round(width * scale), round(height * scale)), resample=Image.BILINEAR)
                pad = Image.new("RGB", (self.width, self.height))
                pad.paste(pad_color, [0, 0, self.width, self.height])
                pad.paste(image)
                return pad, scale

        scale = None
        image = Image.open(image_path)
        image = image.convert(mode='RGB')
        if self.preprocessor == "fixed_shape_resizer" or self.preprocessor == "keep_aspect_ratio_resizer":
            #Resize & Pad with ImageNet mean values and keep as [0,255] Normalization
            image, scale = resize_pad(image, (124, 116, 104))
            image = np.asarray(image, dtype=self.dtype)
        else:
            print("Preprocessing method {} not supported".format(self.preprocessor))
            sys.exit(1)
        if self.format == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        return image/255., scale

    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding three items per iteration: a numpy array holding a batch of images, the list of
        paths to the images loaded within this batch, and the list of resize scales for each image in the batch.
        """
        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            batch_scales = [None] * len(batch_images)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i], batch_scales[i] = self.preprocess_image(image)
            self.batch_index += 1
            yield batch_data, batch_images, batch_scales


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = cuda.mem_alloc(size)
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """
    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30))
        # self.config.max_workspace_size = workspace * (2 ** 30)  # Deprecation

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path, end2end, conf_thres, iou_thres, max_det):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        print("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        assert self.batch_size > 0
        # self.builder.max_batch_size = self.batch_size  # This no effect for networks created with explicit batch dimension mode. Also DEPRECATED.
        
        # but YOLOv7-face landmark detector can't use EfficientNMS_TRT (EfficientNMS_TRT does not return the boxes index, so can't find the keypoint in selected box)
        if end2end:
            raise "please does not use yolov7-face"
            previous_output = self.network.get_output(0)
            self.network.unmark_output(previous_output)
            # output [1, 8400, 85]
            # slice boxes, obj_score, class_scores
            strides = trt.Dims([1,1,1])
            starts = trt.Dims([0,0,0])
            bs, num_boxes, temp = previous_output.shape
            shapes = trt.Dims([bs, num_boxes, 4])
            # [0, 0, 0] [1, 8400, 4] [1, 1, 1]
            boxes = self.network.add_slice(previous_output, starts, shapes, strides)
            num_classes = temp -5 
            starts[2] = 4
            shapes[2] = 1
            # [0, 0, 4] [1, 8400, 1] [1, 1, 1]
            obj_score = self.network.add_slice(previous_output, starts, shapes, strides)
            starts[2] = 5
            shapes[2] = num_classes
            # [0, 0, 5] [1, 8400, 80] [1, 1, 1]
            scores = self.network.add_slice(previous_output, starts, shapes, strides)
            # scores = obj_score * class_scores => [bs, num_boxes, nc]
            updated_scores = self.network.add_elementwise(obj_score.get_output(0), scores.get_output(0), trt.ElementWiseOperation.PROD)

            '''
            "plugin_version": "1",
            "background_class": -1,  # no background class
            "max_output_boxes": detections_per_img,
            "score_threshold": score_thresh,
            "iou_threshold": nms_thresh,
            "score_activation": False,
            "box_coding": 1,
            '''
            registry = trt.get_plugin_registry()
            assert(registry)
            creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
            assert(creator)
            fc = []
            fc.append(trt.PluginField("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("max_output_boxes", np.array([max_det], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("score_threshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            fc.append(trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            fc.append(trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
            
            fc = trt.PluginFieldCollection(fc) 
            nms_layer = creator.create_plugin("nms_layer", fc)

            layer = self.network.add_plugin_v2([boxes.get_output(0), updated_scores.get_output(0)], nms_layer)
            layer.get_output(0).name = "num"
            layer.get_output(1).name = "boxes"
            layer.get_output(2).name = "scores"
            layer.get_output(3).name = "classes"
            for i in range(4):
                self.network.mark_output(layer.get_output(i))


    def create_engine(self, engine_path, precision, calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        print("Building {} Engine in {}".format(precision, engine_path))
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        # TODO: Strict type is only needed If the per-layer precision overrides are used
        # If a better method is found to deal with that issue, this flag can be removed.
        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                print("INT8 is not supported natively on this platform/device")
            else:
                if self.builder.platform_has_fast_fp16:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    self.config.set_flag(trt.BuilderFlag.FP16)
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    self.config.int8_calibrator.set_image_batcher(
                        ImageBatcher(calib_input, calib_shape, calib_dtype, max_num_images=calib_num_images,
                                     exact_batches=True))

        # with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
        with self.builder.build_serialized_network(self.network, self.config) as engine, open(engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(engine_path))
            f.write(engine)  # .serialize()

def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx, args.end2end, args.conf_thres, args.iou_thres, args.max_det)
    builder.create_engine(args.engine, args.precision, args.calib_input, args.calib_cache, args.calib_num_images,
                          args.calib_batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("-w", "--workspace", default=1, type=int, help="The max memory workspace size to allow in Gb, "
                                                                       "default: 1")
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument("--calib_cache", default="./calibration.cache",
                        help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    parser.add_argument("--calib_num_images", default=5000, type=int,
                        help="The maximum number of images to use for calibration, default: 5000")
    parser.add_argument("--calib_batch_size", default=8, type=int,
                        help="The batch size for the calibration process, default: 8")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="export the engine include nms plugin, default: False")
    parser.add_argument("--conf_thres", default=0.4, type=float,
                        help="The conf threshold for the nms, default: 0.4")
    parser.add_argument("--iou_thres", default=0.5, type=float,
                        help="The iou threshold for the nms, default: 0.5")
    parser.add_argument("--max_det", default=100, type=int,
                        help="The total num for results, default: 100")

    args = parser.parse_args()
    print(args)
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision == "int8" and not (args.calib_input or os.path.exists(args.calib_cache)):
        parser.print_help()
        log.error("When building in int8 precision, --calib_input or an existing --calib_cache file is required")
        sys.exit(1)
    
    main(args)