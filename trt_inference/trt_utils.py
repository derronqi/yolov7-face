import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision

class BaseEngine(object):
    def __init__(self, engine_path):
        self.mean = None
        self.std = None
        self.n_classes = 1
        self.nkpt = 5
        self.class_names = ['face']

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})


    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        # 여기에 nms 추가할 수는 없나
        
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data

    def inference(self, img_path, conf=0.5, end2end=False):
        origin_img = cv2.imread(img_path)
        #img, ratio = preproc_pad(origin_img, self.imgsz, self.mean, self.std)
        #data = self.infer(img)
        
        resized_img, resized_img_tran = preproc(origin_img, self.imgsz)
        data = self.infer(resized_img_tran)
        #predictions = np.reshape(data, (1, -1, int(5+self.n_classes + self.nkpt*3)))[0] # does not using batch inference
        #dets = self.postprocess(predictions,ratio)
        predictions = np.reshape(data, (1, -1, int(5+self.n_classes + self.nkpt*3))) # does not using batch inference        
        dets = self.postprocess_ops_nms(predictions=predictions)[0]
        #print("output", len(dets), print(dets[0]))
        if dets is not None:
            print(dets)
            final_boxes, final_scores, final_key_points = dets[:,:4], dets[:, 4], dets[:, 5:]
            origin_img = vis(resized_img, final_boxes, final_scores, final_key_points,
                             conf=conf, class_names=self.class_names)
        return origin_img

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        keypoints = predictions[:, 6:]
        scores = predictions[:,4:5] # * predictions[:,5:6] # obj prod * class prod (face)
        scores_mask = scores > 0.1
        scores_mask = scores_mask.squeeze()
        if scores_mask.sum() == 0:
            return None       

        boxes = boxes[scores_mask, :]
        keypoints = keypoints[scores_mask, :]
        scores = scores[scores_mask, :]        
        
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        keep = nms(boxes_xyxy, scores, nms_thr=0.45)
        dets_boxes = boxes_xyxy[keep]
        dets_scores = scores[keep]
        dets_keypoint = keypoints[keep]
        num_idx = np.zeros([len(dets_boxes), 1])
        dets = np.concatenate([dets_boxes, dets_scores, dets_keypoint], 1)
        return dets
    
    @staticmethod
    def postprocess_ops_nms(predictions, conf_thres =0.25, iou_thres=0.45, classes=None):
        kpt_label=5
        min_wh, max_wh = 2, 4096
        prediction = torch.from_numpy(predictions)
        print(prediction.shape)
        xc = prediction[..., 4] > conf_thres
        output = [torch.zeros((0, kpt_label*3+6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            # Compute conf
            cx, cy, w, h = x[:,0:1], x[:,1:2], x[:,2:3], x[:,3:4]
            obj_conf = x[:, 4:5]
            #print("obj_conf :", obj_conf)
            cls_conf = x[:, 5:6]
            #print("cls_conf :", cls_conf)
            kpts = x[:, 6:]
            cls_conf = obj_conf * cls_conf  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            x_min = cx - (w/2)
            y_min = cy - (h/2)
            x_max = cx + (w/2)
            y_max = cy + (h/2)
            box = torch.cat((x_min, y_min, x_max, y_max), 1)            
            conf, j = cls_conf.max(1, keepdim=True)
            #print("after class conf :", conf)
            x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]
            #print("x shape : ", x.shape)
            c = x[:, 5:6] * 0
            #print("c shape :", c)
            boxes, scores = x[:, :4] +c , x[:, 4]  # boxes (offset by class), scores
            #print("boxes.shape :", boxes.shape)
            #print("box value :", boxes, scores)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            output[xi] = x[i]
        return output
        
    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100/(time.perf_counter() - t0), 'FPS')


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def preproc(image, input_size, swap=(2,0,1)):
    resized_img = cv2.resize(image, input_size)
    #resized_img = resized_img / 255
    resized_img_transpose = resized_img.transpose(swap)
    resized_img_transpose = resized_img_transpose / 255
    resized_img_transpose = np.ascontiguousarray(resized_img_transpose, dtype=np.float32)
    return resized_img, resized_img_transpose

def preproc_pad(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)


_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, keypoints, conf=0.1, class_names=None):
    #img = cv2.resize(img, (640, 640))
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(0)
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def vis_keypoint(img, boxes, scores, keypoints, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        keypoint = keypoints[i]
        cls_id = 0
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img