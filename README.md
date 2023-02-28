# yolov7-face-trt
***
This code is designed to run the yolov7-face in a TensorRT-python environment.

## to-do list

- [x] support webcam and video (but slow image & video)
- [X] support EfficientNMS_TRT
- [ ] simplified code and optimized

## trt build and run

#### prepare python library
This code tested docker image nvcr.io/nvidia/pytorch:22.10-py3 (RTX3090), nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3 (Jetson Orin)

#### GPU setting
```
cd ./docker/gpu
sh compose.sh # setting to docker container
cd yolov7-face
pip install opencv-python # re-build opencv for environment setting and using cv2.imshow()
```

#### jetson setting
```
cd ./docker/jetson
sh compose.sh # setting to docker container
cd yolov7-face
```

### demo in pytorch (+ 6dof)
#### demo yolov7-face image
```
python3 detect.py --weights yolov7-tiny-face.pt --source img_path
# if you want to 6dof result
python3 detect.py --weights yolov7-tiny-face.pt --source img_path --use-dof # (--save-dof for save result) 
```

#### demo yolov7-face webcam (or video)
```
python3 detect.py --weights yolov7-tiny-face.pt --source 0 or video_path # (0 is webcam index)

# if you want 6dof result

python3 detect.py --weights yolov7-tiny-face.pt --source img_path --use-dof # (--save-dof for save result) 

```

#### demo yolov7-face realsense (only run desktop environment, because Orin does not support pyrealsense2)
```
python3 detect.py --weights yolov7-tiny-face.pt --source 'rgb' --use-rs # 'infrared' supported future works

# if you want 6dof result

python3 detect.py --weights yolov7-tiny-face.pt --source 'rgb' --use-rs --use-dof # (--save-dof for save result) 

```


#### First. convert pytorch to onnx model (without include_nms)
```
# convert yolov7-tiny-face.pt to yolov7-tiny-face.onnx
python3 models/export.py --weights yolov7-tiny-face.pt --grid --simplify
```
#### Second. convert onnx model to trt model (use local machine)
```
# convert yolov7-tiny-face.onnx to yolov7-tiny-face.trt (without nms) # (using fp16)
python3 models/export_tensorrt.py -o yolov7-tiny-face.onnx -e yolov7-tiny-face.trt 
```
```
# convert yolov7-tiny-face.onnx to end2end.trt (with nms)
python3 models/export_tensorrt.py -o yolov7-tiny-face.onnx -e end2end.trt --end2end
```
#### Third. run trt model
Run image inference
```
# use pytorch nms
python3 trt_inference/yolo_face_trt_inference.py -e yolov7-tiny-face.trt -i {image_path} -o {output_img_name}
```
```
# using end2end machine
python3 trt_inference/yolo_face_trt_inference.py -e end2end.trt -i image.jpg --end2end
```

### Run webcam inference
#### using torchvision nms
```
python3 trt_inference/yolo_face_trt_inference.py -e yolov7-tiny-face.trt -v 0

```
#### using tensorrt nms
```
python3 trt_inference/yolo_face_trt_inference.py -e end2end.trt --end2end -v 0
```




### New feature

* Dynamic keypoints
* WingLoss
* Efficient backbones
* EIOU and SIOU



| Method           |  Test Size | Easy  | Medium | Hard  | FLOPs (B) @640 | Link  |
| -----------------| ---------- | ----- | ------ | ----- | -------------- | ----- |
| yolov7-lite-t    | 640        | 88.7  | 85.2   | 71.5  |  0.8           | [google](https://drive.google.com/file/d/1HNXd9EdS-BJ4dk7t1xJDFfr1JIHjd5yb/view?usp=sharing) |
| yolov7-lite-s    | 640        | 92.7  | 89.9   | 78.5  |  3.0           | [google](https://drive.google.com/file/d/1MIC5vD4zqRLF_uEZHzjW_f-G3TsfaOAf/view?usp=sharing) |
| yolov7-tiny      | 640        | 94.7  | 92.6   | 82.1  |  13.2          | [google](https://drive.google.com/file/d/1Mona-I4PclJr5mjX1qb8dgDeMpYyBcwM/view?usp=sharing) |
| yolov7s          | 640        | 94.8  | 93.1   | 85.2  |  16.8          | [google](https://drive.google.com/file/d/1_ZjnNF_JKHVlq41EgEqMoGE2TtQ3SYmZ/view?usp=sharing) |
| yolov7           | 640        | 96.9  | 95.5   | 88.0  |  103.4         | [google](https://drive.google.com/file/d/1oIaGXFd4goyBvB1mYDK24GLof53H9ZYo/view?usp=sharing) |
| yolov7+TTA       | 640        | 97.2  | 95.8   | 87.7  |  103.4         | [google](https://drive.google.com/file/d/1oIaGXFd4goyBvB1mYDK24GLof53H9ZYo/view?usp=sharing) |
| yolov7-w6        | 960        | 96.4  | 95.0   | 88.3  |  89.0          | [google](https://drive.google.com/file/d/1U_kH7Xa_9-2RK2hnyvsyMLKdYB0h4MJS/view?usp=sharing) |
| yolov7-w6+TTA    | 1280       | 96.9  | 95.8   | 90.4  |  89.0          | [google](https://drive.google.com/file/d/1U_kH7Xa_9-2RK2hnyvsyMLKdYB0h4MJS/view?usp=sharing) |



#### Dataset

[WiderFace](http://shuoyang1213.me/WIDERFACE/)

[yolov7-face-label](https://drive.google.com/file/d/1FsZ0ACah386yUufi0E_PVsRW_0VtZ1bd/view?usp=sharing)

#### Test

![](data/images/result.jpg)


#### Demo

* [ncnn_Android_face](https://github.com/FeiGeChuanShu/ncnn_Android_face)

* [yolov7-detect-face-onnxrun-cpp-py](https://github.com/hpc203/yolov7-detect-face-onnxrun-cpp-py)

#### References

* [https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

* [https://github.com/ppogg/YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)
