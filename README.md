# yolov7-face

### New feature

* Dynamic keypoints
* WingLoss
* Efficient backbones
* EIOU and SIOU



| Method           |  Test Size | Easy  | Medium | Hard  | FLOPs (B) @640 | Link  |
| -----------------| ---------- | ----- | ------ | ----- | -------------- | ----- |
| yolov7-lite-e    | 640        | 91.8  | 88.6   | 71.8  |  2.6           | [google](https://drive.google.com/file/d/1hvjQ43lrhYWslCiiNoKDmkso98tgdG7P/view?usp=sharing) |
| yolov7-tiny-leak | 640        | 93.2  | 91.3   | 83.0  |  16.6          | [google](https://drive.google.com/file/d/1B2F5YuERfMEfJeRXfz5oMxI8wcZLmvFJ/view?usp=sharing) |
| yolov7s          | 640        | 94.8  | 93.1   | 85.2  |  16.8          | [google](https://drive.google.com/file/d/1_ZjnNF_JKHVlq41EgEqMoGE2TtQ3SYmZ/view?usp=sharing) |
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
