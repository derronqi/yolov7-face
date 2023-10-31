from fractions import Fraction
import subprocess
import argparse
import os
import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, \
    scale_coords
from ffprobe3 import FFProbe
from utils.datasets import letterbox
import numpy as np

torch.backends.cudnn.enabled = True
device = torch.device('cuda:0')


def detect(opt):
    weights, frame_size, kpt_label = opt.weights, opt.frame_size, opt.kpt_label
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, frame_size, frame_size).to(
            device).type_as(next(model.parameters())))  # run once
    videos = os.listdir(opt.input_directory)
    os.makedirs(opt.output_directory, exist_ok=True)
    for video_name in videos:
        blur_video(
            os.path.join(opt.input_directory, video_name),
            opt.output_directory, model, stride, frame_size, kpt_label)


def read_video_fps_and_duration(video_path: str):
    meta_data = FFProbe(video_path)
    fps = None
    duration = 0.0
    for stream in meta_data.streams:
        if stream.is_video():
            try:
                fps = float(Fraction(stream.avg_frame_rate))
                duration = float(Fraction(stream.duration))
            except:
                pass
    if not fps:
        for stream in meta_data.streams:
            if stream.is_video():
                try:
                    fps = float(Fraction(stream.r_frame_rate))
                except:
                    pass
    return {"fps": fps, "duration": duration}


def blur_faces(frame, preprocessed_frame, predictions, kpt_label):
    # Process detections
    for i, det in enumerate(predictions):  # detections per image
        if len(det):
            # Rescale boxes from frame_size to im0 size
            scale_coords(
                preprocessed_frame.shape[1:], det[:, :4], frame.shape,
                kpt_label=False)
            scale_coords(
                preprocessed_frame.shape[1:], det[:, 6:], frame.shape,
                kpt_label=kpt_label, step=3)

            # Write results
            for *xyxy, _, _ in reversed(det[:, :6]):
                y_min, x_min, y_max, x_max = xyxy
                x_min, x_max, y_min, y_max = x_min.item(), x_max.item(), y_min.item(), y_max.item()
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, frame.shape[0])
                y_max = min(y_max, frame.shape[1])
                frame = pixelate(frame, int(x_min), int(x_max), int(
                    y_min), int(y_max))
    return frame


def pixelate(image, a, b, c, d, grain=5):
    if (b - a) * (d - c) < grain**2 or b - a < grain or d - c < grain:
        return image
    grain = min((grain, b - a - 1, d - c - 1))
    flou = cv2.resize(
        cv2.resize(image[a:b, c:d], (grain, grain)),
        (d - c, b - a),
        interpolation=cv2.INTER_NEAREST,
    )
    image[a:b, c:d] = flou
    return image


def detect_faces(frame, model, kpt_label):
    frame_tensor = torch.from_numpy(frame).to(device)
    frame_tensor = frame_tensor.half()
    frame_tensor /= 255.0
    if frame_tensor.ndimension() == 3:
        frame_tensor = frame_tensor.unsqueeze(0)

    predictions = model(frame_tensor, augment=opt.augment)[0]
    predictions = non_max_suppression(
        predictions, opt.conf_thres, opt.iou_thres, classes=opt.classes,
        agnostic=opt.agnostic_nms, kpt_label=kpt_label)
    return predictions


def blur_video(video_path, output_directory, model, stride, frame_size, kpt_label):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    blurred_video_name = os.path.join(
        output_directory, "blurred_" + video_name + ".mp4")
    fps = read_video_fps_and_duration(video_path)["fps"]
    video_reader = cv2.VideoCapture(video_path)
    is_frame_valid, frame = video_reader.read()
    w, h = frame.shape[:2]
    video_writer = create_video_writer(fps, (w, h), blurred_video_name)
    while is_frame_valid:
        preprocessed_frame = letterbox(
            frame, frame_size, stride=stride, auto=False)[0]
        # BGR to RGB, to 3x416x416
        preprocessed_frame = preprocessed_frame[:, :, ::-1].transpose(2, 0, 1)
        preprocessed_frame = np.ascontiguousarray(preprocessed_frame)
        predictions = detect_faces(preprocessed_frame, model, kpt_label)
        frame = blur_faces(frame, preprocessed_frame, predictions, kpt_label)
        try:
            # Write frame to FFMPEG pipe
            video_writer.stdin.write(frame.tobytes())
        except:
            stderr_output = video_writer.stderr.read()
            print(f"FFMPEG STDERR OUTPUT:\n{stderr_output.decode()}")
            break
        is_frame_valid, frame = video_reader.read()
    video_writer.terminate()  # release previous video writer


def create_video_writer(fps, shape, output_name):
    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', f'{shape[1]}x{shape[0]}',
               '-pix_fmt', 'bgr24',
               '-r', str(fps),
               '-i', '-',
               '-an',
               '-vcodec', 'libx264',
               '-crf', '0',
               output_name]

    pipe = subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    return pipe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov7-w6-face.pt', help='model.pt path(s)')
    parser.add_argument('--frame-size', nargs='+', type=int,
                        default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.025, help='object confidence threshold')
    parser.add_argument(
        '--input-directory', '-i', type=str, required=True,
        help='The directory containing the videos to be blurred')
    parser.add_argument(
        '--output-directory', '-o', type=str, required=True,
        help='The directory into which we should save the blurred videos')
    parser.add_argument('--iou-thres', type=float,
                        default=0.1, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--kpt-label', type=int, default=5,
                        help='number of keypoints')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt=opt)
