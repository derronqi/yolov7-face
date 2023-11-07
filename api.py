import boto3
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import torch
import argparse
import os
from blur_videos import detect

app = FastAPI()

class Video(BaseModel):
    bucket_name: str
    video_key: str

# Create an S3 client
s3 = boto3.client('s3')

# Init inference
torch.backends.cudnn.enabled = True
device = torch.device('cuda:0')

# Init videos directories
VIDEO_PATH = '/workspace/videos'
BLURRED_VIDEO_PATH = '/workspace/blurred_videos'

# Check if the directory exists
if not os.path.exists(VIDEO_PATH):
    os.mkdir(VIDEO_PATH)
if not os.path.exists(BLURRED_VIDEO_PATH):
    os.mkdir(BLURRED_VIDEO_PATH)

@app.post("/cloud")
def detect_faces_in_cloud(video: Video, request: Request):
    # Download alert video
    video_name = f'{VIDEO_PATH}/video.mp4'
    try:
        s3.download_file(video.bucket_name, video.video_key, video_name)
        print(f"Object '{video.video_key}' downloaded to '{video_name}'")
    except Exception as e:
        msg = f"An error occurred: {e}"
        print(msg)
        raise HTTPException(status_code=500, detail=msg)

    # Blur video
    opt = argparse.Namespace()
    opt.weights = 'yolov7-w6-face.pt'
    opt.frame_size = 1280
    opt.conf_thres = 0.025
    opt.input_directory = VIDEO_PATH
    opt.output_directory = BLURRED_VIDEO_PATH
    opt.iou_thres = 0.1
    opt.classes = None
    opt.agnostic_nms = False
    opt.augment = False
    opt.update = False
    opt.kpt_label = 5

    with torch.no_grad():
        detect(opt=opt)

    # Upload blurred video
    try:
        blurred_video_name = f'{BLURRED_VIDEO_PATH}/blurred_video.mp4'
        s3.upload_file(Filename=blurred_video_name, Bucket=video.bucket_name, Key=video.video_key, ExtraArgs={"Tagging": f"blurred=True"})
        print(f"Video '{blurred_video_name}' uploaded to bucket '{video.bucket_name}'")
    except Exception as e:
        msg = f"An error occurred: {e}"
        print(msg)
        raise HTTPException(status_code=500, detail=msg)
