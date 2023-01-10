from trt_utils import preproc, vis
from trt_utils import BaseEngine
import numpy as np
import cv2
import time
import os, sys
import argparse
from loguru import logger

class Predictor(BaseEngine):
    def __init__(self, engine_path, logger):
        super(Predictor, self).__init__(engine_path, logger)
        self.n_classes = 1  # your model classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-o", "--output", default="output_trt.png",help="image output path")
    parser.add_argument("-l", "--log", default="./infer_trt.log",help="log path")
    parser.add_argument("-v", "--video",  help="video path or camera index ")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="use end2end engine")

    args = parser.parse_args()
    print(args)
    
    logger.add(args.log)
    
    pred = Predictor(engine_path=args.engine, logger=logger)
    pred.get_fps()
    img_path = args.image
    video = args.video
    if img_path:
      origin_img = pred.inference(img_path, conf=0.1, end2end=args.end2end)
      print(origin_img.shape)
      if origin_img.shape[0] == 3:
        origin_img = origin_img.transpose(1, 2, 0)
      cv2.imwrite("%s" %args.output, origin_img)
    if video:
      pred.detect_video(video, conf=0.1, end2end=args.end2end) # set 0 use a webcam