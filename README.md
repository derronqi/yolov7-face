
# Installation

```
git clone git@github.com:veesion-io/yolov7-face.git
cd yolov7-face
nvidia-docker run --gpus all --name blurring --security-opt seccomp=unconfined \
  --net=host --ipc=host -v /dev/shm:/dev/shm --ulimit memlock=-1 \
  -v /path/to/yolov7-face:/workspace/ -v /path/to/your/videos:/workspace/videos/ \
  --ulimit stack=67108864 -it nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash

cd /workspace/
pip install gdown
gdown https://drive.google.com/uc?id=1U_kH7Xa_9-2RK2hnyvsyMLKdYB0h4MJS
pip install -r requirements.txt
pip install seaborn onnxruntime


pip uninstall -y opencv-contrib-python \
  && rm -rf /usr/local/lib/python3.10/dist-packages/cv2
pip install opencv-contrib-python ffprobe3
apt update && DEBIAN_FRONTEND=noninteractive apt install ffmpeg -y

```

# Detect and blur faces : 

Detect faces with a deep learning model and save blurred videos.
```
python3 blur_videos.py -i videos -o blurred_videos
```

## With docker

1. Build image

```bash
docker build -t yolov7-face: .
```

2. Run container

```bash
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name face -v /root/videos:/workspace/videos -v /root/blurred_videos:/workspace/blurred_videos yolov7-face python3 blur_videos.py -i /workspace/videos -o /workspace/blurred_videos
```

docker run --rm -it -d --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name face -v /root/videos:/workspace/videos -v /root/blurred_videos:/workspace/blurred_videos yolov7-face uvicorn api:app --host 0.0.0.0
