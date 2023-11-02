FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /workspace

RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=1U_kH7Xa_9-2RK2hnyvsyMLKdYB0h4MJS

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install seaborn onnxruntime

RUN pip uninstall -y opencv-contrib-python \
  && rm -rf /usr/local/lib/python3.10/dist-packages/cv2
RUN pip install opencv-contrib-python ffprobe3
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install ffmpeg -y

COPY . .

CMD [ "/bin/bash" ]
