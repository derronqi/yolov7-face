ARG BASE_TORCH_VERSION=pth1.12
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3
FROM ${BASE_IMAGE}


ARG USER_ID
ARG GROUP_ID
ARG USER_NAME

RUN addgroup --gid $GROUP_ID $USER_NAME
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME
# if you want to using torch.cuda to $USER_NAME, your user_name adding to video group
RUN usermod -a -G video $USER_NAME

RUN apt-get clean
# this line change to ubuntu archive
RUN apt-get update -y
RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    curl \
    python3 \
    python3-dev \
    ca-certificates 
###################
# apt-get setting #
###################
RUN apt-get install vim -y 
RUN apt-get install tmux -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install ffmpeg -y
RUN apt-get install libx264-dev -y
RUN apt-get install x11-xserver-utils -y
RUN apt-get install x11-apps -y
RUN apt-get install -y libxinerama-dev libxcursor-dev
RUN apt-get install v4l-utils -y

#########################
#    python setting     #
#########################
RUN pip3 install -U easydict
RUN pip3 install -U onnxruntime
RUN pip3 install -U onnxsim
RUN pip3 install -U seaborn
RUN pip3 install -U loguru
RUN pip3 install -U numpy==1.22.1 # same setting trt
######################################
#   librealsense Base Image Stage    #
######################################

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
RUN git clone https://github.com/pdh930105/yolov7-face.git

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV PATH=/usr/local/cuda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64:
ENV DEBIAN_FRONTEND=noninteractive
