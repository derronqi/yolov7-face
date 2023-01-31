ARG BASE_TORCH_VERSION=pth1.12
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.1.0-${BASE_TORCH_VERSION}-py3
FROM ${BASE_IMAGE} as librealsense-builder

ENV DEBIAN_FRONTEND noninteractive

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
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

RUN apt-get install python3.9-dev -y

WORKDIR /usr/src
RUN echo $PWD
RUN ls $PWD
RUN git clone https://github.com/IntelRealSense/librealsense.git
# Build and install
RUN ls $PWD
RUN cd /usr/src/librealsense \
    && mkdir build && cd build \
    && cmake ../ \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_PYTHON_BINDINGS:bool=true \
    -DCMAKE_BUILD_TYPE=release \
    -DFORCE_RSUSB_BACKEND=false
RUN cd /usr/src/librealsense/build && make clean && make -j$(($(nproc)-1)) all && make install

######################################
#   librealsense Base Image Stage    #
######################################

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
RUN addgroup --gid $GROUP_ID $USER_NAME --force-badname
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME --force-badname
RUN usermod -a -G video $USER_NAME

RUN echo $USER_NAME:$USER_NAME|chpasswd
RUN adduser ${USER_NAME} sudo
RUN echo '${USER_NAME} ALL=NOPASSWD: ALL' >> /etc/sudoers

#RUN echo '${USER_NAME} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Copy binaries from builder stage
#COPY --from=librealsense-builder /opt/librealsense /usr/local/
#COPY --from=librealsense-builder /usr/lib/python3/dist-packages/pyrealsense2 /usr/lib/python3.8/dist-packages/
#COPY --from=librealsense-builder /usr/src/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib

# Install dep packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    libusb-1.0-0 \
    udev \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    python3-pyqt5*

RUN apt-get install vim -y 
RUN apt-get install tmux -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install ffmpeg -y
RUN apt-get install libx264-dev -y
RUN apt-get install x11-xserver-utils -y
RUN apt-get install x11-apps -y
RUN apt-get install -y libxinerama-dev libxcursor-dev
RUN apt-get install v4l-utils -y


RUN apt-get autoremove -y
RUN apt-get clean

#########################
#    python setting     #
#########################
RUN python3 -m pip install --upgrade --no-cache-dir setuptools pip

# RUN python3 -m pip install --no-cache-dir opencv-python
# in Dockerfile, can't use cuda to RUN python3 -m pip install opencv-contrib-python
RUN python3 -m pip install --no-cache-dir --upgrade Cython
RUN python3 -m pip install --no-cache-dir scipy
RUN python3 -m pip install --no-cache-dir tqdm
RUN pip3 install -U easydict
RUN pip3 install -U onnxruntime
RUN pip3 install -U onnx-simplifier==0.3.5
RUN pip3 install -U seaborn
RUN pip3 install -U loguru
RUN pip3 install -U numpy==1.22.1 # same setting trt
RUN pip3 install -U pyyaml

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
RUN git clone https://github.com/pdh930105/yolov7-face.git

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV PATH=/usr/local/cuda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64:
ENV DEBIAN_FRONTEND=noninteractive




