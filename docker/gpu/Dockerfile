ARG BASE_VERSION=22.10
FROM nvcr.io/nvidia/pytorch:${BASE_VERSION}-py3 as base

ENV DEBIAN_FRONTEND noninteractive

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
RUN apt-get clean
# this line change to ubuntu archive
RUN apt-get update -y
RUN apt-get install vim -y
RUN apt-get install tmux -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install ffmpeg -y
RUN apt-get install libx264-dev -y
RUN apt-get install sudo -y
RUN apt-get install x11-xserver-utils -y
RUN apt-get install x11-apps -y

RUN addgroup --gid $GROUP_ID $USER_NAME
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME
RUN echo '${USER_NAME}:{USER_NAME}'|chpasswd
RUN adduser pdh sudo
RUN echo '${USER_NAME} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN pip install timm
RUN pip install yacs
RUN pip install pyhocon
RUN pip install -U fvcore
RUN pip install -U transformers
RUN pip install pytorch-lightning
RUN pip install -U easydict
RUN pip install -U pycuda
RUN pip install -U onnxruntime
RUN pip install -U seaborn
RUN pip3 install -U loguru
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
RUN git clone https://github.com/pdh930105/yolov7-face.git

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

