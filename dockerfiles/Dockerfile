FROM nvidia/cuda:10.1-devel

RUN apt-get update

ENV CUDNN_VERSION 7.6.0.64

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get install -y libcupti-dev

RUN apt-get install -y python3.7

RUN apt-get install -y python3-pip

RUN apt-get install -y git

RUN apt-get install nano

RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN apt-get update

#COPY ./requirements.txt requirements.txt

RUN pip3 install pip --upgrade

RUN apt-get update

RUN pip3 install ipython

RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y protobuf-compiler
RUN apt-get install -y htop
RUN apt-get install -y screen

RUN apt-get update
RUN apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LC_ALL en_US.UTF-8 
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en

RUN pip3 install Cython
RUN pip3 install numpy

WORKDIR /code

RUN git clone https://github.com/nilskk/models.git
RUN git clone https://github.com/nilskk/fruit_tf_od_api.git

WORKDIR /code/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py .
RUN python3 -m pip install --editable .

WORKDIR /code/fruit_tf_od_api
RUN python3 -m pip install --editable .

RUN pip3 install tensorflow-addons==0.11.2
RUN pip3 install tensorflow==2.3.3



