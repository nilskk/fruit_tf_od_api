FROM tensorflow/tensorflow:nightly-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow/

# Copy this version of of the model garden into the image
COPY --chown=tensorflow . /home/tensorflow/fruit_tf_od_api

# Compile protobuf configs
RUN (cd /home/tensorflow/fruit_tf_od_api/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/fruit_tf_od_api/

RUN mkdir -p models/pretrained_models/
RUN mkdir -p models/exported_models/

RUN (curl http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz \
    | tar -xzC models/pretrained_models)
RUN (curl http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz \
    | tar -xzC models/pretrained_models)

RUN mkdir -p data/tfrecords/

RUN cp object_detection/packages/tf2_nightly/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

RUN python -m pip install -U pip
RUN python -m pip install .

ENV TF_CPP_MIN_LOG_LEVEL 3

