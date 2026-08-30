FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3-pip \
        build-essential \
        cmake \
        pkg-config \
        libeigen3-dev \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --no-cache-dir --break-system-packages \
        numpy \
        scipy \
        open3d \
        opencv-python-headless \
        onnxruntime \
        pillow \
        matplotlib

WORKDIR /workspace
