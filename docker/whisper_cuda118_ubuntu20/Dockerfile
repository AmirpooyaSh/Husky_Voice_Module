FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# docker build -t whisper-ubuntu20-cuda118 .

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    ffmpeg \
    alsa-utils \
    pulseaudio-utils \
    libasound2 \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda with Python 3.10
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch CUDA 11.8
RUN pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Whisper and microphone dependencies
RUN pip install -U \
    openai-whisper \
    sounddevice \
    soundfile \
    numpy \
    scipy \
    setuptools-rust

WORKDIR /workspace

CMD ["/bin/bash"]
