FROM arm64v8/python:3.10-slim-bullseye

# docker build -f Dockerfile -t whisper-jetson-cpu .

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    alsa-utils \
    pulseaudio-utils \
    portaudio19-dev \
    libasound2 \
    libasound2-dev \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    pkg-config \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# CPU PyTorch for ARM64/aarch64.
# This avoids Jetson CUDA/PyTorch compatibility issues.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Whisper + microphone dependencies
RUN pip install -U \
    openai-whisper \
    sounddevice \
    soundfile \
    numpy \
    scipy

WORKDIR /workspace

CMD ["/bin/bash"]