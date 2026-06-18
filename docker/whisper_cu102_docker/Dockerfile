FROM nvcr.io/nvidia/cuda:10.2-runtime-ubuntu18.04

# docker build -t whisper-cu102 .

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    ffmpeg \
    libportaudio2 \
    libasound2 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu102

RUN pip install openai-whisper sounddevice soundfile numpy

WORKDIR /workspace

CMD ["/bin/bash"]
