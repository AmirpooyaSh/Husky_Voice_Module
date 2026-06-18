FROM nvcr.io/nvidia/cuda:10.2-runtime-ubuntu18.04

# docker build -t vox-profile-cu102:latest .

ENV DEBIAN_FRONTEND=noninteractive

# Basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    ca-certificates \
    build-essential \
    libportaudio2 \
    portaudio19-dev \
    libasound2-dev \
    alsa-utils \
    pulseaudio-utils \
    libsndfile1 \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libice6 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda

RUN wget --progress=dot:giga --tries=5 --timeout=30 \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh

ENV PATH=${CONDA_DIR}/bin:${PATH}

# Create Vox-Profile conda environment
RUN conda create -y -n vox_profile python=3.8 && \
    conda clean -afy

ENV PATH=${CONDA_DIR}/envs/vox_profile/bin:${CONDA_DIR}/bin:${PATH}

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Install CUDA 10.2-compatible Python packages
COPY requirements-cu102.txt /tmp/requirements-cu102.txt
RUN pip install --no-cache-dir -r /tmp/requirements-cu102.txt

# Clone Vox-Profile
RUN git clone https://github.com/tiantiaf0627/vox-profile-release.git /opt/vox-profile-release

# Install Vox-Profile without pulling its original CUDA 12 requirements
RUN cd /opt/vox-profile-release && \
    pip install -e . --no-deps

ENV PYTHONPATH=/opt/vox-profile-release:${PYTHONPATH}

WORKDIR /workspace

CMD ["/bin/bash"]
