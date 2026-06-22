FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/models \
    WHISPER_MODEL=base.en \
    WHISPER_DEVICE=cuda \
    WHISPER_LANGUAGE=en \
    WHISPER_FP16=true \
    WHISPER_BEAM=1 \
    REQUEST_MODEL_NAME=whisper-1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch CUDA 12.6 wheels. This keeps the container on CUDA 12.6 instead of pulling a CUDA 12.9 runtime.
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch \
    torchaudio

# Official OpenAI Whisper Python package + small FastAPI wrapper.
RUN python3 -m pip install \
    openai-whisper \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

WORKDIR /app
COPY openai_whisper_api_server.py /app/api_server.py

EXPOSE 9000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "9000"]
