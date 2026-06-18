# Live Whisper ASR in Docker: Ubuntu 20.04 + CUDA 11.8

This README documents the setup used to run live microphone speech-to-text with OpenAI Whisper inside a Docker container using:

- Ubuntu 20.04 container
- CUDA 11.8
- NVIDIA Container Toolkit
- PyTorch CUDA 11.8
- OpenAI Whisper
- Live microphone input through ALSA/PulseAudio device passthrough

The expected project directory is:

```bash
~/Husky_Voice_Module/docker/whisper_cuda118_ubuntu20
```

This directory should contain at least:

```text
Dockerfile
live_whisper_mic.py
README.md
```

> Important Jetson note: this Dockerfile is intended for an x86_64 NVIDIA desktop/workstation, such as an RTX 3080 machine. It is not a Jetson TX2/L4T Dockerfile. Jetson devices need JetPack/L4T-compatible base images, not standard `nvidia/cuda:11.8.0-...-ubuntu20.04` images.

---

## 0. Go to the project directory

```bash
cd ~/Husky_Voice_Module/docker/whisper_cuda118_ubuntu20
```

Check that the required files are present:

```bash
ls -lh
```

Expected:

```text
Dockerfile
live_whisper_mic.py
```

---

## 1. Check host NVIDIA driver

Run this on the host, not inside Docker:

```bash
nvidia-smi
```

You should see your NVIDIA GPU and driver information. For the working desktop setup, the host showed CUDA 11.8 support through the NVIDIA driver.

If this command fails on the host, Docker GPU support will not work either.

---

## 2. Install Docker Engine on Ubuntu

Skip this section if Docker is already installed and `docker --version` works.

Remove conflicting old Docker packages:

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  sudo apt-get remove -y $pkg 2>/dev/null || true
done
```

Install prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
```

Add Docker's official GPG key:

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

Add the Docker apt repository:

```bash
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF_DOCKER
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF_DOCKER
```

Install Docker:

```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Start and enable Docker:

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

Test Docker:

```bash
sudo docker run --rm hello-world
```

Optional: allow running Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Test again without `sudo`:

```bash
docker run --rm hello-world
```

---

## 3. Install NVIDIA Container Toolkit

Skip this section if this already works:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

Install NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Configure Docker to use the NVIDIA runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Critical fix used for this setup:

```bash
grep -n "no-cgroups" /etc/nvidia-container-runtime/config.toml
```

If you see:

```text
no-cgroups = true
```

edit the file:

```bash
sudo nano /etc/nvidia-container-runtime/config.toml
```

Change it to:

```text
no-cgroups = false
```

Then restart Docker:

```bash
sudo systemctl restart docker
```

Test NVIDIA GPU access in Docker:

```bash
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  nvidia/cuda:11.8.0-base-ubuntu20.04 \
  nvidia-smi
```

Expected: the container prints the same GPU table as host `nvidia-smi`.

---

## 4. Dockerfile

The project directory should contain this Dockerfile:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

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

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install -U \
    openai-whisper \
    sounddevice \
    soundfile \
    numpy \
    scipy \
    setuptools-rust

WORKDIR /workspace

CMD ["/bin/bash"]
```

---

## 5. Build the Docker image

From the project directory:

```bash
cd ~/Husky_Voice_Module/docker/whisper_cuda118_ubuntu20
```

Build:

```bash
docker build -t whisper-ubuntu20-cuda118 .
```

---

## 6. Test the built image with GPU

First test `nvidia-smi` inside the image:

```bash
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  whisper-ubuntu20-cuda118 \
  nvidia-smi
```

Then test PyTorch CUDA:

```bash
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  whisper-ubuntu20-cuda118 \
  python - <<'PY'
import torch

print("torch version:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("CUDA tensor test passed:", y.shape)
PY
```

Expected output includes:

```text
cuda available: True
CUDA tensor test passed
```

---

## 7. Run the container with GPU and microphone access

Remove old container with the same name, if it exists:

```bash
docker rm -f whisper-live 2>/dev/null || true
```

Run the container:

```bash
docker run --rm -it \
  --name whisper-live \
  --runtime=nvidia \
  --gpus all \
  --net=host \
  --device /dev/snd \
  --group-add audio \
  -v "$PWD":/workspace \
  whisper-ubuntu20-cuda118
```

You should now be inside the container at:

```text
/workspace
```

---

## 8. Test CUDA inside the running container

Inside the container:

```bash
python - <<'PY'
import torch

print("torch version:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

Expected:

```text
cuda available: True
```

---

## 9. List microphone devices

Inside the container:

```bash
python live_whisper_mic.py --list-devices
```

In the tested setup, the working microphone was:

```text
18 AB13X USB Audio
```

If your mic index is different, replace `--mic 18` in the commands below.

---

## 10. Run live Whisper with the fastest model

Use `tiny.en` for the fastest English-only Whisper model:

```bash
python live_whisper_mic.py \
  --model tiny.en \
  --mic 18 \
  --device cuda \
  --rms-threshold 0.006 \
  --silence-ms 350 \
  --min-seconds 0.25 \
  --max-ms 3000
```

Expected startup lines:

```text
Loading Whisper model: tiny.en
Device: cuda
FP16: True
```

Speak a short command and pause. Example phrases:

```text
stop now
slow down
proceed
move forward
stay where you are
```

---

## 11. Slightly more accurate command

If `tiny.en` is too inaccurate, use `base.en`:

```bash
python live_whisper_mic.py \
  --model base.en \
  --mic 18 \
  --device cuda \
  --rms-threshold 0.006 \
  --silence-ms 500 \
  --min-seconds 0.3 \
  --max-ms 4000
```

---

## 12. Tuning microphone sensitivity

If it does not detect your voice, lower the RMS threshold:

```bash
--rms-threshold 0.003
```

If it triggers too often from noise, increase it:

```bash
--rms-threshold 0.01
```

If response is too slow after you stop talking, reduce silence wait:

```bash
--silence-ms 250
```

If it cuts you off too early, increase silence wait:

```bash
--silence-ms 600
```

---

## 13. Common fixes

### Problem: `Failed to initialize NVML: Unknown Error`

Check this file:

```bash
grep -n "no-cgroups" /etc/nvidia-container-runtime/config.toml
```

Set:

```text
no-cgroups = false
```

Then:

```bash
sudo systemctl restart docker
```

Test:

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

---

### Problem: container name already exists

```bash
docker rm -f whisper-live
```

Then run the container again.

---

### Problem: CUDA is still false in PyTorch

Inside the container:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

Interpretation:

```text
nvidia-smi fails        -> Docker GPU passthrough problem
torch.version.cuda None -> CPU-only PyTorch installed
torch CUDA 11.8 false   -> runtime/driver/toolkit issue
torch CUDA 11.8 true    -> GPU is ready
```

---

### Problem: wrong microphone

List devices:

```bash
python live_whisper_mic.py --list-devices
```

Try the correct device index:

```bash
python live_whisper_mic.py --model tiny.en --mic 18 --device cuda
```

If needed, try another mic index from the list.

---

## 14. One-command run after everything is built

From the host:

```bash
cd ~/Husky_Voice_Module/docker/whisper_cuda118_ubuntu20

docker rm -f whisper-live 2>/dev/null || true

docker run --rm -it \
  --name whisper-live \
  --runtime=nvidia \
  --gpus all \
  --net=host \
  --device /dev/snd \
  --group-add audio \
  -v "$PWD":/workspace \
  whisper-ubuntu20-cuda118
```

Then inside the container:

```bash
python live_whisper_mic.py \
  --model tiny.en \
  --mic 18 \
  --device cuda \
  --rms-threshold 0.006 \
  --silence-ms 350 \
  --min-seconds 0.25 \
  --max-ms 3000
```

---

## 15. Notes

- `tiny.en` is fastest but less accurate.
- `base.en` is slower but usually better for short command phrases.
- CUDA 10.2 is not recommended for RTX 30-series GPUs. It can produce errors such as `no suitable kernel image` or `no kernel image is available for execution on the device`.
- For RTX 3080, use CUDA 11.8 or newer. This README uses CUDA 11.8 because it works well with Ubuntu 20.04 containers.
