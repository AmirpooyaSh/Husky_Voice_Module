# Whisper Docker Setup for ROS Speech-to-Text

This guide explains how to install, start, auto-start, verify, and change the Whisper Docker server used by the ROS Noetic speech-to-text node.

The ROS Noetic Docker image, ROS launch file, and Python ROS node are assumed to already exist and work. This file only covers the Whisper Docker container.

---

## 1. System assumptions

This setup assumes:

- The computer has an NVIDIA GPU.
- NVIDIA driver and NVIDIA Container Toolkit are already installed.
- Docker is already installed.
- The ROS Noetic Docker container is already built.
- The ROS speech-to-text launch file and Python node already exist.
- The ROS node sends audio to the Whisper API at:

```text
http://127.0.0.1:9000/v1/audio/transcriptions
```

The Whisper Docker image used here is:

```text
hwdsl2/whisper-server:cuda
```

The container name used here is:

```text
whisper
```

---

## 2. Make Docker work without sudo

Check whether Docker works without `sudo`:

```bash
docker ps
```

If you get a permission error, run:

```bash
sudo groupadd docker 2>/dev/null || true
sudo usermod -aG docker $USER
newgrp docker
```

Then test again:

```bash
docker ps
```

If it still fails, log out and log back in, then test again:

```bash
docker ps
```

---

## 3. Check GPU access

Confirm the host can see the NVIDIA GPU:

```bash
nvidia-smi
```

Then confirm Docker can use the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu20.04 nvidia-smi
```

If this fails, fix NVIDIA Container Toolkit before continuing.

---

## 4. Pull the Whisper Docker image

Pull the CUDA-enabled Whisper server image:

```bash
docker pull hwdsl2/whisper-server:cuda
```

This image provides an OpenAI-compatible transcription API.

The API endpoint used by the ROS node is:

```text
POST /v1/audio/transcriptions
```

The exposed port is:

```text
9000
```

---

## 5. Start the Whisper Docker container

Recommended starting model for short robot voice commands:

```text
base.en
```

Start the container:

```bash
docker run -d \
  --name whisper \
  --restart unless-stopped \
  --gpus all \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_MODEL=base.en \
  -e WHISPER_LANGUAGE=en \
  -e WHISPER_COMPUTE_TYPE=float16 \
  -e WHISPER_BEAM=1 \
  -v whisper-data:/var/lib/whisper \
  -p 9000:9000 \
  hwdsl2/whisper-server:cuda
```

Important options:

```text
--name whisper
```

Names the container `whisper`.

```text
--restart unless-stopped
```

Makes the container start automatically when the computer starts, unless it was manually stopped.

```text
--gpus all
```

Allows the container to use the NVIDIA GPU.

```text
WHISPER_DEVICE=cuda
```

Runs Whisper on GPU.

```text
WHISPER_MODEL=base.en
```

Sets the actual Whisper backend model.

```text
WHISPER_LANGUAGE=en
```

Uses English transcription.

```text
WHISPER_COMPUTE_TYPE=float16
```

Uses FP16 GPU inference.

```text
WHISPER_BEAM=1
```

Uses faster decoding, which is useful for short commands.

```text
-v whisper-data:/var/lib/whisper
```

Stores downloaded model files in a persistent Docker volume.

```text
-p 9000:9000
```

Exposes the Whisper API on port `9000`.

---

## 6. Wait for the server to finish loading

After starting the container, the server may need time to start.

On first run, it may also download the Whisper model. During this time, the ROS node may show errors like:

```text
Connection refused
```

or:

```text
Connection reset by peer
```

This can happen while the server is still loading.

Check logs:

```bash
docker logs -f whisper
```

Wait until the server is fully running before launching the ROS node.

In a separate terminal, test:

```bash
curl http://127.0.0.1:9000/docs
```

If this returns the API documentation page, the server is reachable.

You can also test:

```bash
curl http://127.0.0.1:9000/v1/models
```

---

## 7. Check if Whisper is running

List running containers:

```bash
docker ps
```

You should see something like:

```text
whisper   hwdsl2/whisper-server:cuda   0.0.0.0:9000->9000/tcp
```

If the container exists but is stopped:

```bash
docker ps -a
```

Start it again:

```bash
docker start whisper
```

Then check:

```bash
docker ps
curl http://127.0.0.1:9000/docs
```

---

## 8. Test transcription manually

Record a short test file.

Example using the Razer Kiyo Pro mic:

```bash
arecord -D plughw:3,0 -f S16_LE -c 1 -r 16000 -d 5 test.wav
```

Example using the motherboard mic input:

```bash
arecord -D plughw:2,0 -f S16_LE -c 1 -r 16000 -d 5 test.wav
```

If you are not sure which device to use:

```bash
arecord -l
```

Then send the file to Whisper:

```bash
curl http://127.0.0.1:9000/v1/audio/transcriptions \
  -F file=@test.wav \
  -F model=whisper-1 \
  -F language=en
```

Expected output:

```json
{
  "text": "stop the robot"
}
```

---

## 9. ROS launch file reminder

The ROS node should send requests to:

```text
http://127.0.0.1:9000/v1/audio/transcriptions
```

The ROS launch file should use:

```xml
<param name="whisper_url" value="http://127.0.0.1:9000/v1/audio/transcriptions" />
<param name="model" value="whisper-1" />
```

Do not put this in the ROS launch file:

```xml
<param name="model" value="base.en" />
```

For this Docker image, the ROS-side API model should remain:

```text
whisper-1
```

The actual Whisper backend model is changed through the Docker environment variable:

```text
WHISPER_MODEL
```

---

## 10. See which Whisper model is currently being used

Check the active backend model:

```bash
docker exec whisper printenv WHISPER_MODEL
```

Example output:

```text
base.en
```

Check all Whisper-related environment variables:

```bash
docker inspect whisper --format '{{range .Config.Env}}{{println .}}{{end}}' | grep WHISPER
```

---

## 11. Change the Whisper model

To change the Whisper backend model, remove and recreate the container with a different `WHISPER_MODEL`.

The persistent model/cache volume is:

```text
whisper-data
```

Removing the container does not delete this volume.

---

### Option A: Use `tiny.en` for fastest response

```bash
docker rm -f whisper
```

```bash
docker run -d \
  --name whisper \
  --restart unless-stopped \
  --gpus all \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_MODEL=tiny.en \
  -e WHISPER_LANGUAGE=en \
  -e WHISPER_COMPUTE_TYPE=float16 \
  -e WHISPER_BEAM=1 \
  -v whisper-data:/var/lib/whisper \
  -p 9000:9000 \
  hwdsl2/whisper-server:cuda
```

---

### Option B: Use `base.en` for balanced speed and accuracy

```bash
docker rm -f whisper
```

```bash
docker run -d \
  --name whisper \
  --restart unless-stopped \
  --gpus all \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_MODEL=base.en \
  -e WHISPER_LANGUAGE=en \
  -e WHISPER_COMPUTE_TYPE=float16 \
  -e WHISPER_BEAM=1 \
  -v whisper-data:/var/lib/whisper \
  -p 9000:9000 \
  hwdsl2/whisper-server:cuda
```

---

### Option C: Use `small.en` for better accuracy but slower response

```bash
docker rm -f whisper
```

```bash
docker run -d \
  --name whisper \
  --restart unless-stopped \
  --gpus all \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_MODEL=small.en \
  -e WHISPER_LANGUAGE=en \
  -e WHISPER_COMPUTE_TYPE=float16 \
  -e WHISPER_BEAM=1 \
  -v whisper-data:/var/lib/whisper \
  -p 9000:9000 \
  hwdsl2/whisper-server:cuda
```

---

## 12. Recommended model choices for robot commands

For short command recognition, use:

```text
tiny.en   = fastest, lower accuracy
base.en   = good balance, recommended starting point
small.en  = better accuracy, slower
```

Recommended default:

```text
WHISPER_MODEL=base.en
WHISPER_BEAM=1
WHISPER_COMPUTE_TYPE=float16
```

If latency is too high:

```text
WHISPER_MODEL=tiny.en
```

If accuracy is not good enough:

```text
WHISPER_MODEL=small.en
```

---

## 13. Auto-start behavior after reboot

The container should auto-start because it was created with:

```bash
--restart unless-stopped
```

Verify the restart policy:

```bash
docker inspect whisper --format '{{.HostConfig.RestartPolicy.Name}}'
```

Expected output:

```text
unless-stopped
```

If needed, update it:

```bash
docker update --restart unless-stopped whisper
```

After reboot:

```bash
docker ps
```

If the container is not running but exists:

```bash
docker ps -a
docker start whisper
```

Important:

If you manually stop the container using:

```bash
docker stop whisper
```

then Docker may keep it stopped even after reboot because the policy is `unless-stopped`.

Start it manually again:

```bash
docker start whisper
```

---

## 14. Common errors and fixes

### Error: Connection refused

Example:

```text
Failed to establish a new connection: [Errno 111] Connection refused
```

Meaning:

Nothing is listening on port `9000`.

Fix:

```bash
docker ps
docker ps -a
docker start whisper
curl http://127.0.0.1:9000/docs
```

If the container does not exist, recreate it using the `docker run` command above.

---

### Error: Connection reset by peer

Example:

```text
ConnectionResetError(104, 'Connection reset by peer')
```

Meaning:

The ROS node reached the Whisper server, but the server closed the connection.

Common reasons:

- The Whisper container is still starting.
- The model is still downloading.
- The model is still loading into GPU memory.
- The server is restarting after the first request.

Fix:

```bash
docker logs -f whisper
```

Wait until the server is ready, then test:

```bash
curl http://127.0.0.1:9000/docs
```

Then manually test transcription:

```bash
curl http://127.0.0.1:9000/v1/audio/transcriptions \
  -F file=@test.wav \
  -F model=whisper-1 \
  -F language=en
```

---

### Error: ROS node uses wrong model name

If the ROS launch file has:

```xml
<param name="model" value="base.en" />
```

change it to:

```xml
<param name="model" value="whisper-1" />
```

The Docker container uses `WHISPER_MODEL=base.en`, but the API request should use `model=whisper-1`.

---

### Error: CUDA or GPU issue

Check:

```bash
nvidia-smi
docker logs --tail=200 whisper
```

If needed, recreate the container with a safer compute type:

```bash
docker rm -f whisper
```

```bash
docker run -d \
  --name whisper \
  --restart unless-stopped \
  --gpus all \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_MODEL=base.en \
  -e WHISPER_LANGUAGE=en \
  -e WHISPER_COMPUTE_TYPE=int8_float16 \
  -e WHISPER_BEAM=1 \
  -v whisper-data:/var/lib/whisper \
  -p 9000:9000 \
  hwdsl2/whisper-server:cuda
```

---

## 15. Useful management commands

Start Whisper:

```bash
docker start whisper
```

Stop Whisper:

```bash
docker stop whisper
```

Restart Whisper:

```bash
docker restart whisper
```

View logs:

```bash
docker logs -f whisper
```

View recent logs:

```bash
docker logs --tail=100 whisper
```

Remove the container:

```bash
docker rm -f whisper
```

List volumes:

```bash
docker volume ls
```

Inspect the model/cache volume:

```bash
docker volume inspect whisper-data
```

Remove the model/cache volume only if you want to delete downloaded models:

```bash
docker volume rm whisper-data
```

Normally, do not remove `whisper-data`.