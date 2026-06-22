# CUDA 12.6-Compatible OpenAI Whisper Docker Server

This replaces the `hwdsl2/whisper-server:cuda` container that starts with CUDA 12.9.1 and fails on machines whose NVIDIA driver only supports CUDA 12.6.

The server keeps the same API used by the existing ROS/test script:

```text
http://127.0.0.1:9000/v1/audio/transcriptions
```

The request-side model name remains:

```text
whisper-1
```

The real backend model is controlled by:

```text
WHISPER_MODEL=base.en
```

This version uses the official OpenAI Whisper Python package inside an NVIDIA CUDA 12.6 container, with a small FastAPI wrapper to expose an OpenAI-compatible transcription endpoint.

---

## 1. Stop the old broken container

```bash
docker rm -f whisper 2>/dev/null || true
```

---

## 2. Build the new image

From this directory:

```bash
./build_whisper_cuda126.sh
```

Equivalent manual command:

```bash
docker build -t local/openai-whisper-cuda126:latest .
```

---

## 3. Run the new server

```bash
./run_whisper_cuda126.sh
```

This creates a container named `whisper`, exposes port `9000`, uses the GPU, and auto-starts after reboot unless manually stopped.

---

## 4. Watch logs

```bash
docker logs -f whisper
```

On first run, Whisper will download the model. The persistent cache volume is:

```text
whisper-openai-cache
```

---

## 5. Test the API

```bash
curl http://127.0.0.1:9000/health
```

```bash
curl http://127.0.0.1:9000/v1/models
```

Record a short test file:

```bash
arecord -D plughw:2,0 -f S16_LE -c 1 -r 16000 -d 5 test.wav
```

Then transcribe:

```bash
curl http://127.0.0.1:9000/v1/audio/transcriptions \
  -F file=@test.wav \
  -F model=whisper-1 \
  -F language=en
```

Expected response:

```json
{"text":"stop the robot"}
```

---

## 6. Use your existing Python test file

Your existing `live_whisper.py` should keep working because this server preserves:

```text
WHISPER_URL=http://127.0.0.1:9000/v1/audio/transcriptions
API_MODEL=whisper-1
```

Run:

```bash
python3 live_whisper.py
```

---

## 7. Change model for speed/accuracy

Fastest:

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
  -e WHISPER_FP16=true \
  -e WHISPER_BEAM=1 \
  -e REQUEST_MODEL_NAME=whisper-1 \
  -v whisper-openai-cache:/models \
  -p 9000:9000 \
  local/openai-whisper-cuda126:latest
```

Balanced default:

```text
WHISPER_MODEL=base.en
```

Better accuracy but slower:

```text
WHISPER_MODEL=small.en
```

---

## 8. If CUDA still fails

Check Docker GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

Check the container logs:

```bash
docker logs --tail=200 whisper
```

If GPU still fails but you need a CPU fallback, run without `--gpus all` and set:

```bash
-e WHISPER_DEVICE=cpu \
-e WHISPER_FP16=false
```

CPU will be slower, but it avoids CUDA completely.
