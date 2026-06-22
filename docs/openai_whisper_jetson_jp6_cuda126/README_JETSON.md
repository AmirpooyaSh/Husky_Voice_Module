# OpenAI Whisper API Server for Jetson Orin Nano Super / JetPack 6 / CUDA 12.6

This keeps the same API used by the desktop setup:

- Container name: `whisper`
- Endpoint: `http://127.0.0.1:9000/v1/audio/transcriptions`
- Request model name: `whisper-1`
- Backend model default: `base.en`

## Build

```bash
cd openai_whisper_jetson_jp6_cuda126
./build_jetson_whisper.sh
```

## Run

```bash
./run_jetson_whisper.sh
```

## Check logs

```bash
docker logs -f whisper
```

## Health check

```bash
curl http://127.0.0.1:9000/health
curl http://127.0.0.1:9000/v1/models
```

## Change model

Edit `run_jetson_whisper.sh` and change:

```bash
-e WHISPER_MODEL=base.en
```

to one of:

```bash
-e WHISPER_MODEL=tiny.en
-e WHISPER_MODEL=base.en
-e WHISPER_MODEL=small.en
```

For Jetson Orin Nano Super, start with `base.en`. Use `tiny.en` if latency is too high. Use `small.en` only if you have enough free memory.

## Important Jetson notes

Do not use the desktop Dockerfile based on `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` on Jetson. Jetson is ARM64 and uses JetPack/L4T/iGPU-compatible containers.

Do not install PyTorch using `https://download.pytorch.org/whl/cu126` inside this Jetson image. The base image already contains Jetson-compatible GPU PyTorch.
