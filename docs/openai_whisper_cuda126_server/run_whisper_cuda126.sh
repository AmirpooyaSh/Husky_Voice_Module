#!/usr/bin/env bash
set -euo pipefail

# Stop/remove the old crash-looping hwdsl2 container if it exists.
docker rm -f whisper 2>/dev/null || true

# Start the CUDA 12.6-compatible local OpenAI Whisper API server.
docker run -d \
  --name whisper \
  --restart unless-stopped \
  --gpus all \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_MODEL=base.en \
  -e WHISPER_LANGUAGE=en \
  -e WHISPER_FP16=true \
  -e WHISPER_BEAM=1 \
  -e REQUEST_MODEL_NAME=whisper-1 \
  -v whisper-openai-cache:/models \
  -p 9000:9000 \
  local/openai-whisper-cuda126:latest

echo "Container started. Follow logs with: docker logs -f whisper"
