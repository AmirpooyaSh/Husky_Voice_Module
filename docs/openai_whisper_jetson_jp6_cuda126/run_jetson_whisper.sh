#!/usr/bin/env bash
set -euo pipefail

# Remove old whisper container if it exists.
docker rm -f whisper 2>/dev/null || true

# Jetson usually uses --runtime nvidia. --network host makes 127.0.0.1:9000 available on the Jetson.
docker run -d \
  --name whisper \
  --restart unless-stopped \
  --runtime nvidia \
  --network host \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_MODEL=base.en \
  -e WHISPER_LANGUAGE=en \
  -e WHISPER_FP16=true \
  -e WHISPER_BEAM=1 \
  -e REQUEST_MODEL_NAME=whisper-1 \
  -v whisper-openai-cache:/models \
  local/openai-whisper-jetson-jp6-cu126:latest

echo "Container started. Follow logs with: docker logs -f whisper"
echo "Health check: curl http://127.0.0.1:9000/health"
