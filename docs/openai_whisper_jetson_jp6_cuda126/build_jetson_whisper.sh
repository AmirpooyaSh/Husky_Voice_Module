#!/usr/bin/env bash
set -euo pipefail

docker build \
  -f Dockerfile.jetson-jp6-cu126 \
  -t local/openai-whisper-jetson-jp6-cu126:latest \
  .
