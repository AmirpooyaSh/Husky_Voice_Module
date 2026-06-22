#!/usr/bin/env bash
set -euo pipefail

docker build -t local/openai-whisper-cuda126:latest .
