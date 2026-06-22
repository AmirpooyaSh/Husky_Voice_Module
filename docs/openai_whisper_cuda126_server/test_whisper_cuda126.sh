#!/usr/bin/env bash
set -euo pipefail

curl http://127.0.0.1:9000/health
printf '\n\n'
curl http://127.0.0.1:9000/v1/models
printf '\n\n'

if [[ -f test.wav ]]; then
  curl http://127.0.0.1:9000/v1/audio/transcriptions \
    -F file=@test.wav \
    -F model=whisper-1 \
    -F language=en
  printf '\n'
else
  echo "No test.wav found. Record one first, for example:"
  echo "arecord -D plughw:2,0 -f S16_LE -c 1 -r 16000 -d 5 test.wav"
fi
