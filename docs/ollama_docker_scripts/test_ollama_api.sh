#!/usr/bin/env bash
set -euo pipefail

# test_ollama_api.sh
#
# Tests:
#   1) Ollama native /api/tags
#   2) Ollama native /api/chat
#   3) OpenAI-compatible /v1/chat/completions
#
# Usage:
#   ./test_ollama_api.sh
#
# Optional:
#   MODEL=qwen2.5:0.5b ./test_ollama_api.sh
#   MODEL=llama3.1:8b ./test_ollama_api.sh

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-11434}"
MODEL="${MODEL:-llama3.1:8b}"
PROMPT="${PROMPT:-Classify this robot command as STOP, SLOW_DOWN, PROCEED, or UNKNOWN: stop the robot right now}"

BASE_URL="http://${HOST}:${PORT}"

echo "=== Ollama API test ==="
echo "Base URL: ${BASE_URL}"
echo "Model:    ${MODEL}"
echo

echo "[1/4] Checking /api/tags..."
curl -fsS "${BASE_URL}/api/tags" | python3 -m json.tool || curl -fsS "${BASE_URL}/api/tags"
echo
echo

echo "[2/4] Ensuring model exists locally..."
if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx "ollama"; then
    if ! docker exec ollama ollama list | awk '{print $1}' | grep -qx "${MODEL}"; then
        echo "Model '${MODEL}' not found. Pulling it now..."
        docker exec ollama ollama pull "${MODEL}"
    else
        echo "Model '${MODEL}' is already installed."
    fi
else
    echo "Skipping docker-based model check because container name 'ollama' was not found."
fi
echo

echo "[3/4] Testing native Ollama /api/chat..."
curl -fsS "${BASE_URL}/api/chat" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"stream\": false,
      \"messages\": [
        {
          \"role\": \"user\",
          \"content\": \"${PROMPT}\"
        }
      ]
    }" | python3 -m json.tool || true
echo
echo

echo "[4/4] Testing OpenAI-compatible /v1/chat/completions..."
curl -fsS "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"stream\": false,
      \"temperature\": 0,
      \"max_tokens\": 80,
      \"messages\": [
        {
          \"role\": \"system\",
          \"content\": \"You classify robot commands. Return only STOP, SLOW_DOWN, PROCEED, or UNKNOWN.\"
        },
        {
          \"role\": \"user\",
          \"content\": \"${PROMPT}\"
        }
      ]
    }" | python3 -m json.tool || true

echo
echo "API test finished."
