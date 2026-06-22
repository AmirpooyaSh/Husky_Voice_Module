#!/usr/bin/env bash
set -euo pipefail

# Change the model here, or override it from the terminal:
#   OLLAMA_MODEL="llama3.1:8b" ./run_ollama_command_benchmark.sh
#   OLLAMA_MODEL="qwen2.5:0.5b" ./run_ollama_command_benchmark.sh

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:0.5b}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
DATASET="${DATASET:-robot_command_phrases_tricky.tsv}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
LIMIT="${LIMIT:-0}"
WARMUP="${WARMUP:-1}"

echo "=== Ollama tricky command benchmark ==="
echo "Model:    ${OLLAMA_MODEL}"
echo "Base URL: ${OLLAMA_BASE_URL}"
echo "Dataset:  ${DATASET}"
echo "Limit:    ${LIMIT}"
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found."
    exit 1
fi

if ! python3 -c "import openai" >/dev/null 2>&1; then
    echo "Installing Python openai package for current user..."
    python3 -m pip install --user openai
fi

if ! curl -fsS "http://localhost:11434/api/tags" >/dev/null; then
    echo "ERROR: Ollama is not reachable at http://localhost:11434"
    echo "Check:"
    echo "  docker ps | grep ollama"
    echo "  docker logs ollama"
    exit 1
fi

if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx "ollama"; then
    if ! docker exec ollama ollama list | awk '{print $1}' | grep -qx "${OLLAMA_MODEL}"; then
        echo "Model '${OLLAMA_MODEL}' is not installed. Pulling it now..."
        docker exec ollama ollama pull "${OLLAMA_MODEL}"
    fi
fi

ARGS=(--model "${OLLAMA_MODEL}" --base-url "${OLLAMA_BASE_URL}" --dataset "${DATASET}" --timeout "${REQUEST_TIMEOUT}")

if [[ "${WARMUP}" == "1" ]]; then
    ARGS+=(--warmup)
fi

if [[ "${LIMIT}" != "0" ]]; then
    ARGS+=(--limit "${LIMIT}")
fi

python3 benchmark_ollama_robot_commands.py "${ARGS[@]}"
