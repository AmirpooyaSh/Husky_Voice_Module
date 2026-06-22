#!/usr/bin/env bash
set -euo pipefail

# run_ollama.sh
#
# Starts/recreates the Ollama Docker container for local robot-command classification.
# It keeps the named Docker volume, so downloaded models survive container recreation.
#
# Usage:
#   ./run_ollama.sh
#
# Useful examples:
#   MODELS="qwen2.5:0.5b" ./run_ollama.sh
#   MODELS="llama3.1:8b qwen2.5:0.5b" ./run_ollama.sh
#   USE_GPU=0 ./run_ollama.sh
#   RECREATE=0 ./run_ollama.sh
#
# Defaults:
#   - Uses official ollama/ollama image
#   - Container name: ollama
#   - Port: 11434
#   - Volume: ollama:/root/.ollama
#   - Restart policy: unless-stopped
#   - Pulls llama3.1:8b because your Python tool-call test script uses it
#   - Also pulls qwen2.5:0.5b because it is your low-latency model candidate

IMAGE="${IMAGE:-ollama/ollama:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-ollama}"
VOLUME_NAME="${VOLUME_NAME:-ollama}"
HOST_PORT="${HOST_PORT:-11434}"
CONTAINER_PORT="${CONTAINER_PORT:-11434}"
USE_GPU="${USE_GPU:-1}"
RECREATE="${RECREATE:-1}"
PULL_MODELS="${PULL_MODELS:-1}"
MODELS="${MODELS:-llama3.1:8b qwen2.5:0.5b}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-10m}"

echo "=== Ollama Docker run script ==="
echo "Image:          ${IMAGE}"
echo "Container:      ${CONTAINER_NAME}"
echo "Volume:         ${VOLUME_NAME}:/root/.ollama"
echo "Port:           ${HOST_PORT}:${CONTAINER_PORT}"
echo "Use GPU:        ${USE_GPU}"
echo "Recreate:       ${RECREATE}"
echo "Pull models:    ${PULL_MODELS}"
echo "Models:         ${MODELS}"
echo "Keep alive:     ${OLLAMA_KEEP_ALIVE}"
echo

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker command not found."
    exit 1
fi

if ! docker ps >/dev/null 2>&1; then
    echo "ERROR: Docker daemon not reachable by this user."
    exit 1
fi

echo "[1/6] Ensuring Docker service is enabled if systemd is available..."
if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable docker >/dev/null 2>&1 || true
    sudo systemctl start docker >/dev/null 2>&1 || true
fi
echo "Docker service check finished."
echo

echo "[2/6] Pulling image..."
docker pull "${IMAGE}"
echo

EXISTS="0"
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    EXISTS="1"
fi

if [[ "${EXISTS}" == "1" ]]; then
    if [[ "${RECREATE}" == "1" ]]; then
        echo "[3/6] Removing existing container '${CONTAINER_NAME}' but keeping volume '${VOLUME_NAME}'..."
        docker rm -f "${CONTAINER_NAME}"
    else
        echo "[3/6] Reusing existing container '${CONTAINER_NAME}'..."
        docker start "${CONTAINER_NAME}" >/dev/null || true
        docker update --restart unless-stopped "${CONTAINER_NAME}" >/dev/null
    fi
else
    echo "[3/6] No existing container named '${CONTAINER_NAME}'."
fi
echo

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[4/6] Creating container..."
    GPU_ARGS=()
    if [[ "${USE_GPU}" == "1" ]]; then
        GPU_ARGS=(--gpus all)
    fi

    set +e
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --restart unless-stopped \
        "${GPU_ARGS[@]}" \
        -e "OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}" \
        -v "${VOLUME_NAME}:/root/.ollama" \
        -p "${HOST_PORT}:${CONTAINER_PORT}" \
        "${IMAGE}"
    RUN_STATUS=$?
    set -e

    if [[ "${RUN_STATUS}" != "0" && "${USE_GPU}" == "1" ]]; then
        echo
        echo "GPU container start failed. Retrying CPU-only so Ollama is at least usable..."
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        docker run -d \
            --name "${CONTAINER_NAME}" \
            --restart unless-stopped \
            -e "OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}" \
            -v "${VOLUME_NAME}:/root/.ollama" \
            -p "${HOST_PORT}:${CONTAINER_PORT}" \
            "${IMAGE}"
    elif [[ "${RUN_STATUS}" != "0" ]]; then
        echo "ERROR: Failed to start Ollama container."
        exit "${RUN_STATUS}"
    fi
else
    echo "[4/6] Container already running."
fi
echo

echo "[5/6] Waiting for Ollama API..."
for i in {1..60}; do
    if curl -fsS "http://127.0.0.1:${HOST_PORT}/api/tags" >/dev/null 2>&1; then
        echo "Ollama API is ready."
        break
    fi

    if [[ "$i" == "60" ]]; then
        echo "ERROR: Ollama API did not become ready."
        echo "Recent logs:"
        docker logs --tail=80 "${CONTAINER_NAME}" || true
        exit 1
    fi

    sleep 2
done
echo

if [[ "${PULL_MODELS}" == "1" ]]; then
    echo "[6/6] Pulling models..."
    for MODEL in ${MODELS}; do
        echo
        echo "Pulling model: ${MODEL}"
        docker exec "${CONTAINER_NAME}" ollama pull "${MODEL}"
    done
else
    echo "[6/6] Skipping model pulls because PULL_MODELS=0"
fi

echo
echo "Installed models:"
docker exec "${CONTAINER_NAME}" ollama list || true

echo
echo "Restart policy:"
docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' "${CONTAINER_NAME}"

echo
echo "Done. Test with:"
echo "  ./test_ollama_api.sh"
echo "  ./test_ollama_python_classifier.sh"
