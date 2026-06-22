#!/usr/bin/env bash
set -euo pipefail

# build_ollama.sh
#
# Ollama does not need a custom Dockerfile build for this workflow.
# This script "builds/prepares" the environment by:
#   1) checking Docker,
#   2) pulling/updating the official Ollama image,
#   3) optionally checking host NVIDIA visibility.
#
# Usage:
#   ./build_ollama.sh
#
# Optional environment variables:
#   IMAGE=ollama/ollama:latest ./build_ollama.sh
#   CHECK_GPU=0 ./build_ollama.sh

IMAGE="${IMAGE:-ollama/ollama:latest}"
CHECK_GPU="${CHECK_GPU:-1}"

echo "=== Ollama Docker prepare/build script ==="
echo "Image: ${IMAGE}"
echo

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker command not found."
    echo "Install Docker first, then re-run this script."
    exit 1
fi

echo "[1/4] Docker version:"
docker --version
echo

echo "[2/4] Checking Docker daemon access..."
if ! docker ps >/dev/null 2>&1; then
    echo "ERROR: Docker is installed, but this user cannot run docker ps."
    echo "Fix Docker permissions or run with sudo."
    echo
    echo "Common fix:"
    echo "  sudo groupadd docker 2>/dev/null || true"
    echo "  sudo usermod -aG docker \$USER"
    echo "  newgrp docker"
    exit 1
fi
echo "Docker access OK."
echo

echo "[3/4] Pulling/updating Ollama image..."
docker pull "${IMAGE}"
echo

if [[ "${CHECK_GPU}" == "1" ]]; then
    echo "[4/4] Checking host NVIDIA GPU visibility..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi || true
        echo
        echo "Host NVIDIA check finished."
        echo "The run script will request GPU access with: --gpus all"
    else
        echo "nvidia-smi not found on host."
        echo "If this is a CPU-only machine, that is fine."
        echo "If this should use GPU, install/fix the NVIDIA driver and NVIDIA Container Toolkit."
    fi
else
    echo "[4/4] Skipping GPU check because CHECK_GPU=0"
fi

echo
echo "Done. Next:"
echo "  ./run_ollama.sh"
