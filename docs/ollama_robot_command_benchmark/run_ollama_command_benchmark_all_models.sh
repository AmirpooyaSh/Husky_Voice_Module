#!/usr/bin/env bash
set -euo pipefail

# run_ollama_command_benchmark_all_models.sh
#
# Reads Ollama model names from models.txt and runs:
#   benchmark_ollama_robot_commands.py
# once per model.
#
# Expected files in the same directory:
#   models.txt
#   robot_command_phrases_tricky.tsv
#   benchmark_ollama_robot_commands.py
#
# Output:
#   benchmark_results/<model>.csv
#   benchmark_results/model_accuracy_summary.csv
#   benchmark_results/model_accuracy_summary.md
#
# models.txt format:
#   One model name per line.
#   Empty lines and lines starting with # are ignored.
#
# Example:
#   qwen2.5:0.5b
#   qwen2.5:1.5b
#   llama3.2:1b

MODELS_FILE="${MODELS_FILE:-models.txt}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
DATASET="${DATASET:-robot_command_phrases_tricky.tsv}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
LIMIT="${LIMIT:-0}"
WARMUP="${WARMUP:-1}"
RESULTS_DIR="${RESULTS_DIR:-benchmark_results}"

mkdir -p "${RESULTS_DIR}"

echo "=== Ollama multi-model command benchmark ==="
echo "Models file: ${MODELS_FILE}"
echo "Base URL:    ${OLLAMA_BASE_URL}"
echo "Dataset:     ${DATASET}"
echo "Results dir: ${RESULTS_DIR}"
echo "Limit:       ${LIMIT}"
echo

if [[ ! -f "${MODELS_FILE}" ]]; then
    echo "ERROR: Cannot find ${MODELS_FILE}"
    echo "Create it with one Ollama model per line, for example:"
    echo "  qwen2.5:0.5b"
    echo "  qwen2.5:1.5b"
    echo "  llama3.2:1b"
    exit 1
fi

if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: Cannot find dataset file: ${DATASET}"
    exit 1
fi

if [[ ! -f "benchmark_ollama_robot_commands.py" ]]; then
    echo "ERROR: Cannot find benchmark_ollama_robot_commands.py"
    exit 1
fi

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

# Convert a model name into a safe filename.
safe_model_name() {
    echo "$1" | sed 's/:/_/g' | sed 's#[^A-Za-z0-9_.-]#_#g'
}

TOTAL_MODELS=0
while IFS= read -r MODEL || [[ -n "${MODEL}" ]]; do
    # Trim leading/trailing whitespace.
    MODEL="$(echo "${MODEL}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

    # Skip empty lines and comments.
    if [[ -z "${MODEL}" || "${MODEL}" == \#* ]]; then
        continue
    fi

    TOTAL_MODELS=$((TOTAL_MODELS + 1))
done < "${MODELS_FILE}"

if [[ "${TOTAL_MODELS}" -eq 0 ]]; then
    echo "ERROR: ${MODELS_FILE} has no usable model names."
    exit 1
fi

echo "Found ${TOTAL_MODELS} model(s)."
echo

MODEL_INDEX=0
while IFS= read -r MODEL || [[ -n "${MODEL}" ]]; do
    MODEL="$(echo "${MODEL}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

    if [[ -z "${MODEL}" || "${MODEL}" == \#* ]]; then
        continue
    fi

    MODEL_INDEX=$((MODEL_INDEX + 1))
    SAFE_NAME="$(safe_model_name "${MODEL}")"
    OUTPUT_CSV="${RESULTS_DIR}/${SAFE_NAME}.csv"

    echo
    echo "======================================================================"
    echo "[${MODEL_INDEX}/${TOTAL_MODELS}] Benchmarking model: ${MODEL}"
    echo "Output CSV: ${OUTPUT_CSV}"
    echo "======================================================================"

    if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx "ollama"; then
        if ! docker exec ollama ollama list | awk '{print $1}' | grep -qx "${MODEL}"; then
            echo "Model '${MODEL}' is not installed. Pulling it now..."
            docker exec ollama ollama pull "${MODEL}"
        fi
    else
        echo "Could not check/pull models through Docker, but Ollama API is reachable. Continuing."
    fi

    ARGS=(
        --model "${MODEL}"
        --base-url "${OLLAMA_BASE_URL}"
        --dataset "${DATASET}"
        --timeout "${REQUEST_TIMEOUT}"
        --output "${OUTPUT_CSV}"
    )

    if [[ "${WARMUP}" == "1" ]]; then
        ARGS+=(--warmup)
    fi

    if [[ "${LIMIT}" != "0" ]]; then
        ARGS+=(--limit "${LIMIT}")
    fi

    python3 benchmark_ollama_robot_commands.py "${ARGS[@]}"

done < "${MODELS_FILE}"

echo
echo "======================================================================"
echo "Creating summary table..."
echo "======================================================================"

python3 summarize_ollama_model_results.py \
    --results-dir "${RESULTS_DIR}" \
    --models-file "${MODELS_FILE}" \
    --csv-output "${RESULTS_DIR}/model_accuracy_summary.csv" \
    --md-output "${RESULTS_DIR}/model_accuracy_summary.md"

echo
echo "Done."
echo
echo "Summary CSV:"
echo "  ${RESULTS_DIR}/model_accuracy_summary.csv"
echo
echo "Markdown table:"
echo "  ${RESULTS_DIR}/model_accuracy_summary.md"
echo
echo "Preview:"
cat "${RESULTS_DIR}/model_accuracy_summary.md"
