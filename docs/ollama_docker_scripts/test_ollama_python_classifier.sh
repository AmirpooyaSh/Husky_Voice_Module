#!/usr/bin/env bash
set -euo pipefail

# test_ollama_python_classifier.sh
#
# Runs your OpenAI-client Python classifier script non-interactively if it is present.
# Your uploaded Python script uses:
#   base_url="http://localhost:11434/v1"
#   MODEL="llama3.1:8b"
#
# Usage:
#   ./test_ollama_python_classifier.sh
#
# It sends one test command, then q to exit.

SCRIPT="${SCRIPT:-./test_ollama_command_classifier.py}"
TEST_COMMAND="${TEST_COMMAND:-stop the robot right now}"

echo "=== Ollama Python classifier test ==="
echo "Script:       ${SCRIPT}"
echo "Test command: ${TEST_COMMAND}"
echo

if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: Cannot find ${SCRIPT}"
    echo "Place this script in the same folder as test_ollama_command_classifier.py, or run:"
    echo "  SCRIPT=/path/to/test_ollama_command_classifier.py ./test_ollama_python_classifier.sh"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found."
    exit 1
fi

python3 - <<'PY'
try:
    import openai  # noqa: F401
except Exception:
    raise SystemExit(
        "ERROR: Python package 'openai' is not installed in this environment.\n"
        "Install it with:\n"
        "  python3 -m pip install openai\n"
    )
PY

printf "%s\nq\n" "${TEST_COMMAND}" | python3 "${SCRIPT}"
