# Ollama Docker Scripts

These scripts document the Ollama Docker workflow for the local robot command-classification setup.

## Files

- `build_ollama.sh`  
  Pulls/updates the official `ollama/ollama` Docker image and checks Docker/GPU availability.

- `run_ollama.sh`  
  Creates or recreates the `ollama` container with:
  - `--restart unless-stopped`
  - GPU access via `--gpus all`
  - persistent model volume: `ollama:/root/.ollama`
  - API port: `11434:11434`

- `test_ollama_api.sh`  
  Tests:
  - `http://127.0.0.1:11434/api/tags`
  - `http://127.0.0.1:11434/api/chat`
  - `http://127.0.0.1:11434/v1/chat/completions`

- `test_ollama_python_classifier.sh`  
  Runs your `test_ollama_command_classifier.py` script automatically with one command, then exits.

## Quick use

```bash
chmod +x *.sh
./build_ollama.sh
./run_ollama.sh
./test_ollama_api.sh
./test_ollama_python_classifier.sh
```

## Defaults

`run_ollama.sh` pulls:

```text
llama3.1:8b qwen2.5:0.5b
```

Reason:

- Your uploaded Python tool-call classifier uses `llama3.1:8b` by default.
- Your documentation recommends `qwen2.5:0.5b` as the low-latency model for command classification.

To pull only the fast model:

```bash
# qwen3:0.6b
MODELS="qwen3:0.6b" ./run_ollama.sh
```

To test a different model:

```bash
MODEL="qwen2.5:0.5b" ./test_ollama_api.sh
```

## CPU fallback

```bash
USE_GPU=0 ./run_ollama.sh
```

## Reuse existing container without recreating it

```bash
RECREATE=0 ./run_ollama.sh
```

## Important note

Removing the container does not remove downloaded models because the scripts use the named Docker volume:

```text
ollama:/root/.ollama
```

Do not run `docker volume rm ollama` unless you intentionally want to delete the downloaded models.
