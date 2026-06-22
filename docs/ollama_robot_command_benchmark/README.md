# Tricky Ollama Robot Command Benchmark

This is a harder 400-example benchmark for short robot-command classification.

Labels:

- `STOP`
- `SLOW_DOWN`
- `PROCEED`
- `UNKNOWN`

The dataset intentionally includes tricky phrases such as:

```text
do not proceed          -> STOP
do not stop             -> PROCEED
don't stop, slow down   -> SLOW_DOWN
stop the recording only -> UNKNOWN
the word stop           -> UNKNOWN
```

Run:

```bash
chmod +x *.sh
./run_ollama_command_benchmark.sh
```

Choose a model:

```bash
OLLAMA_MODEL="qwen2.5:0.5b" ./run_ollama_command_benchmark.sh
OLLAMA_MODEL="qwen2.5:1.5b" ./run_ollama_command_benchmark.sh
OLLAMA_MODEL="llama3.2:1b" ./run_ollama_command_benchmark.sh
OLLAMA_MODEL="llama3.1:8b" ./run_ollama_command_benchmark.sh
```

Quick debug:

```bash
LIMIT=20 ./run_ollama_command_benchmark.sh
```

The script saves a CSV with prediction, correctness, response time, confidence, raw tool-call arguments, and errors.
