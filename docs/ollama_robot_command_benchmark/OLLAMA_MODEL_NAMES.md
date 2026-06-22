# Ollama model names to try with the benchmark

Use any of these like:

```bash
OLLAMA_MODEL="qwen2.5:0.5b" ./run_ollama_command_benchmark.sh
```

## Best first tests for your robot-command classifier

```text
qwen2.5:0.5b
qwen2.5:1.5b
llama3.2:1b
llama3.2:3b
qwen3:0.6b
qwen3:1.7b
gemma3:1b
phi3:3.8b
llama3.1:8b
```

## Other common Ollama text/chat models

```text
qwen2.5:3b
qwen2.5:7b
qwen2.5:14b
qwen2.5:32b
qwen2.5:72b

qwen3:4b
qwen3:8b
qwen3:14b
qwen3:30b
qwen3:32b
qwen3:235b

llama3.1:8b
llama3.1:70b
llama3.1:405b

llama3.2:1b
llama3.2:3b

llama3:8b
llama3:70b

gemma3:270m
gemma3:1b
gemma3:4b
gemma3:12b
gemma3:27b

gemma2:2b
gemma2:9b
gemma2:27b

gemma4:e2b
gemma4:e4b
gemma4:12b
gemma4:26b
gemma4:31b

deepseek-r1:1.5b
deepseek-r1:7b
deepseek-r1:8b
deepseek-r1:14b
deepseek-r1:32b
deepseek-r1:70b
deepseek-r1:671b

mistral:7b
phi3:3.8b
phi3:14b
qwen2.5-coder:0.5b
qwen2.5-coder:1.5b
qwen2.5-coder:3b
qwen2.5-coder:7b
qwen2.5-coder:14b
qwen2.5-coder:32b
```

## See what is installed locally

```bash
docker exec -it ollama ollama list
```

## Search/pull models

Ollama's online library is updated over time, so the truly complete list is the official library page:

```text
https://ollama.com/library
```
