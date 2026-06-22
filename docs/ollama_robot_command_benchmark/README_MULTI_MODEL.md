# Multi-model Ollama benchmark

Put these files in the same directory as:

```text
models.txt
robot_command_phrases_tricky.tsv
benchmark_ollama_robot_commands.py
```

Then run:

```bash
chmod +x run_ollama_command_benchmark_all_models.sh
./run_ollama_command_benchmark_all_models.sh
```

It will run all models in `models.txt`, then create:

```text
benchmark_results/model_accuracy_summary.csv
benchmark_results/model_accuracy_summary.md
```

The summary table has one row per model and columns for:

```text
STOP %
SLOW_DOWN %
PROCEED %
UNKNOWN %
Total %
Mean response time
N
```

For quick debugging:

```bash
LIMIT=20 ./run_ollama_command_benchmark_all_models.sh
```
