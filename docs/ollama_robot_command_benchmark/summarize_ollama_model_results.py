#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


LABELS = ["STOP", "SLOW_DOWN", "PROCEED", "UNKNOWN"]


def safe_model_name(model):
    safe = model.replace(":", "_")
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in safe)
    return safe


def read_models(path):
    models = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            model = line.strip()
            if not model or model.startswith("#"):
                continue
            models.append(model)
    return models


def summarize_one(csv_path):
    counts = {label: {"total": 0, "correct": 0} for label in LABELS}
    total = 0
    correct = 0
    response_times = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected = row.get("expected", "")
            is_correct = str(row.get("correct", "")).strip().lower() == "true"

            if expected in counts:
                counts[expected]["total"] += 1
                if is_correct:
                    counts[expected]["correct"] += 1

            total += 1
            if is_correct:
                correct += 1

            try:
                response_times.append(float(row.get("response_time_sec", "")))
            except Exception:
                pass

    summary = {}

    for label in LABELS:
        label_total = counts[label]["total"]
        label_correct = counts[label]["correct"]
        if label_total:
            summary[label] = 100.0 * label_correct / label_total
        else:
            summary[label] = 0.0

    summary["TOTAL"] = 100.0 * correct / total if total else 0.0
    summary["N"] = total
    summary["MEAN_TIME_SEC"] = sum(response_times) / len(response_times) if response_times else 0.0

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="benchmark_results")
    parser.add_argument("--models-file", default="models.txt")
    parser.add_argument("--csv-output", default="benchmark_results/model_accuracy_summary.csv")
    parser.add_argument("--md-output", default="benchmark_results/model_accuracy_summary.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    models = read_models(args.models_file)

    rows = []

    for model in models:
        csv_path = results_dir / f"{safe_model_name(model)}.csv"

        if not csv_path.exists():
            rows.append({
                "model": model,
                "STOP": "MISSING",
                "SLOW_DOWN": "MISSING",
                "PROCEED": "MISSING",
                "UNKNOWN": "MISSING",
                "TOTAL": "MISSING",
                "mean_response_time_sec": "MISSING",
                "n": 0,
            })
            continue

        s = summarize_one(csv_path)
        rows.append({
            "model": model,
            "STOP": f"{s['STOP']:.2f}",
            "SLOW_DOWN": f"{s['SLOW_DOWN']:.2f}",
            "PROCEED": f"{s['PROCEED']:.2f}",
            "UNKNOWN": f"{s['UNKNOWN']:.2f}",
            "TOTAL": f"{s['TOTAL']:.2f}",
            "mean_response_time_sec": f"{s['MEAN_TIME_SEC']:.3f}",
            "n": s["N"],
        })

    csv_output = Path(args.csv_output)
    md_output = Path(args.md_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "STOP",
        "SLOW_DOWN",
        "PROCEED",
        "UNKNOWN",
        "TOTAL",
        "mean_response_time_sec",
        "n",
    ]

    with csv_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_output.open("w", encoding="utf-8") as f:
        f.write("| Model | STOP % | SLOW_DOWN % | PROCEED % | UNKNOWN % | Total % | Mean time (s) | N |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {model} | {STOP} | {SLOW_DOWN} | {PROCEED} | {UNKNOWN} | {TOTAL} | {mean_response_time_sec} | {n} |\n".format(**row)
            )

    print(f"Saved: {csv_output}")
    print(f"Saved: {md_output}")


if __name__ == "__main__":
    main()
