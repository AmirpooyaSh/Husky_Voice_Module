#!/usr/bin/env python3
"""
benchmark_ollama_robot_commands_native_tools.py

Robot-command benchmark using Ollama's NATIVE Python tool-calling API.

This version intentionally does NOT use:
    OpenAI(...)
    client.chat.completions.create(...)

Instead it uses:
    from ollama import Client
    client.chat(..., tools=[...])

This matches Ollama's official Python tool-calling style.

Expected dataset format:
    label<TAB>text

Labels:
    STOP
    SLOW_DOWN
    PROCEED
    UNKNOWN

Default Ollama host:
    http://localhost:11434

Compatibility note:
    Your older shell scripts may still pass:
        --base-url http://localhost:11434/v1

    This script accepts that and automatically converts it to:
        http://localhost:11434
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from ollama import Client
except Exception:
    print("ERROR: missing Python package 'ollama'.", file=sys.stderr)
    print("Install it with:", file=sys.stderr)
    print("  python3 -m pip install -U ollama", file=sys.stderr)
    raise


VALID = {"STOP", "SLOW_DOWN", "PROCEED", "UNKNOWN"}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_robot_command",
            "description": (
                "Classify a short user phrase into exactly one robot command: "
                "STOP, SLOW_DOWN, PROCEED, or UNKNOWN."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["STOP", "SLOW_DOWN", "PROCEED", "UNKNOWN"],
                        "description": "The classified robot command."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence from 0.0 to 1.0."
                    }
                },
                "required": ["command"],
                "additionalProperties": False
            }
        }
    }
]


def normalize_label(x):
    if x is None:
        return "INVALID"

    x = str(x).strip().upper()
    x = x.replace("-", "_")
    x = x.replace(" ", "_")

    if x in {"SLOWDOWN", "SLOW"}:
        return "SLOW_DOWN"

    return x if x in VALID else "INVALID"


def normalize_confidence(conf):
    if conf in ["", None]:
        return ""

    try:
        conf = float(conf)
    except Exception:
        return ""

    if 1.0 < conf <= 100.0:
        conf = conf / 100.0

    return max(0.0, min(1.0, conf))


def safe_model_name(model):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model.replace(":", "_"))


def normalize_ollama_host(base_url_or_host):
    """
    Accepts either:
        http://localhost:11434
    or the older OpenAI-compatible:
        http://localhost:11434/v1

    Returns:
        http://localhost:11434
    """
    host = (base_url_or_host or "http://localhost:11434").strip()
    host = host.rstrip("/")

    if host.endswith("/v1"):
        host = host[:-3].rstrip("/")

    if host.endswith("/api"):
        host = host[:-4].rstrip("/")

    return host


def load_dataset(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        if not reader.fieldnames or "label" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("Dataset must have header: label<TAB>text")

        for i, row in enumerate(reader, start=2):
            label = normalize_label(row["label"])
            text = (row["text"] or "").strip()

            if label not in VALID:
                raise ValueError(f"Invalid label on line {i}: {row['label']}")

            if not text:
                raise ValueError(f"Empty text on line {i}")

            rows.append({"label": label, "text": text})

    return rows


def get_attr_or_key(obj, name, default=None):
    """
    Ollama Python responses are usually objects with attributes, but depending
    on package version they may also behave like dictionaries.
    This helper supports both.
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def get_message(response):
    return get_attr_or_key(response, "message", None)


def get_tool_calls(message):
    calls = get_attr_or_key(message, "tool_calls", None)

    if calls is None:
        return []

    return calls


def parse_tool_call(message):
    """
    Returns:
        predicted, confidence, status, raw_arguments, raw_content

    Status values:
        ok
        no_tool_call
        invalid_tool_name
        invalid_arguments
        invalid_label
    """
    raw_content = get_attr_or_key(message, "content", "") or ""
    calls = get_tool_calls(message)

    if not calls:
        return "NO_TOOL_CALL", "", "no_tool_call", "", raw_content

    call = calls[0]
    function_obj = get_attr_or_key(call, "function", None)

    function_name = get_attr_or_key(function_obj, "name", "")
    if function_name != "classify_robot_command":
        raw_args = get_attr_or_key(function_obj, "arguments", "")
        return "INVALID_TOOL", "", "invalid_tool_name", raw_args, raw_content

    args = get_attr_or_key(function_obj, "arguments", {})

    # Official Ollama Python examples show call.function.arguments as a dict.
    # But keep support for JSON strings just in case.
    raw_args = args

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return "INVALID_ARGS", "", "invalid_arguments", raw_args, raw_content

    if not isinstance(args, dict):
        return "INVALID_ARGS", "", "invalid_arguments", raw_args, raw_content

    predicted = normalize_label(args.get("command"))
    confidence = normalize_confidence(args.get("confidence", ""))

    if predicted not in VALID:
        return predicted, confidence, "invalid_label", raw_args, raw_content

    return predicted, confidence, "ok", raw_args, raw_content


def ollama_chat(client, model, messages, timeout, think):
    """
    Calls Ollama native chat.

    Some Ollama/Python-client versions support think=...
    Some older ones may not. If passing think causes TypeError, retry without it.
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 60
        }
    }

    if think is not None:
        kwargs["think"] = think

    try:
        return client.chat(**kwargs)
    except TypeError as e:
        # Retry without think for older ollama Python clients.
        if "think" in kwargs:
            kwargs.pop("think", None)
            return client.chat(**kwargs)
        raise e


def classify(client, model, text, timeout, think=False):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a robot command classifier. "
                "Classify the user's text into exactly one command: "
                "STOP, SLOW_DOWN, PROCEED, or UNKNOWN. "
                "Always use the classify_robot_command tool. "
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]

    t0 = time.perf_counter()

    response = ollama_chat(
        client=client,
        model=model,
        messages=messages,
        timeout=timeout,
        think=think
    )

    dt = time.perf_counter() - t0

    message = get_message(response)
    pred, conf, status, raw_args, raw_content = parse_tool_call(message)

    return pred, conf, status, raw_args, raw_content, dt


def skip_marker_path(out):
    out = Path(out)
    return out.with_suffix(out.suffix + ".SKIPPED.txt")


def write_skip_marker(out, model, reason, details):
    marker = skip_marker_path(out)
    marker.parent.mkdir(parents=True, exist_ok=True)

    with marker.open("w", encoding="utf-8") as f:
        f.write("SKIPPED MODEL\n")
        f.write("=============\n")
        f.write(f"Model: {model}\n")
        f.write(f"Reason: {reason}\n")
        f.write("\nDetails:\n")
        f.write(str(details).strip() + "\n")

    return marker


def precheck_model(client, model, timeout, precheck_text, precheck_expected, think):
    try:
        pred, conf, status, raw_args, raw_content, dt = classify(
            client=client,
            model=model,
            text=precheck_text,
            timeout=timeout,
            think=think
        )
    except Exception as e:
        return False, {
            "reason": "exception",
            "error": repr(e),
            "precheck_text": precheck_text,
            "expected": precheck_expected
        }

    info = {
        "reason": status,
        "predicted": pred,
        "expected": precheck_expected,
        "confidence": conf,
        "response_time_sec": dt,
        "raw_arguments": raw_args,
        "raw_content": raw_content,
        "precheck_text": precheck_text
    }

    if status != "ok":
        return False, info

    if pred not in VALID:
        info["reason"] = "invalid_prediction"
        return False, info

    if precheck_expected and pred != precheck_expected:
        info["reason"] = "wrong_precheck_prediction"
        return False, info

    info["reason"] = "passed"
    return True, info


def print_summary(results):
    correct = sum(1 for r in results if r["correct"])
    times = [r["response_time_sec"] for r in results if isinstance(r["response_time_sec"], float)]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total:    {len(results)}")
    print(f"Correct:  {correct}")
    print(f"Wrong:    {len(results) - correct}")
    print(f"Accuracy: {100 * correct / max(1, len(results)):.2f}%")

    if times:
        s = sorted(times)
        print("\nResponse time:")
        print(f"  mean:   {statistics.mean(times):.3f}s")
        print(f"  median: {statistics.median(times):.3f}s")
        print(f"  min:    {min(times):.3f}s")
        print(f"  max:    {max(times):.3f}s")
        print(f"  p90:    {s[int(0.90 * (len(s) - 1))]:.3f}s")
        print(f"  p95:    {s[int(0.95 * (len(s) - 1))]:.3f}s")

    print("\nAccuracy by label:")

    for lab in ["STOP", "SLOW_DOWN", "PROCEED", "UNKNOWN"]:
        sub = [r for r in results if r["expected"] == lab]
        c = sum(1 for r in sub if r["correct"])
        print(f"  {lab:9s}: {c:3d}/{len(sub):3d} = {100 * c / max(1, len(sub)):.2f}%")


def make_client(host, timeout):
    """
    Construct Ollama client.

    Newer ollama clients accept timeout=...
    If not, fall back to host only.
    """
    try:
        return Client(host=host, timeout=timeout)
    except TypeError:
        return Client(host=host)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b"))

    # Keep --base-url for compatibility with your existing .sh files.
    # It may be http://localhost:11434/v1 from the old OpenAI-compatible script.
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

    # Native name, optional.
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", ""))

    ap.add_argument("--dataset", default=os.environ.get("DATASET", "robot_command_phrases_tricky.tsv"))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("REQUEST_TIMEOUT", "120")))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")))
    ap.add_argument("--output", default=None)
    ap.add_argument("--warmup", action="store_true")

    # Ollama docs often show think=True in examples. For this classifier,
    # default is False to reduce latency and avoid extra thinking overhead.
    ap.add_argument("--think", action="store_true", help="Pass think=True to Ollama chat if supported.")

    # Precheck behavior.
    ap.add_argument("--no-precheck", action="store_true")
    ap.add_argument("--precheck-text", default=os.environ.get("PRECHECK_TEXT", "stop"))
    ap.add_argument("--precheck-expected", default=os.environ.get("PRECHECK_EXPECTED", "STOP"))
    ap.add_argument(
        "--precheck-timeout",
        type=float,
        default=float(os.environ.get("PRECHECK_TIMEOUT", os.environ.get("REQUEST_TIMEOUT", "120")))
    )

    args = ap.parse_args()

    rows = load_dataset(args.dataset)
    if args.limit > 0:
        rows = rows[:args.limit]

    host = normalize_ollama_host(args.host if args.host else args.base_url)
    client = make_client(host=host, timeout=args.timeout)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output or f"ollama_tricky_benchmark_{safe_model_name(args.model)}_{stamp}.csv")

    print("Ollama native-tool robot-command benchmark")
    print(f"Model:    {args.model}")
    print(f"Host:     {host}")
    print(f"Dataset:  {args.dataset}")
    print(f"Examples: {len(rows)}")
    print(f"Output:   {out}")
    print(f"Think:    {args.think}\n")

    marker = skip_marker_path(out)
    if marker.exists():
        marker.unlink()

    if not args.no_precheck:
        print("Precheck: verifying native Ollama tool-call behavior before full benchmark...")

        expected = normalize_label(args.precheck_expected) if args.precheck_expected else ""

        ok, info = precheck_model(
            client=client,
            model=args.model,
            timeout=args.precheck_timeout,
            precheck_text=args.precheck_text,
            precheck_expected=expected,
            think=args.think
        )

        if not ok:
            reason = info.get("reason", "unknown")
            marker = write_skip_marker(out, args.model, reason, json.dumps(info, indent=2))

            print()
            print("=" * 70)
            print("SKIPPING MODEL")
            print("=" * 70)
            print(f"Model:  {args.model}")
            print(f"Reason: {reason}")
            print(f"Marker: {marker}")
            print()
            print("Precheck details:")
            print(json.dumps(info, indent=2))
            print()
            print("The script exits with code 0 so a multi-model shell loop can continue.")
            return 0

        print(
            "Precheck passed: predicted={} expected={} time={:.3f}s\n".format(
                info.get("predicted"),
                info.get("expected"),
                info.get("response_time_sec", 0.0)
            )
        )

    if args.warmup:
        try:
            print("Warm-up...")
            classify(client, args.model, "stop", args.timeout, think=args.think)
            print("Warm-up finished.\n")
        except Exception as e:
            print(f"Warm-up failed: {e}\n")

    results = []

    for i, r in enumerate(rows, 1):
        expected = r["label"]
        text = r["text"]

        try:
            pred, conf, status, raw_args, raw_content, dt = classify(
                client=client,
                model=args.model,
                text=text,
                timeout=args.timeout,
                think=args.think
            )
            err = ""
        except Exception as e:
            pred, conf, status, raw_args, raw_content, dt, err = "ERROR", "", "exception", "", "", "", repr(e)

        correct = (pred == expected)

        results.append({
            "index": i,
            "expected": expected,
            "predicted": pred,
            "correct": correct,
            "response_time_sec": dt,
            "confidence": conf,
            "status": status,
            "text": text,
            "raw_arguments": raw_args,
            "raw_content": raw_content,
            "error": err
        })

        tstr = f"{dt:.3f}s" if isinstance(dt, float) else "NA"
        print(f"[{i:03d}/{len(rows):03d}] {'OK' if correct else 'WRONG':5s} expected={expected:9s} predicted={pred:12s} time={tstr} text={text!r}")

    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "index",
            "expected",
            "predicted",
            "correct",
            "response_time_sec",
            "confidence",
            "status",
            "text",
            "raw_arguments",
            "raw_content",
            "error"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print_summary(results)
    print(f"\nSaved CSV: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
