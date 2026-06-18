#!/usr/bin/env python3
"""
Live HuBERT emotion recognizer and visualizer.

Default model:
    superb/hubert-large-superb-er

This model is a HuBERT-Large model fine-tuned for categorical
speech emotion recognition. It outputs emotion-class logits/probabilities,
not valence/arousal regression values.

Example:
    python live_hubert_emotion_visualizer.py \
      --input-sr 48000 \
      --device DEVICE_INDEX \
      --window-sec 3.0 \
      --hop-sec 1.0 \
      --silence-rms 0.005 \
      --matplotlib-backend Qt5Agg \
      --log-csv logs/hubert_emotions.csv
"""

import argparse
import csv
import os
import queue
import sys
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


DEFAULT_MODEL = "superb/hubert-large-superb-er"


def get_backend_from_argv() -> str:
    """Read --matplotlib-backend before importing pyplot."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matplotlib-backend", default=None)
    args, _ = parser.parse_known_args()
    return args.matplotlib_backend


backend = get_backend_from_argv()
if backend:
    import matplotlib
    matplotlib.use(backend)

import matplotlib.pyplot as plt


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert sounddevice chunk to mono float32 in [-1, 1]."""
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


def resample_audio(audio: np.ndarray, input_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to model sampling rate."""
    if input_sr == target_sr:
        return audio.astype(np.float32)

    gcd = np.gcd(input_sr, target_sr)
    up = target_sr // gcd
    down = input_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def get_labels(model) -> List[str]:
    """
    Read labels from Hugging Face config.
    Falls back to LABEL_0, LABEL_1, ...
    """
    num_labels = int(getattr(model.config, "num_labels", 0))
    id2label = getattr(model.config, "id2label", None) or {}

    labels = []
    for i in range(num_labels):
        label = id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))
        labels.append(str(label))

    return labels


def infer_output_mode(labels: List[str], model) -> str:
    """
    Decide whether the model is classification or continuous V/A/D style regression.

    The default HuBERT SUPERB ER model is classification.
    This only switches to regression if labels/config strongly suggest it.
    """
    problem_type = str(getattr(model.config, "problem_type", "") or "").lower()
    label_text = " ".join(labels).lower()

    vad_terms = ["valence", "arousal", "dominance"]
    has_vad_terms = any(term in label_text for term in vad_terms)

    if "regression" in problem_type or has_vad_terms:
        return "regression"

    return "classification"


def maybe_scale_va_value(label: str, value: float) -> float:
    """
    If a regression model returns valence/arousal in [0, 1], rescale to [-1, 1].
    This is not used for the default categorical HuBERT emotion model.
    """
    label_l = label.lower()

    if ("valence" in label_l or "arousal" in label_l) and 0.0 <= value <= 1.0:
        return float(np.clip((2.0 * value) - 1.0, -1.0, 1.0))

    return float(value)


def run_model(
    model,
    feature_extractor,
    device: torch.device,
    audio: np.ndarray,
    model_sr: int,
    labels: List[str],
    output_mode: str,
) -> Dict:
    """
    Run HuBERT model.

    For classification:
        returns probabilities and top emotion.

    For V/A/D-style regression:
        returns raw/rescaled scores by label.
    """
    inputs = feature_extractor(
        audio,
        sampling_rate=model_sr,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)

    logits_cpu = logits.detach().float().cpu()

    if output_mode == "regression":
        raw_values = logits_cpu.numpy().astype(float)
        values = {
            label: maybe_scale_va_value(label, float(value))
            for label, value in zip(labels, raw_values)
        }
        return {
            "mode": "regression",
            "values": values,
            "raw_values": {label: float(value) for label, value in zip(labels, raw_values)},
        }

    probs = F.softmax(logits_cpu, dim=-1).numpy()
    top_idx = int(np.argmax(probs))

    return {
        "mode": "classification",
        "probs": probs,
        "top_label": labels[top_idx],
        "top_prob": float(probs[top_idx]),
    }


def setup_classification_plot(labels: List[str]):
    plt.ion()

    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2])

    ax_bar = fig.add_subplot(grid[0, 0])
    ax_hist = fig.add_subplot(grid[1, 0])

    x = np.arange(len(labels))
    bars = ax_bar.bar(x, np.zeros(len(labels)))

    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Probability")
    ax_bar.set_title("HuBERT Emotion Recognition")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=30, ha="right")
    ax_bar.grid(True, axis="y", alpha=0.3)

    info_text = ax_bar.text(
        0.02,
        0.95,
        "Waiting for voice...",
        transform=ax_bar.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.35"),
    )

    ax_hist.set_title("Rolling top-emotion confidence")
    ax_hist.set_xlim(0, 60)
    ax_hist.set_ylim(0, 1)
    ax_hist.set_xlabel("Time (s)")
    ax_hist.set_ylabel("Top probability")
    ax_hist.grid(True, alpha=0.3)

    conf_line, = ax_hist.plot([], [], label="Top emotion probability")
    ax_hist.legend(loc="lower right")

    fig.tight_layout()

    return fig, ax_bar, ax_hist, bars, info_text, conf_line


def setup_regression_plot(labels: List[str]):
    plt.ion()

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1)

    x = np.arange(len(labels))
    bars = ax.bar(x, np.zeros(len(labels)))

    ax.set_ylim(-1, 1)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Score")
    ax.set_title("HuBERT Emotion Regression Outputs")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    info_text = ax.text(
        0.02,
        0.95,
        "Waiting for voice...",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.35"),
    )

    fig.tight_layout()

    return fig, ax, bars, info_text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face HuBERT emotion model.",
    )

    parser.add_argument(
        "--input-sr",
        type=int,
        default=16000,
        help="Microphone sample rate. Try 48000 if 16000 fails.",
    )

    parser.add_argument(
        "--audio-device",
        type=int,
        default=None,
        help="Microphone input device index. Use `python -m sounddevice` to list devices.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Deprecated alias for --audio-device. This is the microphone device, not CUDA.",
    )

    parser.add_argument(
        "--compute-device",
        default="auto",
        choices=["auto", "cuda", "cuda:0", "cpu"],
        help="Torch compute device. Use auto unless you want to force cuda:0 or cpu.",
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit with an error if CUDA is not available.",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=3.0,
        help="Audio duration used for each prediction.",
    )

    parser.add_argument(
        "--hop-sec",
        type=float,
        default=1.0,
        help="How often to update prediction.",
    )

    parser.add_argument(
        "--max-model-sec",
        type=float,
        default=15.0,
        help="Maximum audio duration passed to the model.",
    )

    parser.add_argument(
        "--silence-rms",
        type=float,
        default=0.005,
        help="Skip prediction when audio RMS is below this value.",
    )

    parser.add_argument(
        "--ema",
        type=float,
        default=0.35,
        help="Smoothing factor from 0 to 1. Higher = more responsive.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )

    parser.add_argument(
        "--matplotlib-backend",
        default=None,
        help="Example: Qt5Agg. Must be available in the environment.",
    )

    parser.add_argument(
        "--log-csv",
        default=None,
        help="Optional CSV file to save live outputs.",
    )

    args = parser.parse_args()

    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive.")

    if args.hop_sec <= 0:
        raise ValueError("--hop-sec must be positive.")

    if args.max_model_sec <= 0:
        raise ValueError("--max-model-sec must be positive.")

    # Backward compatibility: --device is the microphone index.
    if args.audio_device is None and args.device is not None:
        args.audio_device = args.device

    # Torch compute device selection.
    if args.cpu:
        compute_device = torch.device("cpu")
    elif args.compute_device in ["cuda", "cuda:0"]:
        if not torch.cuda.is_available():
            if args.require_cuda:
                raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
            print("WARNING: CUDA was requested but is not available. Falling back to CPU.")
            compute_device = torch.device("cpu")
        else:
            compute_device = torch.device("cuda:0")
    else:
        compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.require_cuda and compute_device.type != "cuda":
        raise RuntimeError("CUDA is required, but this script is running on CPU.")

    print(f"Using Torch compute device: {compute_device}")
    if compute_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"Torch CUDA version: {torch.version.cuda}")

        # Quick CUDA sanity check before loading the model.
        try:
            x = torch.randn(128, 128, device=compute_device)
            y = x @ x
            torch.cuda.synchronize()
            print("CUDA tensor test: OK")
        except Exception as e:
            if args.require_cuda:
                raise RuntimeError(f"CUDA tensor test failed: {e}") from e
            print(f"WARNING: CUDA tensor test failed: {e}")
            print("Falling back to CPU.")
            compute_device = torch.device("cpu")

    print(f"Loading HuBERT emotion model: {args.model}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModelForAudioClassification.from_pretrained(args.model).to(compute_device)
    model.eval()

    labels = get_labels(model)
    output_mode = infer_output_mode(labels, model)

    model_sr = int(getattr(feature_extractor, "sampling_rate", 16000))
    print(f"Model sample rate: {model_sr} Hz")
    print(f"Model labels: {labels}")
    print(f"Detected output mode: {output_mode}")

    if output_mode == "classification":
        print("This model outputs categorical emotion probabilities.")
        print("No valence/arousal scale adjustment is needed.")
        fig, ax_bar, ax_hist, bars, info_text, conf_line = setup_classification_plot(labels)
        hist_times = []
        hist_conf = []
        smoothed_probs = None
    else:
        print("This model appears to output continuous scores.")
        print("Valence/arousal values in [0, 1] will be rescaled to [-1, 1].")
        fig, ax_reg, bars, info_text = setup_regression_plot(labels)
        smoothed_values = None

    print(f"Opening microphone input device: {args.audio_device}")
    print("Press Ctrl+C to stop.")

    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_q.put(indata.copy())

    buffer_len = int(model_sr * args.window_sec)
    max_model_len = int(model_sr * args.max_model_sec)
    audio_buffer = deque(maxlen=max(buffer_len, max_model_len))

    csv_file = None
    csv_writer = None

    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        csv_file = open(args.log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)

        if output_mode == "classification":
            csv_writer.writerow(
                ["time_sec", "top_emotion", "top_probability", "rms", *[f"prob_{label}" for label in labels]]
            )
        else:
            csv_writer.writerow(
                ["time_sec", "rms", *[f"value_{label}" for label in labels], *[f"raw_{label}" for label in labels]]
            )

    t0 = time.time()
    last_prediction_time = 0.0

    try:
        with sd.InputStream(
            samplerate=args.input_sr,
            channels=1,
            dtype="float32",
            device=args.audio_device,
            blocksize=int(args.input_sr * 0.1),
            callback=audio_callback,
        ):
            while True:
                try:
                    chunk = audio_q.get(timeout=0.05)
                except queue.Empty:
                    plt.pause(0.01)
                    continue

                chunk = to_mono_float32(chunk)
                chunk_model_sr = resample_audio(chunk, args.input_sr, model_sr)
                audio_buffer.extend(chunk_model_sr.tolist())

                if len(audio_buffer) < buffer_len:
                    plt.pause(0.01)
                    continue

                now = time.time()
                elapsed = now - t0

                if now - last_prediction_time < args.hop_sec:
                    plt.pause(0.01)
                    continue

                last_prediction_time = now

                segment = np.array(audio_buffer, dtype=np.float32)[-max_model_len:]
                segment_rms = rms(segment)

                if segment_rms < args.silence_rms:
                    info_text.set_text(f"Waiting for clear voice...\nRMS: {segment_rms:.4f}")
                    plt.pause(0.01)
                    continue

                result = run_model(
                    model=model,
                    feature_extractor=feature_extractor,
                    device=compute_device,
                    audio=segment,
                    model_sr=model_sr,
                    labels=labels,
                    output_mode=output_mode,
                )

                if output_mode == "classification":
                    probs = result["probs"]

                    if smoothed_probs is None:
                        smoothed_probs = probs
                    else:
                        alpha = args.ema
                        smoothed_probs = alpha * probs + (1.0 - alpha) * smoothed_probs

                    top_idx = int(np.argmax(smoothed_probs))
                    top_label = labels[top_idx]
                    top_prob = float(smoothed_probs[top_idx])

                    for bar, prob in zip(bars, smoothed_probs):
                        bar.set_height(float(prob))

                    info_text.set_text(
                        f"Top emotion: {top_label} ({top_prob:.2f})\n"
                        f"RMS: {segment_rms:.4f}\n"
                        f"Model: {args.model}"
                    )

                    hist_times.append(elapsed)
                    hist_conf.append(top_prob)

                    while hist_times and hist_times[0] < elapsed - 60:
                        hist_times.pop(0)
                        hist_conf.pop(0)

                    conf_line.set_data(hist_times, hist_conf)
                    ax_hist.set_xlim(max(0, elapsed - 60), max(60, elapsed))

                    probs_text = ", ".join(
                        f"{label}:{float(prob):.2f}" for label, prob in zip(labels, smoothed_probs)
                    )
                    print(
                        f"[{elapsed:7.2f}s] Emotion: {top_label:<12} "
                        f"P={top_prob:.3f} | RMS={segment_rms:.4f} | {probs_text}",
                        flush=True,
                    )

                    if csv_writer:
                        csv_writer.writerow(
                            [
                                f"{elapsed:.3f}",
                                top_label,
                                f"{top_prob:.6f}",
                                f"{segment_rms:.6f}",
                                *[f"{float(p):.6f}" for p in smoothed_probs],
                            ]
                        )
                        csv_file.flush()

                else:
                    values = result["values"]
                    raw_values = result["raw_values"]
                    current_values = np.array([values[label] for label in labels], dtype=float)

                    if smoothed_values is None:
                        smoothed_values = current_values
                    else:
                        alpha = args.ema
                        smoothed_values = alpha * current_values + (1.0 - alpha) * smoothed_values

                    for bar, value in zip(bars, smoothed_values):
                        bar.set_height(float(value))

                    value_text = "\n".join(
                        f"{label}: {float(value):+.3f}" for label, value in zip(labels, smoothed_values)
                    )

                    info_text.set_text(f"{value_text}\nRMS: {segment_rms:.4f}")

                    print(
                        f"[{elapsed:7.2f}s] "
                        + " | ".join(f"{label}: {float(value):+.3f}" for label, value in zip(labels, smoothed_values))
                        + f" | RMS={segment_rms:.4f}",
                        flush=True,
                    )

                    if csv_writer:
                        csv_writer.writerow(
                            [
                                f"{elapsed:.3f}",
                                f"{segment_rms:.6f}",
                                *[f"{float(v):.6f}" for v in smoothed_values],
                                *[f"{float(raw_values[label]):.6f}" for label in labels],
                            ]
                        )
                        csv_file.flush()

                fig.canvas.draw_idle()
                plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
