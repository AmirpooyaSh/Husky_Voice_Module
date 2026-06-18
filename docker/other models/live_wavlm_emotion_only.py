#!/usr/bin/env python3
"""
Live categorical speech-emotion visualization using a WavLM-Large emotion model.

This script uses the Vox-Profile WavLM-Large categorical emotion wrapper:
    src.model.emotion.wavlm_emotion.WavLMWrapper

Expected repo/package layout inside the Docker container:
    /opt/vox-profile-release
or any environment where `src.model.emotion.wavlm_emotion` is importable.

Example:
    python live_wavlm_emotion_only.py \
      --input-sr 48000 \
      --device 5 \
      --window-sec 3.0 \
      --hop-sec 1.0 \
      --matplotlib-backend Qt5Agg \
      --log-csv logs/wavlm_emotions.csv
"""

import argparse
import csv
import os
import queue
import time
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

# Matplotlib backend is selected in main() before importing pyplot.

from src.model.emotion.wavlm_emotion import WavLMWrapper as CatEmotionWrapper


MODEL_SAMPLE_RATE = 16000
MAX_MODEL_SECONDS = 15

EMOTION_LABELS = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
    "Other",
]


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert input audio block to clean mono float32 in [-1, 1]."""
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


def resample_to_16k(audio: np.ndarray, input_sr: int) -> np.ndarray:
    """Resample microphone audio to the WavLM model's expected 16 kHz."""
    if input_sr == MODEL_SAMPLE_RATE:
        return audio.astype(np.float32)

    gcd = np.gcd(input_sr, MODEL_SAMPLE_RATE)
    up = MODEL_SAMPLE_RATE // gcd
    down = input_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    """Root mean square amplitude for simple silence detection."""
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def predict_emotions(cat_model, device: torch.device, audio_16k: np.ndarray):
    """Run WavLM-Large categorical emotion model and return probabilities."""
    max_len = MAX_MODEL_SECONDS * MODEL_SAMPLE_RATE
    audio_16k = audio_16k[-max_len:]

    x = torch.from_numpy(audio_16k).float().unsqueeze(0).to(device)

    with torch.inference_mode():
        output = cat_model(x)
        if isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        probs = F.softmax(logits, dim=1).squeeze(0)

    probs_np = probs.detach().cpu().numpy().astype(np.float32)
    top_idx = int(np.argmax(probs_np))
    top_emotion = EMOTION_LABELS[top_idx]
    top_prob = float(probs_np[top_idx])

    return top_emotion, top_prob, probs_np


def setup_plot(plt):
    """Create live probability bar chart and top-emotion confidence plot."""
    plt.ion()

    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0])

    ax_bar = fig.add_subplot(grid[0, 0])
    ax_hist = fig.add_subplot(grid[1, 0])

    bars = ax_bar.bar(EMOTION_LABELS, np.zeros(len(EMOTION_LABELS)))
    ax_bar.set_title("Live WavLM-Large Categorical Speech Emotion")
    ax_bar.set_ylabel("Probability")
    ax_bar.set_ylim(0, 1)
    ax_bar.tick_params(axis="x", rotation=25)
    ax_bar.grid(True, axis="y", alpha=0.25)

    info_text = ax_bar.text(
        0.02,
        0.95,
        "Waiting for clear voice...",
        transform=ax_bar.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", boxstyle="round,pad=0.35"),
        zorder=5,
    )

    ax_hist.set_title("Rolling Top-Emotion Confidence")
    ax_hist.set_xlabel("Time (s)")
    ax_hist.set_ylabel("Top probability")
    ax_hist.set_ylim(0, 1)
    ax_hist.set_xlim(0, 60)
    ax_hist.grid(True, alpha=0.3)
    confidence_line, = ax_hist.plot([], [], label="Top confidence")
    ax_hist.legend(loc="lower right")

    fig.tight_layout()
    return fig, ax_bar, ax_hist, bars, info_text, confidence_line


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live WavLM-Large categorical emotion recognition and visualization."
    )

    parser.add_argument(
        "--cat-model",
        default="tiantiaf/wavlm-large-categorical-emotion",
        help="Hugging Face model ID for the WavLM-Large categorical emotion classifier.",
    )

    parser.add_argument(
        "--input-sr",
        type=int,
        default=16000,
        help="Microphone sample rate. Use 48000 for many USB microphones.",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=3.0,
        help="Audio duration used for each prediction. Keep >= 3 seconds.",
    )

    parser.add_argument(
        "--hop-sec",
        type=float,
        default=1.0,
        help="How often to update prediction, in seconds.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Microphone input device index. Use `python -m sounddevice` to list devices.",
    )

    parser.add_argument(
        "--silence-rms",
        type=float,
        default=0.005,
        help="Skip prediction when RMS is below this threshold.",
    )

    parser.add_argument(
        "--ema",
        type=float,
        default=0.35,
        help="Probability smoothing factor from 0 to 1. Higher = more responsive.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even when CUDA is available.",
    )

    parser.add_argument(
        "--matplotlib-backend",
        default=None,
        help="Optional backend, e.g., Qt5Agg inside Docker with GUI support.",
    )

    parser.add_argument(
        "--log-csv",
        default=None,
        help="Optional CSV file to save emotion probabilities.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.window_sec < 3.0:
        raise ValueError("Use --window-sec >= 3.0 for more reliable emotion predictions.")

    if not (0.0 <= args.ema <= 1.0):
        raise ValueError("--ema must be between 0 and 1.")

    import matplotlib
    if args.matplotlib_backend:
        matplotlib.use(args.matplotlib_backend)
    import matplotlib.pyplot as plt

    if args.cpu:
        compute_device = torch.device("cpu")
    else:
        compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {compute_device}")
    if compute_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not used. Running on CPU.")

    print("Loading WavLM-Large categorical emotion model...")
    cat_model = CatEmotionWrapper.from_pretrained(args.cat_model).to(compute_device)
    cat_model.eval()
    print("Model loaded.")

    print("Opening microphone...")
    print("Use Ctrl+C to stop.")

    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_q.put(indata.copy())

    buffer_len = int(MODEL_SAMPLE_RATE * args.window_sec)
    audio_buffer = deque(maxlen=buffer_len)

    fig, ax_bar, ax_hist, bars, info_text, confidence_line = setup_plot(plt)

    t0 = time.time()
    last_prediction_time = 0.0
    times = []
    top_confidences = []
    smoothed_probs: Optional[np.ndarray] = None

    csv_file = None
    csv_writer = None
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        csv_file = open(args.log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "time_sec",
                "top_emotion",
                "top_emotion_probability",
                "rms",
                *[f"prob_{label}" for label in EMOTION_LABELS],
            ]
        )

    try:
        with sd.InputStream(
            samplerate=args.input_sr,
            channels=1,
            dtype="float32",
            device=args.device,
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
                chunk_16k = resample_to_16k(chunk, args.input_sr)
                audio_buffer.extend(chunk_16k.tolist())

                if len(audio_buffer) < buffer_len:
                    plt.pause(0.01)
                    continue

                now = time.time()
                elapsed = now - t0

                if now - last_prediction_time < args.hop_sec:
                    plt.pause(0.01)
                    continue

                last_prediction_time = now

                segment = np.array(audio_buffer, dtype=np.float32)
                segment_rms = rms(segment)

                if segment_rms < args.silence_rms:
                    info_text.set_text(f"Waiting for clear voice...\nRMS: {segment_rms:.4f}")
                    plt.pause(0.01)
                    continue

                raw_top_emotion, raw_top_prob, probs = predict_emotions(
                    cat_model, compute_device, segment
                )

                if smoothed_probs is None:
                    smoothed_probs = probs
                else:
                    alpha = args.ema
                    smoothed_probs = alpha * probs + (1.0 - alpha) * smoothed_probs

                top_idx = int(np.argmax(smoothed_probs))
                top_emotion = EMOTION_LABELS[top_idx]
                top_prob = float(smoothed_probs[top_idx])

                # Terminal output.
                prob_summary = ", ".join(
                    f"{label}:{float(prob):.2f}"
                    for label, prob in zip(EMOTION_LABELS, smoothed_probs)
                )
                print(
                    f"[{elapsed:7.2f}s] Emotion: {top_emotion:<9s} "
                    f"P={top_prob:.3f} | RMS={segment_rms:.4f} | {prob_summary}",
                    flush=True,
                )

                # Update bar chart.
                for bar, prob in zip(bars, smoothed_probs):
                    bar.set_height(float(prob))

                info_text.set_text(
                    f"Top emotion: {top_emotion} ({top_prob:.2f})\n"
                    f"Raw top: {raw_top_emotion} ({raw_top_prob:.2f})\n"
                    f"RMS: {segment_rms:.4f}"
                )

                # Update confidence history.
                times.append(elapsed)
                top_confidences.append(top_prob)
                while times and times[0] < elapsed - 60:
                    times.pop(0)
                    top_confidences.pop(0)

                confidence_line.set_data(times, top_confidences)
                ax_hist.set_xlim(max(0, elapsed - 60), max(60, elapsed))

                if csv_writer:
                    csv_writer.writerow(
                        [
                            f"{elapsed:.3f}",
                            top_emotion,
                            f"{top_prob:.6f}",
                            f"{segment_rms:.6f}",
                            *[f"{float(p):.6f}" for p in smoothed_probs],
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
