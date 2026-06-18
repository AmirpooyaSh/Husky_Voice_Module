#!/usr/bin/env python3
"""
Live Wav2Vec2-Large emotion recognition from microphone audio.

Default model:
    superb/wav2vec2-large-superb-er

This script visualizes categorical emotion probabilities only. The selected
SUPERB Wav2Vec2 emotion-recognition model does not output valence/arousal.

Typical run:
    python live_wav2vec2_large_emotion_only.py \
      --input-sr 48000 \
      --device DEVICE_INDEX \
      --window-sec 3.0 \
      --hop-sec 1.0 \
      --silence-rms 0.005 \
      --matplotlib-backend Qt5Agg \
      --log-csv logs/wav2vec2_large_emotions.csv
"""

import argparse
import csv
import os
import queue
import time
from collections import deque
from typing import Dict, List, Tuple

# Parse only the backend argument before importing pyplot.
_backend_parser = argparse.ArgumentParser(add_help=False)
_backend_parser.add_argument("--matplotlib-backend", default=None)
_backend_args, _ = _backend_parser.parse_known_args()

if _backend_args.matplotlib_backend:
    import matplotlib
    matplotlib.use(_backend_args.matplotlib_backend)
elif os.environ.get("MPLBACKEND"):
    # Respect the environment variable set by Docker, e.g. MPLBACKEND=Qt5Agg.
    pass

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


MODEL_SAMPLE_RATE = 16000

# Make labels more readable if the model config uses abbreviations.
LABEL_NORMALIZATION = {
    "ang": "Anger",
    "anger": "Anger",
    "hap": "Happiness",
    "happy": "Happiness",
    "happiness": "Happiness",
    "sad": "Sadness",
    "sadness": "Sadness",
    "neu": "Neutral",
    "neutral": "Neutral",
    "sur": "Surprise",
    "surprise": "Surprise",
    "fea": "Fear",
    "fear": "Fear",
    "dis": "Disgust",
    "disgust": "Disgust",
    "con": "Contempt",
    "contempt": "Contempt",
    "oth": "Other",
    "other": "Other",
}


def readable_label(label: str) -> str:
    key = str(label).strip().lower()
    return LABEL_NORMALIZATION.get(key, str(label).strip().replace("_", " ").title())


def get_model_labels(model) -> List[str]:
    """Return model labels in output-logit order."""
    id2label: Dict[int, str] = model.config.id2label
    labels = []
    for i in range(model.config.num_labels):
        raw = id2label.get(i, id2label.get(str(i), f"label_{i}"))
        labels.append(readable_label(raw))
    return labels


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


def resample_to_16k(audio: np.ndarray, input_sr: int) -> np.ndarray:
    if input_sr == MODEL_SAMPLE_RATE:
        return audio.astype(np.float32)

    gcd = np.gcd(input_sr, MODEL_SAMPLE_RATE)
    up = MODEL_SAMPLE_RATE // gcd
    down = input_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def predict_emotion(
    feature_extractor,
    model,
    device: torch.device,
    audio_16k: np.ndarray,
) -> np.ndarray:
    """Return class probabilities for one audio segment."""
    max_len = 15 * MODEL_SAMPLE_RATE
    audio_16k = audio_16k[-max_len:]

    inputs = feature_extractor(
        audio_16k,
        sampling_rate=MODEL_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0)

    return probs.detach().cpu().numpy()


def setup_plot(labels: List[str]):
    plt.ion()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, np.zeros(len(labels)))

    ax.set_title("Live Wav2Vec2-Large Emotion Recognition")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    info_text = ax.text(
        0.02,
        0.98,
        "Waiting for voice...",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", boxstyle="round,pad=0.35"),
    )

    fig.tight_layout()
    return fig, ax, bars, info_text


def main():
    parser = argparse.ArgumentParser(
        description="Live categorical speech emotion recognition using a Wav2Vec2-Large emotion model."
    )

    parser.add_argument(
        "--model",
        default="superb/wav2vec2-large-superb-er",
        help="Hugging Face audio-classification model. Default uses Wav2Vec2-Large fine-tuned for SUPERB ER.",
    )

    parser.add_argument(
        "--input-sr",
        type=int,
        default=16000,
        help="Microphone sample rate. Use 48000 for many USB microphones, 16000 if supported.",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=3.0,
        help="Audio duration used for each prediction. Keep >= 3 for emotion recognition.",
    )

    parser.add_argument(
        "--hop-sec",
        type=float,
        default=1.0,
        help="How often to update prediction.",
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
        help="Skip prediction when audio RMS is below this value.",
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
        help="Force CPU even if CUDA is available.",
    )

    parser.add_argument(
        "--log-csv",
        default=None,
        help="Optional CSV file to save emotion probabilities.",
    )

    parser.add_argument(
        "--matplotlib-backend",
        default=None,
        help="GUI backend, e.g. Qt5Agg. Must be set before pyplot import; this script handles that early.",
    )

    args = parser.parse_args()

    if args.window_sec < 3.0:
        raise ValueError("Use --window-sec >= 3.0 for more reliable emotion recognition.")

    compute_device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"Using compute device: {compute_device}")
    if compute_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading feature extractor: {args.model}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)

    print(f"Loading Wav2Vec2 emotion model: {args.model}")
    model = AutoModelForAudioClassification.from_pretrained(args.model).to(compute_device)
    model.eval()

    labels = get_model_labels(model)
    print("Emotion labels:", ", ".join(labels))
    print("Note: this model outputs categorical emotion probabilities only; it does not output valence/arousal.")
    print("Opening microphone...")
    print("Press Ctrl+C to stop.")

    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_q.put(indata.copy())

    buffer_len = int(MODEL_SAMPLE_RATE * args.window_sec)
    audio_buffer = deque(maxlen=buffer_len)

    fig, ax, bars, info_text = setup_plot(labels)

    t0 = time.time()
    last_prediction_time = 0.0
    smoothed_probs = None

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
                *[f"prob_{label}" for label in labels],
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

                probs = predict_emotion(feature_extractor, model, compute_device, segment)

                if smoothed_probs is None:
                    smoothed_probs = probs
                else:
                    alpha = args.ema
                    smoothed_probs = alpha * probs + (1.0 - alpha) * smoothed_probs

                top_idx = int(np.argmax(smoothed_probs))
                top_emotion = labels[top_idx]
                top_prob = float(smoothed_probs[top_idx])

                for bar, prob in zip(bars, smoothed_probs):
                    bar.set_height(float(prob))

                info_text.set_text(
                    f"Emotion: {top_emotion} ({top_prob:.2f})\n"
                    f"RMS: {segment_rms:.4f}\n"
                    f"Model: {args.model}"
                )

                prob_text = ", ".join(
                    f"{label}:{float(prob):.2f}" for label, prob in zip(labels, smoothed_probs)
                )
                print(
                    f"[{elapsed:7.2f}s] Emotion: {top_emotion:<12} "
                    f"P={top_prob:.3f} | RMS={segment_rms:.4f} | {prob_text}",
                    flush=True,
                )

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
