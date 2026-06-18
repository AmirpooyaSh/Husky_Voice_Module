import argparse
import csv
import os
import queue
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from src.model.emotion.wavlm_emotion_dim import WavLMWrapper as DimEmotionWrapper
from src.model.emotion.wavlm_emotion import WavLMWrapper as CatEmotionWrapper


MODEL_SAMPLE_RATE = 16000

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


def scale_0_1_to_neg1_1(value: float) -> float:
    """Convert model output from [0, 1] to [-1, 1]."""
    return float(np.clip((2.0 * value) - 1.0, -1.0, 1.0))


def predict_all(dim_model, cat_model, device, audio_16k: np.ndarray):
    max_len = 15 * MODEL_SAMPLE_RATE
    audio_16k = audio_16k[-max_len:]

    x = torch.from_numpy(audio_16k).float().unsqueeze(0).to(device)

    with torch.inference_mode():
        # Dimensional model
        arousal, valence, dominance = dim_model(x)

        # Categorical model
        cat_output = cat_model(x)
        if isinstance(cat_output, (tuple, list)):
            logits = cat_output[0]
        else:
            logits = cat_output

        probs = F.softmax(logits, dim=1).squeeze(0)

    # Vox-Profile dimensional outputs are in [0, 1].
    # Rescale valence and arousal to [-1, 1] for the VA plane:
    #   -1 = low/negative side, 0 = neutral center, +1 = high/positive side.
    arousal = scale_0_1_to_neg1_1(float(arousal.squeeze().detach().cpu()))
    valence = scale_0_1_to_neg1_1(float(valence.squeeze().detach().cpu()))

    # Keep dominance in its original model range [0, 1].
    dominance = float(dominance.squeeze().detach().cpu())

    probs_np = probs.detach().cpu().numpy()
    top_idx = int(np.argmax(probs_np))
    top_emotion = EMOTION_LABELS[top_idx]
    top_prob = float(probs_np[top_idx])

    return valence, arousal, dominance, top_emotion, top_prob, probs_np


def setup_plot(background_path):
    plt.ion()

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1])

    ax_va = fig.add_subplot(grid[:, 0])
    ax_hist = fig.add_subplot(grid[0, 1])
    ax_cat = fig.add_subplot(grid[1, 1])

    # Background image for VA plane.
    # The model outputs are rescaled to [-1, 1], so the plot is now:
    #   x-axis = Valence, from -1 negative/unpleasant to +1 positive/pleasant
    #   y-axis = Arousal, from -1 low/calm to +1 high/alert
    bg = plt.imread(background_path)
    ax_va.imshow(bg, extent=(-1, 1, -1, 1), origin="upper")
    ax_va.set_xlim(-1, 1)
    ax_va.set_ylim(-1, 1)
    ax_va.set_aspect("equal")
    ax_va.set_xlabel("Valence (-1 to 1)")
    ax_va.set_ylabel("Arousal (-1 to 1)")
    ax_va.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax_va.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax_va.axhline(0.0, linewidth=1.0, alpha=0.35)
    ax_va.axvline(0.0, linewidth=1.0, alpha=0.35)

    # Live point
    point, = ax_va.plot(
        [0.0], [0.0],
        marker="o",
        markersize=14,
        markerfacecolor="red",
        markeredgecolor="white",
        markeredgewidth=2.0,
        linestyle="None",
        zorder=5,
    )

    point_shadow, = ax_va.plot(
        [0.0], [0.0],
        marker="o",
        markersize=22,
        markerfacecolor=(1, 0, 0, 0.18),
        markeredgecolor="none",
        linestyle="None",
        zorder=4,
    )

    info_text = ax_va.text(
        0.03,
        0.97,
        "Waiting for voice...",
        transform=ax_va.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", boxstyle="round,pad=0.35"),
        zorder=6,
    )

    # Rolling VA plot
    ax_hist.set_title("Rolling Valence / Arousal")
    ax_hist.set_xlim(0, 60)
    ax_hist.set_ylim(-1, 1)
    ax_hist.set_xlabel("Time (s)")
    ax_hist.set_ylabel("Score (-1 to 1)")
    ax_hist.grid(True, alpha=0.3)
    ax_hist.axhline(0.0, linewidth=1.0, alpha=0.35)

    valence_line, = ax_hist.plot([], [], label="Valence")
    arousal_line, = ax_hist.plot([], [], label="Arousal")
    ax_hist.legend(loc="lower right")

    # Emotion probabilities
    ax_cat.set_title("Categorical Voice Emotion")
    ax_cat.set_ylim(0, 1)
    ax_cat.set_ylabel("Probability")
    bars = ax_cat.bar(EMOTION_LABELS, np.zeros(len(EMOTION_LABELS)))
    ax_cat.tick_params(axis="x", rotation=35)

    fig.tight_layout()

    return fig, ax_va, ax_hist, ax_cat, point, point_shadow, info_text, valence_line, arousal_line, bars


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dim-model",
        default="tiantiaf/wavlm-large-msp-podcast-emotion-dim",
        help="Dimensional emotion model.",
    )

    parser.add_argument(
        "--cat-model",
        default="tiantiaf/wavlm-large-categorical-emotion",
        help="Categorical emotion model.",
    )

    parser.add_argument(
        "--va-background",
        default="v-a-indicator.png",
        help="Path to the valence-arousal indicator image.",
    )

    parser.add_argument(
        "--input-sr",
        type=int,
        default=16000,
        help="Microphone sample rate. Try 48000 if 16000 fails.",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=3.0,
        help="Audio duration used for each prediction. Keep >= 3.",
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
        help="Smoothing factor from 0 to 1. Higher = more responsive.",
    )

    parser.add_argument(
        "--log-csv",
        default=None,
        help="Optional CSV file to save outputs.",
    )

    args = parser.parse_args()

    if args.window_sec < 3.0:
        raise ValueError("Use --window-sec >= 3.0. Shorter clips may be unreliable.")

    if not os.path.exists(args.va_background):
        raise FileNotFoundError(
            f"Could not find background image: {args.va_background}\n"
            f"Put your image in this folder or pass --va-background /full/path/to/v-a-indicator.png"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch in this environment."
        )

    compute_device = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    print("Loading dimensional model...")
    dim_model = DimEmotionWrapper.from_pretrained(args.dim_model).to(compute_device)
    dim_model.eval()

    print("Loading categorical model...")
    cat_model = CatEmotionWrapper.from_pretrained(args.cat_model).to(compute_device)
    cat_model.eval()

    print("Models loaded.")
    print("Opening microphone...")
    print("Press Ctrl+C to stop.")

    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_q.put(indata.copy())

    buffer_len = int(MODEL_SAMPLE_RATE * args.window_sec)
    audio_buffer = deque(maxlen=buffer_len)

    (
        fig,
        ax_va,
        ax_hist,
        ax_cat,
        point,
        point_shadow,
        info_text,
        valence_line,
        arousal_line,
        bars,
    ) = setup_plot(args.va_background)

    t0 = time.time()
    last_prediction_time = 0.0

    times = []
    valence_values = []
    arousal_values = []

    smoothed_valence = None
    smoothed_arousal = None
    smoothed_dominance = None
    smoothed_probs = None

    csv_file = None
    csv_writer = None

    if args.log_csv:
        csv_file = open(args.log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "time_sec",
                "valence_rescaled_neg1_to_1",
                "arousal_rescaled_neg1_to_1",
                "dominance_original_0_to_1",
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
                    info_text.set_text(
                        f"Waiting for clear voice...\nRMS: {segment_rms:.4f}"
                    )
                    plt.pause(0.01)
                    continue

                (
                    valence,
                    arousal,
                    dominance,
                    top_emotion,
                    top_prob,
                    probs,
                ) = predict_all(dim_model, cat_model, compute_device, segment)

                if smoothed_valence is None:
                    smoothed_valence = valence
                    smoothed_arousal = arousal
                    smoothed_dominance = dominance
                    smoothed_probs = probs
                else:
                    alpha = args.ema
                    smoothed_valence = alpha * valence + (1.0 - alpha) * smoothed_valence
                    smoothed_arousal = alpha * arousal + (1.0 - alpha) * smoothed_arousal
                    smoothed_dominance = alpha * dominance + (1.0 - alpha) * smoothed_dominance
                    smoothed_probs = alpha * probs + (1.0 - alpha) * smoothed_probs

                smoothed_top_idx = int(np.argmax(smoothed_probs))
                smoothed_top_emotion = EMOTION_LABELS[smoothed_top_idx]
                smoothed_top_prob = float(smoothed_probs[smoothed_top_idx])

                # Print live values in terminal once per prediction update.
                print(
                    f"[{elapsed:7.2f}s] "
                    f"Valence [-1,1]: {smoothed_valence:+.3f} | "
                    f"Arousal [-1,1]: {smoothed_arousal:+.3f} | "
                    f"Dominance [0,1]: {smoothed_dominance:.3f} | "
                    f"Emotion: {smoothed_top_emotion} ({smoothed_top_prob:.2f}) | "
                    f"RMS: {segment_rms:.4f}",
                    flush=True,
                )

                # Update point on VA image
                point.set_data([smoothed_valence], [smoothed_arousal])
                point_shadow.set_data([smoothed_valence], [smoothed_arousal])

                info_text.set_text(
                    f"Valence [-1,1]:   {smoothed_valence:.3f}\n"
                    f"Arousal [-1,1]:   {smoothed_arousal:.3f}\n"
                    f"Dominance [0,1]:  {smoothed_dominance:.3f}\n"
                    f"Emotion:   {smoothed_top_emotion} ({smoothed_top_prob:.2f})\n"
                    f"RMS:       {segment_rms:.4f}"
                )

                # Update rolling plot
                times.append(elapsed)
                valence_values.append(smoothed_valence)
                arousal_values.append(smoothed_arousal)

                while times and times[0] < elapsed - 60:
                    times.pop(0)
                    valence_values.pop(0)
                    arousal_values.pop(0)

                valence_line.set_data(times, valence_values)
                arousal_line.set_data(times, arousal_values)
                ax_hist.set_xlim(max(0, elapsed - 60), max(60, elapsed))

                # Update emotion bars
                for bar, prob in zip(bars, smoothed_probs):
                    bar.set_height(float(prob))

                if csv_writer:
                    csv_writer.writerow(
                        [
                            f"{elapsed:.3f}",
                            f"{smoothed_valence:.6f}",
                            f"{smoothed_arousal:.6f}",
                            f"{smoothed_dominance:.6f}",
                            smoothed_top_emotion,
                            f"{smoothed_top_prob:.6f}",
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