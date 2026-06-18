#!/usr/bin/env python3
"""
Live Whisper VAD + categorical emotion visualizer.

VAD here means:
    Valence, Arousal, Dominance

This script uses the Vox-Profile Whisper emotion models:

Dimensional VAD model:
    tiantiaf/whisper-large-v3-msp-podcast-emotion-dim

Categorical emotion model:
    tiantiaf/whisper-large-v3-msp-podcast-emotion

The dimensional model returns:
    arousal, valence, dominance

The categorical model returns emotion logits.

The model input is kept as:
    16 kHz, mono, max 15 seconds

Dual-GPU mode:
    VAD model      -> --vad-device, default cuda:0
    Emotion model  -> --emotion-device, default cuda:1

Use Docker with both GPUs visible, for example:
    docker run --gpus all ...

Valence and arousal are rescaled from [0, 1] to [-1, 1].
Dominance is kept in its original [0, 1] range.

Example:
    python live_whisper_vad_emotion_gpu.py \
      --va-background v-a-indicator.png \
      --input-sr 48000 \
      --audio-device DEVICE_INDEX \
      --vad-device cuda:0 \
      --emotion-device cuda:1 \
      --window-sec 3.0 \
      --hop-sec 1.0 \
      --silence-rms 0.005 \
      --matplotlib-backend Qt5Agg \
      --log-csv logs/whisper_vad_emotion_live.csv
"""

import argparse
import csv
import os
import queue
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

# Limit CPU thread oversubscription.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly


def _get_requested_matplotlib_backend() -> Optional[str]:
    """Read --matplotlib-backend before importing pyplot."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matplotlib-backend", default=None)
    args, _ = parser.parse_known_args()
    return args.matplotlib_backend


_requested_backend = _get_requested_matplotlib_backend()
if _requested_backend:
    import matplotlib
    matplotlib.use(_requested_backend)

import matplotlib.pyplot as plt


# Robust imports for running either from /workspace or inside the cloned repo.
try:
    from src.model.emotion.whisper_emotion_dim import WhisperWrapper as DimEmotionWrapper
    from src.model.emotion.whisper_emotion import WhisperWrapper as CatEmotionWrapper
except Exception:
    this_file = Path(os.path.realpath(__file__))
    candidate_roots = [
        this_file.parent,
        this_file.parent.parent,
        Path("/opt/vox-profile-release"),
        Path("/workspace/vox-profile-release"),
    ]

    for root in candidate_roots:
        if str(root) not in sys.path:
            sys.path.append(str(root))
        emotion_dir = root / "src" / "model" / "emotion"
        if emotion_dir.exists() and str(emotion_dir) not in sys.path:
            sys.path.append(str(emotion_dir))

    try:
        from src.model.emotion.whisper_emotion_dim import WhisperWrapper as DimEmotionWrapper
        from src.model.emotion.whisper_emotion import WhisperWrapper as CatEmotionWrapper
    except Exception:
        from whisper_emotion_dim import WhisperWrapper as DimEmotionWrapper
        from whisper_emotion import WhisperWrapper as CatEmotionWrapper


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
    """Convert microphone chunks to mono float32 in [-1, 1]."""
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


def resample_to_16k(audio: np.ndarray, input_sr: int) -> np.ndarray:
    """Resample input audio to 16 kHz for the Whisper emotion models."""
    if input_sr == MODEL_SAMPLE_RATE:
        return audio.astype(np.float32)

    gcd = np.gcd(input_sr, MODEL_SAMPLE_RATE)
    up = MODEL_SAMPLE_RATE // gcd
    down = input_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    """Root-mean-square energy used as a simple silence gate."""
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def scale_0_1_to_neg1_1(value: float) -> float:
    """Convert model output from [0, 1] to [-1, 1]."""
    return float(np.clip((2.0 * value) - 1.0, -1.0, 1.0))


def force_module_to_device(module, device: torch.device, module_name: str = "model"):
    """
    Robustly move a Vox-Profile wrapper and its nested Hugging Face / Whisper
    modules to one target device.

    Some custom wrappers keep nested modules such as `backbone_model` as
    attributes. Calling wrapper.to(device) should move them, but in practice
    a nested Whisper encoder can remain on cuda:0 while the input is on cuda:1.
    This helper explicitly moves common nested module attributes too.
    """
    module.to(device)

    for attr_name in [
        "backbone_model",
        "model",
        "encoder",
        "decoder",
        "feature_extractor",
        "classifier",
        "projector",
        "layer_norm",
    ]:
        if hasattr(module, attr_name):
            attr = getattr(module, attr_name)
            if isinstance(attr, torch.nn.Module):
                attr.to(device)

    # Final recursive move catches registered submodules, parameters, and buffers.
    module.to(device)

    # Verify registered parameters/buffers.
    bad_items = []

    for name, param in module.named_parameters(recurse=True):
        if param.device != device:
            bad_items.append((f"parameter:{name}", str(param.device)))

    for name, buf in module.named_buffers(recurse=True):
        if buf.device != device:
            bad_items.append((f"buffer:{name}", str(buf.device)))

    if bad_items:
        preview = ", ".join([f"{name} on {dev}" for name, dev in bad_items[:8]])
        raise RuntimeError(
            f"{module_name} is not fully on {device}. Mismatched items: {preview}"
        )

    return module


def print_module_device_summary(module, module_name: str):
    """Print a compact summary of where a model's parameters live."""
    devices = {}

    for _, param in module.named_parameters(recurse=True):
        devices[str(param.device)] = devices.get(str(param.device), 0) + param.numel()

    for _, buf in module.named_buffers(recurse=True):
        devices[str(buf.device)] = devices.get(str(buf.device), 0) + buf.numel()

    print(f"{module_name} device summary:")
    for dev, count in sorted(devices.items()):
        print(f"  {dev}: {count:,} parameter/buffer elements")


def _extract_logits_from_cat_output(cat_output):
    """
    Vox-Profile Whisper categorical model commonly returns:
        logits, embedding, _, _, _, _ = model(x, return_feature=True)

    This helper also supports simpler outputs for compatibility.
    """
    if isinstance(cat_output, (tuple, list)):
        return cat_output[0]
    return cat_output


def predict_all(
    dim_model,
    cat_model,
    vad_device: torch.device,
    emotion_device: torch.device,
    audio_16k: np.ndarray,
):
    """
    Run both Whisper models on separate devices:
        1. VAD dimensional model      -> vad_device
        2. Categorical emotion model  -> emotion_device

    The audio input is tiny compared with the model size, so we create one
    tensor on each GPU rather than moving model outputs between GPUs.
    """
    max_len = 15 * MODEL_SAMPLE_RATE
    audio_16k = audio_16k[-max_len:]

    x_vad = torch.from_numpy(audio_16k).float().unsqueeze(0).to(vad_device)
    x_emotion = torch.from_numpy(audio_16k).float().unsqueeze(0).to(emotion_device)

    with torch.inference_mode():
        # Dimensional model returns arousal, valence, dominance.
        if vad_device.type == "cuda":
            with torch.cuda.device(vad_device):
                arousal, valence, dominance = dim_model(x_vad)
        else:
            arousal, valence, dominance = dim_model(x_vad)

        # Categorical model returns logits plus optional features.
        if emotion_device.type == "cuda":
            with torch.cuda.device(emotion_device):
                try:
                    cat_output = cat_model(x_emotion, return_feature=True)
                except TypeError:
                    cat_output = cat_model(x_emotion)
        else:
            try:
                cat_output = cat_model(x_emotion, return_feature=True)
            except TypeError:
                cat_output = cat_model(x_emotion)

        logits = _extract_logits_from_cat_output(cat_output)
        probs = F.softmax(logits, dim=1).squeeze(0)

    # Vox-Profile dimensional outputs are expected in [0, 1].
    # Rescale valence and arousal to [-1, 1] for the VA plane.
    arousal = scale_0_1_to_neg1_1(float(arousal.squeeze().detach().cpu()))
    valence = scale_0_1_to_neg1_1(float(valence.squeeze().detach().cpu()))

    # Keep dominance in its original [0, 1] range.
    dominance = float(dominance.squeeze().detach().cpu())

    probs_np = probs.detach().cpu().numpy()
    top_idx = int(np.argmax(probs_np))
    top_emotion = EMOTION_LABELS[top_idx]
    top_prob = float(probs_np[top_idx])

    return valence, arousal, dominance, top_emotion, top_prob, probs_np


def setup_plot(background_path: str):
    plt.ion()

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1])

    ax_va = fig.add_subplot(grid[:, 0])
    ax_hist = fig.add_subplot(grid[0, 1])
    ax_cat = fig.add_subplot(grid[1, 1])

    # Valence-arousal background.
    # x-axis = Valence, y-axis = Arousal, both in [-1, 1].
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
    ax_va.set_title("Whisper VAD: Valence / Arousal")

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

    # Rolling V/A plot.
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

    # Categorical emotion probabilities.
    ax_cat.set_title("Whisper Categorical Emotion")
    ax_cat.set_ylim(0, 1)
    ax_cat.set_ylabel("Probability")
    bars = ax_cat.bar(EMOTION_LABELS, np.zeros(len(EMOTION_LABELS)))
    ax_cat.tick_params(axis="x", rotation=35)

    fig.tight_layout()

    return (
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
    )


def parse_torch_device(device_name: str, role_name: str, require_cuda: bool = False) -> torch.device:
    """
    Parse and validate a Torch device string such as:
        cpu
        cuda
        cuda:0
        cuda:1
        auto

    For this dual-GPU script, prefer explicit cuda:0 and cuda:1.
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if require_cuda:
            raise RuntimeError(f"{role_name}: CUDA is required, but torch.cuda.is_available() is False.")
        return torch.device("cpu")

    if device_name == "cpu":
        if require_cuda:
            raise RuntimeError(f"{role_name}: CUDA is required, but device was set to CPU.")
        return torch.device("cpu")

    if device_name == "cuda":
        device_name = "cuda:0"

    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{role_name}: CUDA device {device_name} was requested, but CUDA is not available.")

        device = torch.device(device_name)
        index = 0 if device.index is None else int(device.index)
        count = torch.cuda.device_count()

        if index < 0 or index >= count:
            raise RuntimeError(
                f"{role_name}: requested {device_name}, but only {count} CUDA device(s) are visible. "
                f"Run Docker with both GPUs visible, e.g. `--gpus all`, and check `nvidia-smi`."
            )

        return torch.device(f"cuda:{index}")

    raise ValueError(f"{role_name}: unsupported device string: {device_name}")


def cuda_sanity_check(device: torch.device, role_name: str):
    """Small tensor test on a specific GPU."""
    if device.type != "cuda":
        print(f"{role_name}: using CPU")
        return

    with torch.cuda.device(device):
        print(f"{role_name}: using {device} -> {torch.cuda.get_device_name(device.index)}")
        print(f"{role_name}: CUDA capability: {torch.cuda.get_device_capability(device.index)}")
        x = torch.randn(128, 128, device=device)
        y = x @ x
        torch.cuda.synchronize(device)
        print(f"{role_name}: CUDA tensor test OK")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dim-model",
        default="tiantiaf/whisper-large-v3-msp-podcast-emotion-dim",
        
        help="Whisper dimensional VAD model.",
    )

    parser.add_argument(
        "--cat-model",
        default="tiantiaf/whisper-large-v3-msp-podcast-emotion",
        help="Whisper categorical emotion model.",
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
        "--audio-device",
        type=int,
        default=None,
        help="Microphone input device index. Use `python -m sounddevice` to list devices.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Deprecated alias for --audio-device. This is the microphone device index, not CUDA.",
    )

    parser.add_argument(
        "--vad-device",
        default="cuda:0",
        help="Torch device for the Whisper VAD model. Example: cuda:0",
    )

    parser.add_argument(
        "--emotion-device",
        default="cuda:1",
        help="Torch device for the Whisper categorical emotion model. Example: cuda:1",
    )

    parser.add_argument(
        "--compute-device",
        default=None,
        help=(
            "Deprecated single-device option kept for backward compatibility. "
            "If provided, both models use this device unless --vad-device/--emotion-device are explicitly changed."
        ),
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force both models to CPU.",
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit if either selected model device is not CUDA.",
    )

    parser.add_argument(
        "--matplotlib-backend",
        default=None,
        help="Example: Qt5Agg. Must be installed in the environment.",
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

    if args.audio_device is None and args.device is not None:
        args.audio_device = args.device

    if args.window_sec < 3.0:
        raise ValueError("Use --window-sec >= 3.0. Shorter clips may be unreliable.")

    if args.hop_sec <= 0:
        raise ValueError("--hop-sec must be positive.")

    if not os.path.exists(args.va_background):
        raise FileNotFoundError(
            f"Could not find background image: {args.va_background}\n"
            f"Put your image in this folder or pass --va-background /full/path/to/v-a-indicator.png"
        )

    # Backward compatibility:
    # If the old --compute-device is used, put both models on that same device.
    # Otherwise, default is VAD -> cuda:0 and Emotion -> cuda:1.
    if args.cpu:
        args.vad_device = "cpu"
        args.emotion_device = "cpu"
    elif args.compute_device is not None:
        args.vad_device = args.compute_device
        args.emotion_device = args.compute_device

    vad_device = parse_torch_device(args.vad_device, "VAD model", require_cuda=args.require_cuda)
    emotion_device = parse_torch_device(args.emotion_device, "Emotion model", require_cuda=args.require_cuda)

    if args.require_cuda and (vad_device.type != "cuda" or emotion_device.type != "cuda"):
        raise RuntimeError("CUDA is required, but at least one model is not running on CUDA.")

    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Visible CUDA device count: {torch.cuda.device_count()}")
        print(f"Torch CUDA version: {torch.version.cuda}")

    cuda_sanity_check(vad_device, "VAD model")
    cuda_sanity_check(emotion_device, "Emotion model")

    print("Loading Whisper dimensional VAD model...")
    if vad_device.type == "cuda":
        torch.cuda.set_device(vad_device)
        with torch.cuda.device(vad_device):
            dim_model = DimEmotionWrapper.from_pretrained(args.dim_model)
    else:
        dim_model = DimEmotionWrapper.from_pretrained(args.dim_model)

    dim_model = force_module_to_device(dim_model, vad_device, "VAD model")
    dim_model.eval()
    print_module_device_summary(dim_model, "VAD model")

    print("Loading Whisper categorical emotion model...")
    if emotion_device.type == "cuda":
        torch.cuda.set_device(emotion_device)
        with torch.cuda.device(emotion_device):
            cat_model = CatEmotionWrapper.from_pretrained(args.cat_model)
    else:
        cat_model = CatEmotionWrapper.from_pretrained(args.cat_model)

    cat_model = force_module_to_device(cat_model, emotion_device, "Emotion model")
    cat_model.eval()
    print_module_device_summary(cat_model, "Emotion model")

    print("Models loaded.")
    print(f"Opening microphone input device: {args.audio_device}")
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
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
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
                ) = predict_all(dim_model, cat_model, vad_device, emotion_device, segment)

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

                print(
                    f"[{elapsed:7.2f}s] "
                    f"Valence [-1,1]: {smoothed_valence:+.3f} | "
                    f"Arousal [-1,1]: {smoothed_arousal:+.3f} | "
                    f"Dominance [0,1]: {smoothed_dominance:.3f} | "
                    f"Emotion: {smoothed_top_emotion} ({smoothed_top_prob:.2f}) | "
                    f"RMS: {segment_rms:.4f}",
                    flush=True,
                )

                # Update point on VA image.
                point.set_data([smoothed_valence], [smoothed_arousal])
                point_shadow.set_data([smoothed_valence], [smoothed_arousal])

                info_text.set_text(
                    f"Valence [-1,1]:   {smoothed_valence:.3f}\n"
                    f"Arousal [-1,1]:   {smoothed_arousal:.3f}\n"
                    f"Dominance [0,1]:  {smoothed_dominance:.3f}\n"
                    f"Emotion:   {smoothed_top_emotion} ({smoothed_top_prob:.2f})\n"
                    f"RMS:       {segment_rms:.4f}"
                )

                # Update rolling V/A plot.
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

                # Update categorical emotion bars.
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