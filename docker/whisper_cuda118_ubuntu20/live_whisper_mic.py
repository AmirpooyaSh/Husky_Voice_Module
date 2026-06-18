#!/usr/bin/env python3

#!/usr/bin/env python3

import argparse
import queue
import sys
import time

import numpy as np
import sounddevice as sd
import torch
import whisper


WHISPER_SAMPLE_RATE = 16000


def block_rms(x):
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def resample_to_16k(audio, original_rate):
    """
    Lightweight resampling without extra dependencies.
    Good enough for live command/speech transcription.
    """
    if original_rate == WHISPER_SAMPLE_RATE:
        return audio.astype(np.float32)

    duration = len(audio) / float(original_rate)
    old_t = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_len = int(duration * WHISPER_SAMPLE_RATE)
    new_t = np.linspace(0.0, duration, num=new_len, endpoint=False)

    resampled = np.interp(new_t, old_t, audio)
    return resampled.astype(np.float32)


def get_default_input_rate(device_index=None):
    if device_index is not None:
        dev = sd.query_devices(device_index)
    else:
        default_input = sd.default.device[0]
        if default_input is None or default_input < 0:
            raise RuntimeError("No default input device found. Use --list-devices and then --mic INDEX.")
        dev = sd.query_devices(default_input)

    return int(dev["default_samplerate"])


def main():
    parser = argparse.ArgumentParser(
        description="Live microphone transcription using OpenAI Whisper."
    )

    parser.add_argument("--model", default="base.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument(
        "--mic-sample-rate",
        default="auto",
        help="Mic capture sample rate. Use auto, 48000, 44100, etc.",
    )

    parser.add_argument("--block-ms", type=int, default=100)
    parser.add_argument("--rms-threshold", type=float, default=0.006)
    parser.add_argument("--silence-ms", type=int, default=700)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--max-ms", type=int, default=7000)
    parser.add_argument("--no-speech-threshold", type=float, default=0.6)

    parser.add_argument("--mic", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        print("\nDefault device:", sd.default.device)
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested, but torch.cuda.is_available() is False.")
        sys.exit(1)

    if args.mic_sample_rate == "auto":
        mic_sample_rate = get_default_input_rate(args.mic)
    else:
        mic_sample_rate = int(args.mic_sample_rate)

    language = args.language.strip() or None
    use_fp16 = device == "cuda"

    print(f"Loading Whisper model: {args.model}")
    print(f"Device: {device}")
    print(f"FP16: {use_fp16}")
    print(f"Mic sample rate: {mic_sample_rate}")
    print(f"Whisper sample rate: {WHISPER_SAMPLE_RATE}")

    model = whisper.load_model(args.model, device=device)

    audio_queue = queue.Queue()
    blocksize = int(mic_sample_rate * args.block_ms / 1000)

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    print("\nAvailable input device:", args.mic if args.mic is not None else "default")
    print("Listening. Press Ctrl+C to stop.")
    print("Tip: say a short phrase, then pause.\n")
    print("> ", end="", flush=True)

    triggered = False
    speech_blocks = []
    speech_ms = 0
    silence_ms = 0

    try:
        with sd.InputStream(
            samplerate=mic_sample_rate,
            blocksize=blocksize,
            dtype="float32",
            channels=1,
            callback=callback,
            device=args.mic,
        ):
            while True:
                block = audio_queue.get()
                level = block_rms(block)
                is_speech = level >= args.rms_threshold

                if is_speech:
                    if not triggered:
                        triggered = True
                        speech_blocks = []
                        speech_ms = 0
                        silence_ms = 0

                    speech_blocks.append(block)
                    speech_ms += args.block_ms
                    silence_ms = 0

                elif triggered:
                    speech_blocks.append(block)
                    speech_ms += args.block_ms
                    silence_ms += args.block_ms

                should_transcribe = (
                    triggered
                    and (
                        silence_ms >= args.silence_ms
                        or speech_ms >= args.max_ms
                    )
                )

                if should_transcribe:
                    triggered = False
                    silence_ms = 0

                    audio = np.concatenate(speech_blocks).astype(np.float32)
                    duration = len(audio) / float(mic_sample_rate)

                    speech_blocks = []
                    speech_ms = 0

                    if duration < args.min_seconds:
                        print("> ", end="", flush=True)
                        continue

                    audio_16k = resample_to_16k(audio, mic_sample_rate)

                    print("\rTranscribing...        ", end="", flush=True)

                    result = model.transcribe(
                        audio_16k,
                        language=language,
                        task="transcribe",
                        fp16=use_fp16,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        no_speech_threshold=args.no_speech_threshold,
                    )

                    text = result.get("text", "").strip()

                    if text:
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"\r[{timestamp}] {text}")
                    else:
                        print("\r[no speech detected]")

                    print("> ", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print("\nERROR:", e)
        print("\nTry listing devices:")
        print("  python live_whisper_mic.py --list-devices")
        print("\nThen run with a specific mic and sample rate, for example:")
        print("  python live_whisper_mic.py --model base.en --mic 0 --mic-sample-rate 48000")


if __name__ == "__main__":
    main()