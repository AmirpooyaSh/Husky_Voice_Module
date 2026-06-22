#!/usr/bin/env python3

import os
import time
import wave
import audioop
import tempfile
import subprocess
from collections import deque

import requests


WHISPER_URL = "http://127.0.0.1:9000/v1/audio/transcriptions"

# For hwdsl2/whisper-server, keep this as whisper-1.
# The real backend model is controlled by Docker WHISPER_MODEL=base.en/tiny.en/small.en
API_MODEL = "whisper-1"
LANGUAGE = "en"

# Your mic:
# plughw:2,0 = motherboard mic
# plughw:3,0 = Razer Kiyo Pro
# Lambda PC = plughw:1,0
AUDIO_DEVICE = "plughw:1,0"

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
SAMPLE_FORMAT = "S16_LE"

# Python reads this many seconds from the continuous arecord stream.
# This can be fractional because Python handles it, not arecord -d.
FRAME_SECONDS = 0.25

# VAD thresholds.
# Increase if background noise triggers speech.
# Decrease if quiet speech is missed.
RMS_START_THRESHOLD = 1000
RMS_CONTINUE_THRESHOLD = 180

# How much silence means the command is finished.
END_SILENCE_SECONDS = 0.8

# Keep a little audio before speech starts so the first word is not clipped.
PRE_ROLL_SECONDS = 0.5

# Avoid sending tiny noise bursts.
MIN_UTTERANCE_SECONDS = 0.5

# Prevent very long captures.
MAX_UTTERANCE_SECONDS = 6.0


BYTES_PER_FRAME = int(SAMPLE_RATE * FRAME_SECONDS) * CHANNELS * SAMPLE_WIDTH_BYTES
PRE_ROLL_FRAMES = max(1, int(PRE_ROLL_SECONDS / FRAME_SECONDS))


def start_arecord_stream():
    cmd = [
        "arecord",
        "-D", AUDIO_DEVICE,
        "-f", SAMPLE_FORMAT,
        "-c", str(CHANNELS),
        "-r", str(SAMPLE_RATE),
        "-t", "raw",
        "-q"
    ]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )


def read_exact(pipe, num_bytes):
    chunks = []
    remaining = num_bytes

    while remaining > 0:
        chunk = pipe.read(remaining)
        if not chunk:
            break

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def rms_of_frame(frame):
    if not frame:
        return 0
    return audioop.rms(frame, SAMPLE_WIDTH_BYTES)


def save_wav(frames_list):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = tmp.name
    tmp.close()

    audio_data = b"".join(frames_list)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)

    return wav_path


def transcribe(wav_path):
    """
    Send one saved utterance to the Whisper Docker server.

    Returns:
        text: Transcribed text.
        api_elapsed: Total time spent waiting for the Docker Whisper API response.
                     This includes HTTP transfer + model inference inside the container.
    """
    with open(wav_path, "rb") as f:
        files = {
            "file": ("audio.wav", f, "audio/wav")
        }

        data = {
            "model": API_MODEL,
            "language": LANGUAGE
        }

        api_start = time.time()
        response = requests.post(
            WHISPER_URL,
            files=files,
            data=data,
            timeout=60
        )
        api_elapsed = time.time() - api_start

    if response.status_code != 200:
        raise RuntimeError(
            "Whisper request failed. Status: {} Body: {}".format(
                response.status_code,
                response.text
            )
        )

    result = response.json()
    text = result.get("text", "").strip()
    return text, api_elapsed


def main():
    print("Continuous VAD-style Whisper Docker transcription")
    print("Whisper URL:", WHISPER_URL)
    print("Audio device:", AUDIO_DEVICE)
    print("Frame seconds:", FRAME_SECONDS)
    print("Press Ctrl+C to stop.")
    print()

    proc = start_arecord_stream()

    speaking = False
    utterance_frames = []
    pre_roll = deque(maxlen=PRE_ROLL_FRAMES)

    silence_time = 0.0
    utterance_time = 0.0

    try:
        while True:
            frame = read_exact(proc.stdout, BYTES_PER_FRAME)

            if len(frame) < BYTES_PER_FRAME:
                err = proc.stderr.read().decode(errors="ignore")
                raise RuntimeError("arecord stopped unexpectedly. stderr: {}".format(err))

            rms = rms_of_frame(frame)

            if not speaking:
                pre_roll.append(frame)

                if rms >= RMS_START_THRESHOLD:
                    speaking = True
                    utterance_frames = list(pre_roll)
                    silence_time = 0.0
                    utterance_time = len(utterance_frames) * FRAME_SECONDS

                    print("[speech started] rms={}".format(rms))
                else:
                    print("[silence] rms={}".format(rms))

                continue

            utterance_frames.append(frame)
            utterance_time += FRAME_SECONDS

            if rms < RMS_CONTINUE_THRESHOLD:
                silence_time += FRAME_SECONDS
            else:
                silence_time = 0.0

            should_end = silence_time >= END_SILENCE_SECONDS
            too_long = utterance_time >= MAX_UTTERANCE_SECONDS

            if should_end or too_long:
                if utterance_time < MIN_UTTERANCE_SECONDS:
                    speaking = False
                    utterance_frames = []
                    silence_time = 0.0
                    utterance_time = 0.0
                    continue

                wav_path = save_wav(utterance_frames)

                try:
                    text, whisper_api_time = transcribe(wav_path)

                    # How long the spoken audio was.
                    audio_duration = utterance_time

                    # Real-time factor: < 1.0 means faster than real time.
                    # Example: 0.25 means 4x faster than the audio length.
                    real_time_factor = whisper_api_time / audio_duration if audio_duration > 0 else 0.0

                    if text:
                        print(
                            "[whisper docker time: {:.3f}s | audio: {:.2f}s | RTF: {:.2f}x] {}".format(
                                whisper_api_time,
                                audio_duration,
                                real_time_factor,
                                text
                            )
                        )
                    else:
                        print(
                            "[whisper docker time: {:.3f}s | audio: {:.2f}s | RTF: {:.2f}x] No text".format(
                                whisper_api_time,
                                audio_duration,
                                real_time_factor
                            )
                        )

                finally:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)

                speaking = False
                utterance_frames = []
                pre_roll.clear()
                silence_time = 0.0
                utterance_time = 0.0

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()