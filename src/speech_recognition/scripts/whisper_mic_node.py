#!/usr/bin/env python3

import os
import json
import time
import wave
import audioop
import tempfile
import subprocess
from collections import deque

import rospy
import requests
from std_msgs.msg import String


class WhisperMicNode:
    def __init__(self):
        rospy.init_node("whisper_mic_node", anonymous=False)

        # Whisper Docker API
        self.whisper_url = rospy.get_param(
            "~whisper_url",
            "http://127.0.0.1:9000/v1/audio/transcriptions"
        )

        # For hwdsl2/whisper-server, keep this as whisper-1.
        # The real model is controlled by Docker WHISPER_MODEL=base.en/tiny.en/small.en.
        self.model = rospy.get_param("~model", "whisper-1")
        self.language = rospy.get_param("~language", "en")

        # Audio device
        # plughw:2,0 = motherboard mic input
        # plughw:3,0 = Razer Kiyo Pro
        self.audio_device = rospy.get_param("~audio_device", "plughw:2,0")

        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 1))
        self.sample_width_bytes = int(rospy.get_param("~sample_width_bytes", 2))
        self.sample_format = rospy.get_param("~sample_format", "S16_LE")

        # VAD parameters
        self.frame_seconds = float(rospy.get_param("~frame_seconds", 0.25))

        # Your requested value
        self.rms_start_threshold = int(rospy.get_param("~rms_start_threshold", 1000))

        # Usually lower than start threshold so speech does not get cut off too early
        self.rms_continue_threshold = int(rospy.get_param("~rms_continue_threshold", 600))

        self.end_silence_seconds = float(rospy.get_param("~end_silence_seconds", 0.8))
        self.pre_roll_seconds = float(rospy.get_param("~pre_roll_seconds", 0.5))
        self.min_utterance_seconds = float(rospy.get_param("~min_utterance_seconds", 0.5))
        self.max_utterance_seconds = float(rospy.get_param("~max_utterance_seconds", 6.0))

        self.timeout = float(rospy.get_param("~timeout", 60.0))

        # Publishers
        self.pub_text = rospy.Publisher("/voice/text", String, queue_size=10)
        self.pub_json = rospy.Publisher("/voice/text_json", String, queue_size=10)
        self.pub_final_json = rospy.Publisher("/voice/final_text_json", String, queue_size=10)

        self.bytes_per_frame = int(
            self.sample_rate * self.frame_seconds
        ) * self.channels * self.sample_width_bytes

        self.pre_roll_frames = max(1, int(self.pre_roll_seconds / self.frame_seconds))

        self.arecord_proc = None

        rospy.loginfo("Whisper mic VAD node started")
        rospy.loginfo("Whisper URL: %s", self.whisper_url)
        rospy.loginfo("Model/API name: %s", self.model)
        rospy.loginfo("Language: %s", self.language)
        rospy.loginfo("Audio device: %s", self.audio_device)
        rospy.loginfo("Frame seconds: %.3f", self.frame_seconds)
        rospy.loginfo("RMS start threshold: %d", self.rms_start_threshold)
        rospy.loginfo("RMS continue threshold: %d", self.rms_continue_threshold)
        rospy.loginfo("End silence seconds: %.3f", self.end_silence_seconds)

    def start_arecord_stream(self):
        cmd = [
            "arecord",
            "-D", self.audio_device,
            "-f", self.sample_format,
            "-c", str(self.channels),
            "-r", str(self.sample_rate),
            "-t", "raw",
            "-q"
        ]

        rospy.loginfo("Starting arecord stream: %s", " ".join(cmd))

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

    def stop_arecord_stream(self):
        if self.arecord_proc is not None:
            try:
                self.arecord_proc.terminate()
                self.arecord_proc.wait(timeout=2)
            except Exception:
                try:
                    self.arecord_proc.kill()
                except Exception:
                    pass

            self.arecord_proc = None

    def read_exact(self, pipe, num_bytes):
        chunks = []
        remaining = num_bytes

        while remaining > 0 and not rospy.is_shutdown():
            chunk = pipe.read(remaining)

            if not chunk:
                break

            chunks.append(chunk)
            remaining -= len(chunk)

        return b"".join(chunks)

    def rms_of_frame(self, frame):
        if not frame:
            return 0

        return audioop.rms(frame, self.sample_width_bytes)

    def save_wav(self, frames_list):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()

        audio_data = b"".join(frames_list)

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width_bytes)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)

        return wav_path

    def transcribe(self, wav_path):
        with open(wav_path, "rb") as f:
            files = {
                "file": ("audio.wav", f, "audio/wav")
            }

            data = {
                "model": self.model,
                "language": self.language
            }

            response = requests.post(
                self.whisper_url,
                files=files,
                data=data,
                timeout=self.timeout
            )

        if response.status_code != 200:
            raise RuntimeError(
                "Whisper request failed. Status: {} Body: {}".format(
                    response.status_code,
                    response.text
                )
            )

        result = response.json()
        text = result.get("text", "").strip()

        return text, result

    def publish_transcription(self, text, raw_result, utterance_seconds, rms_start, inference_time):
        if not text:
            rospy.loginfo("Whisper returned no text")
            return

        rospy.loginfo("Transcribed: %s", text)

        self.pub_text.publish(text)

        msg = {
            "text": text,
            "model": self.model,
            "language": self.language,
            "audio_device": self.audio_device,
            "utterance_seconds": utterance_seconds,
            "rms_start": rms_start,
            "rms_start_threshold": self.rms_start_threshold,
            "rms_continue_threshold": self.rms_continue_threshold,
            "inference_time_sec": inference_time,
            "timestamp": rospy.Time.now().to_sec(),
            "raw": raw_result
        }

        json_msg = json.dumps(msg)

        self.pub_json.publish(json_msg)
        self.pub_final_json.publish(json_msg)

    def spin(self):
        speaking = False
        utterance_frames = []
        pre_roll = deque(maxlen=self.pre_roll_frames)

        silence_time = 0.0
        utterance_time = 0.0
        speech_start_rms = 0

        self.arecord_proc = self.start_arecord_stream()

        rospy.on_shutdown(self.stop_arecord_stream)

        while not rospy.is_shutdown():
            try:
                frame = self.read_exact(self.arecord_proc.stdout, self.bytes_per_frame)

                if len(frame) < self.bytes_per_frame:
                    err = ""

                    try:
                        err = self.arecord_proc.stderr.read().decode(errors="ignore")
                    except Exception:
                        pass

                    rospy.logerr("arecord stopped unexpectedly. stderr: %s", err)
                    self.stop_arecord_stream()
                    rospy.sleep(1.0)
                    self.arecord_proc = self.start_arecord_stream()

                    speaking = False
                    utterance_frames = []
                    pre_roll.clear()
                    silence_time = 0.0
                    utterance_time = 0.0
                    speech_start_rms = 0
                    continue

                rms = self.rms_of_frame(frame)

                if not speaking:
                    pre_roll.append(frame)

                    if rms >= self.rms_start_threshold:
                        speaking = True
                        utterance_frames = list(pre_roll)
                        silence_time = 0.0
                        utterance_time = len(utterance_frames) * self.frame_seconds
                        speech_start_rms = rms

                        rospy.loginfo("Speech started. rms=%d", rms)
                    else:
                        rospy.logdebug("Silence. rms=%d", rms)

                    continue

                utterance_frames.append(frame)
                utterance_time += self.frame_seconds

                if rms < self.rms_continue_threshold:
                    silence_time += self.frame_seconds
                else:
                    silence_time = 0.0

                should_end = silence_time >= self.end_silence_seconds
                too_long = utterance_time >= self.max_utterance_seconds

                if should_end or too_long:
                    if utterance_time < self.min_utterance_seconds:
                        speaking = False
                        utterance_frames = []
                        pre_roll.clear()
                        silence_time = 0.0
                        utterance_time = 0.0
                        speech_start_rms = 0
                        continue

                    rospy.loginfo(
                        "Speech ended. utterance=%.2f sec, silence=%.2f sec",
                        utterance_time,
                        silence_time
                    )

                    wav_path = self.save_wav(utterance_frames)

                    try:
                        start = time.time()
                        text, raw_result = self.transcribe(wav_path)
                        inference_time = time.time() - start

                        self.publish_transcription(
                            text=text,
                            raw_result=raw_result,
                            utterance_seconds=utterance_time,
                            rms_start=speech_start_rms,
                            inference_time=inference_time
                        )

                    except Exception as e:
                        rospy.logerr("Whisper mic node error: %s", str(e))

                    finally:
                        if os.path.exists(wav_path):
                            os.remove(wav_path)

                    speaking = False
                    utterance_frames = []
                    pre_roll.clear()
                    silence_time = 0.0
                    utterance_time = 0.0
                    speech_start_rms = 0

            except Exception as e:
                rospy.logerr("Whisper VAD loop error: %s", str(e))
                rospy.sleep(1.0)


if __name__ == "__main__":
    node = WhisperMicNode()
    node.spin()
