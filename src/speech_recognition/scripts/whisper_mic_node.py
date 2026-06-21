#!/usr/bin/env python3

import os
import json
import time
import tempfile
import subprocess

import rospy
import requests

from std_msgs.msg import String


class WhisperMicNode:
    def __init__(self):
        rospy.init_node("whisper_mic_node", anonymous=False)

        self.whisper_url = rospy.get_param(
            "~whisper_url",
            "http://127.0.0.1:8000/v1/audio/transcriptions"
        )

        self.model = rospy.get_param("~model", "base.en")
        self.language = rospy.get_param("~language", "en")

        self.chunk_seconds = float(rospy.get_param("~chunk_seconds", 3.0))
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 1))

        # For default mic, keep this as "default".
        # To list devices inside Docker:
        #   arecord -l
        self.audio_device = rospy.get_param("~audio_device", "default")

        self.timeout = float(rospy.get_param("~timeout", 30.0))

        self.pub_text = rospy.Publisher("/voice/text", String, queue_size=10)
        self.pub_json = rospy.Publisher("/voice/text_json", String, queue_size=10)

        rospy.loginfo("Whisper mic node started")
        rospy.loginfo("Whisper URL: %s", self.whisper_url)
        rospy.loginfo("Model: %s", self.model)
        rospy.loginfo("Audio device: %s", self.audio_device)

    def record_chunk(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()

        cmd = [
            "arecord",
            "-D", self.audio_device,
            "-f", "S16_LE",
            "-c", str(self.channels),
            "-r", str(self.sample_rate),
            "-d", str(int(self.chunk_seconds)),
            "-q",
            wav_path
        ]

        try:
            subprocess.check_call(cmd)
            return wav_path
        except subprocess.CalledProcessError as e:
            rospy.logerr("arecord failed: %s", str(e))
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return None

    def transcribe(self, wav_path):
        with open(wav_path, "rb") as f:
            files = {
                "file": ("audio.wav", f, "audio/wav")
            }

            data = {
                "model": self.model,
                "language": self.language,
                "response_format": "json"
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

        # OpenAI-compatible Whisper APIs usually return:
        #   {"text": "..."}
        text = result.get("text", "").strip()

        return text, result

    def spin(self):
        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            wav_path = None

            try:
                wav_path = self.record_chunk()

                if wav_path is None:
                    rospy.sleep(1.0)
                    continue

                start_time = time.time()
                text, raw_result = self.transcribe(wav_path)
                elapsed = time.time() - start_time

                if text:
                    rospy.loginfo("Transcribed: %s", text)

                    self.pub_text.publish(text)

                    msg = {
                        "text": text,
                        "model": self.model,
                        "language": self.language,
                        "chunk_seconds": self.chunk_seconds,
                        "transcription_time_sec": elapsed,
                        "timestamp": rospy.Time.now().to_sec(),
                        "raw": raw_result
                    }

                    self.pub_json.publish(json.dumps(msg))

            except Exception as e:
                rospy.logerr("Whisper mic node error: %s", str(e))
                rospy.sleep(1.0)

            finally:
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)

            rate.sleep()


if __name__ == "__main__":
    node = WhisperMicNode()
    node.spin()