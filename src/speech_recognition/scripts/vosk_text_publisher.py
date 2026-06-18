#!/usr/bin/env python3

import os
import sys
import json
import queue
import time

import rospy
import rospkg

from std_msgs.msg import String

import sounddevice as sd
from vosk import Model, KaldiRecognizer


class VoskTextPublisher:
    def __init__(self):
        rospy.init_node("vosk_speech_to_text", anonymous=False)

        rospack = rospkg.RosPack()
        package_path = rospack.get_path("speech_recognition")

        default_model_path = os.path.join(
            package_path,
            "models",
            "vosk-model-small-en-us-0.15"
        )

        self.model_path = rospy.get_param("~model_path", default_model_path)
        self.topic_name = rospy.get_param("~topic", "/voice/final_text_json")
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.block_size = int(rospy.get_param("~block_size", 8000))

        # Leave empty for default microphone
        self.device = rospy.get_param("~device", "")
        if self.device == "":
            self.device = None

        self.audio_queue = queue.Queue()

        self.pub = rospy.Publisher(
            self.topic_name,
            String,
            queue_size=10
        )

        if not os.path.isdir(self.model_path):
            rospy.logerr("Vosk model path does not exist: %s", self.model_path)
            rospy.signal_shutdown("Missing Vosk model")
            return

        rospy.loginfo("Loading Vosk model from: %s", self.model_path)
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

        # Used to measure how long each spoken segment takes to finalize
        self.utterance_start_time = None
        self.last_partial_text = ""

        rospy.loginfo("Vosk speech-to-text JSON node started.")
        rospy.loginfo("Publishing JSON text on topic: %s", self.topic_name)
        rospy.loginfo("Sample rate: %d", self.sample_rate)
        rospy.loginfo("Audio device: %s", str(self.device))

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            rospy.logwarn("Audio input status: %s", str(status))

        self.audio_queue.put(bytes(indata))

    def publish_json_result(self, text, finalization_time_sec):
        payload = {
            "text": text,
            "finalization_time_sec": round(finalization_time_sec, 3)
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

        rospy.loginfo("Published JSON: %s", msg.data)

    def run(self):
        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype="int16",
                channels=1,
                callback=self.audio_callback,
                device=self.device
            ):
                rospy.loginfo("Listening...")

                while not rospy.is_shutdown():
                    try:
                        data = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "").strip()

                        if text:
                            now = time.perf_counter()

                            if self.utterance_start_time is None:
                                finalization_time_sec = 0.0
                            else:
                                finalization_time_sec = now - self.utterance_start_time

                            self.publish_json_result(text, finalization_time_sec)

                        # Reset timing for next spoken segment
                        self.utterance_start_time = None
                        self.last_partial_text = ""

                    else:
                        partial_result = json.loads(self.recognizer.PartialResult())
                        partial_text = partial_result.get("partial", "").strip()

                        # Start timing when Vosk first detects speech content
                        if partial_text and self.utterance_start_time is None:
                            self.utterance_start_time = time.perf_counter()

                        self.last_partial_text = partial_text

        except Exception as e:
            rospy.logerr("Vosk node error: %s", str(e))

        finally:
            try:
                final_result = json.loads(self.recognizer.FinalResult())
                final_text = final_result.get("text", "").strip()

                if final_text:
                    now = time.perf_counter()

                    if self.utterance_start_time is None:
                        finalization_time_sec = 0.0
                    else:
                        finalization_time_sec = now - self.utterance_start_time

                    self.publish_json_result(final_text, finalization_time_sec)

            except Exception:
                pass


if __name__ == "__main__":
    node = VoskTextPublisher()
    node.run()