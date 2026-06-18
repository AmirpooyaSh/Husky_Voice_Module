#!/usr/bin/env python3
"""
ROS Vosk speech-to-text publisher with optional GPU BatchRecognizer support.

Publishes JSON on /voice/final_text_json by default:
    {
      "text": "...",
      "finalization_time_sec": 0.123,
      "engine": "vosk_gpu_batch" or "vosk_cpu_stream"
    }

Important:
- GPU mode requires a Vosk build compiled with CUDA support.
- The normal pip `vosk` package usually does NOT include CUDA support.
- GPU mode uses BatchModel / BatchRecognizer / GpuInit.
- CPU mode uses Model / KaldiRecognizer and keeps the original partial-result logic.

ROS params:
    ~model_path              path to Vosk model
    ~topic                   output topic
    ~sample_rate             default 16000
    ~block_size              default 8000
    ~device                  sounddevice input device, empty/default means default mic
    ~use_gpu                 default true
    ~gpu_required            default false; if true, shutdown if GPU init fails
    ~speech_rms_threshold    default 0.01, used only for GPU timing because BatchRecognizer has no PartialResult
"""

import os
import json
import queue
import time

import rospy
import rospkg

from std_msgs.msg import String

import numpy as np
import sounddevice as sd

# CPU imports are always expected in a normal Vosk installation.
from vosk import Model, KaldiRecognizer

# GPU imports are optional because they only exist when Vosk was built with CUDA batch support.
try:
    from vosk import BatchModel, BatchRecognizer, GpuInit
    VOSK_GPU_IMPORT_OK = True
except Exception:
    BatchModel = None
    BatchRecognizer = None
    GpuInit = None
    VOSK_GPU_IMPORT_OK = False


class VoskTextPublisher:
    def __init__(self):
        rospy.init_node("vosk_speech_to_text", anonymous=False)

        rospack = rospkg.RosPack()
        package_path = rospack.get_path("speech_recognition")

        default_model_path = os.path.join(
            package_path,
            "models",
            "vosk-model-en-us-0.22"
        )

        self.model_path = rospy.get_param("~model_path", default_model_path)
        self.topic_name = rospy.get_param("~topic", "/voice/final_text_json")
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.block_size = int(rospy.get_param("~block_size", 8000))

        # GPU options
        self.use_gpu = bool(rospy.get_param("~use_gpu", True))
        self.gpu_required = bool(rospy.get_param("~gpu_required", False))

        # GPU BatchRecognizer does not provide PartialResult like KaldiRecognizer.
        # We use audio RMS to estimate when a spoken segment starts for latency timing.
        self.speech_rms_threshold = float(rospy.get_param("~speech_rms_threshold", 0.01))

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

        self.engine_name = "vosk_cpu_stream"
        self.model = None
        self.recognizer = None

        # Used to measure how long each spoken segment takes to finalize
        self.utterance_start_time = None
        self.last_partial_text = ""
        self.last_gpu_raw_result = ""

        if self.use_gpu:
            self._init_gpu_or_fallback()
        else:
            self._init_cpu()

        rospy.loginfo("Vosk speech-to-text JSON node started.")
        rospy.loginfo("Publishing JSON text on topic: %s", self.topic_name)
        rospy.loginfo("Sample rate: %d", self.sample_rate)
        rospy.loginfo("Block size: %d", self.block_size)
        rospy.loginfo("Audio device: %s", str(self.device))
        rospy.loginfo("Vosk engine: %s", self.engine_name)

    def _init_cpu(self):
        rospy.loginfo("Loading CPU Vosk model from: %s", self.model_path)
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.engine_name = "vosk_cpu_stream"

    def _init_gpu_or_fallback(self):
        if not VOSK_GPU_IMPORT_OK:
            msg = (
                "Vosk GPU classes are not available. "
                "Your installed vosk Python package/libvosk was probably not built with CUDA. "
                "Expected imports: BatchModel, BatchRecognizer, GpuInit."
            )
            if self.gpu_required:
                rospy.logerr(msg)
                rospy.signal_shutdown("Vosk GPU support missing")
                return
            rospy.logwarn(msg)
            rospy.logwarn("Falling back to CPU Vosk recognizer.")
            self._init_cpu()
            return

        try:
            rospy.loginfo("Initializing Vosk CUDA context with GpuInit()...")
            GpuInit()

            rospy.loginfo("Loading GPU BatchModel from: %s", self.model_path)
            self.model = BatchModel(self.model_path)
            self.recognizer = BatchRecognizer(self.model, self.sample_rate)
            self.engine_name = "vosk_gpu_batch"

            rospy.loginfo("Vosk GPU BatchRecognizer initialized.")

        except Exception as e:
            msg = "Failed to initialize Vosk GPU BatchRecognizer: %s" % str(e)
            if self.gpu_required:
                rospy.logerr(msg)
                rospy.signal_shutdown("Vosk GPU initialization failed")
                return
            rospy.logwarn(msg)
            rospy.logwarn("Falling back to CPU Vosk recognizer.")
            self._init_cpu()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            rospy.logwarn("Audio input status: %s", str(status))

        self.audio_queue.put(bytes(indata))

    def publish_json_result(self, text, finalization_time_sec):
        payload = {
            "text": text,
            "finalization_time_sec": round(finalization_time_sec, 3),
            "engine": self.engine_name
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

        rospy.loginfo("Published JSON: %s", msg.data)

    def _chunk_rms_float(self, data):
        """Return normalized RMS for int16 PCM bytes, roughly 0.0 to 1.0."""
        if not data:
            return 0.0

        audio = np.frombuffer(data, dtype=np.int16)
        if audio.size == 0:
            return 0.0

        audio_f = audio.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(np.square(audio_f)) + 1e-12))

    def _run_cpu_loop(self):
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

    def _run_gpu_loop(self):
        """
        GPU BatchRecognizer loop.

        BatchRecognizer is different from KaldiRecognizer:
        - You feed chunks with AcceptWaveform(data).
        - You call model.Wait() so CUDA processing catches up.
        - You read completed recognition chunks with recognizer.Result().
        - There is no PartialResult() path in the batch API, so we use RMS to time speech start.
        """
        while not rospy.is_shutdown():
            try:
                data = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Start timing on first non-silent chunk.
            if self.utterance_start_time is None:
                chunk_rms = self._chunk_rms_float(data)
                if chunk_rms >= self.speech_rms_threshold:
                    self.utterance_start_time = time.perf_counter()

            try:
                self.recognizer.AcceptWaveform(data)

                # Wait for queued CUDA work before pulling Result().
                self.model.Wait()

                raw_result = self.recognizer.Result()

            except Exception as e:
                rospy.logerr("Vosk GPU recognition error: %s", str(e))
                continue

            # BatchRecognizer may return "" until a segment is ready.
            if not raw_result:
                continue

            # Avoid duplicate publication if a binding returns the same result repeatedly.
            if raw_result == self.last_gpu_raw_result:
                continue

            self.last_gpu_raw_result = raw_result

            try:
                result = json.loads(raw_result)
            except Exception:
                rospy.logwarn("Could not parse Vosk GPU result as JSON: %s", raw_result)
                continue

            text = result.get("text", "").strip()
            if not text:
                continue

            now = time.perf_counter()
            if self.utterance_start_time is None:
                finalization_time_sec = 0.0
            else:
                finalization_time_sec = now - self.utterance_start_time

            self.publish_json_result(text, finalization_time_sec)

            # Reset timing for next completed speech segment.
            self.utterance_start_time = None

    def run(self):
        if self.recognizer is None:
            rospy.logerr("Recognizer was not initialized.")
            return

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

                if self.engine_name == "vosk_gpu_batch":
                    self._run_gpu_loop()
                else:
                    self._run_cpu_loop()

        except Exception as e:
            rospy.logerr("Vosk node error: %s", str(e))

        finally:
            self._publish_remaining_final_result()

    def _publish_remaining_final_result(self):
        try:
            if self.engine_name == "vosk_gpu_batch":
                try:
                    self.recognizer.FinishStream()
                except Exception:
                    pass

                try:
                    self.model.Wait()
                except Exception:
                    pass

                raw_result = self.recognizer.Result()
                if not raw_result:
                    return

                final_result = json.loads(raw_result)

            else:
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
