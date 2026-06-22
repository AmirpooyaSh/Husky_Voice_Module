#!/usr/bin/env python3

import json
import time
import threading

import rospy
import requests
from std_msgs.msg import String


def normalize_command(command):
    if command is None:
        return "UNKNOWN"

    print(command)

    command = str(command).strip().upper()
    command = command.replace(" ", "_")
    command = command.replace("-", "_")

    valid_commands = {"STOP", "SLOW_DOWN", "PROCEED", "UNKNOWN"}

    if command in valid_commands:
        return command

    return "UNKNOWN"


def normalize_confidence(confidence):
    try:
        confidence = float(confidence)
    except Exception:
        return 0.0

    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0

    if confidence < 0.0:
        confidence = 0.0

    if confidence > 1.0:
        confidence = 1.0

    return confidence


class LLMCommandClassifierNode:
    def __init__(self):
        rospy.init_node("llm_command_classifier_node", anonymous=False)

        self.input_topic = rospy.get_param("~input_topic", "/voice/text")
        self.command_topic = rospy.get_param("~command_topic", "/voice/command")
        self.json_topic = rospy.get_param("~json_topic", "/voice/command_json")

        self.ollama_base_url = rospy.get_param(
            "~ollama_base_url",
            "http://127.0.0.1:11434/v1"
        )

        self.chat_url = self.ollama_base_url.rstrip("/") + "/chat/completions"

        self.model = rospy.get_param("~model", "llama3.1:8b")
        self.temperature = float(rospy.get_param("~temperature", 0.0))
        self.max_tokens = int(rospy.get_param("~max_tokens", 80))
        self.timeout = float(rospy.get_param("~timeout", 30.0))

        self.ignore_empty_text = bool(rospy.get_param("~ignore_empty_text", True))
        self.publish_unknown_on_error = bool(rospy.get_param("~publish_unknown_on_error", True))

        self.busy = False
        self.lock = threading.Lock()

        self.pub_command = rospy.Publisher(self.command_topic, String, queue_size=10)
        self.pub_json = rospy.Publisher(self.json_topic, String, queue_size=10)

        self.sub_text = rospy.Subscriber(
            self.input_topic,
            String,
            self.text_callback,
            queue_size=1
        )

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "classify_robot_command",
                    "description": "Classify this command as STOP, SLOW_DOWN, PROCEED, or UNKNOWN with selection confidence.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "enum": ["STOP", "SLOW_DOWN", "PROCEED", "UNKNOWN"]
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Selection confidence from 0.0 to 1.0."
                            }
                        },
                        "required": ["command", "confidence"],
                        "additionalProperties": False
                    }
                }
            }
        ]

        rospy.loginfo("LLM command classifier node started")
        rospy.loginfo("Input topic: %s", self.input_topic)
        rospy.loginfo("Command topic: %s", self.command_topic)
        rospy.loginfo("JSON topic: %s", self.json_topic)
        rospy.loginfo("Ollama chat URL: %s", self.chat_url)
        rospy.loginfo("Model: %s", self.model)

    def local_classify_robot_command(self, command, confidence):
        return {
            "command": normalize_command(command),
            "confidence": normalize_confidence(confidence)
        }

    def classify_with_ollama_tool_call(self, user_text):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a robot command classifier. "
                    "Classify the user's text into exactly one command: "
                    "STOP, SLOW_DOWN, PROCEED, or UNKNOWN. "
                    "Always use the classify_robot_command tool. "
                )
            },
            {
                "role": "user",
                "content": user_text
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": {
                "type": "function",
                "function": {
                    "name": "classify_robot_command"
                }
            },
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        start_time = time.time()

        response = requests.post(
            self.chat_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self.timeout
        )

        tool_call_time = time.time() - start_time

        if response.status_code != 200:
            return {
                "success": False,
                "text": user_text,
                "error": "Ollama request failed. Status: {} Body: {}".format(
                    response.status_code,
                    response.text
                ),
                "tool_call_time_sec": tool_call_time,
                "total_time_sec": time.time() - start_time
            }

        try:
            data = response.json()
        except Exception:
            return {
                "success": False,
                "text": user_text,
                "error": "Ollama response was not valid JSON.",
                "raw_response": response.text,
                "tool_call_time_sec": tool_call_time,
                "total_time_sec": time.time() - start_time
            }

        try:
            message = data["choices"][0]["message"]
        except Exception:
            return {
                "success": False,
                "text": user_text,
                "error": "Unexpected Ollama response format.",
                "raw_response": data,
                "tool_call_time_sec": tool_call_time,
                "total_time_sec": time.time() - start_time
            }

        tool_calls = message.get("tool_calls", None)

        if not tool_calls:
            return {
                "success": False,
                "text": user_text,
                "error": "Model did not return a tool call.",
                "raw_content": message.get("content", ""),
                "raw_response": data,
                "tool_call_time_sec": tool_call_time,
                "total_time_sec": time.time() - start_time
            }

        tool_call = tool_calls[0]
        function_info = tool_call.get("function", {})
        function_name = function_info.get("name", "")

        if function_name != "classify_robot_command":
            return {
                "success": False,
                "text": user_text,
                "error": "Unknown tool/function requested: {}".format(function_name),
                "raw_response": data,
                "tool_call_time_sec": tool_call_time,
                "total_time_sec": time.time() - start_time
            }

        raw_arguments = function_info.get("arguments", {})

        try:
            if isinstance(raw_arguments, str):
                function_args = json.loads(raw_arguments)
            elif isinstance(raw_arguments, dict):
                function_args = raw_arguments
            else:
                function_args = {}
        except Exception:
            return {
                "success": False,
                "text": user_text,
                "error": "Tool arguments were not valid JSON.",
                "raw_arguments": raw_arguments,
                "tool_call_time_sec": tool_call_time,
                "total_time_sec": time.time() - start_time
            }

        final_result = self.local_classify_robot_command(
            command=function_args.get("command"),
            confidence=function_args.get("confidence")
        )

        total_time = time.time() - start_time

        return {
            "success": True,
            "text": user_text,
            "command": final_result["command"],
            "confidence": final_result["confidence"],
            "model": self.model,
            "model_arguments": function_args,
            "function_response": final_result,
            "tool_call_time_sec": tool_call_time,
            "total_time_sec": total_time,
            "timestamp": rospy.Time.now().to_sec()
        }

    def publish_result(self, result):
        if result.get("success", False):
            command = result.get("command", "UNKNOWN")
            self.pub_command.publish(command)
            self.pub_json.publish(json.dumps(result))

            rospy.loginfo(
                "Text: '%s' -> Command: %s | Confidence: %.3f | Time: %.3f sec",
                result.get("text", ""),
                command,
                result.get("confidence", 0.0),
                result.get("total_time_sec", 0.0)
            )
        else:
            rospy.logerr("LLM classification failed: %s", result.get("error", "unknown error"))

            if self.publish_unknown_on_error:
                error_result = {
                    "success": False,
                    "text": result.get("text", ""),
                    "command": "UNKNOWN",
                    "confidence": 0.0,
                    "model": self.model,
                    "error": result.get("error", "unknown error"),
                    "raw_content": result.get("raw_content", None),
                    "raw_arguments": result.get("raw_arguments", None),
                    "raw_response": result.get("raw_response", None),
                    "tool_call_time_sec": result.get("tool_call_time_sec", 0.0),
                    "total_time_sec": result.get("total_time_sec", 0.0),
                    "timestamp": rospy.Time.now().to_sec()
                }

                self.pub_command.publish("UNKNOWN")
                self.pub_json.publish(json.dumps(error_result))

    def text_callback(self, msg):
        text = msg.data.strip()

        if self.ignore_empty_text and not text:
            return

        with self.lock:
            if self.busy:
                rospy.logwarn("LLM classifier is busy. Skipping text: %s", text)
                return
            self.busy = True

        try:
            rospy.loginfo("Received transcribed text: %s", text)
            result = self.classify_with_ollama_tool_call(text)
            self.publish_result(result)

        except Exception as e:
            rospy.logerr("LLM command classifier error: %s", str(e))

            if self.publish_unknown_on_error:
                error_result = {
                    "success": False,
                    "text": text,
                    "command": "UNKNOWN",
                    "confidence": 0.0,
                    "model": self.model,
                    "error": str(e),
                    "timestamp": rospy.Time.now().to_sec()
                }

                self.pub_command.publish("UNKNOWN")
                self.pub_json.publish(json.dumps(error_result))

        finally:
            with self.lock:
                self.busy = False

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = LLMCommandClassifierNode()
    node.spin()