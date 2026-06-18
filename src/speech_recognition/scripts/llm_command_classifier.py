#!/usr/bin/env python

import rospy
import json
import urllib2
import time

from std_msgs.msg import String


pub = None

OLLAMA_URL = None
MODEL = None


def normalize_label(raw_label):
    if raw_label is None:
        return "UNKNOWN"

    label = raw_label.strip().upper()
    label = label.replace("-", "_")
    label = label.replace(" ", "_")

    if "STOP" in label:
        return "STOP"
    elif "SLOW" in label:
        return "SLOW_DOWN"
    elif "PROCEED" in label or "RESUME" in label or "CONTINUE" in label or "GO" in label:
        return "PROCEED"
    else:
        return "UNKNOWN"


def ask_ollama(text):
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify the input only:"
                    "STOP, SLOW_DOWN, PROCEED, UNKNOWN. "
                )
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "options": {
            "temperature": 0.0,
            "num_predict": 10
        }
    }

    req = urllib2.Request(
        OLLAMA_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )

    response = urllib2.urlopen(req, timeout=120)
    result = json.loads(response.read())

    raw_label = result["message"]["content"]

    print(raw_label)
    return normalize_label(raw_label)


def publish_json_command(label, llm_response_time):
    payload = {
        "command": label,
        "llm_response_time_sec": round(llm_response_time, 3)
    }

    msg = String()
    msg.data = json.dumps(payload)

    pub.publish(msg)

    rospy.loginfo("Published LLM JSON: %s", msg.data)


def json_text_callback(msg):
    global pub

    rospy.loginfo("Received Vosk JSON: %s", msg.data)

    try:
        data = json.loads(msg.data)
        text = data.get("text", "").strip()
        stt_finalization_time = data.get("finalization_time_sec", None)

    except Exception as e:
        rospy.logwarn("Could not parse incoming JSON. Error: %s", str(e))
        rospy.logwarn("Trying to use raw message as text instead.")
        text = msg.data.strip()
        stt_finalization_time = None

    if not text:
        rospy.logwarn("Received empty text. Skipping LLM classification.")
        return

    rospy.loginfo("Text sent to Ollama: %s", text)

    start_time = time.time()

    try:
        label = ask_ollama(text)
    except Exception as e:
        rospy.logerr("Ollama request failed: %s", str(e))
        label = "UNKNOWN"

    llm_response_time = time.time() - start_time

    rospy.loginfo("LLM classified command as: %s", label)
    rospy.loginfo("LLM response time: %.3f sec", llm_response_time)

    if stt_finalization_time is not None:
        rospy.loginfo("Previous STT finalization time: %.3f sec", float(stt_finalization_time))

    publish_json_command(label, llm_response_time)


if __name__ == "__main__":
    rospy.init_node("llm_json_command_classifier")

    input_topic = rospy.get_param("~input_topic", "/voice/final_text_json")
    output_topic = rospy.get_param("~output_topic", "/llm_command_classification")

    OLLAMA_URL = rospy.get_param("~ollama_url", "http://127.0.0.1:11434/api/chat")
    MODEL = rospy.get_param("~model", "gpt-oss:20b")

    pub = rospy.Publisher(
        output_topic,
        String,
        queue_size=10
    )

    rospy.Subscriber(
        input_topic,
        String,
        json_text_callback,
        queue_size=10
    )

    rospy.loginfo("LLM JSON command classifier node started.")
    rospy.loginfo("Subscribing to: %s", input_topic)
    rospy.loginfo("Publishing to: %s", output_topic)
    rospy.loginfo("Ollama URL: %s", OLLAMA_URL)
    rospy.loginfo("Model: %s", MODEL)

    rospy.spin()