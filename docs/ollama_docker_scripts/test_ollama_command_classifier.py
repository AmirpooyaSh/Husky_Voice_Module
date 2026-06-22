#!/usr/bin/env python3

import json
import time
from openai import OpenAI


# Use the exact model name shown by: docker exec -it ollama ollama list 
# Good first test:
MODEL = "llama3.1:8b"

# You can later try:
# MODEL = "gpt-oss:20b"
# MODEL = "qwen2.5:1.5b"
# MODEL = "qwen3:0.6b"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # placeholder, ignored by Ollama
)


def classify_robot_command(command: str, confidence: float):
    """
    Local Python function executed after the model selects the command.
    The model decides the arguments; this function validates and returns them.
    """

    command = normalize_command(command)
    confidence = normalize_confidence(confidence)

    return json.dumps({
        "command": command,
        "confidence": confidence
    })


tools = [
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
                        "enum": ["STOP", "SLOW_DOWN", "PROCEED", "UNKNOWN"],
                        "description": "The classified robot command"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "How confidence your classification is from 0.0 to 1.0."
                    }
                },
                "required": ["command", "confidence"],
                "additionalProperties": False
            }
        }
    }
]


available_functions = {
    "classify_robot_command": classify_robot_command
}


def normalize_command(command):
    if command is None:
        return "UNKNOWN"

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

    # Convert 0-100 style confidence to 0-1 if the model gives percentage
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0

    if confidence < 0.0:
        confidence = 0.0

    if confidence > 1.0:
        confidence = 1.0

    return confidence


def classify_with_tool_call(user_text):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a robot command classifier. "
                "Always use the classify_robot_command tool. "
                "Do not answer in normal text."
            )
        },
        {
            "role": "user",
            "content": user_text
        }
    ]

    start_time = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,

        # Force the local model to call this exact function.
        # If your Ollama/model ignores this, switch model to llama3.1:8b.
        tool_choice={
            "type": "function",
            "function": {
                "name": "classify_robot_command"
            }
        },

        temperature=0.0,
        max_tokens=80
    )

    tool_call_time = time.time() - start_time

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        return {
            "success": False,
            "error": "Model did not return a tool call.",
            "raw_content": response_message.content,
            "tool_call_time_sec": tool_call_time
        }

    # Append model's tool call intent to message history
    messages.append(response_message)

    tool_outputs = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name

        if function_name not in available_functions:
            return {
                "success": False,
                "error": "Unknown tool/function requested: {}".format(function_name),
                "raw_content": response_message.content,
                "tool_call_time_sec": tool_call_time
            }

        function_to_call = available_functions[function_name]

        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            return {
                "success": False,
                "error": "Tool arguments were not valid JSON.",
                "raw_arguments": tool_call.function.arguments,
                "tool_call_time_sec": tool_call_time
            }

        # Execute local Python function
        function_response = function_to_call(
            command=function_args.get("command"),
            confidence=function_args.get("confidence")
        )

        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response
        })

        tool_outputs.append({
            "function_name": function_name,
            "model_arguments": function_args,
            "function_response": json.loads(function_response)
        })

    total_time = time.time() - start_time

    # For your classifier, we do NOT need a second model call.
    # The function_response already contains the final command/confidence.
    final_result = tool_outputs[0]["function_response"]

    return {
        "success": True,
        "command": final_result["command"],
        "confidence": final_result["confidence"],
        "model_arguments": tool_outputs[0]["model_arguments"],
        "function_response": final_result,
        "tool_call_time_sec": tool_call_time,
        "total_time_sec": total_time,
        "num_tool_calls": len(tool_calls)
    }


def main():
    print("Local Ollama OpenAI-tool command classifier")
    print("Model:", MODEL)
    print("Endpoint: http://localhost:11434/v1")
    print("Type q, quit, or exit to stop.")
    print()

    while True:
        try:
            user_text = input("Input command > ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except EOFError:
            print("\nExiting.")
            break

        if user_text.lower() in ["q", "quit", "exit"]:
            print("Exiting.")
            break

        if not user_text:
            continue

        result = classify_with_tool_call(user_text)

        print()

        if not result["success"]:
            print("ERROR:", result["error"])
            if "raw_content" in result:
                print("Raw model content:", result["raw_content"])
            if "raw_arguments" in result:
                print("Raw tool arguments:", result["raw_arguments"])
            print("Tool-call time: {:.3f} sec".format(result.get("tool_call_time_sec", 0.0)))
            print("-" * 50)
            continue

        print("Classification:", result["command"])
        print("Confidence:    {:.3f}".format(result["confidence"]))
        print("Tool-call time:{:.3f} sec".format(result["tool_call_time_sec"]))
        print("Total time:    {:.3f} sec".format(result["total_time_sec"]))
        print("Raw args:      {}".format(result["model_arguments"]))
        print("-" * 50)


if __name__ == "__main__":
    main()
