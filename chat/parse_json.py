import re


def parse_chat_output(text: str) -> dict:
    result = {}

    # channel
    channel_match = re.search(r"<\|channel\|>(.*?)<\|message\|>", text, re.DOTALL)
    if channel_match:
        result["channel"] = channel_match.group(1).strip()

    # message
    message_match = re.search(r"<\|message\|>(.*?)<\|end\|>", text, re.DOTALL)
    if message_match:
        result["message"] = message_match.group(1).strip()

    # assistant block
    assistant_match = re.search(
        r"<\|start\|>assistant<\|channel\|>(.*?)<\|message\|>(.*?)<\|return\|>",
        text,
        re.DOTALL
    )
    if assistant_match:
        result["assistant"] = {
            "channel": assistant_match.group(1).strip(),
            "message": assistant_match.group(2).strip()
        }

    return result
