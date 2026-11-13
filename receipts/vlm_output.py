"""Prompt and output parsing for the vision-language model."""
import json
import re
from typing import Optional

from .normalize import FIELDS

PROMPT = (
    "Read this receipt and return only a JSON object with the keys "
    '"vendor", "date" and "total". Use the store name for vendor, the purchase '
    "date for date, and the final amount paid for total. Use null for anything "
    "you cannot read."
)


def _balanced_block(text: str, start: int) -> Optional[str]:
    """Return the balanced {...} block starting at text[start], or None.

    Braces inside JSON strings are ignored.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _first_json_dict(text: str) -> Optional[dict]:
    """Try each '{' in order and return the first block that decodes to a dict."""
    start = text.find("{")
    while start != -1:
        block = _balanced_block(text, start)
        if block is not None:
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                return data
        start = text.find("{", start + 1)
    return None


def parse_model_output(text: str) -> Optional[dict]:
    """Turn raw model text into a record, or None if no usable JSON was found.

    Handles code fences and extra prose around the JSON. Only the expected
    keys are kept, and values are coerced to strings (or None).
    """
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    data = _first_json_dict(text)
    if data is None:
        return None
    record = {}
    for field in FIELDS:
        value = data.get(field)
        record[field] = None if value is None else str(value)
    return record
