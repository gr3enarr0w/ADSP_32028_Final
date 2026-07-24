"""Shared Gemini response parsing utilities."""

import json


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM response text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def parse_json_response(text: str) -> dict | list:
    """Strip code fences and parse JSON from an LLM response."""
    return json.loads(strip_code_fences(text))
