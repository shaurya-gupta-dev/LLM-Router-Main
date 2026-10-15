"""Test that thinking/reasoning preambles are stripped from model responses."""
import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.providers.gemini import _strip_thinking_preamble, _THINK_TAG_PATTERNS
from app.providers.openai_compat import _THINK_TAG_RE


# ---------- Gemini heuristic stripping tests ----------

def test_gemma_preamble_stripped():
    """Exact string from the user's reported bug."""
    raw = (
        'The user just said "hey". According to guidelines, respond concisely, '
        'friendly, maybe ask what they need. No need to use tools. Also note '
        'response style: short, crisp, useful only; no filler. '
        'So a brief greeting and ask how can help.'
        'Hey! How can I assist you today?'
    )
    result = _strip_thinking_preamble(raw)
    assert result == "Hey! How can I assist you today?", f"Got: {result!r}"
    print("PASS: Gemma preamble stripped (user-reported bug case)")


def test_think_tag_gemini():
    """<think> tags stripped by the Gemini heuristic."""
    raw = "<think>let me reason...</think>Sure, I can help with that."
    result = _strip_thinking_preamble(raw)
    assert result == "Sure, I can help with that.", f"Got: {result!r}"
    print("PASS: <think> tag stripped in Gemini provider")


def test_clean_response_untouched():
    """Normal responses should not be altered."""
    raw = "Hey! How can I assist you today?"
    result = _strip_thinking_preamble(raw)
    assert result == raw, f"Got: {result!r}"
    print("PASS: Clean response untouched")


# ---------- OpenAI-compat <think> stripping tests ----------

def test_openai_compat_think_tags():
    """DeepSeek-R1 / QwQ style <think>...</think> blocks stripped."""
    raw = "<think>I need to carefully consider this request.\nLet me think...</think>\nSure, I can help!"
    result = _THINK_TAG_RE.sub('', raw).strip()
    assert result == "Sure, I can help!", f"Got: {result!r}"
    print("PASS: OpenAI compat <think> tag stripped")


def test_openai_compat_no_think_tags():
    """Normal content untouched."""
    raw = "Hello! What can I do for you?"
    result = _THINK_TAG_RE.sub('', raw).strip()
    assert result == raw, f"Got: {result!r}"
    print("PASS: OpenAI compat clean response untouched")


if __name__ == "__main__":
    test_gemma_preamble_stripped()
    test_think_tag_gemini()
    test_clean_response_untouched()
    test_openai_compat_think_tags()
    test_openai_compat_no_think_tags()
    print("\nAll 5 thinking-strip tests passed.")
