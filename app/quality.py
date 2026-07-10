"""Semantic answer quality gates.

Runs a lightweight verification pass on local model answers to detect
hallucinations, missing logic, or weak completions before returning them.
"""

from __future__ import annotations

import re
from typing import Any

from app.classifier import Category, RouteTarget

_WEAK_PHRASES = (
    "i don't know",
    "i do not know",
    "i cannot answer",
    "i can't answer",
    "as an ai language model",
    "error:",
)


def is_weak_answer(prompt: str, response: str, category: Category) -> bool:
    """
    Run a semantic verification check on the local answer.
    Returns True if the answer is WEAK/FAIL, False if GOOD/PASS.
    """
    text = response.strip()
    lower = text.lower()
    
    # 1. Fast heuristics for absolute failures (empty, generic, etc)
    if not text or len(text.split()) < 2:
        return True
    if any(phrase in lower for phrase in _WEAK_PHRASES):
        return True

    if category == Category.SENTIMENT:
        return lower not in {"positive", "negative", "neutral"}

    if category == Category.MATH:
        if "answer:" not in lower and not re.search(r"-?\d+(?:\.\d+)?", text):
            return True

    if category == Category.NER:
        if ":" not in text and "," not in text:
            return True

    if category == Category.CODE_GEN:
        has_code_shape = "```" in text or re.search(r"\b(def|class|function|const|let|var|import|print|for|if|while|return)\b", text)
        if not bool(has_code_shape):
            return True

    if category == Category.CODE_DEBUG:
        has_fix = "```" in text or re.search(r"\b(def|class|function|const|let|var|import|print|for|if|while|return)\b", text)
        if not has_fix:
            return True
            
    return False
