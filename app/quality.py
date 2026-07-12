"""Semantic answer quality gates.

Runs a lightweight verification pass on local model answers to detect
hallucinations, missing logic, or weak completions before returning them.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from app.classifier import Category, RouteTarget
from app.zero_token import _extract_arithmetic_expression, _safe_eval, _solve_linear_equation

def fix_ner_response(prompt: str, response: str) -> str:
    if ":" in response:
        _, response = response.split(":", 1)
        
    entities = [e.strip() for e in response.split(",") if e.strip()]
    if len(entities) < 2:
        return response
        
    prompt_lower = prompt.lower()
    
    merged = []
    curr = entities[0]
    
    for nxt in entities[1:]:
        combined = f"{curr} {nxt}"
        if combined.lower() in prompt_lower:
            curr = combined
        else:
            merged.append(curr)
            curr = nxt
    merged.append(curr)
    
    return ", ".join(merged)

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
    if not text or (category != Category.SENTIMENT and len(text.split()) < 2):
        return True
    if any(phrase in lower for phrase in _WEAK_PHRASES):
        return True

    if category == Category.SENTIMENT:
        return lower not in {"positive", "negative", "neutral"}

    if category == Category.MATH:
        if "answer:" not in lower and not re.search(r"-?\d+(?:\.\d+)?", text):
            return True
            
        try:
            expr = _extract_arithmetic_expression(prompt)
            if expr:
                expected = float(_safe_eval(expr))
                expected_str1 = str(int(expected)) if expected.is_integer() else str(expected)
                expected_str2 = f"{expected:.6g}"
                if expected_str1 not in text and expected_str2 not in text:
                    return True
            eq = _solve_linear_equation(prompt)
            if eq:
                _, _, _, expected = eq
                expected_str1 = str(int(expected)) if expected.is_integer() else str(expected)
                expected_str2 = f"{expected:.6g}"
                if expected_str1 not in text and expected_str2 not in text:
                    return True
        except Exception:
            pass

    if category == Category.FACTUAL:
        # Without a knowledge base, we cannot locally verify factual correctness.
        # Force escalation to remote.
        return True

    if category == Category.LOGIC:
        # Cannot easily verify logical conclusion against premises without a solver.
        return True

    if category == Category.SUMMARIZATION:
        # Summarization is too open-ended and cannot be strongly verified locally.
        # Force escalation to remote.
        return True

    if category == Category.NER:
        if ":" not in text and "," not in text:
            return True
        
        # Parse entities
        if ":" in text:
            entities_str = text.split(":", 1)[1]
        else:
            entities_str = text
            
        entities = [e.strip() for e in entities_str.split(",")]
        entities = [e for e in entities if e]
        
        prompt_lower = prompt.lower()
        for e in entities:
            # Strip trailing/leading punctuation
            e_clean = e.strip(".'\"?![]{}()").lower()
            if not e_clean:
                continue
            if e_clean not in prompt_lower:
                return True
        
        if not entities:
            return True

    if category in (Category.CODE_GEN, Category.CODE_DEBUG):
        code_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
        if code_blocks:
            code = code_blocks[0]
            try:
                ast.parse(code)
                # Attempt to execute snippet
                import subprocess
                try:
                    res = subprocess.run(
                        ["python", "-c", code],
                        capture_output=True,
                        timeout=1.0
                    )
                    if res.returncode != 0:
                        # Only fail if it's not a clear input EOF error
                        err = res.stderr.decode('utf-8', errors='ignore')
                        if "EOFError" not in err and "ModuleNotFoundError" not in err:
                            return True
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass
            except SyntaxError:
                if "```python" in text.lower():
                    return True
        else:
            has_code_shape = re.search(r"\b(def|class|function|const|let|var|import|print|for|if|while|return)\b", text)
            if not has_code_shape:
                return True
                
    return False
