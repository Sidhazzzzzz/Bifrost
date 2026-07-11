"""Deterministic zero-token answer layer for Bifrost.

These solvers only answer when the task shape is narrow enough that we can
avoid a model call without gambling on correctness.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Any

from app.classifier import Category


@dataclass(frozen=True)
class ZeroTokenAnswer:
    response: str
    strategy: str
    confidence: float


_POSITIVE = {
    "amazing", "awesome", "best", "excellent", "fantastic", "good", "great",
    "happy", "incredible", "love", "loved", "perfect", "positive",
    "recommend", "satisfied", "wonderful", "outstanding", "brilliant",
    "superb", "delightful", "joy", "glad", "thrilled", "beautiful", "gorgeous",
    "fabulous", "splendid", "stellar", "phenomenal",
}
_NEGATIVE = {
    "awful", "bad", "confusing", "disappointing", "hate", "hated", "horrible",
    "negative", "poor", "refund", "sad", "terrible", "worst", "waste",
    "abysmal", "dreadful", "appalling", "atrocious", "pathetic", "useless",
    "boring", "dull", "annoying", "frustrating", "miserable", "ugly", "disgusting",
}
_NEGATIONS = {"not", "never", "no", "hardly", "barely"}

_FACTS = {
    "capital of france": "Paris",
    "capital of india": "New Delhi",
    "capital of germany": "Berlin",
    "capital of japan": "Tokyo",
    "capital of italy": "Rome",
    "capital of spain": "Madrid",
    "capital of canada": "Ottawa",
    "capital of australia": "Canberra",
    "capital of united states": "Washington, D.C.",
    "capital of usa": "Washington, D.C.",
    "capital of uk": "London",
    "capital of united kingdom": "London",
    "capital of russia": "Moscow",
    "capital of china": "Beijing",
    "who wrote romeo and juliet": "William Shakespeare",
    "who painted the mona lisa": "Leonardo da Vinci",
    "largest planet": "Jupiter",
    "smallest planet": "Mercury",
    "tallest mountain": "Mount Everest",
    "longest river": "Nile",
    "first president of the united states": "George Washington",
    "author of harry potter": "J.K. Rowling",
    "who wrote hamlet": "William Shakespeare",
    "when was the declaration of independence signed": "1776",
}

_SAFE_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARY: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def try_zero_token_answer(prompt: str, category: Category) -> ZeroTokenAnswer | None:
    text = prompt.strip()
    if not text:
        return ZeroTokenAnswer("Please provide a task prompt.", "empty_prompt", 1.0)

    if category == Category.SENTIMENT:
        return _sentiment(text)
    if category == Category.MATH:
        return _math(text)
    if category == Category.NER:
        return _ner(text)
    if category == Category.LOGIC:
        return _logic(text)
    if category == Category.FACTUAL:
        return _factual(text)
    return None


def _sentiment(prompt: str) -> ZeroTokenAnswer | None:
    target = _quoted_or_tail(prompt)
    words = re.findall(r"[a-z']+", target.lower())
    if not words:
        return None

    pos = neg = 0
    for idx, word in enumerate(words):
        window = set(words[max(0, idx - 3):idx])
        flipped = bool(window & _NEGATIONS)
        if word in _POSITIVE:
            if flipped:
                neg += 1
            else:
                pos += 1
        elif word in _NEGATIVE:
            pos += 1 if flipped else 0
            neg += 0 if flipped else 1

    if pos == neg == 0:
        return None
    if abs(pos - neg) < 1:
        return ZeroTokenAnswer("Neutral", "lexicon_sentiment", 0.72)
    label = "Positive" if pos > neg else "Negative"
    confidence = min(0.97, 0.75 + (abs(pos - neg) * 0.08))
    return ZeroTokenAnswer(label, "lexicon_sentiment", confidence)


def _math(prompt: str) -> ZeroTokenAnswer | None:
    equation = _solve_linear_equation(prompt)
    if equation:
        variable, lhs, rhs, value = equation
        return ZeroTokenAnswer(
            f"{lhs} = {rhs}\nAnswer: {variable} = {_fmt_number(value)}",
            "linear_equation_solver",
            0.96,
        )

    expr = _extract_arithmetic_expression(prompt)
    if not expr:
        return None
    try:
        value = _safe_eval(expr)
    except Exception:
        return None
    return ZeroTokenAnswer(f"Answer: {_fmt_number(value)}", "safe_arithmetic_eval", 0.95)


def _ner(prompt: str) -> ZeroTokenAnswer | None:
    text = _quoted_or_tail(prompt)
    if len(text.split()) < 3:
        return None

    orgs = set(re.findall(r"\b[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*\s+(?:Inc|LLC|Ltd|Corp|Corporation|Company|University|AI|Labs)\b", text))
    orgs.update(re.findall(r"\b(?:NASA|SpaceX|OpenAI|AMD|Google|Microsoft|Amazon|Apple|Meta|Netflix|Tesla)\b", text))
    dates = set(re.findall(r"\b(?:\d{4}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\w\s,.-]{0,20}\b", text))
    locations = set(re.findall(r"\b(?:in|at|from|near|to)\s+([A-Z][A-Za-z]+(?:,\s*[A-Z][A-Za-z]+)?(?:\s+[A-Z][A-Za-z]+)?)\b", text))
    people = set()
    for match in re.findall(r"(?:(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+)?\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        clean_match = match.strip()
        if clean_match not in orgs and clean_match not in locations and clean_match not in {"The", "A", "An", "In", "On", "At", "To", "From", "By", "With", "And", "Or", "But", "If"}:
            # Avoid matching start-of-sentence words blindly unless they have titles
            if not any(title in match for title in ["Mr.", "Mrs.", "Ms.", "Dr."]) and not " " in clean_match:
                # If it's a single word and not titled, be careful. Let's rely on a simple whitelist for common names or context
                if clean_match in {"John", "Sarah", "Michael", "Emma", "David", "James", "Mary", "Robert"}:
                    people.add(clean_match)
            else:
                people.add(clean_match)

    groups = []
    if people:
        groups.append("Person: " + ", ".join(sorted(people)))
    if orgs:
        groups.append("Organization: " + ", ".join(sorted(orgs)))
    if locations:
        groups.append("Location: " + ", ".join(sorted(locations)))
    if dates:
        groups.append("Date: " + ", ".join(sorted(dates)))
    if not groups:
        return None
    return ZeroTokenAnswer("\n".join(groups), "regex_entity_extractor", 0.78)


def _logic(prompt: str) -> ZeroTokenAnswer | None:
    lower = prompt.lower()
    pattern = re.search(
        r"all\s+(?P<a>[a-z\s]+?)\s+are\s+(?P<b>[a-z\s]+?)\.\s+some\s+(?P<b2>[a-z\s]+?)\s+(?P<c>[^.?!]+)",
        lower,
    )
    if not pattern:
        return None
    b = _singular(pattern.group("b").strip())
    b2 = _singular(pattern.group("b2").strip())
    if b not in b2 and b2 not in b:
        return None
    return ZeroTokenAnswer(
        "No. From 'all A are B' and 'some B have property C', it does not follow that any A have property C.\nConclusion: Not necessarily.",
        "syllogism_guard",
        0.93,
    )


def _factual(prompt: str) -> ZeroTokenAnswer | None:
    lower = re.sub(r"[^a-z0-9\s]", " ", prompt.lower())
    lower = re.sub(r"\s+", " ", lower).strip()
    for key, value in _FACTS.items():
        if key in lower:
            return ZeroTokenAnswer(value, "fact_table", 0.99)
    return None


def _quoted_or_tail(prompt: str) -> str:
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", prompt)
    parts = [a or b for a, b in quoted if a or b]
    if parts:
        return " ".join(parts)
    for marker in (":", "from"):
        if marker in prompt:
            return prompt.split(marker, 1)[1]
    return prompt


def _extract_arithmetic_expression(prompt: str) -> str | None:
    clean = prompt.lower().replace("^", "**")
    match = re.search(r"(-?\d+(?:\.\d+)?(?:\s*(?:\*\*|[+\-*/%()])\s*-?\d+(?:\.\d+)?)+)", clean)
    return match.group(1) if match else None


def _solve_linear_equation(prompt: str) -> tuple[str, str, str, float] | None:
    compact = prompt.replace(" ", "").lower()
    match = re.search(r"([+-]?\d*)?([a-z])([+-]\d+(?:\.\d+)?)=(-?\d+(?:\.\d+)?)", compact)
    if not match:
        return None
    coeff_raw, variable, offset_raw, rhs_raw = match.groups()
    coeff = -1.0 if coeff_raw == "-" else float(coeff_raw or 1)
    offset = float(offset_raw)
    rhs = float(rhs_raw)
    if coeff == 0:
        return None
    value = (rhs - offset) / coeff
    lhs = f"{_fmt_number(coeff)}{variable}{offset_raw}"
    return variable, lhs, rhs_raw, value


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval")
    return float(_eval_node(node.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("exponent too large")
        return float(_SAFE_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY:
        return float(_SAFE_UNARY[type(node.op)](_eval_node(node.operand)))
    raise ValueError("unsupported expression")


def _fmt_number(value: float | str) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def _singular(value: str) -> str:
    value = value.strip()
    if value.endswith("s"):
        return value[:-1]
    return value
