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


def try_zero_token_answer(prompt: str, category: Category) -> ZeroTokenAnswer | None:
    text = prompt.strip()
    if not text:
        return ZeroTokenAnswer("Please provide a task prompt.", "empty_prompt", 1.0)

    if category == Category.MATH:
        return _math(text)
    return None

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
