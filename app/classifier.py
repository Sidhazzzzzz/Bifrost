"""
Bifrost Task Complexity Classifier
Evaluates complexity of incoming prompts to determine whether they should be
routed to the LOCAL base tier (Ollama / local_model) or the REMOTE cloud tier
(Groq / llama-3.1-8b-instant).

Routing Philosophy:
  - LOCAL  → Simple, factual, short-answer, sentiment, basic math, NER
  - REMOTE → Code generation/debugging, long reasoning, multi-step logic,
             creative writing, complex summarization
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class RouteTarget(str, Enum):
    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


class Category(str, Enum):
    FACTUAL       = "factual"
    MATH          = "mathematical"
    SENTIMENT     = "sentiment"
    SUMMARIZATION = "summarization"
    NER           = "ner"
    CODE_DEBUG    = "code_debugging"
    LOGIC         = "logical_reasoning"
    CODE_GEN      = "code_generation"
    UNKNOWN       = "unknown"


@dataclass
class Classification:
    category: Category
    complexity_score: float  # 0.0 to 1.0
    routed_to: RouteTarget


# ---------------------------------------------------------------------------
# Pattern banks (compiled once at import time for performance)
# ---------------------------------------------------------------------------

_CODE_PATTERNS = [
    re.compile(r"```"),
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bclass\s+\w+(?:\(|:)"),
    re.compile(r"\bfunction\s+\w+"),
    re.compile(r"\bconst\s+\w+\s*="),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bprint\s*\("),
    re.compile(r"\bconsole\.\w+\("),
    re.compile(r"\bfor\s*\(.*;\s*.*;\s*.*\)"),
    re.compile(r"\bif\s*\(.*\)\s*\{"),
]

_MATH_PATTERN = re.compile(
    r"(?:\d+\s*[\+\-\*\/\^\%]\s*\d+)"
    r"|\b(?:solve|calculate|compute|evaluate|simplify|derive|integrate|differentiate)\b"
    r"|\b(?:equation|formula|expression|polynomial|matrix|vector)\b"
    r"|(?:what\s+is\s+\d+)"
    r"|(?:how\s+(?:much|many)\s+is)"
    r"|\b(?:square\s+root|factorial|logarithm|percentage|ratio|proportion|half|apples?)\b"
    r"|(?:cost|total).*?(?:\$?\d+(?:\.\d+)?)"
    r"|(?:if\s+a\s+.*?\s+leaves)"
    r"|(?:how\s+many\s+.*?\s+are\s+in)"
    r"|\b(?:priced\s+at|discount|probability\s+of|rate|distance|speed|joint\s+probability)\b"
    r"|\b(?:geometry|algebra|calculus|statistics|probability|fractions)\b",
    re.IGNORECASE,
)

_SENTIMENT_KEYWORDS = {
    "sentiment", "feel", "feeling", "mood", "emotion", "tone",
    "positive", "negative", "neutral", "opinion", "attitude",
    "happy", "sad", "angry", "review",
    "toxic", "safe", "polarity", "complaining", "praising",
    "hostility", "aggressive", "friendly", "spam", "abusive", "offensive",
}

_NER_KEYWORDS = {
    "extract", "entities", "ner", "named entity", "identify names",
    "find all names", "people mentioned", "organizations mentioned",
    "locations mentioned", "dates mentioned",
    "key players", "companies", "places", "geographical",
    "geopolitical", "monetary", "brands", "products",
    "who is mentioned", "what organizations",
}

_SUMMARY_KEYWORDS = {
    "summarize", "summary", "tldr", "tl;dr", "condense", "brief",
    "key points", "main idea", "gist", "recap", "overview",
    "long story short", "what is this about", "what is this article about", "what is this text about", "boil this down",
    "explain this document", "one paragraph", "too long", "outline",
    "abstract", "synopsis", "wrap up",
}

_LOGIC_KEYWORDS = {
    "logic", "logical", "prove", "proof", "reason", "reasoning",
    "deduce", "deduction", "infer", "inference", "syllogism",
    "paradox", "fallacy", "premise", "conclusion", "therefore",
    "if then", "implies", "conclude", "if all",
    "riddle", "puzzle", "brain teaser", "who is the tallest",
    "how do you measure", "cross a river", "brothers and sisters",
    "who is lying",
}

_CODE_GEN_VERBS = {
    "write", "generate", "create", "implement", "build", "make", "develop",
    "draft", "whip", "translate", "convert", "design", "setup", "configure", "scaffold",
    "need", "want",
}
_CODE_GEN_NOUNS = {
    "function", "class", "script", "program", "method", "module",
    "api", "endpoint", "component", "app", "application", "bot",
    "algorithm", "data structure", "loop", "regex",
    "python", "javascript", "java", "c++", "rust", "go", "sql",
    "html", "css", "typescript", "react", "node", "flask", "django",
    "bash", "shell", "query", "json", "xml", "struct", "interface",
    "ui", "frontend", "backend", "database", "vue", "angular", "postgres", "postgresql", "mysql",
    "layout", "dependency",
}

_CODE_DEBUG_KEYWORDS = {
    "debug", "bug", "error", "fix", "crash", "traceback", "exception",
    "segfault", "wrong output", "wrong value", "why does my code",
    "fails", "failing", "broken", "issue", "problem", "hangs", "stuck",
    "infinite loop", "memory leak", "not working", "resolve", "doesn't work",
    "nullreference", "typeerror", "valueerror", "indexerror", "panic",
    "compile", "overflows",
}

_COMPLEXITY_BOOSTERS = {
    "explain in detail", "step-by-step", "comprehensive", "thorough",
    "compare and contrast", "pros and cons", "advantages and disadvantages",
    "architect", "design pattern", "trade-off", "optimization",
    "multi-step", "complex", "advanced", "in-depth",
}

_SIMPLE_FACTUAL_PATTERNS = [
    re.compile(r"^(?:what|who|when|where|which)\s+(?:is|are|was|were)\s+", re.IGNORECASE),
    re.compile(r"^(?:name|list|tell me)\s+", re.IGNORECASE),
    re.compile(r"^(?:define|meaning of)\s+", re.IGNORECASE),
    re.compile(r"^(?:capital of|president of|pm of|ceo of)\s+", re.IGNORECASE),
    re.compile(r"^(?:how old|how tall|how far)\s+", re.IGNORECASE),
]


def classify(prompt: str, threshold: float = 0.5) -> Classification:
    """Classify prompt complexity to route between LOCAL and REMOTE."""
    if not prompt.strip():
        return Classification(
            category=Category.FACTUAL,
            complexity_score=0.1,
            routed_to=RouteTarget.LOCAL,
        )

    lower = prompt.lower().strip()
    words = lower.split()
    word_count = len(words)

    # ------------------------------------------------------------------
    # Phase 1: Detect category
    # ------------------------------------------------------------------
    category = _detect_category(prompt, lower, word_count)

    # ------------------------------------------------------------------
    # Phase 2: Compute complexity score (0.0 – 1.0)
    # ------------------------------------------------------------------
    score = _compute_complexity(prompt, lower, word_count, category)

    # ------------------------------------------------------------------
    # Phase 3: Route decision
    # ------------------------------------------------------------------
    routed_to = RouteTarget.LOCAL
    if round(score, 2) >= threshold:
        routed_to = RouteTarget.REMOTE

    return Classification(
        category=category,
        complexity_score=round(score, 2),
        routed_to=routed_to,
    )


def _detect_category(prompt: str, lower: str, word_count: int) -> Category:
    """Determine the task category using layered heuristics."""

    # Check code presence first (highest priority)
    has_code = any(p.search(prompt) for p in _CODE_PATTERNS)
    
    has_debug_kw = any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in _CODE_DEBUG_KEYWORDS)
    has_gen_verb = any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in _CODE_GEN_VERBS)
    has_gen_noun = any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in _CODE_GEN_NOUNS)

    if has_code or (has_gen_verb and has_gen_noun):
        if has_debug_kw:
            return Category.CODE_DEBUG
        return Category.CODE_GEN

    if has_debug_kw and (has_code or has_gen_noun or "code" in lower):
        return Category.CODE_DEBUG

    # Math detection (arithmetic expressions or math keywords)
    if _MATH_PATTERN.search(lower):
        return Category.MATH

    # Sentiment
    if any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in _SENTIMENT_KEYWORDS):
        return Category.SENTIMENT

    # NER
    if any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in _NER_KEYWORDS):
        return Category.NER

    # Summarization
    if any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in _SUMMARY_KEYWORDS):
        return Category.SUMMARIZATION

    # Logic / Reasoning
    if any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in _LOGIC_KEYWORDS):
        return Category.LOGIC

    return Category.FACTUAL


def _compute_complexity(
    prompt: str,
    lower: str,
    word_count: int,
    category: Category,
) -> float:
    """Return a 0.0–1.0 complexity score incorporating multiple signals."""
    score = 0.0

    # ── Signal 1: Prompt length ──────────────────────────────
    if word_count > 100:
        score += 0.35
    elif word_count > 60:
        score += 0.25
    elif word_count > 30:
        score += 0.15
    elif word_count > 15:
        score += 0.05

    # ── Signal 2: Simple factual pattern (strong LOCAL signal) ──
    is_simple_factual = any(p.search(lower) for p in _SIMPLE_FACTUAL_PATTERNS)
    if is_simple_factual and word_count <= 15:
        score -= 0.3  # strong pull toward LOCAL

    # ── Signal 3: Complexity booster phrases ─────────────────
    booster_hits = sum(1 for phrase in _COMPLEXITY_BOOSTERS if phrase in lower)
    score += min(booster_hits * 0.15, 0.35)

    # ── Signal 4: Multiple questions ─────────────────────────
    question_count = lower.count("?")
    if question_count > 2:
        score += 0.2
    elif question_count > 1:
        score += 0.1

    # ── Signal 5: Category-based baseline ────────────────────
    category_baselines = {
        Category.FACTUAL:       0.15,
        Category.MATH:          0.85,
        Category.SENTIMENT:     0.1,
        Category.NER:           0.2,
        Category.SUMMARIZATION: 0.3,
        Category.LOGIC:         0.85,
        Category.CODE_DEBUG:    0.6,
        Category.CODE_GEN:      0.7,
        Category.UNKNOWN:       0.25,
    }
    score += category_baselines.get(category, 0.2)

    # ── Signal 6: Code presence boost ────────────────────────
    has_code = any(p.search(prompt) for p in _CODE_PATTERNS)
    if has_code:
        score += 0.2

    # ── Clamp ────────────────────────────────────────────────
    return min(max(score, 0.0), 1.0)
