"""
Bifrost Prompt Templates
Token-optimised system prompts and formatters for each of the eight task
categories.  Each template is designed to elicit concise, accurate responses
while minimising both input and output token usage.
"""

from __future__ import annotations

from app.classifier import Category

# ---------------------------------------------------------------------------
# System prompts — kept deliberately short to save input tokens
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[Category, str] = {
    Category.SENTIMENT: "You are a sentiment classifier. Output exactly ONE word: 'positive', 'negative', or 'neutral'. Do NOT output any other text, reasoning, or explanation. Note: reviews that express disappointment, lack of expected quality, or subtle criticism should be classified as 'negative'.",
    Category.NER: "Extract ALL entities (Person, Organization, Location, Date) as a comma-separated list. Keep adjacent capitalized words, organizational names, and multi-word proper nouns together as a single entity (e.g., 'United Nations Security Council' not 'United Nations, Security Council'). Ignore if the user only asks for 'names', extract all types. Output ONLY the list, no labels or extra text.",
    Category.FACTUAL: "Answer the factual question in exactly 1 sentence. Facts only. No introductory filler.",
    Category.SUMMARIZATION: "Summarize short.",
    Category.MATH: "Think step-by-step and show your reasoning. However, you MUST end your response strictly with 'Answer: [value]' on a new line.",
    Category.LOGIC: "Think step-by-step and show your reasoning. However, you MUST end your response strictly with 'Conclusion: [yes, no, or not necessarily]' on a new line.",
    Category.CODE_DEBUG: "Fix code. Code only.",
    Category.CODE_GEN: "Output ONLY raw code. NO markdown formatting. NO backticks. NO explanations.",
    Category.UNKNOWN: "Answer short.",
}


def build_messages(
    prompt: str,
    category: Category,
    complexity_score: float = 0.0,
    *,
    max_prompt_chars: int = 4000,
) -> list[dict[str, str]]:
    """
    Build the messages array for the chat completions API.

    Parameters
    ----------
    prompt : str
        The user's task prompt.
    category : Category
        Classified category of the task.
    max_prompt_chars : int
        Safety cap on prompt length to avoid runaway token usage.

    Returns
    -------
    list[dict]
        OpenAI-compatible messages array.
    """
    system_msg = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS[Category.UNKNOWN])

    # Truncate extremely long prompts to protect token budget
    user_content = prompt[:max_prompt_chars] if len(prompt) > max_prompt_chars else prompt

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Max-tokens hint per category (controls output length)
# ---------------------------------------------------------------------------

MAX_TOKENS_HINT: dict[Category, int] = {
    Category.SENTIMENT:     5,
    Category.NER:           50,
    Category.FACTUAL:       150,
    Category.SUMMARIZATION: 80,
    Category.MATH:          300,
    Category.LOGIC:         500,
    Category.CODE_DEBUG:    300,
    Category.CODE_GEN:      200,
    Category.UNKNOWN:       100,
}
