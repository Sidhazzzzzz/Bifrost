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
    Category.SENTIMENT: "Sentiment: Positive, Negative, Neutral. 1 word.",
    Category.NER: "Extract entities (Person, Org, Loc, Date) as CSV. No text.",
    Category.FACTUAL: "Answer in 1 sentence. Facts only.",
    Category.SUMMARIZATION: "Summarize short.",
    Category.MATH: "Solve step-by-step. End with 'Answer: [value]'.",
    Category.LOGIC: "Step-by-step logic. End with 'Conclusion: yes' or 'Conclusion: no'.",
    Category.CODE_DEBUG: "Fix code. Code only.",
    Category.CODE_GEN: "Output ONLY raw code. NO markdown formatting. NO backticks. NO explanations.",
    Category.UNKNOWN: "Answer short.",
}


def build_messages(
    prompt: str,
    category: Category,
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
    Category.FACTUAL:       30,
    Category.SUMMARIZATION: 80,
    Category.MATH:          300,
    Category.LOGIC:         300,
    Category.CODE_DEBUG:    300,
    Category.CODE_GEN:      200,
    Category.UNKNOWN:       100,
}
