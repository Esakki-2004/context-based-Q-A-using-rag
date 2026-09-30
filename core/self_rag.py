"""
Self-RAG — context relevance check.

Before generating the final answer, scores whether the retrieved context
is actually sufficient to answer the user's question.

Returns a confidence score (0.0–1.0) and a boolean indicating whether
to proceed with generation or flag low confidence.
"""

import json
from dataclasses import dataclass

from openai import OpenAI

from core.config import OPENAI_API_KEY, CHAT_MODEL

CONFIDENCE_THRESHOLD = 0.5


@dataclass
class RelevanceCheck:
    is_sufficient: bool
    confidence: float     # 0.0 – 1.0
    reasoning: str


_SYSTEM_PROMPT = """
You are a context relevance judge for a RAG system.

Given a user question and retrieved context, decide whether the context
contains enough information to answer the question accurately.

Respond ONLY with a JSON object:
{
  "is_sufficient": true | false,
  "confidence": 0.0–1.0,
  "reasoning": "<one sentence>"
}
""".strip()


def check_relevance(question: str, context: str) -> RelevanceCheck:
    """Score whether the context is sufficient to answer the question."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved context:\n{context[:3000]}"  # cap to avoid huge prompts
                ),
            },
        ],
        max_tokens=150,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        confidence = float(data.get("confidence", 0.5))
        return RelevanceCheck(
            is_sufficient=data.get("is_sufficient", confidence >= CONFIDENCE_THRESHOLD),
            confidence=confidence,
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return RelevanceCheck(
            is_sufficient=True,
            confidence=0.5,
            reasoning="Could not parse relevance check; proceeding with generation.",
        )
