"""
modules/judge.py
LLM-as-a-Judge evaluator for generated social media posts.

Scores each post on:
  - faithfulness: alignment with brand guidelines (source)
  - relevance:    how well the post addresses the intended topic/platform

Usage:
    from modules.judge import judge_variants
"""

import json
import os

import anthropic
from dotenv import load_dotenv


def _get_client() -> anthropic.Anthropic:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_dotenv(dotenv_path=os.path.join(repo_root, ".env"))
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")
    return anthropic.Anthropic(api_key=api_key)


JUDGE_PROMPT = """You are an evaluation judge for social media content. Score the response on two independent criteria:

1. FAITHFULNESS: How well does the post reflect the brand guidelines (source)? Posts that contradict or ignore brand voice score low.
2. RELEVANCE: How well does the post address the intended topic and platform requirements, regardless of brand alignment?

Scoring anchors:
- 9-10: Fully meets the criterion
- 6-8:  Mostly meets it, minor gaps
- 3-5:  Significant problems
- 1-2:  Directly contradicts or ignores the criterion

Calibration examples:
- Guidelines: "Bold, empowering, athlete-first voice." Intent: "Nike Twitter post about new shoe launch." Post: "Just do it. New Pegasus 41 drops tomorrow. Speed meets comfort. #Nike #Running"
  -> faithfulness: 10, relevance: 10
- Guidelines: "Bold, empowering voice." Intent: "Nike Twitter post about new shoe." Post: "Check out our sale on shoes."
  -> faithfulness: 4, relevance: 7  (on-topic but ignores brand voice)
- Guidelines: "Professional, thought-leadership tone." Intent: "LinkedIn post about AI trends." Post: "OMG AI is so lit rn!!! 🔥🔥 #AI"
  -> faithfulness: 2, relevance: 6  (addresses topic but wrong voice)

Evaluate:

Brand Guidelines (source): {source}
Intended topic and platform (question): {question}
Generated post (response): {response}

Respond with ONLY valid JSON, no markdown fences, no preamble:
{{"faithfulness": <int 1-10>, "relevance": <int 1-10>, "explanation": "<one or two sentences>"}}"""


def judge(
    source: str,
    question: str,
    response: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """
    Scores a single post for brand faithfulness and topic relevance.

    Args:
        source:   brand guidelines used for generation
        question: generation intent (e.g. "LinkedIn post for Adobe about X in professional tone")
        response: the generated post text
        model:    Claude model to use as judge

    Returns:
        dict with keys: faithfulness (int 1-10), relevance (int 1-10), explanation (str)
        On failure returns None scores with an explanation message.
    """
    try:
        client = _get_client()
        msg = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    source=source[:2000],
                    question=question,
                    response=response,
                ),
            }],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].removeprefix("json").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"⚠️ Judge call failed: {e}")
        return {
            "faithfulness": None,
            "relevance": None,
            "explanation": f"Judge unavailable: {e}",
        }


def judge_variants(
    variants: list,
    brand_context: str,
    question: str,
    model: str = "claude-haiku-4-5-20251001",
) -> list:
    """
    Runs the judge on all variants in-place, adding judge scores to each.

    Args:
        variants:      list of variant dicts from generate_posts()
        brand_context: brand guidelines / RAG context used during generation
        question:      generation intent string
        model:         Claude model to use as judge

    Returns:
        same list, each variant augmented with:
            judge_faithfulness (int|None)
            judge_relevance    (int|None)
            judge_explanation  (str)
    """
    source = brand_context or "No brand guidelines provided."
    print("Running LLM-as-a-Judge on generated variants...")

    for i, variant in enumerate(variants):
        scores = judge(
            source=source,
            question=question,
            response=variant.get("post_text", ""),
            model=model,
        )
        variant["judge_faithfulness"] = scores.get("faithfulness")
        variant["judge_relevance"] = scores.get("relevance")
        variant["judge_explanation"] = scores.get("explanation", "")
        print(
            f"  Variant {i + 1}: "
            f"faithfulness={scores.get('faithfulness')} "
            f"relevance={scores.get('relevance')}"
        )

    return variants
