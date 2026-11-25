"""Final verdict decision logic combining toxicity, similarity, and LLM judge outputs."""

from typing import Any

import config


def decide(
    tox_before: dict[str, Any],
    tox_after: dict[str, Any],
    sim: float,
    judge_result: dict[str, Any],
) -> dict[str, str]:
    """Compute the final evaluation verdict for a candidate rewrite.

    Args:
        tox_before: Toxicity score dict for the original comment.
        tox_after: Toxicity score dict for the candidate rewrite.
        sim: Cosine similarity score in [0.0, 1.0].
        judge_result: Output dict from judge.evaluate().

    Returns:
        Dict with keys 'label' ('Safe' | 'Meaning lost' | 'Still toxic') and 'explanation'.
    """
    before_score = float(tox_before.get("toxic", 0.0))
    after_score = float(tox_after.get("toxic", 0.0))
    drop = before_score - after_score
    judge_available = bool(judge_result.get("available", False))

    insufficient_drop = before_score > config.TOXIC_THRESHOLD and drop < config.MIN_DROP
    if after_score > config.TOXIC_THRESHOLD or insufficient_drop:
        reasons: list[str] = []
        if after_score > config.TOXIC_THRESHOLD:
            reasons.append(
                f"candidate toxicity ({after_score:.2f}) exceeds threshold ({config.TOXIC_THRESHOLD:.2f})"
            )
        if insufficient_drop:
            reasons.append(
                f"toxicity drop ({drop:.2f}) is below minimum required drop ({config.MIN_DROP:.2f})"
            )
        judge_note = (
            ""
            if judge_available
            else " Note: Gemini judge was unavailable."
        )
        return {
            "label": config.VERDICT_STILL_TOXIC,
            "explanation": (
                f"Still toxic because {' and '.join(reasons)}. "
                f"Supporting MiniLM similarity: {sim:.2f}.{judge_note}"
            ),
        }

    if judge_available:
        meaning = str(judge_result.get("meaning_preserved", "partial")).lower()
        if meaning == "no":
            return {
                "label": config.VERDICT_MEANING_LOST,
                "explanation": (
                    f"Toxicity dropped sufficiently ({before_score:.2f} -> {after_score:.2f}), "
                    f"but the Gemini judge flagged that the original meaning was not preserved. "
                    f"Supporting MiniLM similarity: {sim:.2f}."
                ),
            }
        return {
            "label": config.VERDICT_SAFE,
            "explanation": (
                f"Toxicity dropped sufficiently ({before_score:.2f} -> {after_score:.2f}) "
                f"and meaning preservation was rated '{meaning}' by the Gemini judge. "
                f"Supporting MiniLM similarity: {sim:.2f}."
            ),
        }

    # Fallback when Gemini judge is unavailable
    if sim < config.FALLBACK_SIMILARITY_THRESHOLD:
        return {
            "label": config.VERDICT_MEANING_LOST,
            "explanation": (
                f"Gemini judge was unavailable; using similarity fallback. "
                f"Toxicity dropped ({before_score:.2f} -> {after_score:.2f}), "
                f"but MiniLM similarity ({sim:.2f}) is below {config.FALLBACK_SIMILARITY_THRESHOLD:.2f}."
            ),
        }

    return {
        "label": config.VERDICT_SAFE,
        "explanation": (
            f"Gemini judge was unavailable; using similarity fallback. "
            f"Toxicity dropped ({before_score:.2f} -> {after_score:.2f}) "
            f"and MiniLM similarity ({sim:.2f}) is at or above {config.FALLBACK_SIMILARITY_THRESHOLD:.2f}."
        ),
    }


if __name__ == "__main__":
    print(
        "Case Safe (judge):",
        decide(
            {"toxic": 0.95, "top_category": "toxic", "top_score": 0.95},
            {"toxic": 0.02, "top_category": "toxic", "top_score": 0.02},
            0.68,
            {"meaning_preserved": "yes", "what_was_lost": "Profanity", "reason": "Kept intent", "available": True},
        ),
    )
    print(
        "Case Still toxic:",
        decide(
            {"toxic": 0.95, "top_category": "toxic", "top_score": 0.95},
            {"toxic": 0.60, "top_category": "toxic", "top_score": 0.60},
            0.85,
            {"meaning_preserved": "yes", "what_was_lost": "None", "reason": "Still rude", "available": True},
        ),
    )
    print(
        "Case Meaning lost (judge):",
        decide(
            {"toxic": 0.95, "top_category": "toxic", "top_score": 0.95},
            {"toxic": 0.01, "top_category": "toxic", "top_score": 0.01},
            0.20,
            {"meaning_preserved": "no", "what_was_lost": "Main topic", "reason": "Unrelated", "available": True},
        ),
    )
    print(
        "Case Fallback Meaning lost:",
        decide(
            {"toxic": 0.95, "top_category": "toxic", "top_score": 0.95},
            {"toxic": 0.01, "top_category": "toxic", "top_score": 0.01},
            0.22,
            {"meaning_preserved": "partial", "what_was_lost": "Judge unavailable.", "reason": "No key", "available": False},
        ),
    )

