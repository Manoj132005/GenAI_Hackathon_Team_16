"""Revert report.py to its original form (no config import added here)."""

from typing import Any

import judge
import similarity
import toxicity
import verdict


def build(original: str, candidate: str) -> dict[str, Any]:
    """Run toxicity, similarity, judge, and verdict evaluation and mask all text fields.

    Args:
        original: Original toxic comment.
        candidate: Candidate neutral rewrite.

    Returns:
        Dictionary containing scores, masked text fields, judge output, and final verdict.
    """
    tox_before = toxicity.score(original)
    tox_after = toxicity.score(candidate)
    sim = similarity.score(original, candidate)
    judge_result = judge.evaluate(original, candidate)
    verdict_result = verdict.decide(tox_before, tox_after, sim, judge_result)

    masked_judge = {
        "meaning_preserved": str(judge_result.get("meaning_preserved", "partial")),
        "what_was_lost": toxicity.mask(str(judge_result.get("what_was_lost", ""))),
        "reason": toxicity.mask(str(judge_result.get("reason", ""))),
        "available": bool(judge_result.get("available", False)),
    }

    masked_verdict = {
        "label": str(verdict_result.get("label", "")),
        "explanation": toxicity.mask(str(verdict_result.get("explanation", ""))),
    }

    return {
        "masked_original": toxicity.mask(original),
        "masked_candidate": toxicity.mask(candidate),
        "toxicity_before": tox_before,
        "toxicity_after": tox_after,
        "toxicity_drop": round(float(tox_before["toxic"]) - float(tox_after["toxic"]), 4),
        "similarity": sim,
        "judge": masked_judge,
        "verdict": masked_verdict,
    }
