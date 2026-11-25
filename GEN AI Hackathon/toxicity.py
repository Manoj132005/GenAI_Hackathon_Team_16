"""Lightweight local toxicity estimates and offensive-word masking.

This fallback deliberately avoids PyTorch. It uses better-profanity plus a small
explicit insult/threat lexicon, so its scores are approximate and should not be
treated as a trained toxicity classifier.
"""

import re

from better_profanity import profanity

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[@$*]+[A-Za-z0-9]+)*")
_INSULTS = {
    "asshole", "bastard", "bitch", "dick", "dumb", "idiot", "moron",
    "loser", "stupid", "worthless",
}
_THREATS = {"attack", "bomb", "kill", "murder", "shoot", "threaten"}


def _normalize(token: str) -> str:
    return token.lower().replace("@", "a").replace("$", "s").replace("*", "")


def _is_offensive(token: str) -> bool:
    normalized = _normalize(token)
    return (
        normalized in _INSULTS
        or normalized in _THREATS
        or profanity.contains_profanity(normalized)
    )


def get_model() -> None:
    """Compatibility no-op; no neural model is loaded in fallback mode."""
    return None


def score(text: str) -> dict[str, float | str]:
    """Estimate toxicity from a local profanity and insult lexicon.

    Each matched term raises the score; the score is a heuristic, not a model
    probability. The limited lexicon can miss context, coded language, and
    identity-directed abuse.
    """
    if not text or not text.strip():
        return {"toxic": 0.0, "top_category": "toxic", "top_score": 0.0}

    tokens = [m.group(0) for m in _TOKEN_PATTERN.finditer(text)]
    matched = [token for token in tokens if _is_offensive(token)]
    if not matched:
        return {"toxic": 0.0, "top_category": "toxic", "top_score": 0.0}

    score_value = round(min(1.0, 0.55 + 0.15 * (len(matched) - 1)), 4)
    normalized = [_normalize(token) for token in matched]
    if any(token in _THREATS for token in normalized):
        category = "threat"
    elif any(token in _INSULTS for token in normalized):
        category = "insult"
    else:
        category = "obscene"
    return {"toxic": score_value, "top_category": category, "top_score": score_value}


def mask(text: str) -> str:
    """Mask profanity and explicit insults while preserving the first letter."""
    if not text:
        return ""
    parts: list[str] = []
    last_idx = 0
    for match in _TOKEN_PATTERN.finditer(text):
        parts.append(text[last_idx:match.start()])
        token = match.group(0)
        if _is_offensive(token):
            parts.append(token[0] + "*" * (len(token) - 1) if len(token) > 1 else token)
        else:
            parts.append(token)
        last_idx = match.end()
    parts.append(text[last_idx:])
    return "".join(parts)


if __name__ == "__main__":
    sample_toxic = "You are such a stupid idiot, shut the fuck up!"
    sample_clean = "I strongly disagree with your approach to this problem."
    print("Toxicity estimate:", score(sample_toxic))
    print("Clean text estimate:", score(sample_clean))
    print("Masked toxic:", mask(sample_toxic))
    print("Masked clean:", mask(sample_clean))
