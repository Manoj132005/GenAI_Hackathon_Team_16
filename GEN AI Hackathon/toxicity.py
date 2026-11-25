"""Toxic-BERT scoring and offensive-word masking utilities."""

import re
from typing import Optional

from better_profanity import profanity
from detoxify import Detoxify

import config

# Matches alphabetic/numeric words, including inline symbol substitutions like f@ck
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[@$*]+[A-Za-z0-9]+)*")

_model: Optional[Detoxify] = None
_profanity_initialized: bool = False
_word_offensive_cache: dict[str, bool] = {}


def get_model() -> Detoxify:
    """Load and return the singleton Detoxify('original') Toxic-BERT model."""
    global _model
    if _model is None:
        _model = Detoxify(config.TOXICITY_MODEL_TYPE)
    return _model


def _ensure_profanity_loaded() -> None:
    """Load the better-profanity default word list once."""
    global _profanity_initialized
    if not _profanity_initialized:
        profanity.load_censor_words()
        _profanity_initialized = True


def _mask_token(token: str) -> str:
    """Keep the first character and replace the remaining characters with asterisks."""
    if len(token) <= 1:
        return token
    return token[0] + ("*" * (len(token) - 1))


_DETOXIFY_TO_CANONICAL = {
    "toxicity": "toxic",
    "severe_toxicity": "severe_toxic",
    "obscene": "obscene",
    "threat": "threat",
    "insult": "insult",
    "identity_attack": "identity_hate",
}


def _normalize_predictions(raw_scores: dict) -> dict:
    """Map Detoxify's standardized keys back to the six Toxic-BERT label names."""
    normalized = {}
    for key, value in raw_scores.items():
        canonical = _DETOXIFY_TO_CANONICAL.get(key, key)
        normalized[canonical] = value
    return normalized


def _check_words_offensive(words: list[str]) -> None:
    """Populate _word_offensive_cache for any unseen lowercase words."""
    _ensure_profanity_loaded()
    words_for_model: list[str] = []

    for word in words:
        if word in _word_offensive_cache:
            continue
        if len(word) <= 1:
            _word_offensive_cache[word] = False
            continue
        if profanity.contains_profanity(word):
            _word_offensive_cache[word] = True
            continue
        words_for_model.append(word)

    if words_for_model:
        model = get_model()
        predictions = _normalize_predictions(model.predict(words_for_model))
        toxic_scores = predictions["toxic"]
        if isinstance(toxic_scores, (int, float)):
            toxic_scores = [toxic_scores]
        for word, toxic_score in zip(words_for_model, toxic_scores):
            _word_offensive_cache[word] = float(toxic_score) > config.WORD_TOXIC_THRESHOLD


def score(text: str) -> dict[str, float | str]:
    """Score text toxicity using Toxic-BERT ('original').

    Args:
        text: Input comment text.

    Returns:
        Dict with keys 'toxic' (float), 'top_category' (str), and 'top_score' (float).
    """
    if not text or not text.strip():
        return {"toxic": 0.0, "top_category": "toxic", "top_score": 0.0}

    model = get_model()
    raw_scores = _normalize_predictions(model.predict(text))
    scores = {label: float(raw_scores[label]) for label in config.TOXICITY_LABELS}
    top_category = max(scores, key=lambda label: scores[label])
    return {
        "toxic": round(scores["toxic"], 4),
        "top_category": top_category,
        "top_score": round(scores[top_category], 4),
    }


def mask(text: str) -> str:
    """Mask offensive words, keeping the first letter and replacing the rest with '*'.

    Detects offensive words using both better-profanity and a per-word Toxic-BERT
    check (masking when single-word 'toxic' score > 0.5).
    """
    if not text:
        return ""

    matches = list(_TOKEN_PATTERN.finditer(text))
    if not matches:
        return text

    unique_words = list(dict.fromkeys(m.group(0).lower() for m in matches))
    _check_words_offensive(unique_words)

    parts: list[str] = []
    last_idx = 0
    for match in matches:
        start, end = match.span()
        parts.append(text[last_idx:start])
        token = match.group(0)
        if _word_offensive_cache.get(token.lower(), False):
            parts.append(_mask_token(token))
        else:
            parts.append(token)
        last_idx = end

    parts.append(text[last_idx:])
    return "".join(parts)


if __name__ == "__main__":
    sample_toxic = "You are such a stupid idiot, shut the fuck up!"
    sample_clean = "I strongly disagree with your approach to this problem."
    print("Toxic comment score:", score(sample_toxic))
    print("Clean comment score:", score(sample_clean))
    print("Masked toxic:", mask(sample_toxic))
    print("Masked clean:", mask(sample_clean))

