"""Semantic meaning similarity scoring using MiniLM sentence embeddings."""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Load and return the singleton SentenceTransformer('all-MiniLM-L6-v2') model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.SIMILARITY_MODEL_NAME)
    return _model


def score(a: str, b: str) -> float:
    """Compute cosine similarity between two texts, clamped to [0.0, 1.0].

    Args:
        a: First text (e.g., original comment).
        b: Second text (e.g., candidate rewrite).

    Returns:
        Cosine similarity float in [0.0, 1.0].
    """
    if not a or not a.strip() or not b or not b.strip():
        return 0.0
    if a.strip() == b.strip():
        return 1.0

    model = get_model()
    embeddings = model.encode([a, b], normalize_embeddings=True)
    cosine_sim = float(np.dot(embeddings[0], embeddings[1]))
    clamped = max(0.0, min(1.0, cosine_sim))
    return round(clamped, 4)


if __name__ == "__main__":
    s1 = "Please fix the broken login page immediately."
    s2 = "Kindly repair the login screen as soon as possible."
    s3 = "The weather looks nice today."
    print("Similar pair score:", score(s1, s2))
    print("Unrelated pair score:", score(s1, s3))
    print("Identical pair score:", score(s1, s1))

