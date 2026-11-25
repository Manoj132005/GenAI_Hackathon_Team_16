"""Build a simplified 120-case labelled evaluation CSV from the local ParaDetox dataset."""

import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import data

EVAL_SAMPLE_SIZE = 40


def build_eval_set() -> pd.DataFrame:
    """Sample 40 ParaDetox pairs and produce 120 labelled evaluation cases (3 per pair).

    Case types per pair:
      - safe:         toxic text + its human rewrite   -> expected: Safe
      - still_toxic:  toxic text as its own candidate  -> expected: Still toxic
      - meaning_lost: toxic text + a different pair's rewrite -> expected: Meaning lost
    """
    pairs = data.load_pairs()
    rng = random.Random(config.EVAL_RANDOM_SEED)
    sample_size = min(EVAL_SAMPLE_SIZE, len(pairs))
    sampled = rng.sample(pairs, sample_size)

    rows: list[dict[str, str]] = []
    for idx, (toxic_text, neutral_text) in enumerate(sampled):
        # safe: original toxic + its human rewrite
        rows.append({
            "case_id": f"safe_{idx:03d}",
            "case_type": "safe",
            "expected_label": config.VERDICT_SAFE,
            "original": toxic_text,
            "candidate": neutral_text,
        })

        # still_toxic: toxic text paired with itself
        rows.append({
            "case_id": f"still_toxic_{idx:03d}",
            "case_type": "still_toxic",
            "expected_label": config.VERDICT_STILL_TOXIC,
            "original": toxic_text,
            "candidate": toxic_text,
        })

        # meaning_lost: toxic text + a rewrite from a different pair
        other_idx = (idx + 7) % sample_size
        if sampled[other_idx][1] == neutral_text and sample_size > 1:
            other_idx = (idx + 1) % sample_size
        rows.append({
            "case_id": f"meaning_lost_{idx:03d}",
            "case_type": "meaning_lost",
            "expected_label": config.VERDICT_MEANING_LOST,
            "original": toxic_text,
            "candidate": sampled[other_idx][1],
        })

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.EVAL_SET_PATH, index=False, encoding="utf-8")
    print(f"Saved evaluation set: {len(df)} cases ({sample_size} pairs x 3 types) -> {config.EVAL_SET_PATH}")
    return df


if __name__ == "__main__":
    build_eval_set()
