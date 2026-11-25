"""Configuration constants and thresholds for Safe Public Comment Rewrite Reviewer."""

from pathlib import Path

# Project paths
BASE_DIR: Path = Path(__file__).resolve().parent
DATASET_PATH: Path = BASE_DIR / "Dataset Paradetox.xlsx"
JUDGE_CACHE_PATH: Path = BASE_DIR / "judge_cache.json"
EVAL_DIR: Path = BASE_DIR / "eval"
EVAL_SET_PATH: Path = EVAL_DIR / "eval_set.csv"
EVAL_RESULTS_PATH: Path = EVAL_DIR / "eval_results.csv"
OVER_SANITIZED_CACHE_PATH: Path = EVAL_DIR / "over_sanitized_cache.json"

# Verdict thresholds
TOXIC_THRESHOLD: float = 0.3
MIN_DROP: float = 0.2
FALLBACK_SIMILARITY_THRESHOLD: float = 0.35

# Toxicity classifier & masking settings
TOXICITY_MODEL_TYPE: str = "original"
TOXICITY_LABELS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)
WORD_TOXIC_THRESHOLD: float = 0.5

# Meaning similarity model
SIMILARITY_MODEL_NAME: str = "all-MiniLM-L6-v2"

# UI settings
UI_EXAMPLE_COUNT: int = 12

# Gemini LLM judge settings
GEMINI_TEMPERATURE: float = 0.0
GEMINI_MAX_RETRIES: int = 3
GEMINI_BASE_BACKOFF_SECONDS: float = 1.0

# Evaluation settings
EVAL_RANDOM_SEED: int = 42
EVAL_SAMPLE_SIZE: int = 150
OVER_SANITIZED_COUNT: int = 30

# Verdict labels
VERDICT_SAFE: str = "Safe"
VERDICT_MEANING_LOST: str = "Meaning lost"
VERDICT_STILL_TOXIC: str = "Still toxic"

