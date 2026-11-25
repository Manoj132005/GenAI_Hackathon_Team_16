"""Local ParaDetox dataset loading utilities."""

from pathlib import Path
from typing import Optional

import pandas as pd

import config


def _read_dataframe(path: Path) -> pd.DataFrame:
    """Read a tabular dataset file into a DataFrame based on its extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset file format: {suffix}")


def _clean_text(value: object) -> str:
    """Convert a cell value to a stripped string, returning '' for missing values."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_pairs(dataset_path: Optional[Path | str] = None) -> list[tuple[str, str]]:
    """Load (toxic, neutral) comment pairs from the local ParaDetox dataset file.

    Args:
        dataset_path: Optional override path. Defaults to config.DATASET_PATH.

    Returns:
        A list of (toxic, neutral) string tuples with empty/missing rows removed.
    """
    path = Path(dataset_path) if dataset_path is not None else config.DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = _read_dataframe(path)

    toxic_candidates = [col for col in df.columns if str(col).strip().lower() == "toxic"]
    if not toxic_candidates:
        raise ValueError(f"Could not find 'toxic' column in {list(df.columns)}")
    toxic_col = toxic_candidates[0]

    neutral_cols = [
        col for col in df.columns if str(col).strip().lower().startswith("neutral")
    ]
    if not neutral_cols:
        raise ValueError(f"Could not find any neutral column in {list(df.columns)}")

    pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        toxic_text = _clean_text(row[toxic_col])
        if not toxic_text:
            continue

        neutral_text = ""
        for col in neutral_cols:
            candidate = _clean_text(row[col])
            if candidate:
                neutral_text = candidate
                break

        if not neutral_text:
            continue

        pairs.append((toxic_text, neutral_text))

    return pairs


if __name__ == "__main__":
    loaded_pairs = load_pairs()
    print(f"Dataset path: {config.DATASET_PATH}")
    print(f"Loaded valid (toxic, neutral) pairs: {len(loaded_pairs)}")

