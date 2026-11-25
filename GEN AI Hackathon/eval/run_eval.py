"""Run the evaluation pipeline on the 120-case labelled test set and report accuracy."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import report
from eval.build_eval_set import build_eval_set

LABELS = [config.VERDICT_SAFE, config.VERDICT_MEANING_LOST, config.VERDICT_STILL_TOXIC]


def run_evaluation() -> None:
    """Score all cases in eval_set.csv, save masked results, and print accuracy metrics."""
    if not config.EVAL_SET_PATH.exists():
        print("eval_set.csv not found; building now...")
        eval_df = build_eval_set()
    else:
        eval_df = pd.read_csv(config.EVAL_SET_PATH)

    total = len(eval_df)
    print(f"Running pipeline on {total} cases (judge cache reused for already-scored pairs)...\n")

    result_rows: list[dict] = []
    for row_num, (_, row) in enumerate(eval_df.iterrows(), start=1):
        rpt = report.build(str(row["original"]), str(row["candidate"]))
        result_rows.append({
            "case_id": row["case_id"],
            "case_type": row["case_type"],
            "expected_label": str(row["expected_label"]),
            "predicted_label": rpt["verdict"]["label"],
            "judge_available": rpt["judge"]["available"],
            "correct": rpt["verdict"]["label"] == str(row["expected_label"]),
            "toxicity_before": rpt["toxicity_before"]["toxic"],
            "toxicity_after": rpt["toxicity_after"]["toxic"],
            "similarity": rpt["similarity"],
        })
        if row_num % 10 == 0 or row_num == total:
            done = sum(r["correct"] for r in result_rows)
            print(f"  [{row_num:>3}/{total}] correct so far: {done}/{row_num}")

    results_df = pd.DataFrame(result_rows)
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.EVAL_RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nSaved masked results -> {config.EVAL_RESULTS_PATH}")

    # Judge availability summary
    fallback_cases = results_df[~results_df["judge_available"]]
    judge_cases = results_df[results_df["judge_available"]]
    print(f"\nJudge availability: {len(judge_cases)} used Gemini, {len(fallback_cases)} used similarity fallback.")
    if len(fallback_cases) > 0:
        print("Fallback cases excluded from the accuracy table below.")

    scored = judge_cases.copy()
    if scored.empty:
        print("No judge-scored cases to evaluate.")
        return

    y_true = scored["expected_label"].tolist()
    y_pred = scored["predicted_label"].tolist()

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True: {lbl}" for lbl in LABELS],
        columns=[f"Pred: {lbl}" for lbl in LABELS],
    )
    print("\nConfusion Matrix (judge-scored cases only):")
    print(cm_df.to_string())

    print("\nPer-class Accuracy (judge-scored cases only):")
    for lbl in LABELS:
        subset = scored[scored["expected_label"] == lbl]
        if len(subset) == 0:
            print(f"  {lbl:<14}: N/A")
        else:
            acc = float(subset["correct"].mean())
            print(f"  {lbl:<14}: {acc:.4f}  ({int(subset['correct'].sum())}/{len(subset)})")

    overall = float(scored["correct"].mean())
    print(f"\nOverall Accuracy (judge-scored, n={len(scored)}): {overall:.4f}  ({int(scored['correct'].sum())}/{len(scored)})")

    if len(fallback_cases) > 0:
        fb_acc = float(fallback_cases["correct"].mean())
        print(f"Fallback Accuracy  (similarity only, n={len(fallback_cases)}): {fb_acc:.4f}  ({int(fallback_cases['correct'].sum())}/{len(fallback_cases)})")


if __name__ == "__main__":
    run_evaluation()
