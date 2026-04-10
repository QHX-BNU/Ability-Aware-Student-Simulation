from __future__ import annotations

from pathlib import Path
import argparse
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_from_project_root(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def penalized_balanced_accuracy(acc_correct: float, acc_incorrect: float) -> float:
    return ((acc_correct + acc_incorrect) / 2) * (1 - abs(acc_correct - acc_incorrect))


def compute_group_accuracy_sum(
    record_csv: str | Path,
    ability_csv: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute ability-grouped metrics.

    Expected columns in record_csv:
      - id, llm_answer, correct_answer, stu_ans

    Expected columns in ability_csv:
      - stu_id, ablility_level

    Notes:
      - This function preserves the original notebook column name 'ablility_level'.
      - record_csv and ability_csv are resolved relative to PROJECT_ROOT if given as relative paths.
    """

    record_csv_path = resolve_from_project_root(record_csv)
    if ability_csv is None:
        ability_csv = Path("scripts/data/selected_students_by_ability.csv")
    ability_csv_path = resolve_from_project_root(ability_csv)

    if not record_csv_path.exists():
        raise FileNotFoundError(f"record_csv not found: {record_csv_path}")
    if not ability_csv_path.exists():
        raise FileNotFoundError(
            f"ability_csv not found: {ability_csv_path}. "
            "Pass --ability-csv to point to your selected_students_by_ability.csv"
        )

    df = pd.read_csv(str(record_csv_path))
    ability_df = pd.read_csv(str(ability_csv_path))

    df = df.merge(
        ability_df,
        left_on="id",
        right_on="stu_id",
        how="inner",
    )

    df["llm_correct"] = df["llm_answer"] == df["correct_answer"]
    df["stu_correct"] = df["stu_ans"] == 1

    results: dict[str, dict[str, Any]] = {}

    for level, group in df.groupby("ablility_level"):
        correct_mask = group["stu_correct"]
        incorrect_mask = ~correct_mask

        if correct_mask.sum() == 0 or incorrect_mask.sum() == 0:
            results[str(level)] = {
                "correct_accuracy": 0.0,
                "incorrect_accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "num_correct": int(correct_mask.sum()),
                "num_incorrect": int(incorrect_mask.sum()),
            }
            continue

        acc_correct = (group["llm_correct"] & correct_mask).sum() / correct_mask.sum()
        acc_incorrect = ((~group["llm_correct"]) & incorrect_mask).sum() / incorrect_mask.sum()

        results[str(level)] = {
            "correct_accuracy": float(acc_correct),
            "incorrect_accuracy": float(acc_incorrect),
            "balanced_accuracy": float(penalized_balanced_accuracy(float(acc_correct), float(acc_incorrect))),
            "num_correct": int(correct_mask.sum()),
            "num_incorrect": int(incorrect_mask.sum()),
        }

    correct_mask = df["stu_correct"]
    incorrect_mask = ~correct_mask

    if correct_mask.sum() > 0 and incorrect_mask.sum() > 0:
        overall_correct = (df["llm_correct"] & correct_mask).sum() / correct_mask.sum()
        overall_incorrect = ((~df["llm_correct"]) & incorrect_mask).sum() / incorrect_mask.sum()

        results["overall"] = {
            "correct_accuracy": float(overall_correct),
            "incorrect_accuracy": float(overall_incorrect),
            "balanced_accuracy": float(
                penalized_balanced_accuracy(float(overall_correct), float(overall_incorrect))
            ),
            "num_correct": int(correct_mask.sum()),
            "num_incorrect": int(incorrect_mask.sum()),
        }
    else:
        results["overall"] = {
            "correct_accuracy": 0.0,
            "incorrect_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "num_correct": int(correct_mask.sum()),
            "num_incorrect": int(incorrect_mask.sum()),
        }

    return results


def results_to_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "ability_level"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ability-grouped accuracy metrics.")
    parser.add_argument(
        "--record-csv",
        nargs="+",
        required=True,
        help="One or more merged_results.csv paths (relative to project root or absolute).",
    )
    parser.add_argument(
        "--ability-csv",
        default="scripts/data/selected_students_by_ability.csv",
        help=(
            "selected_students_by_ability.csv path (relative to project root or absolute). "
            "Default: scripts/data/selected_students_by_ability.csv"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for record_csv in args.record_csv:
        results = compute_group_accuracy_sum(record_csv=record_csv, ability_csv=args.ability_csv)
        df_results = results_to_frame(results)

        print("=" * 80)
        print(f"record_csv: {resolve_from_project_root(record_csv)}")
        print(df_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
