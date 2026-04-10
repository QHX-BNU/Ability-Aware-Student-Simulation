from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_from_project_root(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def penalized_balanced_accuracy(acc_plus: float, acc_minus: float) -> float:
    return ((acc_plus + acc_minus) / 2) * (1 - abs(acc_plus - acc_minus))


def compute_overall_metrics(record_csv: str | Path) -> dict[str, float]:
    """Compute overall metrics without ability grouping.

    Expected columns in record_csv:
      - llm_answer, correct_answer, stu_ans

    Definitions (same as evaluation/data.py 'overall'):
      - acc+ : P(LLM correct | student correct)
      - acc- : P(LLM incorrect | student incorrect)
      - BAA  : penalized balanced accuracy of (acc+, acc-)
    """

    record_csv_path = resolve_from_project_root(record_csv)
    if not record_csv_path.exists():
        raise FileNotFoundError(f"record_csv not found: {record_csv_path}")

    df = pd.read_csv(str(record_csv_path))

    required_cols = {"llm_answer", "correct_answer", "stu_ans"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"record_csv missing columns: {sorted(missing)}. "
            f"Expected at least: {sorted(required_cols)}"
        )

    llm_correct = df["llm_answer"] == df["correct_answer"]
    stu_correct = df["stu_ans"] == 1

    correct_mask = stu_correct
    incorrect_mask = ~stu_correct

    num_correct = int(correct_mask.sum())
    num_incorrect = int(incorrect_mask.sum())

    if num_correct == 0 or num_incorrect == 0:
        return {
            "acc+": 0.0,
            "acc-": 0.0,
            "BAA": 0.0,
            "num_correct": float(num_correct),
            "num_incorrect": float(num_incorrect),
        }

    acc_plus = float((llm_correct & correct_mask).sum() / num_correct)
    acc_minus = float(((~llm_correct) & incorrect_mask).sum() / num_incorrect)
    baa = float(penalized_balanced_accuracy(acc_plus, acc_minus))

    return {
        "acc+": acc_plus,
        "acc-": acc_minus,
        "BAA": baa,
        "num_correct": float(num_correct),
        "num_incorrect": float(num_incorrect),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute overall acc+/acc-/BAA without ability grouping."
    )
    parser.add_argument(
        "--record-csv",
        nargs="+",
        required=True,
        help="One or more merged_results.csv paths (relative to project root or absolute).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print acc+/acc-/BAA (one line per record_csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for record_csv in args.record_csv:
        metrics = compute_overall_metrics(record_csv)
        record_csv_path = resolve_from_project_root(record_csv)

        if args.quiet:
            print(
                f"{record_csv_path}\t"
                f"acc+={metrics['acc+']:.6f}\tacc-={metrics['acc-']:.6f}\tBAA={metrics['BAA']:.6f}"
            )
        else:
            print("=" * 80)
            print(f"record_csv: {record_csv_path}")
            print(f"acc+ : {metrics['acc+']:.6f}")
            print(f"acc- : {metrics['acc-']:.6f}")
            print(f"BAA  : {metrics['BAA']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
