from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_from_project_root(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_student_ability_from_embedding(
    emb_npy: str | Path,
    num_students: int,
    id_base: int = 1,
) -> pd.DataFrame:
    """Compute per-student ability from embedding matrix.

    Ability definition (same as the original script):
      ability = mean(sigmoid(embedding_row))

    Returns a DataFrame with columns: id, ability
    where id starts from id_base (default 1).
    """

    emb_npy_path = resolve_from_project_root(emb_npy)
    if not emb_npy_path.exists():
        raise FileNotFoundError(f"Embedding npy not found: {emb_npy_path}")

    data = np.load(str(emb_npy_path))
    if data.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape={data.shape}")
    if num_students <= 0 or num_students > data.shape[0]:
        raise ValueError(f"Invalid num_students={num_students} for data shape={data.shape}")

    students = data[:num_students]
    ability = sigmoid(students).mean(axis=1)

    df = pd.DataFrame({
        "id": np.arange(id_base, id_base + len(ability)),
        "ability": ability.astype(float),
    })
    return df


# Compatibility alias for the naming you mentioned (common misspelling: calu_stu_ablility)
def calu_stu_ablility(
    emb_npy: str | Path,
    output_csv: str | Path,
    num_students: int,
    id_base: int = 1,
) -> Path:
    """Compute student ability and save to CSV (columns: id, ability)."""

    output_csv_path = resolve_from_project_root(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = compute_student_ability_from_embedding(emb_npy=emb_npy, num_students=num_students, id_base=id_base)
    df.to_csv(str(output_csv_path), index=False)
    return output_csv_path


def select_students_by_ability_quantile(
    ability_csv: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
    total_per_level: int = 100,
    levels: Iterable[str] = ("low", "medium", "high"),
    mandatory_id_min: int = 1,
    mandatory_id_max: int = 101,
) -> tuple[Path, Path]:
    """Select students by ability quantiles and export CSV + JSON.

    - Reads ability CSV containing columns: id, ability
    - Splits into 3 levels by 1/3 and 2/3 quantiles
    - Ensures students with stu_id in [mandatory_id_min, mandatory_id_max] (within the filtered set)
      are included in the selection
    - Exports:
        output_csv with columns: stu_id, ablility_level
        output_json as a sorted list of selected stu_id
    """

    ability_csv_path = resolve_from_project_root(ability_csv)
    output_csv_path = resolve_from_project_root(output_csv)
    output_json_path = resolve_from_project_root(output_json)

    if not ability_csv_path.exists():
        raise FileNotFoundError(f"Ability csv not found: {ability_csv_path}")

    df = pd.read_csv(str(ability_csv_path))
    df = df.rename(columns={"id": "stu_id"})

    required_cols = {"stu_id", "ability"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    df = df.drop_duplicates(subset="stu_id")

    min_needed = total_per_level * 3
    if len(df) < min_needed:
        raise ValueError(f"Only {len(df)} students available, cannot select {min_needed} students")

    q1 = df["ability"].quantile(1 / 3)
    q2 = df["ability"].quantile(2 / 3)

    def assign_level(a: float) -> str:
        if a <= q1:
            return "low"
        if a <= q2:
            return "medium"
        return "high"

    # Keep the column name consistent with existing evaluation scripts
    # (the original notebook spelling is 'ablility_level')
    df["ablility_level"] = df["ability"].apply(assign_level)

    mandatory_ids = set(df[df["stu_id"].between(mandatory_id_min, mandatory_id_max)]["stu_id"])

    selected_rows: list[pd.DataFrame] = []
    for level in levels:
        level_df = df[df["ablility_level"] == level]

        must_have = level_df[level_df["stu_id"].isin(mandatory_ids)].drop_duplicates(subset="stu_id")

        # If the number of mandatory students in a level exceeds the per-level quota,
        # we cannot satisfy both constraints at the same time:
        # (1) fixed total_per_level per level, and (2) mandatory inclusion of all mandatory IDs.
        if len(must_have) > total_per_level:
            raise ValueError(
                f"Too many mandatory students in level '{level}': {len(must_have)} > total_per_level={total_per_level}. "
                "Increase total_per_level or shrink the mandatory id range."
            )

        if len(must_have) < total_per_level:
            remaining = level_df[~level_df["stu_id"].isin(must_have["stu_id"])]
            supplement = remaining.head(total_per_level - len(must_have))
            final_level = pd.concat([must_have, supplement])
        else:
            final_level = must_have.head(total_per_level)

        if len(final_level) < total_per_level:
            raise ValueError(f"Not enough students to fill {level} level to {total_per_level}")

        selected_rows.append(final_level)

    final_df = pd.concat(selected_rows, ignore_index=True)
    expected_total = total_per_level * 3
    if len(final_df) != expected_total:
        raise ValueError(f"Total selected students={len(final_df)}, expected {expected_total}")

    selected_ids = set(final_df["stu_id"])
    missing_mandatory = mandatory_ids - selected_ids
    if missing_mandatory:
        raise ValueError(f"Missing mandatory students: {missing_mandatory}")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    final_out = final_df[["stu_id", "ablility_level"]]
    final_out.to_csv(str(output_csv_path), index=False)

    stu_ids = [int(x) for x in sorted(final_out["stu_id"].unique())]
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(stu_ids, f, ensure_ascii=False)

    return output_csv_path, output_json_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute student ability from embedding, then select students by ability quantiles."
    )

    parser.add_argument(
        "--emb-npy",
        default="cdm/stu_embedding/student_emb.npy",
        help="Embedding npy path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--num-students",
        type=int,
        default=1299,
        help="Number of student rows at the top of the embedding matrix.",
    )
    parser.add_argument(
        "--id-base",
        type=int,
        default=1,
        help="Student id base in ability CSV (1 means ids are 1..N, 0 means 0..N-1).",
    )

    parser.add_argument(
        "--ability-csv",
        default="scripts/data/student_ability.csv",
        help="Output ability CSV path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--selected-csv",
        default="scripts/data/selected_students_by_ability.csv",
        help="Output selected CSV path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--selected-json",
        default="agent/data/agent_id_list.json",
        help="Output selected IDs JSON path (relative to project root or absolute).",
    )

    parser.add_argument("--total-per-level", type=int, default=100)
    parser.add_argument(
        "--mandatory-id-min",
        type=int,
        default=1,
        help="Mandatory student id range min (inclusive). Student id is row_index+1 by default.",
    )
    parser.add_argument(
        "--mandatory-id-max",
        type=int,
        default=101,
        help="Mandatory student id range max (inclusive). Student id is row_index+1 by default.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    ability_csv_path = calu_stu_ablility(
        emb_npy=args.emb_npy,
        output_csv=args.ability_csv,
        num_students=args.num_students,
        id_base=args.id_base,
    )

    selected_csv_path, selected_json_path = select_students_by_ability_quantile(
        ability_csv=ability_csv_path,
        output_csv=args.selected_csv,
        output_json=args.selected_json,
        total_per_level=args.total_per_level,
        mandatory_id_min=args.mandatory_id_min,
        mandatory_id_max=args.mandatory_id_max,
    )

    df_selected = pd.read_csv(str(selected_csv_path))
    print("Done.")
    print(f"Ability CSV: {ability_csv_path}")
    print(f"Selected CSV: {selected_csv_path}")
    print(f"Selected JSON: {selected_json_path}")
    print("Counts per level:")
    print(df_selected["ablility_level"].value_counts())
    print(f"Students selected: {len(df_selected)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
