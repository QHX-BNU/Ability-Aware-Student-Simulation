# evaluation

This directory contains evaluation scripts.

> Note: Each script infers the project root `PROJECT_ROOT` from its own file location.
> If you pass a relative path, it will be resolved as `PROJECT_ROOT/<relative path>`, so the scripts do not depend on the current working directory.

## Input format

Both scripts take `merged_results.csv` as input (typically created or appended by `scripts/extra_ans.py`).

`merged_results.csv` must contain at least the following columns:

- `llm_answer`: the option selected by the model (e.g., `A/B/C/D/E`)
- `correct_answer`: the gold answer (e.g., `A/B/C/D/E`)
- `stu_ans`: whether the student is correct (current convention: `stu_ans == 1` means student correct; otherwise student incorrect)

For ability-grouped evaluation, you also need an ability grouping file:

- `selected_students_by_ability.csv` (default location: `scripts/data/selected_students_by_ability.csv`)
  - required columns: `stu_id`, `ablility_level`
  - the spelling `ablility_level` is kept for compatibility (`low | medium | high`)

## Metric definitions (as used by the scripts)

Let:

- “LLM correct” be `llm_answer == correct_answer`
- “student correct” be `stu_ans == 1`

Then:

- `acc+`: $P(\mathrm{LLM\ correct}\mid\mathrm{student\ correct})$
- `acc-`: $P(\mathrm{LLM\ incorrect}\mid\mathrm{student\ incorrect})$
- `BAA`:

$$
\mathrm{BAA} = \frac{acc+ + acc-}{2} \cdot (1 - |acc+ - acc-|)
$$

## 1) Ability-grouped evaluation (data.py)

Script: [evaluation/data.py](evaluation/data.py)

What it does:

- merges `merged_results.csv` with `selected_students_by_ability.csv` by student id
- computes metrics for `low/medium/high` and overall `overall`:
  - `correct_accuracy` (corresponds to `acc+`)
  - `incorrect_accuracy` (corresponds to `acc-`)
  - `balanced_accuracy` (corresponds to `BAA`)

Usage:

```bash
python evaluation/data.py \
  --record-csv agent/code/router/merged_results.csv \
  --ability-csv scripts/data/selected_students_by_ability.csv
```

Arguments:

- `--record-csv`: one or more `merged_results.csv` paths
- `--ability-csv`: ability grouping CSV (optional; defaults to `scripts/data/selected_students_by_ability.csv`)

Notes:

- in `merged_results.csv`, the student id column must be named `id`
- in `selected_students_by_ability.csv`, the student id column must be named `stu_id`

## 2) Overall evaluation without ability grouping (overall.py)

Script: [evaluation/overall.py](evaluation/overall.py)

What it does:

- does not read the ability grouping file
- computes overall `acc+ / acc- / BAA` on all records

Usage:

```bash
python evaluation/overall.py \
  --record-csv agent/code/router/merged_results.csv
```

Print a single line (useful for running multiple files in batch):

```bash
python evaluation/overall.py \
  --record-csv agent/code/router/merged_results.csv \
  --quiet
```
