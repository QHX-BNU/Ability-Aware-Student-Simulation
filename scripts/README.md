# scripts

This directory contains runnable helper scripts.

## Directory structure

```text
scripts/
  README.md
  cal_stu_ablility.py
  extra_ans.py
  llm_client.py
```

## cal_stu_ability (script: `cal_stu_ablility.py`)

This script does two things:

1. Compute each student's ability value from `student_emb.npy`
2. Split students into `low/medium/high` by ability tertiles and sample a fixed number per level, then export the selected student ids

(Updated behavior: student ids are generated directly from the embedding row index; the script no longer depends on a test-set id list file.)

> Note: The script infers the project root `PROJECT_ROOT` from its own location. If you pass a relative path, it is resolved as `PROJECT_ROOT/<relative path>`, so the script does not depend on the current working directory.

### Input

- student embedding: `cdm/stu_embedding/student_emb.npy`
  - shape: `[N, D]`
  - by default, only the first `--num-students` rows are used as student vectors

Student id rule: by default `stu_id = row_index + 1` (so the first row maps to `1`).

If you set `--id-base` to `0`, then `stu_id = row_index + 0`.

### Outputs (default)

- `cdm/stu_embedding/student_ability.csv`
  - columns: `id, ability`
  - ability definition: apply `sigmoid` to each row vector, then take the mean
- `cdm/stu_embedding/selected_students_by_ability.csv`
  - columns: `stu_id, ablility_level`
  - `ablility_level` is `low | medium | high` (spelling is kept for compatibility with the existing evaluation scripts)
- `agent/data/agent_id_list.json`
  - JSON array: sorted `stu_id` values extracted from `selected_students_by_ability.csv`

### Selection rules

- split into `low/medium/high` by ability tertiles
- sample `--total-per-level` per level (default 100, total 300)
- try to force-include students with `stu_id` in `[--mandatory-id-min, --mandatory-id-max]`
  - if the mandatory count in a level already exceeds that level's quota, the script raises an error and asks you to increase the quota or shrink the mandatory range

Note: since the script no longer filters by a test set, the meaning of “mandatory” becomes: force-include ids in that range among all students (if the ids exist).

### Usage

Run from the project root (recommended):

```bash
python scripts/cal_stu_ablility.py \
  --emb-npy cdm/stu_embedding/student_emb.npy \
  --num-students 1299
```

Common optional arguments:

- `--ability-csv`: output path for `student_ability.csv`
- `--selected-csv`: output path for `selected_students_by_ability.csv`
- `--selected-json`: output path for the “selected student ids” JSON (default `agent/data/agent_id_list.json`)
- `--total-per-level`: number selected per ability level (default 100)
- `--mandatory-id-min/--mandatory-id-max`: mandatory id range (default 1–101, inclusive)
- `--id-base`: id base used in the ability CSV (default 1)

See all arguments:

```bash
python scripts/cal_stu_ablility.py --help
```

## extra_ans (script: `extra_ans.py`)

This script converts the model's raw text outputs into a structured CSV for downstream evaluation/analysis.

It will:

- read `stu_logs.json` (question text, gold answer, student score, etc.)
- iterate `*_results.json` under a result directory (one file per student)
- for each question, take the `raw` field from the result JSON (usually containing Task3 / reasoning text), and call an LLM to extract the final selected option letter (A/B/C/D/E)
- append the structured records into a CSV (already-processed students are skipped to avoid duplicate appends)

### Configuration you must edit (script constants)

This script currently specifies input/output locations via constants inside the script. You must edit them for your environment.

> Note: The script infers the project root `PROJECT_ROOT` from its own location. If you pass a relative path, it is resolved as `PROJECT_ROOT/<relative path>`, so the script does not depend on the current working directory.

- `RESULT_DIR_OUTPUT_MAP`: mapping from results directory → output CSV
- `STU_LOGS_PATH`: path to `stu_logs.json`

In `RESULT_DIR_OUTPUT_MAP`, the key is the simulation output directory generated when running student simulation under `agent/code` (it contains many `*_results.json`). You need to change it to match your actual output location.

Also, you must configure `base_url / api_key / model` for `local_llm` (the current file contains placeholders).

### Output CSV columns

Each row in the output CSV corresponds to one question attempt, with columns:

- `id`: student id (parsed from the `*_results.json` filename prefix)
- `exer_id`: exercise id
- `llm_answer`: extracted option letter (A–E; empty string if unclear)
- `correct_answer`: gold answer
- `llm_panduan`: `ans.task4` in the result JSON (the script currently writes this field directly)
- `stu_ans`: student field (the script currently writes `score`)

### Dependencies

- `openai` (API calls)
- `tqdm` (progress bar)
