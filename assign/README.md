# assign

This directory is used to **compute similarity based on student/LLM vector representations and generate matching results**.

Currently includes:

- `cal_sim.py`: splits student and LLM vectors from a `.npy` matrix, computes a cosine similarity matrix, and exports the Top-1 most similar LLM for each student.

## cal_sim.py

### 输入

- `npy_path`: a 2D `numpy` array (`.npy` file) with shape `[N, D]`
  - the first `num_students` rows: student vectors
  - the remaining `N - num_students` rows: LLM vectors

The script applies a `sigmoid` normalization to the entire matrix before computing cosine similarity.

### 输出

- `output_csv`: Top-1 LLM matching result per student, with fixed columns:
  - `stu_id`: student row index (starting from 0)
  - `llm_id`: global LLM id (`top1_llm_idx + llm_id_offset`)
  - `cos_value`: the cosine similarity value

If you also want to save the full cosine similarity matrix (shape `[num_students, num_llm]`), pass `output_npy`.

### 运行方式

Run from the repository root:

```bash
python assign/cal_sim.py
```

Notes: `cal_sim.py` infers the project root directory `PROJECT_ROOT` from the script location.

- When running the script directly, the entrypoint uses `PROJECT_ROOT / ...` to build default paths
- When calling `main()` in code, relative paths are also resolved as `PROJECT_ROOT / <relative path>`

Default read/write paths when running the script directly (relative to project root):

- Input: `cdm/stu_embedding/student_emb.npy`
- Output: `agent/data/stu_llm_mapping.csv`

Or call it from code:

```python
from pathlib import Path
from assign.cal_sim import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]

main(
    # 通常使用 cdm 训练得到的 embedding
  npy_path=str(PROJECT_ROOT / "cdm" / "stu_embedding" / "student_emb.npy"),
    num_students=1264,
    # 输出路径可自定义（会自动创建父目录）
  output_csv=str(PROJECT_ROOT / "agent" / "data" / "stu_llm_mapping.csv"),
    llm_id_offset=1264,
    # 默认不保存完整相似度矩阵；如需保存可传入路径，例如：
  # output_npy=str(PROJECT_ROOT / "cdm" / "stu_embedding" / "llm_student_cos_sim.npy"),
    output_npy=None,
)
```

### 参数说明

- `num_students`: number of students (used to split the input matrix)
- `llm_id_offset`: offset for LLM ids; typically equals `num_students` (so global LLM ids start from `num_students`)

### 依赖

- Python 3
- `numpy`
- `pandas`
