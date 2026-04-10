import numpy as np
import pandas as pd
from pathlib import Path


# Project root: .../Ability-Aware-Student-Simulation
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_from_project_root(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    # Prevent exp overflow
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity matrix between two sets of vectors.

    Args:
        x: shape [N, D]
        y: shape [M, D]
    Returns:
        shape [N, M], where (i, j) = cos(x_i, y_j)
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be 2D arrays, got x.ndim={x.ndim}, y.ndim={y.ndim}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must have same embedding dim, got {x.shape} vs {y.shape}")

    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)

    x_norm = np.clip(x_norm, a_min=1e-12, a_max=None)
    y_norm = np.clip(y_norm, a_min=1e-12, a_max=None)

    return (x @ y.T) / (x_norm * y_norm.T)


def main(
    npy_path: str,
    num_students: int = 1264,
    output_csv: str = "stu_top1_llm_cosine.csv",
    llm_id_offset: int = 1264,
    output_npy: str | None = None,
):
    npy_path = resolve_from_project_root(Path(npy_path))
    output_csv_path = resolve_from_project_root(Path(output_csv))
    output_npy_path = resolve_from_project_root(Path(output_npy)) if output_npy else None

    # 1. Load the ability matrix
    ability = np.load(str(npy_path))  # shape: [N, D]

    # 2. Sigmoid normalization (required)
    ability = sigmoid(ability)

    # 3. Split students & LLMs
    stu_ability = ability[:num_students]
    llm_ability = ability[num_students:]

    num_llm = llm_ability.shape[0]

    # 4. Compute cosine similarity matrix (vector-level)
    sim_matrix = cosine_similarity_matrix(stu_ability.astype(np.float32), llm_ability.astype(np.float32)).astype(
        np.float32
    )

    # 5. (Optional) Save similarity matrix as npy
    if output_npy_path:
        if output_npy_path.parent != Path('.'):
            output_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_npy_path), sim_matrix)
        print(f"Cosine similarity matrix saved to {output_npy_path}")
        print(f"Shape: {sim_matrix.shape}")

    top1_llm_idx = np.argmax(sim_matrix, axis=1)
    top1_sim = np.max(sim_matrix, axis=1)
    top1_llm_global_idx = top1_llm_idx + llm_id_offset

    df = pd.DataFrame({
        "stu_id": np.arange(num_students),
        "llm_id": top1_llm_global_idx,
        "cos_value": top1_sim,
    })
    if output_csv_path.parent != Path('.'):
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_csv_path), index=False)

    print(f"Top-1 student-LLM pairs saved to {output_csv_path}")


if __name__ == "__main__":
    main(
        npy_path=str(PROJECT_ROOT / "cdm" / "stu_embedding" / "student_emb.npy"),
        num_students=1264,
        output_csv=str(PROJECT_ROOT / "agent" / "data" / "stu_llm_mapping.csv"),
        llm_id_offset=1264,
    )
