from __future__ import annotations

"""Unified simulation entrypoint.

This file consolidates the duplicated logic from multiple `main_*.py` entrypoints.
Main goals:

- Centralize the easy-to-change variables (model, result directory, log name, data path, etc.).
- Reuse one simulation pipeline (load data → build Profile/Memory → call AgentAction → write results).
- Support different LLM backends:
    - Fixed preset configuration (e.g., gpt_5_mini / llama_3b)
    - Per-student dynamic routing (router_cosine: student_id -> model/base_url/api_key)

Conventions:
- Result file: `<result_dir>/<student_id>_results.json`
- Failure list: `<result_dir>/failed_students.json`
"""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import re
import sys
from typing import Callable, Iterable, Set

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from agent.code.config import (
        DATA_PATH,
        BASE_URL_GPT_5_MINI,
        MODEL_GPT_5_MINI,
        OPENAI_API_KEY_GPT_5_MINI,
        RESULTS_DIR_GPT_5_MINI,
        BASE_URL_LLAMA_3b,
        MODEL_LLAMA_3b,
        OPENAI_API_KEY_LLAMA_3b,
        RESULT_PATH_LLAMA_3b,
        RESULT_PATH_ROUTER,
    )
    from agent.code.utils import load_json, save_json
except ImportError:
    from config import (
        DATA_PATH,
        BASE_URL_GPT_5_MINI,
        MODEL_GPT_5_MINI,
        OPENAI_API_KEY_GPT_5_MINI,
        RESULTS_DIR_GPT_5_MINI,
        BASE_URL_LLAMA_3b,
        MODEL_LLAMA_3b,
        OPENAI_API_KEY_LLAMA_3b,
        RESULT_PATH_LLAMA_3b,
        RESULT_PATH_ROUTER,
    )
    from utils import load_json, save_json


@dataclass(frozen=True)
class SimConfig:
    """Configuration for a single run.

    If you want to tweak variables later, you usually only need to change this:
    - result_dir: output directory
    - base_url/api_key/model: LLM call parameters
    - log_name: LLM call log filename (used by AgentAction's internal logger)
    - data_path: data directory (defaults to DATA_PATH)

    For dynamic per-student routing like router_cosine, use llm_resolver.
    """

    name: str
    result_dir: str | Path

    base_url: str
    api_key: str
    model: str

    log_name: str
    data_path: str | Path = DATA_PATH

    # Optional: choose (model, base_url, api_key) per student.
    # Typical use: in router_cosine, bind each student to a model configuration.
    llm_resolver: Callable[[int], tuple[str, str, str]] | None = None

    def failed_path(self) -> str:
        return f"{self.result_dir}/failed_students.json"


# PRESETS usage notes:
# - For remote API models, base_url is the API endpoint and api_key is the real key.
# - For locally deployed models, api_key must be "local".
# - For locally deployed models, base_url must be the local model path.
# - Example: llama_3b uses api_key="local" and base_url=BASE_URL_LLAMA_3b.
# - router_cosine follows the same rule for each routed model loaded from CSV.
# - log_name stores LLM call logs and is crucial for debugging/analysis.

PRESETS: dict[str, SimConfig] = {
    # Fixed config: all students share the same LLM configuration.
    "gpt_5_mini": SimConfig(
        name="gpt_5_mini",
        result_dir=RESULTS_DIR_GPT_5_MINI,
        base_url=BASE_URL_GPT_5_MINI,
        api_key=OPENAI_API_KEY_GPT_5_MINI,
        model=MODEL_GPT_5_MINI,
        log_name="gpt5mini.log",
    ),
    "llama_3b": SimConfig(
        name="llama_3b",
        result_dir=RESULT_PATH_LLAMA_3b,
        base_url=BASE_URL_LLAMA_3b,
        api_key=OPENAI_API_KEY_LLAMA_3b,
        model=MODEL_LLAMA_3b,
        log_name="llama3b.log",
    ),

    # Dynamic routing: each student uses a different (model/base_url/api_key).
    # The mapping is loaded from CSVs under DATA_PATH by make_router_llm_resolver().
    "router_cosine": SimConfig(
        name="router_cosine",
        result_dir=RESULT_PATH_ROUTER,
        base_url="",
        api_key="",
        model="",
        log_name="router.log",
    ),
}

PRESET_ALIASES = {
    "router_consine": "router_cosine",
}


def make_router_llm_resolver(data_path: str | Path = DATA_PATH) -> Callable[[int], tuple[str, str, str]]:
    """Build a resolver that maps student_id -> (model, base_url, api_key) using CSVs.

    This matches the behavior in main_router_cosine.py, but caches CSV reads.

    Dependency: requires pandas (the original implementation also uses pandas to read CSVs).
    CSV files read (all under data_path):
    - stu_llm_map.csv / stu_llm_mapping.csv: stu_id -> llm_id
    - model_id_map.csv: id(llm_id) -> model_name
    - model_api.csv: model(model_name) -> base_url/api_key
    """

    # Lazy import so other presets don't require pandas.
    import pandas as pd

    data_path = Path(data_path)
    student_llm_map_csv = data_path / "stu_llm_map.csv"
    if not student_llm_map_csv.exists():
        student_llm_map_csv = data_path / "stu_llm_mapping.csv"
    llm_modelname_csv = data_path / "model_id_map.csv"
    llm_config_csv = data_path / "model_api.csv"

    df_map = pd.read_csv(student_llm_map_csv)
    df_modelname = pd.read_csv(llm_modelname_csv)
    df_cfg = pd.read_csv(llm_config_csv)

    # Build lookups for speed.
    student_to_llm_id = dict(zip(df_map["stu_id"].astype(int), df_map["llm_id"].astype(int)))
    llm_id_to_name = dict(zip(df_modelname["id"].astype(int), df_modelname["model_name"]))

    cfg_by_model = {
        row["model"]: (row["model"], row["base_url"], row["api_key"])
        for _, row in df_cfg.iterrows()
    }

    def resolver(student_id: int) -> tuple[str, str, str]:
        if student_id not in student_to_llm_id:
            raise ValueError(f"student_id {student_id} not found in {student_llm_map_csv}")

        llm_id = int(student_to_llm_id[student_id])
        if llm_id not in llm_id_to_name:
            raise ValueError(f"llm_id {llm_id} not found in {llm_modelname_csv}")

        model_name = llm_id_to_name[llm_id]
        if model_name not in cfg_by_model:
            raise ValueError(f"model_name {model_name} not found in {llm_config_csv}")

        model, base_url, api_key = cfg_by_model[model_name]
        return str(model), str(base_url), str(api_key)

    return resolver


def normalize_preset_name(preset: str) -> str:
    """Normalize preset names and handle common typos."""
    return PRESET_ALIASES.get(preset, preset)


def get_simulated_student_ids(result_dir: str | Path) -> Set[int]:
    """Return student_ids that already have results under result_dir.

    Used for resume: if `<id>_results.json` exists, skip that student.
    """

    result_dir = Path(result_dir)
    if not result_dir.exists():
        return set()

    pattern = re.compile(r"^(\d+)_results\.json$")
    simulated_ids: Set[int] = set()

    for file in result_dir.iterdir():
        if not file.is_file():
            continue

        match = pattern.match(file.name)
        if match:
            simulated_ids.add(int(match.group(1)))

    return simulated_ids


def _find_student_logs(all_logs: list[dict], student_id: int) -> dict:
    """Find one student's log block in stu_logs.json by user_id."""
    for item in all_logs:
        if item.get("user_id") == student_id:
            return item
    raise ValueError(f"No log found for student_id={student_id}")


def run_for_student(student_id: int, *, cfg: SimConfig, failed_students: list[int]) -> None:
    """Run the full simulation for one student and persist results.

    Main steps:
    1) Load that student's logs
    2) Initialize Profile/Memory
    3) Create AgentAction and loop simulate_step
    4) Save <student_id>_results.json
    """
    try:
        print(f"Start student: {student_id}")

        try:
            from agent.code.action import AgentAction
            from agent.code.memory import Memory
            from agent.code.stu_profile import Profile
        except ImportError:
            from action import AgentAction
            from memory import Memory
            from stu_profile import Profile

        all_logs = load_json(f"{cfg.data_path}/stu_logs.json")
        student_data = _find_student_logs(all_logs, student_id)

        logs = student_data["logs"]

        kcg = load_json(f"{cfg.data_path}/kcg.json")
        know_name = load_json(f"{cfg.data_path}/know_name_list.json")

        profile = Profile(student_id)
        memory = Memory(kcg, know_name)

        # By default, use the fixed LLM parameters from cfg.
        # If llm_resolver is set (router), override them per student_id.
        base_url = cfg.base_url
        api_key = cfg.api_key
        model = cfg.model
        if cfg.llm_resolver is not None:
            model, base_url, api_key = cfg.llm_resolver(student_id)

        action = AgentAction(
            profile,
            memory,
            base_url=base_url,
            api_key=api_key,
            model=model,
            log_name=cfg.log_name,
        )

        results: list[dict] = []

        for rec_id, rec in enumerate(logs, start=1):
            ans, raw, corr, summ = action.simulate_step(
                rec,
                rec_id,
                similarity_fn=memory.reinforce,
            )
            results.append({"ans": ans, "raw": raw, "corr": corr, "summ": summ})

        save_json(f"{cfg.result_dir}/{student_id}_results.json", results)
        print(f"Finished student: {student_id}")

        # action.py may implement close() (e.g., to free GPU memory for local LLMs).
        # Call it if present.
        if hasattr(action, "close"):
            action.close()

    except Exception as e:
        print(f"[ERROR] Student {student_id}: {e}")
        failed_students.append(student_id)


def run_simulation(cfg: SimConfig, *, student_ids: Iterable[int] | None = None) -> None:
    """Run simulation in batch (by default, all students in agent_id_list.json)."""
    failed_students: list[int] = []

    if cfg.name == "router_cosine" and cfg.llm_resolver is None:
        cfg = replace(cfg, llm_resolver=make_router_llm_resolver(cfg.data_path))

    if student_ids is None:
        agent_id_list = load_json(f"{cfg.data_path}/agent_id_list.json")
        student_ids = agent_id_list

    # Normalize to list once so we can compute lengths and iterate safely.
    if not isinstance(student_ids, list):
        student_ids = list(student_ids)

    simulated_ids = get_simulated_student_ids(cfg.result_dir)
    students_to_run = [s for s in student_ids if s not in simulated_ids]

    print(f"Run preset: {cfg.name}")
    print(f"Total students: {len(student_ids)}")
    print(f"Already simulated: {len(simulated_ids)}")
    print(f"To be simulated: {len(students_to_run)}")

    for sid in students_to_run:
        run_for_student(sid, cfg=cfg, failed_students=failed_students)

    if failed_students:
        print("Failed students:", failed_students)
        save_json(cfg.failed_path(), failed_students)
    else:
        print("All students processed successfully.")


def main(preset: str = "gpt_5_mini") -> None:
    """Run simulation by preset.

    Available presets are keys of PRESETS, e.g.:
    - "gpt_5_mini"
    - "llama_3b"
    - "router_cosine"
    """
    preset = normalize_preset_name(preset)
    cfg = PRESETS.get(preset)
    if cfg is None:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS)}")

    run_simulation(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the student simulation pipeline with a selected preset."
    )
    parser.add_argument(
        "--preset",
        default="gpt_5_mini",
        help=(
            "Preset to run. Available: "
            f"{', '.join(PRESETS)}. "
            "Common typo 'router_consine' is also accepted."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    # If you want to run this single-file entrypoint directly, pass --preset or hardcode it.
    # Example: main("router_cosine")
    args = parse_args()
    main(args.preset)
