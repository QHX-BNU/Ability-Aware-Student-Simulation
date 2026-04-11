# Agent Running Guide

This document describes how to prepare data, configure, and run the `agent` module. The default data directory is `agent/data`. It is recommended to run commands from the repository root.

## 1. Directory Structure

```text
agent/
├─ code/
│  ├─ action.py
│  ├─ config.py
│  ├─ llm_client.py
│  ├─ local_llm.py
│  ├─ main_simulation.py
│  ├─ memory.py
│  ├─ stu_profile.py
│  └─ utils.py
├─ data/
│  ├─ agent_id_list.json
│  ├─ exer_list.json
│  ├─ kcg.json
│  ├─ know_list.json
│  ├─ know_name_list.json
│  ├─ model_api.csv
│  ├─ model_id_map.csv
│  ├─ profile.json
│  ├─ stu_llm_mapping.csv
│  ├─ stu_logs.json
└─ README.md
```

The roles of the files are as follows.

### 1.1 Files under `code/`

- `main_simulation.py`: main entry script; reads the student list, loads data, selects a preset, and runs the simulation student by student
- `config.py`: central configuration for data dirs, result dirs, model name, `base_url`, `api_key`, and memory-module parameters
- `action.py`: agent action module; builds prompts, calls the LLM, parses Task1–Task4 outputs, and writes reflection results
- `memory.py`: memory module; implements short-term memory, long-term memory, reinforcement, forgetting, and reflection
- `stu_profile.py`: student profile module; reads statistics from `profile.json` and generates the system prompt
- `llm_client.py`: remote model client; calls OpenAI-compatible APIs with retries
- `local_llm.py`: local model client; loads a local HuggingFace model and runs inference
- `utils.py`: common utilities; currently mainly JSON load/save

### 1.2 Files under `data/`

- `agent_id_list.json`: list of student IDs to simulate; also the default entry for batch runs
- `stu_logs.json`: historical exercise logs for each student; core input for simulation
- `profile.json`: student profile features (e.g., activity, diversity, preference, success rate, ability)
- `know_name_list.json`: mapping from knowledge name to knowledge id; used for knowledge alignment in the memory module
- `know_list.json`: knowledge list / base set; useful for data checks or indexing
- `kcg.json`: knowledge concept graph; used to determine knowledge relations between the current exercise and existing memories
- `exer_list.json`: exercise list / basic exercise metadata; helper data for the question bank
- `stu_llm_mapping.csv`: routing table from student to candidate model; read by `router_cosine`
- `model_id_map.csv`: mapping from model id to model name; used to resolve routing results into concrete model configs
- `model_api.csv`: candidate model API configuration; records `base_url` and `api_key` per model

### 1.3 Other files

- `README.md`: this guide

## 2. Environment

Python 3.10+ is recommended.

Install at least the following dependencies:

```bash
pip install openai pandas torch transformers
```

Notes:

- `openai`: remote API calls
- `pandas`: reading CSV routing files for `router_cosine`
- `torch`, `transformers`: local `llama_3b` preset

## 3. Configuration

Edit `agent/code/config.py` and ensure each preset contains at least the following fields:

```python
BASE_URL_XXX = "..."
OPENAI_API_KEY_XXX = "..."
MODEL_XXX = "..."
```

Important configuration rules:

- Remote models: set `base_url` to your remote endpoint and `api_key` to a real key
- Local models: set `base_url` to the local model path and `api_key` must be exactly `"local"`
- Router mode: each row in `agent/data/model_api.csv` must follow the same rule

In router mode:

- If a candidate model is remote, fill in a remote `base_url` and a real `api_key`
- If a candidate model is local, `base_url` must be a local path and `api_key` must be `"local"`

Do not leave the local model `api_key` empty and do not use other strings, otherwise the code will not take the local-model branch.

## 3.1 Configure models in `main_simulation.py`

The runtime entry is the `PRESETS` dict in `agent/code/main_simulation.py`.

For easier navigation, the most relevant code locations are:

- `SimConfig` definition: `agent/code/main_simulation.py` lines 60–88
- Fixed-model presets (`PRESETS`): `agent/code/main_simulation.py` lines 98–127
- CSV router logic for `router_cosine`: `agent/code/main_simulation.py` lines 134–184
- Runtime selection between fixed vs router: `agent/code/main_simulation.py` lines 256–270
- `router_cosine` auto-mounts `llm_resolver`: `agent/code/main_simulation.py` lines 296–301
- CLI `--preset` argument: `agent/code/main_simulation.py` lines 345–365

Each preset corresponds to a `SimConfig` with the following core fields:

```python
SimConfig(
    name="your_preset_name",
    result_dir="output directory",
    base_url="remote endpoint or local model path",
    api_key="remote key or local",
    model="model name",
    log_name="log file name",
)
```

### 1. Fixed single-model preset

If you want all students to use the same model, add or modify a fixed preset in `PRESETS`.

Remote model example:

```python
"your_remote_model": SimConfig(
    name="your_remote_model",
    result_dir="your_result_dir",
    base_url="https://your-api-endpoint/v1",
    api_key="sk-xxxxxx",
    model="your_remote_model_name",
    log_name="your_remote_model.log",
),
```

Local model example:

```python
"your_local_model": SimConfig(
    name="your_local_model",
    result_dir="your_result_dir",
    base_url="/path/to/your/local/model",
    api_key="local",
    model="your_local_model_name",
    log_name="your_local_model.log",
),
```

Notes:

- For remote models, `base_url` is the API endpoint and `api_key` is a real key
- For local models, `base_url` is the local model directory and `api_key` must be `"local"`
- `result_dir` controls where results are saved
- `log_name` controls the log filename; using different log names per preset is recommended

### 2. Router preset (student → model)

If you want different students to be assigned to different models automatically, use `router_cosine` (with `llm_resolver`).

The preset typically looks like:

```python
"router_cosine": SimConfig(
    name="router_cosine",
    result_dir=RESULT_PATH_ROUTER,
    base_url="",
    api_key="",
    model="",
    log_name="router.log",
),
```

Leaving `base_url`, `api_key`, and `model` empty here is expected. The actual model configuration is loaded at runtime by `make_router_llm_resolver()` from these three files:

- `agent/data/stu_llm_mapping.csv`
- `agent/data/model_id_map.csv`
- `agent/data/model_api.csv`

That is, the router-mode configuration steps are:

1. In `model_api.csv`, fill `model`, `base_url`, `api_key` for each candidate model
2. In `model_id_map.csv`, build an `id -> model_name` mapping
3. In `stu_llm_mapping.csv`, build a `student_id -> model_id` mapping
4. Run with `--preset router_cosine` and the code will choose the model by student id

### 3. How to choose a preset

After configuration, run from the repository root:

```bash
python agent/code/main_simulation.py --preset your_preset_name
```

For example:

- If you added `your_remote_model` in `PRESETS`, pass `--preset your_remote_model`
- If you use router mode, pass `--preset router_cosine`

### 4. Recommended places to edit

When adding new models, you typically only need to modify:

- Fixed preset: `PRESETS` in `main_simulation.py`
- Router preset: `agent/data/model_api.csv`, and if needed `model_id_map.csv` and `stu_llm_mapping.csv`

Common navigation targets:

- Fixed remote/local preset: `agent/code/main_simulation.py` lines 100–115
- Router preset: `agent/code/main_simulation.py` lines 119–125
- Router reads these 3 filenames: `agent/code/main_simulation.py` lines 149–154
- Student→model mapping parse: `agent/code/main_simulation.py` lines 161–182

## 4. Data formats

### 4.1 `agent_id_list.json`

List of student IDs to simulate:

```json
[1, 2, 3]
```

### 4.2 `stu_logs.json`

Each student has one record, containing `user_id` and time-ordered `logs`:

```json
[
  {
    "user_id": 1,
    "logs": [
      {
        "exer_id": 197,
        "knowledge_code": [81],
        "score": 1,
        "exer_content": "What operation(s) can a transaction perform ...",
        "exer_answer": "C",
        "know_name": ["Transactions"]
      }
    ]
  }
]
```

Field descriptions:

- `user_id`: student id; must align with `agent_id_list.json` and `profile.json`
- `exer_id`: exercise id
- `knowledge_code`: list of knowledge ids
- `score`: ground-truth correctness (`1` correct, `0` incorrect)
- `exer_content`: exercise text
- `exer_answer`: correct option
- `know_name`: list of knowledge names

### 4.3 `profile.json`

This is a dict keyed by student ID. The value is a tab-separated string. The current code parses this format directly.

```json
{
  "1": "1\t0.4339622641509434\t0.6451612903225806\tfunctional dependencies\t0.9130434782608695\t0.5241104"
}
```

The field order is:

```text
student_id \t activity \t diversity \t preference \t success_rate \t ability
```

### 4.4 `know_name_list.json`

Mapping from knowledge name to knowledge id:

```json
{
  "functional dependencies": 54,
  "transactions": 81
}
```

### 4.5 `kcg.json`

Knowledge concept graph represented as a list of knowledge-id pairs:

```json
[
  [47, 53],
  [53, 47]
]
```

### 4.6 `model_api.csv`

Candidate model call configuration for `router_cosine`:

```csv
model,base_url,api_key
remote_model_name,https://your-api-endpoint/v1,sk-xxxxxx
local_model_name,/path/to/your/local/model,local
```

Notes:

- Remote model row: `base_url` is the remote endpoint and `api_key` is a real key
- Local model row: `base_url` is the local model path and `api_key` must be `local`

### 4.7 `model_id_map.csv`

Mapping from model id to model name:

```csv
id,model_name
1297,remote_model_name
1298,local_model_name
```

### 4.8 `stu_llm_mapping.csv`

Mapping from student to candidate model:

```csv
stu_id,llm_id,cos_value
1,1297,0.9942062497138976
```

Field descriptions:

- `stu_id`: student id
- `llm_id`: model id
- `cos_value`: cosine similarity between the student and the model

Note:

- If you use `router_cosine`, these three CSV files must all be prepared

## 5. Commands

Run from the repository root.

### 5.1 Run with a fixed remote model

```bash
python agent/code/main_simulation.py --preset gpt_5_mini
```

### 5.2 Run with a local Llama model

```bash
python agent/code/main_simulation.py --preset llama_3b
```

### 5.3 Run with ability-aware routing

```bash
python agent/code/main_simulation.py --preset router_cosine
```

### 5.4 View available arguments

```bash
python agent/code/main_simulation.py --help
```

## 6. Outputs

After the run completes, results are generated under:

- `gpt_5_mini`: `agent/code/gpt5mini/`
- `llama_3b`: `agent/code/llama_3b/`
- `router_cosine`: `agent/code/router/`

Output file formats:

- `<student_id>_results.json`: simulation results for one student
- `failed_students.json`: list of student ids that failed
- `agent/code/logs/`: LLM call logs

Example output for one student:

```json
[
  {
    "ans": {
      "task1": "Yes",
      "task2": "Transactions",
      "task3": "...",
      "task4": "Yes"
    },
    "raw": "...",
    "corr": "...",
    "summ": "..."
  }
]
```

## 7. FAQ

### 7.1 Cannot find student logs

Check:

- Whether the ids in `agent_id_list.json` exist in `stu_logs.json`
- Whether `profile.json` also contains the same student ids

### 7.2 `router_cosine` cannot run

Check:

- Whether `pandas` is installed
- Whether `stu_llm_mapping.csv`, `model_id_map.csv`, and `model_api.csv` all exist
- Whether `stu_id`, `llm_id`, and `model_name` can be matched correctly

### 7.3 Local model cannot run

Check:

- Whether `BASE_URL_LLAMA_3b` is set to a real local model directory
- Whether `torch` and `transformers` are installed
- Whether the machine has an available GPU

## 8. Recommended order

1. First, run a single preset end-to-end with `gpt_5_mini`
2. Then switch to `router_cosine` for multi-model assignment experiments
3. Finally, plug the result directory into the evaluation scripts
