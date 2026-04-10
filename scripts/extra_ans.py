from llm_client import LLMClient
from tqdm import tqdm
import os
from pathlib import Path
import json
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_from_project_root(path_like: str | os.PathLike) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

# ================== LLM ==================
local_llm = LLMClient(
    base_url="https://openai.com/v1/chat/completions",
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    model="gpt-5-mini",
)

def extract_llm_answer_by_llm(task3_text: str,question: str) -> str:
    if not task3_text or not task3_text.strip():
        return ""

    prompt = f"""
You are given a student's answer explanation for a multiple-choice question.

Your task:
- Determine which option (A, B, C, D, or E) the student finally selected based ONLY on the content of Task 3.
- Output ONLY a single uppercase letter: A, B, C, D, or E.
- Do NOT output any explanation, punctuation, or additional text.
- If the selected option cannot be clearly determined, output an empty string.
Question:
{question}

Student answer:
\"\"\"
{task3_text}
\"\"\"
""".strip()

    messages = [
        {"role": "system", "content": "You extract structured answers from text."},
        {"role": "user", "content": prompt}
    ]

    ans = local_llm.call(messages)
    return ans 

def get_llm_panduan(llm_answer: str, question: str) -> str:
    if not llm_answer or not question:
        return "none"

    prompt = f"""
    You are given a multiple-choice question and the answer selected by an LLM.
    
    Your task:
    - Extract the LLM's response from Task 4.
    - The response will be either "Yes" or "No".
    - Output EXACTLY "Yes" or "No" as given.
    - If the response is missing or cannot be determined, output "none".

    Question:
    {question}
    LLM's answer:
    {llm_answer}
    """
    messages = [
        {"role": "system", "content": "You extract structured answers from text."},
        {"role": "user", "content": prompt}
    ]
    ans = local_llm.call(messages)
    return ans.strip()

# ================== Directory ↔ Output file ==================
# Notes:
# - key (results_dir) is the results directory produced after running the student simulation under agent/code
#   (the directory contains many `*_results.json` files).
# - value (output_csv) is the CSV file to which the results under that directory are aggregated/appended.
# - If a relative path is used, it will be resolved as `PROJECT_ROOT/<relative path>`, independent of the cwd.
# - Modify this mapping to match your actual output directories.
RESULT_DIR_OUTPUT_MAP = {
    "agent/code/router": 
        "agent/code/router/merged_results.csv",
}

STU_LOGS_PATH = "agent/data/stu_logs.json"


def load_existing_student_ids(csv_path):
    """
    Read student_ids that have already been processed from an existing results CSV.
    """
    if not os.path.exists(csv_path):
        return set()

    existing_ids = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "id" not in reader.fieldnames:
            return set()

        for row in reader:
            try:
                existing_ids.add(int(row["id"]))
            except Exception:
                continue

    return existing_ids


def main():
    # ---------- 1. Read stu_logs ----------
    stu_logs_path = resolve_from_project_root(STU_LOGS_PATH)
    with open(stu_logs_path, "r", encoding="utf-8") as f:
        stu_logs = json.load(f)

    stu_map = {
        stu["user_id"]: stu["logs"]
        for stu in stu_logs
    }

    # ================== 2. Process by directory ==================
    for results_dir, output_csv in RESULT_DIR_OUTPUT_MAP.items():
        results_dir_path = resolve_from_project_root(results_dir)
        output_csv_path = resolve_from_project_root(output_csv)

        if not results_dir_path.is_dir():
            print(f"[WARN] 目录不存在，已跳过：{results_dir}")
            continue

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # ---------- 2.1 Read existing student_id ----------
        existing_student_ids = load_existing_student_ids(str(output_csv_path))
        print(f"[INFO] 已存在学生数：{len(existing_student_ids)}")

        rows = []

        result_files = [
            f for f in os.listdir(results_dir_path)
            if f.endswith("_results.json")
        ]

        # ---------- Student level ----------
        for file_name in tqdm(result_files, desc=f"Processing {os.path.basename(results_dir)}"):
            try:
                user_id = int(file_name.split("_")[0])
            except ValueError:
                continue

            # ⭐ Core logic: skip if already exists
            if user_id in existing_student_ids:
                continue

            if user_id not in stu_map:
                continue

            res_path = results_dir_path / file_name
            with open(res_path, "r", encoding="utf-8") as f:
                results = json.load(f)

            logs = stu_map[user_id]
            n = min(len(results), len(logs))

            # ---------- Exercise level ----------
            for i in tqdm(range(n), desc=f"User {user_id}", leave=False):
                res_item = results[i]
                log_item = logs[i]
                question_text = log_item.get("exer_content", "")

                llm_answer = res_item.get("raw", "").strip()
                llm_selected_option = extract_llm_answer_by_llm(
                    task3_text=llm_answer,
                    question=question_text
                )
                llm_panduan = res_item.get("ans", {}).get("task4", "")
                print(f"User {user_id}, Exer {log_item.get('exer_id', '')}, Selected Option: {llm_selected_option}, Panduan: {llm_panduan}")

                rows.append({
                    "id": user_id,
                    "exer_id": log_item.get("exer_id", ""),
                    "llm_answer": llm_selected_option,
                    "correct_answer": log_item.get("exer_answer", ""),
                    "llm_panduan": llm_panduan,
                    "stu_ans": log_item.get("score", "")
                })

        # ---------- 3. Append to CSV ----------
        if rows:
            file_exists = output_csv_path.exists()

            with open(output_csv_path,
                      "a" if file_exists else "w",
                      newline="",
                      encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "id",
                        "exer_id",
                        "llm_answer",
                        "correct_answer",
                        "llm_panduan",
                        "stu_ans"
                    ]
                )

                if not file_exists:
                    writer.writeheader()

                writer.writerows(rows)

        print(
            f"[OK] 本次新增学生：{len(set(r['id'] for r in rows))}，"
            f"新增记录：{len(rows)}"
        )


if __name__ == "__main__":
    main()
