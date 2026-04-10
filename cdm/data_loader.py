import pandas as pd
import torch  # type: ignore
from pathlib import Path

# Dataset metadata (adjust as needed)
student_n = 1299 # 1299 students in total
question_n = 212 # 212 questions in total
knowledge_n = 93 # 93 knowledge components in total

# Project root: .../Ability-Aware-Student-Simulation
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Training data path (relative to project root)
train_data_path = PROJECT_ROOT / "cdm" / "train_data" / "train_stu_llm.csv"

class TrainDataLoader(object):
    """
    Data loader for training (read from CSV)
    """
    def __init__(self):
        self.batch_size = 32
        self.ptr = 0

        data_file = train_data_path

        # Read CSV
        df = pd.read_csv(str(data_file))

        # Convert into the required format
        self.data = []
        for _, row in df.iterrows():
            kc_str = str(row["knowledgecomponent_id"])
            if kc_str.strip() == "":
                kc_list = []
            else:
                kc_list = list(map(int, kc_str.split(",")))

            self.data.append({
                "user_id": int(row["student_id"]),
                "exer_id": int(row["question_id"]),
                "knowledge_code": kc_list,
                "score": int(row["answers"])
            })

        self.knowledge_dim = knowledge_n

    def next_batch(self):
        if self.is_end():
            return None, None, None, None

        input_stu_ids, input_exer_ids = [], []
        input_knowedge_embs, ys = [], []

        for count in range(self.batch_size):
            log = self.data[self.ptr + count]

            # Build one-hot for knowledge components
            knowledge_emb = [0.0] * self.knowledge_dim
            for kc in log["knowledge_code"]:
                knowledge_emb[kc - 1] = 1.0

            input_stu_ids.append(log["user_id"] - 1)
            input_exer_ids.append(log["exer_id"] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(log["score"])

        self.ptr += self.batch_size

        return (
            torch.LongTensor(input_stu_ids),
            torch.LongTensor(input_exer_ids),
            torch.Tensor(input_knowedge_embs),
            torch.LongTensor(ys)
        )

    def is_end(self):
        return self.ptr + self.batch_size > len(self.data)

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    """
    Data loader for validation / test (CSV)
    Group by user: each user_id is one batch
    """
    def __init__(self, d_type='validation'):
        self.ptr = 0
        self.data = []

        data_file = PROJECT_ROOT / "data" / ("val.csv" if d_type == "validation" else "test.csv")

        df = pd.read_csv(str(data_file))

        self.knowledge_dim = knowledge_n

        # Group by student_id (align with the original JSON format)
        grouped = df.groupby("student_id")

        for stu_id, group in grouped:
            logs = []
            for _, row in group.iterrows():
                kc_str = str(row["knowledgecomponent_id"])
                if kc_str.strip() == "":
                    kc_list = []
                else:
                    kc_list = list(map(int, kc_str.split(",")))

                logs.append({
                    "exer_id": int(row["question_id"]),
                    "knowledge_code": kc_list,
                    "score": int(row["answers"])
                })

            self.data.append({
                "user_id": int(stu_id),
                "logs": logs
            })

    def next_batch(self):
        if self.is_end():
            return None, None, None, None

        batch = self.data[self.ptr]
        user_id = batch["user_id"]
        logs = batch["logs"]

        input_stu_ids, input_exer_ids = [], []
        input_knowedge_embs, ys = [], []

        for log in logs:
            knowledge_emb = [0.0] * self.knowledge_dim
            for kc in log["knowledge_code"]:
                knowledge_emb[kc - 1] = 1.0

            input_stu_ids.append(user_id - 1)
            input_exer_ids.append(log["exer_id"] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(log["score"])

        self.ptr += 1

        return (
            torch.LongTensor(input_stu_ids),
            torch.LongTensor(input_exer_ids),
            torch.Tensor(input_knowedge_embs),
            torch.LongTensor(ys)
        )

    def is_end(self):
        return self.ptr >= len(self.data)

    def reset(self):
        self.ptr = 0
