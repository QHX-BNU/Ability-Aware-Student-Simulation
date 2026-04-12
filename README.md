
# One LLM Does Not Simulate All Students: Ability-Aware Student Simulation via Cognitive Diagnosis Guided LLM Assignment

<p align="center">
  <b>Accepted to ACL 2026</b>
</p>

<p align="center">
  <img src="assets/framwork.png" alt="Method Overview" width="90%">
</p>

<p align="center">
  <em>Overview of our proposed framework.</em>
</p>

---

## Abstract

Large Language Models (LLMs) have become integral to personalized education systems, particularly in the realm of student behavior simulation. By predicting fine-grained learning behaviors, these simulations enable intelligent systems to provide tailored instructional support. However, most existing methods rely on a single high-capacity LLM to represent an entire population of diverse learners. In this work, we demonstrate that this ``one-size-fits-all'' approach induces a systematic **ability-dependent bias**, where high-capacity models tend to overestimate low-ability students while lower-capacity models underestimate high-ability ones. To mitigate this distortion, we propose an _ability-aware student simulation framework_ that dynamically matches students with appropriate LLM backbones through cognitive alignment. We leverage Neural Cognitive Diagnosis (NeuralCD) to extract multidimensional cognitive profiles for both human students and LLM agents within a shared skill space, subsequently pairing each student with the most cognitively representative model. Extensive experiments demonstrate that our approach substantially reduces simulation bias and consistently outperforms single-model baselines across the entire proficiency spectrum. Our findings suggest that faithful behavior simulation necessitates the **alignment of model capacity with student ability**, establishing cognitive diagnosis as a principled mechanism for model assignment in educational AI. 

---

## Highlights

- Official implementation of our **ACL 2026** paper
- A unified framework for **cognitive diagnosis** and **LLM-based student simulation**
- An integrated codebase for **training, simulation, and evaluation**
- Built upon prior open-source educational modeling frameworks

---

## Framework Overview

Our framework consists of three major stages:

### Stage 1: Response Collection from Students and Candidate LLMs

We first collect historical response records from real students. In addition, we define a candidate set of LLMs as potential simulation backbones. Since LLMs do not have prior interaction histories, each candidate LLM is prompted to answer all exercises in the dataset. We use a chain-of-thought (CoT) prompting strategy to elicit structured reasoning and record the final answer correctness. As a result, we construct a training dataset that contains both real student responses and LLM-generated responses on the same set of exercises.

### Stage 2: Cognitive Profiling via NeuralCD

Given the combined response records from students and candidate LLMs, we train a NeuralCD model to infer their latent cognitive profiles. By treating LLMs as pseudo-students during training, NeuralCD provides a unified and interpretable representation of both human learners and LLMs under the same diagnostic framework. This stage produces cognitive proficiency representations for students and candidate LLMs, which are then used for simulation backbone selection.

### Stage 3: Similarity-Based LLM Assignment

For each student, we compare the student’s cognitive proficiency vector with those of all candidate LLMs and select the most similar one as the simulation backbone. In this work, similarity is instantiated using cosine similarity. This assignment strategy ensures that each student is simulated by an LLM whose cognitive proficiency profile is most compatible with the student’s own abilities.

The overall pipeline is illustrated in the figure above.

---

## Codebase

This repository is an **integrated implementation** for the experiments in our paper, rather than a standalone framework developed entirely from scratch.

In particular:

- The **cognitive diagnosis** part is implemented based on [NeuralCD](https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD).
- The **agent-based student simulation** part is implemented based on [Agent4Edu](https://github.com/bigdata-ustc/Agent4Edu).

Our main contribution lies in **integrating these components into a unified experimental pipeline** and adapting them to the experimental settings studied in this paper.

---

## Repository Structure

```bash
.
├── agent/                 # Code for LLM-based student simulation
├── assets/                # Figures and helper assets
├── assign/                # Student↔LLM similarity & assignment
├── cdm/                   # Cognitive diagnosis models and training scripts
├── evaluation/            # Evaluation scripts
├── scripts/               # Runnable helper scripts (grouping, extraction, etc.)
└── README.md
````

---

## Installation

We recommend using **Python 3.10+**.

Clone the repository:

```bash
git clone https://github.com/QHX-BNU/Ability-Aware-Student-Simulation.git
cd Ability-Aware-Student-Simulation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If your experiments require LLM APIs or local model services, please also configure the corresponding environment variables or service credentials.

---

## Data Preparation

Please prepare a raw interaction dataset that contains (at minimum) the following fields:
* student ID
* exercise/question ID
* exercise/question text
* exercise/question knowledge concept(s)
* ground-truth (correct) answer
* whether the student's answer is correct (e.g., 0/1)

After preparing the raw dataset, you need to preprocess/convert it so that it matches the input formats required by this codebase. The required data formats for each stage (CDM training, simulation, evaluation) are introduced in the documentation below.

If the raw data cannot be released due to privacy or licensing restrictions, we recommend providing:

* preprocessing scripts,
* data format descriptions,
* example input files.


---

## Running Experiments

This repo follows an end-to-end pipeline:

### 1) Cognitive Diagnosis (CDM)

You first need to prepare the CDM training data by converting your dataset into the required CSV format.
See: `cdm/README.md`.

Then train NeuralCD to obtain student embeddings:

```bash
python cdm/train.py
```

### 2) Ability Grouping (students)

Compute each student's ability from embeddings and select students per ability level.
Script: `scripts/cal_stu_ablility.py`.
Configuration and outputs are documented in: `scripts/README.md`.

```bash
python scripts/cal_stu_ablility.py \
  --emb-npy cdm/stu_embedding/student_emb.npy
```

### 3) LLM Assignment (student → LLM)

Assign an LLM to each student based on cosine similarity between embeddings.
Script: `assign/cal_sim.py`.
Configuration is documented in: `assign/README.md`.

```bash
python assign/cal_sim.py
```

### 4) Run Simulation

Run the agent-based student simulation.
Entry script: `agent/code/main_simulation.py`.
Configuration is documented in: `agent/README.md`.

```bash
python agent/code/main_simulation.py --preset router_cosine
```

### 5) Answer Extraction (merge_results.csv)

Extract structured answers from simulation outputs and append them into a `merged_results.csv`.
Script: `scripts/extra_ans.py` (you need to update `RESULT_DIR_OUTPUT_MAP` and `STU_LOGS_PATH` inside the script).
See also: `scripts/README.md`.

```bash
python scripts/extra_ans.py
```

### 6) Evaluation

- Ability-grouped evaluation: `evaluation/data.py`
- Overall evaluation (no ability groups): `evaluation/overall.py`
See: `evaluation/README.md`.

```bash
python evaluation/data.py --record-csv agent/code/router/merged_results.csv
python evaluation/overall.py --record-csv agent/code/router/merged_results.csv
```

---

## Reproducibility

To reproduce the main results reported in the paper, follow these steps:

1. Prepare the dataset and configuration files.
2. Train the cognitive diagnosis model to obtain student knowledge states.
3. Run the student simulation pipeline based on the diagnosis results.
4. Run the evaluation scripts to obtain the final metrics.

For stable reproduction, we recommend fixing random seeds and recording all experimental configurations.

---

## Acknowledgment

This repository builds upon the following open-source projects:

* **NeuralCD**: [https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD](https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD)
* **Agent4Edu**: [https://github.com/bigdata-ustc/Agent4Edu](https://github.com/bigdata-ustc/Agent4Edu)

We sincerely thank the authors of these projects for making their code publicly available.

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{
anonymous2026one,
title={One {LLM} Does Not Simulate All Students: Ability-Aware Student Simulation via Cognitive Diagnosis Guided {LLM} Assignment},
author={Anonymous},
booktitle={The 64th Annual Meeting of the Association for Computational Linguistics},
year={2026},
url={https://openreview.net/forum?id=IfIRg71R61}
}
```

---

## Contact

If you have any questions, please open an issue or contact the authors at huixingq@mail.ustc.edu.cn.

