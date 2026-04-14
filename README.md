
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

## 📑 Table of Contents

- [Abstract](#abstract)
- [Highlights](#highlights)
- [Framework Overview](#framework-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running Experiments](#running-experiments)
- [Reproducibility](#reproducibility)
- [Acknowledgment](#acknowledgment)
- [Citation](#citation)
- [Contact](#contact)

---

## Abstract

Large Language Models (LLMs) have become integral to personalized education systems, particularly in the realm of student behavior simulation. By predicting fine-grained learning behaviors, these simulations enable intelligent systems to provide tailored instructional support. However, most existing methods rely on a single high-capacity LLM to represent an entire population of diverse learners. In this work, we demonstrate that this ''one-size-fits-all'' approach induces a systematic **ability-dependent bias**, where high-capacity models tend to overestimate low-ability students while lower-capacity models underestimate high-ability ones. To mitigate this distortion, we propose an _ability-aware student simulation framework_ that dynamically matches students with appropriate LLM backbones through cognitive alignment. We leverage Neural Cognitive Diagnosis (NeuralCD) to extract multidimensional cognitive profiles for both human students and LLM agents within a shared skill space, subsequently pairing each student with the most cognitively representative model. Extensive experiments demonstrate that our approach substantially reduces simulation bias and consistently outperforms single-model baselines across the entire proficiency spectrum. Our findings suggest that faithful behavior simulation necessitates the **alignment of model capacity with student ability**, establishing cognitive diagnosis as a principled mechanism for model assignment in educational AI. 

---

## Highlights

- ✅ **Official implementation** of our ACL 2026 paper
- 🎯 **Unified framework** for cognitive diagnosis and LLM-based student simulation
- 📊 **Integrated codebase** for training, simulation, and evaluation
- 🔧 **Built upon** prior open-source educational modeling frameworks

---

## Framework Overview

Our framework consists of three major stages:

### 🎯 Stage 1: Response Collection from Students and Candidate LLMs

We first collect historical response records from real students. In addition, we define a candidate set of LLMs as potential simulation backbones.

**Process:**
- Collect historical response data from real students
- Define candidate LLMs as potential simulation backbones
- Prompt each LLM to answer all exercises using chain-of-thought (CoT) prompting
- Record final answer correctness
- Construct training dataset containing both real student responses and LLM-generated responses

### 🧠 Stage 2: Cognitive Profiling via NeuralCD

Given the combined response records from students and candidate LLMs, we train a NeuralCD model to infer their latent cognitive profiles.

**Process:**
- Train NeuralCD on combined response records (students + LLMs)
- Treat LLMs as pseudo-students during training
- Obtain unified and interpretable representation of both human learners and LLMs
- Generate cognitive proficiency representations for students and candidate LLMs
- Use representations for simulation backbone selection

### 🔗 Stage 3: Similarity-Based LLM Assignment

For each student, we compare the student’s cognitive proficiency vector with those of all candidate LLMs and select the most similar one as the simulation backbone.

**Process:**
- Compare student’s cognitive proficiency vector with all candidate LLMs
- Select most similar LLM using cosine similarity
- Ensure each student is simulated by an LLM with compatible cognitive proficiency profile

The overall pipeline is illustrated in the figure above.

---

## Codebase

This repository is an **integrated implementation** for the experiments in our paper, rather than a standalone framework developed entirely from scratch.

### Base Frameworks

We build upon the following open-source projects:

- 🧠 **Cognitive Diagnosis**: Implemented based on [NeuralCD](https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD)
- 🤖 **Agent Simulation**: Implemented based on [Agent4Edu](https://github.com/bigdata-ustc/Agent4Edu)

### Our Contribution

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
└── README.md              # This file
```

### Key Directories

- **`agent/`**: Contains the simulation agent code and main simulation script
- **`cdm/`**: Cognitive diagnosis model implementation and training
- **`assign/`**: LLM assignment logic based on cognitive similarity
- **`evaluation/`**: Scripts for evaluating simulation results
- **`scripts/`**: Utility scripts for data processing and result extraction`

---

## Installation

### Prerequisites

- Python 3.10 or higher

### Setup

**Step 1:** Clone the repository

```bash
git clone https://github.com/QHX-BNU/Ability-Aware-Student-Simulation.git
cd Ability-Aware-Student-Simulation
```

**Step 2:** Install dependencies

```bash
pip install -r requirements.txt
```

**Step 3:** Configure LLM access (if needed)

If your experiments require LLM APIs or local model services, please configure the corresponding environment variables or service credentials.

---

## Data Preparation

### Required Data Fields

Please prepare a raw interaction dataset that contains (at minimum) the following fields:
- Student ID
- Exercise/question ID
- Exercise/question text
- Exercise/question knowledge concept(s)
- Ground-truth (correct) answer
- Whether the student's answer is correct (e.g., 0/1)

### Data Processing

After preparing the raw dataset, you need to preprocess/convert it so that it matches the input formats required by this codebase. The required data formats for each stage (CDM training, simulation, evaluation) are introduced in the documentation below.


### Data Accessibility

After preparing the raw dataset, you need to preprocess/convert it so that it matches the input formats required by this codebase. The required data formats for each stage (CDM training, simulation, evaluation) are introduced in the documentation below.

If the raw data cannot be released due to privacy or licensing restrictions, we recommend providing:

- Preprocessing scripts
- Data format descriptions
- Example input files


---

## Running Experiments

This repo follows an end-to-end pipeline with the following steps:

### Step 1: Cognitive Diagnosis (CDM)

Train NeuralCD to obtain student embeddings.

**First**, prepare the CDM training data by converting your dataset into the required CSV format.

See: [`cdm/README.md`](cdm/README.md)

**Then**, train the model:

```bash
python cdm/train.py
```

---

### Step 2: Ability Grouping (Students)

Compute each student's ability from embeddings and select students per ability level.

**Script:** [`scripts/cal_stu_ablility.py`](scripts/cal_stu_ablility.py)

**Documentation:** [`scripts/README.md`](scripts/README.md)

```bash
python scripts/cal_stu_ablility.py \
  --emb-npy cdm/stu_embedding/student_emb.npy
```

---

### Step 3: LLM Assignment (Student → LLM)

Assign an LLM to each student based on cosine similarity between embeddings.

**Script:** [`assign/cal_sim.py`](assign/cal_sim.py)

**Documentation:** [`assign/README.md`](assign/README.md)

```bash
python assign/cal_sim.py
```

---

### Step 4: Run Simulation

Run the agent-based student simulation.

**Entry script:** [`agent/code/main_simulation.py`](agent/code/main_simulation.py)

**Documentation:** [`agent/README.md`](agent/README.md)

```bash
python agent/code/main_simulation.py --preset router_cosine
```

---

### Step 5: Answer Extraction (merge_results.csv)

Extract structured answers from simulation outputs and append them into a `merged_results.csv`.

**Script:** [`scripts/extra_ans.py`](scripts/extra_ans.py)

> **Note:** You need to update `RESULT_DIR_OUTPUT_MAP` and `STU_LOGS_PATH` inside the script.

**Documentation:** [`scripts/README.md`](scripts/README.md)

```bash
python scripts/extra_ans.py
```

---

### Step 6: Evaluation

Run evaluation scripts to obtain metrics.

**Scripts:**

- Ability-grouped evaluation: [`evaluation/data.py`](evaluation/data.py)
- Overall evaluation (no ability groups): [`evaluation/overall.py`](evaluation/overall.py)

**Documentation:** [`evaluation/README.md`](evaluation/README.md)

```bash
python evaluation/data.py --record-csv agent/code/router/merged_results.csv
python evaluation/overall.py --record-csv agent/code/router/merged_results.csv
```

---

## Reproducibility

To reproduce the main results reported in the paper, follow these steps:

1. **Prepare the dataset and configuration files**
2. **Train the cognitive diagnosis model** to obtain student knowledge states
3. **Run the student simulation pipeline** based on the diagnosis results
4. **Run the evaluation scripts** to obtain the final metrics

For stable reproduction, we recommend fixing random seeds and recording all experimental configurations.

---

## Acknowledgment

This repository builds upon the following open-source projects:

### Base Frameworks

- **NeuralCD**: Cognitive Diagnosis Framework
  [https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD](https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD)

- **Agent4Edu**: Educational Agent Framework
  [https://github.com/bigdata-ustc/Agent4Edu](https://github.com/bigdata-ustc/Agent4Edu)

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

If you have any questions, please:

- 📧 **Email:** huixingq@mail.ustc.edu.cn
- 🐛 **Issues:** Open an issue on GitHub

