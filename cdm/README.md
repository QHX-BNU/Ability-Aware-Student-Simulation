# CDM Running Guide

This document explains how to prepare data, configure training, and run the `cdm` module. The main entrypoint of `cdm` is `train.py`. It is recommended to run commands from the repository root.

## 1. Directory Structure

```text
cdm/
├─ data_loader.py
├─ model.py
├─ train.py
├─ README.md
├─ model/            # directory for saving trained model checkpoints
├─ stu_embedding/    # directory for saving student embeddings
└─ train_data/
   └─ train_stu_llm.csv
```

The roles of the files are as follows.

### 1.1 Files under `cdm/`

- `train.py`: main training entry; sets up device, loads data, builds the model, runs the training loop, and saves checkpoints + student embeddings
- `data_loader.py`: data loader; converts the CSV training data into model-ready batches
- `model.py`: defines the NeuralCD model (student embeddings, item difficulty, discrimination, and the prediction network)
- `README.md`: this guide

### 1.2 Files under `train_data/`

- `train_stu_llm.csv`: training CSV containing responses from real students and/or LLM agents; this is the core input for `cdm` training

### 1.3 Training outputs

- `model/`: saved model checkpoints
- `stu_embedding/`: saved student embeddings

## 2. Environment

Python 3.10+ is recommended.

Install at least the following dependencies:

```bash
pip install torch pandas numpy
```

Notes:

- `torch`: model training
- `pandas`: reading the training CSV
- `numpy`: saving student embeddings

## 3. Configuration

The `cdm` module currently does not use a separate configuration file. Training-related settings are directly defined in `cdm/train.py` and `cdm/data_loader.py`.

### 3.1 Configure training parameters in `train.py`

Key code locations:

- dataset size parameters: `exer_n`, `knowledge_n`, `student_n`
- device and epoch parameters: `device`, `epoch_n`
- training loop: `train()`
- output paths: `PROJECT_ROOT`, `MODEL_DIR`, `EMB_DIR`

Current core parameters in the script:

```python
exer_n = 212
knowledge_n = 93
student_n = 1299

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch_n = 5
```

Field meanings:

- `exer_n`: number of exercises
- `knowledge_n`: number of knowledge components
- `student_n`: number of students; if you include LLMs as pseudo-students during training, they should be included here as well
- `device`: training device; defaults to `cuda:0` if available
- `epoch_n`: number of epochs

If your dataset size changes, make sure to update these values accordingly.

### 3.2 Configure data paths in `data_loader.py`

Key code locations:

- dataset size parameters: `student_n`, `question_n`, `knowledge_n`
- training CSV path: `train_data_path`
- training set loader: `TrainDataLoader`
- validation/test loader: `ValTestDataLoader`

By default, validation/test data are read from `data/val.csv` and `data/test.csv` under the project root.

Current training data path (relative to project root):

```python
train_data_path = PROJECT_ROOT / "cdm" / "train_data" / "train_stu_llm.csv"
```

If you change the training file, update this path accordingly.

### 3.3 Configure model structure in `model.py`

Key code locations:

- `Net` definition: `cdm/model.py` lines 5–66
- hidden layer sizes: `cdm/model.py` line 15
- forward pass: `cdm/model.py` lines 34–51
- non-negative clipper: `cdm/model.py` lines 69–77

Current hidden sizes are defined as:

```python
self.prednet_len1, self.prednet_len2 = 512, 256
```

If you want to adjust the model capacity, modify these values.

## 4. Data Format

### 4.1 `train_stu_llm.csv`

The training file is a CSV. The current code expects these columns:

```csv
student_id,question_id,knowledgecomponent_id,answers
1,2,1,1
1,54,"19,18",1
1,87,"44,29,40",1
```

Field meanings:

- `student_id`: student id or agent id
- `question_id`: exercise id
- `knowledgecomponent_id`: knowledge id; can be a single id or multiple ids separated by commas
- `answers`: correctness (`1` correct, `0` incorrect)

### 4.2 Requirements and constraints

Important notes:

- `student_id`, `question_id`, and `knowledgecomponent_id` are assumed to start from `1`
- `data_loader.py` converts them into 0-based indices before feeding the model
- multi-knowledge exercises must be encoded as a CSV string like `"19,18"`
- knowledge ids in `knowledgecomponent_id` must not exceed `knowledge_n`
- `question_id` must not exceed `exer_n`
- `student_id` must not exceed `student_n`

## 5. Commands

Run from the repository root.

### 5.1 Start training

```bash
python cdm/train.py
```

### 5.2 Locate/modify the training entry

The entrypoint is at the end of `cdm/train.py` and it runs `train()` by default.

## 6. Outputs

The code produces two types of outputs:

- trained model checkpoints
- student embeddings

Note that the save paths are defined in `cdm/train.py`, and they are **joined with the project root automatically**, with output directories created as needed.

Default save locations (relative to project root):

```python
MODEL_DIR = PROJECT_ROOT / "cdm" / "model"
EMB_DIR = PROJECT_ROOT / "cdm" / "stu_embedding"

save_snapshot(net, MODEL_DIR / f"model_epoch{epoch + 1}.pt")
np.save(EMB_DIR / "student_emb.npy", stu_emb)
```

So regardless of where you launch training (as long as the repo structure is unchanged), outputs will be saved to:

- `cdm/model/`: `model_epoch*.pt`
- `cdm/stu_embedding/`: `student_emb.npy`

## 7. FAQ

### 7.1 Training data cannot be found

Check:

- whether `cdm/train_data/train_stu_llm.csv` exists
- whether `train_data_path` in `cdm/data_loader.py` points to the intended file

### 7.2 Dimension errors during training

Check:

- whether `student_n`, `exer_n`, `knowledge_n` in `train.py` match your data
- whether any `knowledgecomponent_id` exceeds `knowledge_n`
- whether any `question_id` exceeds `exer_n`
- whether any `student_id` exceeds `student_n`

### 7.3 Training finishes but saving fails

Check:

- whether `cdm/model/` and `cdm/stu_embedding/` are writable (the script creates them automatically)
- whether you have write permissions
- whether you are training in a read-only location or blocked by system permissions/antivirus software

### 7.4 Can I run without a GPU?

Yes. `train.py` automatically switches to CPU when CUDA is unavailable, but training will be slower.

## 8. Recommended order

1. Verify the column names and id ranges in `train_stu_llm.csv`
2. Ensure `student_n`, `exer_n`, `knowledge_n` in `train.py` match your data
3. Ensure the repository is writable (it will write to `cdm/model/` and `cdm/stu_embedding/`)
4. Run `python cdm/train.py` to start training
