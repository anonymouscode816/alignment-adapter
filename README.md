# Improving the Performance of Compressed Transformers with Post-Hoc Embedding Alignment

This repository contains the implementation of **Alignment Adapter (AlAd)**, a lightweight sliding-window adapter designed to improve the downstream performance of compressed BERT-family models. AlAd aligns the token-level representations of a compressed model with those of a larger reference model and can be used either as a frozen-backbone plug-and-play module or jointly fine-tuned with the compressed model.

---

## 1. Overview

Deep learning models are commonly compressed before deployment in resource-constrained environments. However, compression often reduces downstream task performance. AlAd addresses this by learning a lightweight transformation from the compressed model's token embeddings to the representation space of a larger teacher model.

For an input token sequence, the compressed model produces token embeddings. AlAd takes each token embedding together with a small local context window and maps it to the teacher-model embedding dimension. This helps restore local contextual information lost during compression while adding only a small number of parameters.

The experiments cover three token-level NLP tasks:

| Task | Dataset | Main evaluation metrics |
|---|---|---|
| POS tagging | German Universal Dependencies | Accuracy, overall F1 |
| NER | MultiCoNER v2 | Overall F1, macro F1 |
| EQA | SQuAD 2.0 | Exact Match, F1 |

The compressed models considered are:

| Model | Description |
|---|---|
| ASC | Application-specific compressed BERT model |
| BERT-Mini | Compact BERT model with a smaller hidden dimension |
| BERT-Tiny | Highly compressed BERT model |

BERT-base is used as the larger reference model for representation alignment.

---

## 2. Method Summary

AlAd is trained in three stages.

### Stage 1: Task-independent AlAd pretraining

The compressed model is frozen. AlAd is trained to map compressed-model token embeddings to BERT-base token embeddings using a general text corpus, such as English Wikipedia.

### Stage 2: Task-specific continual alignment

The compressed model remains frozen. AlAd is further trained on the task-specific dataset using representation alignment. Task labels are not used for the alignment objective. This stage adapts the adapter to the distribution of the downstream task.

### Stage 3: Task-specific fine-tuning

AlAd is attached to the compressed model, together with a task-specific prediction head. Two fine-tuning modes are supported:

1. **Frozen compressed model:** the compressed-model backbone remains frozen while AlAd and the task-specific prediction head are trained.
2. **Joint fine-tuning:** AlAd, the compressed model, and the task-specific prediction head are trained jointly.

For EQA, adapter-based fine-tuning is implemented using multiple task-specific scripts rather than a single unified fine-tuning script.

---

## 3. Supported Tasks and Models

The same general experimental structure is followed for **POS**, **NER**, and **EQA** using the following compressed models:

- ASC
- BERT-Mini
- BERT-Tiny

For each task-model pair, the workflow is:

```text
1. Task-independent AlAd pretraining
2. Task-specific continual AlAd pretraining
3. Task-specific fine-tuning with AlAd
   ├── frozen compressed model setting
   └── jointly fine-tuned compressed model setting
```

The evaluated AlAd window sizes are:

```text
W1, W3, W5
```

where:

- `W1` uses only the current token embedding.
- `W3` uses the current token and one neighboring token on each side.
- `W5` uses the current token and two neighboring tokens on each side.

---

## 4. Repository Structure

A recommended organization is shown below. Exact paths may vary slightly depending on the experiment scripts included with the artifact.

```text
.
├── README.md
├── requirements.txt
│
├── NER/
│   ├── ASC/
│   │   ├── pretrain-asc-wall-clock.py
│   │   ├── continual-pretraining.py
│   │   └── fine_tune_asc_with_adapter_all_cases.py
│   │
│   ├── BERT-Mini/
│   │   ├── pretrain-bert-mini-wall-clock.py
│   │   ├── continual-pretraining.py
│   │   └── fine_tune_bert_mini_with_adapter_all_cases.py
│   │
│   └── BERT-Tiny/
│       ├── pretrain-bert-tiny-wall-clock.py
│       ├── continual-pretraining.py
│       └── fine_tune_bert_tiny_with_adapter_all_cases.py
│
├── POS/
│   ├── ASC/
│   ├── BERT-Mini/
│   └── BERT-Tiny/
│
├── EQA/
│   ├── ASC/
│   ├── BERT-Mini/
│   └── BERT-Tiny/
│
├── data/
│   ├── pos/
│   ├── ner/
│   └── eqa/
│
├── checkpoints/
│   ├── pretraining/
│   ├── continual_pretraining/
│   └── fine_tuning/
│
└── results/
    ├── pos/
    ├── ner/
    └── eqa/
```

For the **NER + BERT-Mini** setting, the primary scripts are:

```text
pretrain-bert-mini-wall-clock.py
continual-pretraining.py
fine_tune_bert_mini_with_adapter_all_cases.py
```

The file:

```text
fine_tune_bert_mini_with_adapter_all_cases.py
```

contains experiments for both the frozen-backbone and jointly fine-tuned settings.

---

## 5. Environment Setup

Create a clean Python environment:

```bash
conda create -n alad python=3.10 -y
conda activate alad
python -m pip install --upgrade pip
```

Install the dependencies:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

If installation fails because `requirements.txt` contains a machine-specific local entry such as `conda-pack @ file:///...`, create a cleaned requirements file:

```bash
grep -v "conda-pack @ file:" requirements.txt > requirements_clean.txt
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements_clean.txt
```

For CPU-only installation, install the CPU-compatible version of PyTorch first and then install the remaining dependencies:

```bash
pip install torch torchvision torchaudio
pip install -r requirements_clean.txt
```

---

## 6. Running Experiments

The following commands demonstrate the expected experimental workflow. Modify dataset paths, checkpoint paths, GPU IDs, and hyperparameters according to the arguments supported by the scripts in each experiment directory.

### 6.1 NER with BERT-Mini

Move to the BERT-Mini NER experiment directory:

```bash
cd NER/BERT-Mini
```

#### Step 1: Task-independent pretraining

```bash
python pretrain-bert-mini-wall-clock.py
```

This stage trains AlAd on a general corpus while the compressed-model backbone remains frozen.

#### Step 2: Continual pretraining on the NER dataset

```bash
python continual-pretraining.py
```

This stage adapts AlAd to the NER data distribution using the representation-alignment objective.

#### Step 3: Fine-tuning for NER

```bash
python fine_tune_bert_mini_with_adapter_all_cases.py
```

This script covers both experimental settings:

- frozen compressed model with trainable AlAd
- jointly fine-tuned compressed model with trainable AlAd

Run the corresponding configuration separately for each AlAd window size.

For scripts that expose the window size as a command-line argument, the expected convention is:

```bash
python fine_tune_bert_mini_with_adapter_all_cases.py --window_size 1
python fine_tune_bert_mini_with_adapter_all_cases.py --window_size 3
python fine_tune_bert_mini_with_adapter_all_cases.py --window_size 5
```

If a script defines window size directly in its configuration rather than through a command-line argument, update the corresponding configuration before running the experiment.

### 6.2 POS Experiments

POS experiments follow the same three-stage structure for ASC, BERT-Mini, and BERT-Tiny:

```text
1. Task-independent AlAd pretraining
2. Task-specific continual AlAd pretraining
3. Task-specific fine-tuning
   ├── frozen compressed model
   └── jointly fine-tuned compressed model
```

Example for BERT-Mini:

```bash
cd POS/BERT-Mini
python pretrain-bert-mini-wall-clock.py
python continual-pretraining.py
python fine_tune_bert_mini_with_adapter_all_cases.py
```

Run the appropriate configuration for each required AlAd window size and fine-tuning setting.

### 6.3 EQA Experiments

EQA follows the same overall three-stage procedure, but adapter-based fine-tuning is implemented through multiple task-specific scripts rather than a single combined fine-tuning script.

Recommended workflow:

```text
1. Run task-independent AlAd pretraining.
2. Run continual pretraining on SQuAD 2.0.
3. Run the EQA frozen-backbone fine-tuning script, when applicable.
4. Run the EQA joint fine-tuning script, when applicable.
5. Generate predictions and evaluate Exact Match and F1.
```

Refer to the scripts provided in the corresponding EQA model directory for experiment-specific arguments and configurations.

---

## 7. Outputs

Experiment outputs can be organized using the following directory structure:

```text
results/<task>/<model>/<mode>/W<window_size>/
```

Typical output files include:

```text
metrics.json
training_log.txt
model_config.json
adapter_checkpoint.pt
final_model_checkpoint.pt
predictions.json
```

For representation-level analysis, token-level cosine-similarity measurements may additionally be stored as:

```text
cosine_similarity_results.json
```

Checkpoint and output filenames can differ slightly between experiment scripts.

---

## 8. Evaluation

Task-specific evaluation metrics are:

| Task | Metrics |
|---|---|
| POS | Accuracy, overall F1 |
| NER | Overall F1, macro F1 |
| EQA | Exact Match, F1 |

In addition to downstream task performance, the experiments analyze mean token-level embedding cosine similarity between representations produced after AlAd projection and representations from the corresponding fine-tuned BERT-base reference model.

This representation-level metric can be used to examine whether improved downstream performance is accompanied by stronger alignment with the larger reference model.

---

## 9. Common Issues

### CUDA or PyTorch Installation Errors

The provided environment was configured with a CUDA-compatible PyTorch installation. If the specified build does not match the CUDA version available on your system, install the appropriate PyTorch build for your environment before installing the remaining dependencies.

### `conda-pack @ file:///...` Installation Error

A requirement of the form:

```text
conda-pack @ file:///...
```

refers to a machine-specific local path and may not be installable on another system.

Remove the corresponding entry from `requirements.txt` or create `requirements_clean.txt` as described in the environment setup instructions.

### Out-of-Memory Errors

Memory requirements can be reduced by decreasing:

- batch size
- maximum sequence length
- number of data-loading workers
- AlAd window size

EQA experiments typically require more memory than POS and NER because span prediction involves substantially longer input contexts.

---

## 10. Anonymous Review

This code artifact accompanies an **anonymous research submission**.

To preserve double-blind review, author names, affiliations, personal contact information, acknowledgments that reveal identity, and links to identifying repositories or project pages have been intentionally omitted.

Questions during the review period should be communicated through the anonymous submission or review system, where applicable.

Author and repository information can be restored after completion of the anonymous review process.
