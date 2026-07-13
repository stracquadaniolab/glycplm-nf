# glycplm-nf

![](https://img.shields.io/badge/current_version-v0.1.1-blue)

A Nextflow pipeline for residue-level protein glycosylation site prediction. This is a baseline version that uses the ESMC-300M protein language model for embeddings and a lightweight linear classifier to predict glycosylation.

## Overview

Given a TSV of protein sequences with annotated glycosylation sites, the pipeline can either **train** a new classifier or **predict** glycosylation sites with an existing one.

```
train:
  raw TSV ──▶ PREPROCESS ──▶ SPLIT_TEST ──▶ EMBED ──▶ TRAIN ──▶ classifier.pt + threshold.json + history.json

predict:
  raw TSV ──▶ PREPROCESS ──▶ EMBED ──▶ PREDICT ──▶ predictions.json (+ test-metrics.json)
```

`SPLIT_TEST` is only run for the `train` entry (and only when `--split_test` is enabled); it splits off a held-out test set before embedding. `PREDICT` consumes a trained `classifier.pt` (and, optionally, its optimised `threshold.json`) to score new or held-out sequences.

## Pipeline stages

### 1. PREPROCESS (`preprocess.py`)

Reads the raw TSV, drops proteins with no glycosylation annotation (optional), filters out proteins longer than a given length (optional), and converts each protein's `Glycosylation` annotation string (UniProt-style `CARBOHYD <position>` entries) into a binary list the same length as its sequence.

- **Input**: TSV with `Entry`, `Entry Nmae`, `Sequence` and `Glycosylation` columns
- **Output**: `processed.json` — records with `Entry`, `Entry Name`, `Sequence`, `Glycosylation_binary`

### 2. SPLIT_TEST (`split-test.py`) — `train` entry only

Splits the preprocessed proteins into a train+val set and a held-out test set, so the test set can be evaluated later with `PREDICT`.

- **Input**: `processed.json`
- **Output**: `processed-train-val.json`, `processed-test.json`

### 3. EMBED (`get-embeddings.py`)

Loads ESMC-300M (`AutoModelForMaskedLM`) and its tokenizer, batches sequences through the model, strips the `[CLS]`/`[EOS]` tokens, and pairs each residue's embedding with its binary label.

- **Input**: preprocessed/split JSON
- **Output**: `embedding-data.pt` — a list of `{'entry', 'entry_name', 'sequence', 'embedding': Tensor[seq_len, 960], 'label': Tensor[seq_len]}` dicts, one per protein

### 4. TRAIN (`train.py`, `classifier.py`) — `train` entry only

Splits embedded proteins into train/validation sets, flattens all residues across proteins into single tensors, and trains a single-linear-layer classifier (`ResidueClassifier`, with dropout for regularisation) using a class-weighted cross-entropy loss to handle the natural imbalance between glycosylated and non-glycosylated residues. At each epoch, a decision threshold is re-optimised on the validation set (maximising F1 or Youden's J) and metrics are reported at that threshold.

- **Input**: `embedding-data.pt`
- **Output**: `classifier.pt` (model weights), `threshold.json` (final optimised decision threshold), `history.json` (per-epoch train/val loss, precision, recall, F1, ROC-AUC, PR-AUC, best threshold)

### 5. PREDICT (`predict.py`, `classifier.py`) — `predict` entry only

Loads a trained classifier and runs it over new embedding data, producing per-residue predicted probabilities and labels at a given decision threshold (ideally the one optimised during training). If ground-truth labels are present (e.g. on a held-out test set), aggregate metrics are also computed.

- **Input**: `embedding-data.pt`, `classifier.pt`, decision threshold
- **Output**: `predictions.json` (per-residue predictions), `test-metrics.json` (only written when ground-truth labels are available)

## Requirements

- [Nextflow](https://www.nextflow.io/) (DSL2)
- A container engine: [Docker](https://www.docker.com/) or [Singularity/Apptainer](https://apptainer.org/)
- A GPU for the EMBED, TRAIN and PREDICT stages
- [ESMC-300M](https://huggingface.co/biohub/ESMC-300M) model weights available at `/opt/conda/models/ESMC-300M` inside the container image

## Usage

### Training

```bash
nextflow run main.nf \
    --entry train \
    --input_tsv path/to/raw_data.tsv \
    --outdir results \
    -profile docker
```

### Prediction

```bash
nextflow run main.nf \
    --entry predict \
    --input_tsv path/to/new_data.tsv \
    --model_ckpt results/model/classifier.pt \
    --threshold_json results/model/threshold.json \
    --outdir results \
    -profile docker
```

You can also start either entry from an already-preprocessed JSON file with `--input_json` instead of `--input_tsv`.

## Parameters

### General

| Parameter | Default | Description |
|---|---|---|
| `--entry` | *(required)* | `train` or `predict` |
| `--input_tsv` | `null` | Path to raw TSV with `Sequence` and `Glycosylation` columns |
| `--input_json` | `null` | Path to an already-preprocessed JSON file (skips PREPROCESS) |
| `--model_ckpt` | `null` | Path to a trained `classifier.pt` (required for `--entry predict`) |
| `--outdir` | `results` | Output directory |

### Preprocessing

| Parameter | Default | Description |
|---|---|---|
| `--drop_non_glycosylated` | `true` | Drop sequences without glycosylation annotations |
| `--length_filter` | `500` | Filter out sequences longer than this length |
| `--split_test` | `true` | Whether to split off a held-out test set (`train` entry only) |
| `--test_size` | `0.05` | Fraction of proteins held out as the test set |

### Embedding

| Parameter | Default | Description |
|---|---|---|
| `--batch_size_embed` | `8` | Batch size for embedding generation |

### Training

| Parameter | Default | Description |
|---|---|---|
| `--val_size` | `0.15` | Fraction of proteins held out for validation |
| `--random_state` | `42` | Random seed for splits |
| `--num_epochs` | `10` | Number of training epochs |
| `--train_batch_size` | `4096` | Batch size (residues per batch) for training |
| `--lr` | `0.0001` | Learning rate |
| `--hidden_size` | `960` | Embedding hidden size (ESMC-300M = 960) |
| `--dropout` | `0.1` | Dropout applied before the classifier's linear layer |
| `--optimise_metric` | `f1` | Metric used to optimise the decision threshold (`f1` or `youden`) |

### Prediction

| Parameter | Default | Description |
|---|---|---|
| `--threshold_json` | `null` | Path to `threshold.json` from a `train()` run (optional; overrides `--threshold`) |
| `--threshold` | `0.5` | Fallback decision threshold if `--threshold_json` isn't given |

## Output structure

```
results/
├── preprocess/
│   ├── processed.json
│   └── preprocess.log
├── split-test/                      # train entry, if --split_test
│   ├── processed-train-val.json
│   ├── processed-test.json
│   └── split-test.log
├── embeddings/
│   ├── embedding-data.pt
│   └── embed.log
├── model/                           # train entry
│   ├── classifier.pt
│   ├── threshold.json
│   ├── history.json
│   └── train.log
├── predictions/                     # predict entry
│   ├── predictions.json
│   ├── test-metrics.json            # only if ground-truth labels are present
│   └── predict.log
└── pipeline_info/
    ├── timeline.html
    ├── report.html
    ├── trace.txt
    └── dag.html
```

## License

MIT