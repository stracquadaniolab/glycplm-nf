# glycplm-nf

A Nextflow pipeline for residue-level protein glycosylation site prediction. This is a baseline version that uses ESMC-300M protein language model for embeddings and a lightweight linear classifier to predict glycosylation.

## Overview

Given a TSV of protein sequences with annotated glycosylation sites, the pipeline:

1. **Parses** the raw annotations into per-residue binary labels (`1` = glycosylated residue, `0` = not).
2. **Embeds** each sequence with the ESMC-300M protein language model to get a per-residue vector representation.
3. **Trains** a per-residue binary classifier on top of the embeddings to predict glycosylation sites.

```
raw TSV ──▶ PREPROCESS ──▶ EMBED ──▶ TRAIN ──▶ classifier.pt + history.json
```

## Pipeline stages

### 1. PREPROCESS (`preprocess.py`)

Reads the raw TSV, drops proteins with no glycosylation annotation, and converts each protein's `Glycosylation` annotation string (UniProt-style `CARBOHYD <position>` entries) into a binary string the same length as its sequence.

- **Input**: TSV with at least `Sequence` and `Glycosylation` columns
- **Output**: `processed.pkl` — a pickled pandas DataFrame with an added `Glycosylation_binary` column

### 2. EMBED (`get_embeddings.py`)

Loads ESMC-300M (`AutoModelForMaskedLM`) and tokenizer, batches sequences through the model, strips the `[CLS]`/`[EOS]` tokens, and pairs each residue's embedding with its binary label.

- **Input**: `processed.pkl`
- **Output**: `embedding_data.pt` — a list of `{'embedding': Tensor[seq_len, 960], 'label': Tensor[seq_len]}` dicts, one per protein

### 3. TRAIN (`train.py`, `classifier.py`)

Splits embedded proteins into train/validation sets, flattens all residues across proteins into single tensors, and trains a single-linear-layer classifier (`ResidueClassifier`) with a class-weighted cross-entropy loss to handle the natural imbalance between glycosylated and non-glycosylated residues.

- **Input**: `embedding_data.pt`
- **Output**: `classifier.pt` (model weights), `history.json` (per-epoch train/val loss, precision, recall, F1)

## Requirements

- [Nextflow](https://www.nextflow.io/) (DSL2)
- A container engine: [Docker](https://www.docker.com/) (local) or [Singularity/Apptainer](https://apptainer.org/) (HPC)
- A GPU for the EMBED and TRAIN stages (CPU will work but is significantly slower)
- ESMC-300M model weights available at `/opt/conda/models/ESMC-300M` inside the container image

## Usage

```bash
nextflow run main.nf \
    --input_tsv path/to/raw_data.tsv \
    --outdir results \
    -profile docker
```

```bash
nextflow run main.nf \
    --input_tsv path/to/raw_data.tsv \
    --outdir results \
    -profile singularity,hpc_slurm
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--input_tsv` | *(required)* | Path to raw TSV with `Sequence` and `Glycosylation` columns |
| `--outdir` | `results` | Output directory |
| `--batch_size_embed` | `8` | Batch size for embedding generation |
| `--val_size` | `0.2` | Fraction of proteins held out for validation |
| `--random_state` | `42` | Random seed for the train/val split |
| `--num_epochs` | `5` | Number of training epochs |
| `--train_batch_size` | `8` | Batch size (residues per batch) for training |
| `--lr` | `0.0001` | Learning rate |
| `--hidden_size` | `960` | Embedding hidden size (ESMC-300M = 960) |

## Output structure

```
results/
├── preprocess/
│   └── processed.pkl
├── embeddings/
│   └── embedding_data.pt
├── model/
│   ├── classifier.pt
│   └── history.json
└── pipeline_info/
    ├── timeline.html
    ├── report.html
    ├── trace.txt
    └── dag.svg
```

## Repository structure

```
.
├── main.nf              # Nextflow pipeline definition
├── nextflow.config       # Execution profiles (docker / singularity / hpc_slurm) and resource requests
├── preprocess.py          # Stage 1: TSV -> binary label DataFrame
├── get_embeddings.py      # Stage 2: sequences -> ESMC embeddings
├── classifier.py          # ResidueClassifier model definition
└── train.py               # Stage 3: train/val split, training loop
```

## License

MIT