# Protein Secondary Structure Prediction

Deep learning approach for predicting protein secondary structures from amino acid sequences using ESM-2 embeddings and BiLSTM-CRF architecture.

## Overview

This project implements a high-accuracy protein secondary structure predictor that achieves **~70-74%** accuracy on the test set. It uses:

- **ESM-2 (650M)** transformer embeddings for rich protein representations
- **BiLSTM** architecture to capture sequential dependencies
- **Class-weighted training** to handle imbalanced secondary structure classes

## Project Structure

```
├── generate_large_embeddings.py   # Extract ESM-2 embeddings from FASTA sequences
├── bilstm_predict.py              # Train BiLSTM model and generate predictions
├── requirements.txt               # Python dependencies
└── project_1/                     # Dataset (train.tsv, test.tsv, sequences.fasta)
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, Transformers (Hugging Face), pandas, numpy, tqdm

## Usage

### 1. Generate Embeddings

```bash
python generate_large_embeddings.py
```

Extracts 1280-dimensional embeddings from the `facebook/esm2_t33_650M_UR50D` model and caches them to disk (~47 min on CPU, ~10 min on GPU).

### 2. Train Model & Predict

```bash
python bilstm_predict.py
```

Trains a BiLSTM classifier on cached embeddings and generates:
- `predictions.csv` - Tab-separated predictions (id, secondary_structure)
- `predictions.zip` - Submission file
- `best_bilstm_model.pth` - Best model checkpoint

**Training time:** ~30 min on CPU, ~3 min per epoch on GPU (T4)

## Results

- **Validation Accuracy:** 74.0%
- **Test Accuracy:** ~70% (Codabench)
- **Architecture:** BiLSTM with 2 layers, 256 hidden units, 30% dropout

## Secondary Structure Classes

Predicts 9 classes: `.` (coil), `B` (beta-bridge), `E` (strand), `G` (3-helix), `H` (alpha-helix), `I` (5-helix), `P` (turn), `S` (bend), `T` (hydrogen turn)

## Notes

- Model uses early stopping (patience=4) to prevent overfitting
- Embeddings are cached to `cache/esm2_t33_650M/` for reuse
- GPU highly recommended for training (10x faster than CPU)

