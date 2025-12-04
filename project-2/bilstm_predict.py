#!/usr/bin/env python3
"""
High-accuracy BiLSTM sequence tagger trained on large ESM2 embeddings.

Pipeline:
    1. Run generate_large_embeddings.py to cache per-protein embeddings with
       facebook/esm2_t33_650M_UR50D under cache/esm2_t33_650M/.
    2. Execute this script to train a BiLSTM on the cached features and produce
       predictions.csv / predictions.zip ready for Codabench.

Design choices:
    - Protein-level train/validation split to avoid residue leakage.
    - BiLSTM (2 layers, hidden=256, bidirectional) + dropout for context modeling.
    - Cross-entropy with class weights + label masking for padded residues.
    - Inference iterates over proteins to keep memory usage < 12 GB.
"""

import json
import os
import random
import zipfile
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = "project_1"
TRAIN_PATH = os.path.join(PROJECT_DIR, "train.tsv")
TEST_PATH = os.path.join(PROJECT_DIR, "test.tsv")

EMBED_ROOT = os.path.join("cache", "esm2_t33_650M")
PROTEIN_DIR = os.path.join(EMBED_ROOT, "proteins")
METADATA_PATH = os.path.join(EMBED_ROOT, "metadata.json")

MODEL_SAVE_PATH = "best_bilstm_model.pth"
PREDICTION_CSV = "predictions.csv"
PREDICTION_ZIP = "predictions.zip"

LABELS = ['.', 'B', 'E', 'G', 'H', 'I', 'P', 'S', 'T']
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
INDEX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}
IGNORE_INDEX = -100

RANDOM_STATE = 42
BATCH_SIZE = 6
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
MAX_EPOCHS = 20
PATIENCE = 4
LEARNING_RATE = 3e-4
LR_FACTOR = 0.5
MIN_LR = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_id(identifier: str):
    protein, amino_acid, position = identifier.split("_")
    return protein, amino_acid, int(position) - 1


def load_metadata() -> Dict[str, Dict[str, int]]:
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}. "
            "Run generate_large_embeddings.py first."
        )
    with open(METADATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def filter_df_by_embeddings(df: pd.DataFrame, metadata: Dict[str, Dict[str, int]], split_name: str) -> pd.DataFrame:
    """Drop rows whose proteins lack embeddings or whose positions exceed sequence length."""
    valid_frames = []
    dropped_proteins = []
    truncated_rows = 0

    for protein, group in df.groupby("protein"):
        meta = metadata.get(protein)
        if not meta:
            dropped_proteins.append(protein)
            continue
        protein_len = meta.get("length", 0)
        positions = group["position"].to_numpy(dtype=np.int64)
        mask = positions < protein_len
        if not mask.any():
            dropped_proteins.append(protein)
            continue
        if not mask.all():
            truncated_rows += (~mask).sum()
        valid_idx = group.index[mask]
        valid_frames.append(group.loc[valid_idx])

    if not valid_frames:
        raise RuntimeError(f"No usable residues found for split '{split_name}'.")

    filtered = pd.concat(valid_frames, axis=0).sort_index()
    print(
        f"{split_name}: kept {len(filtered):,} rows | "
        f"dropped proteins={len(dropped_proteins)} | truncated rows={truncated_rows}"
    )
    if dropped_proteins:
        sample = ", ".join(sorted(dropped_proteins)[:10])
        print(f"  Dropped proteins due to missing embeddings: {sample} ...")
    return filtered


class ProteinDataset(Dataset):
    """Protein-level dataset returning entire residue sequences for training."""

    def __init__(self, df: pd.DataFrame, embeddings_dir: str, label_map: Dict[str, int], cache_size: int = 16):
        self.embeddings_dir = embeddings_dir
        self.label_map = label_map
        self.cache_size = cache_size
        self.cache: OrderedDict[str, np.memmap] = OrderedDict()
        self.samples = self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> List[Dict[str, np.ndarray]]:
        samples = []
        for protein, group in df.groupby("protein"):
            sorted_group = group.sort_values("position")
            positions = sorted_group["position"].to_numpy(dtype=np.int32)
            labels = sorted_group["secondary_structure"].to_list()
            label_ids = np.array([self.label_map[label] for label in labels], dtype=np.int64)
            samples.append(
                {
                    "protein": protein,
                    "positions": positions,
                    "label_ids": label_ids,
                }
            )
        return samples

    def _load_embedding(self, protein: str) -> np.memmap:
        if protein in self.cache:
            array = self.cache.pop(protein)
            self.cache[protein] = array
            return array

        path = os.path.join(self.embeddings_dir, f"{protein}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding not found for protein {protein} at {path}")

        array = np.load(path, mmap_mode="r")
        self.cache[protein] = array
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return array

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        protein = sample["protein"]
        positions = sample["positions"]
        label_ids = sample["label_ids"]

        embeddings = self._load_embedding(protein)
        seq_emb = embeddings[positions]
        seq_tensor = torch.from_numpy(np.asarray(seq_emb, dtype=np.float32))
        label_tensor = torch.from_numpy(label_ids)
        return seq_tensor, label_tensor


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return padded_sequences, padded_labels, lengths


class BiLSTMTagger(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor):
        sequences = self.layer_norm(sequences)
        packed = pack_padded_sequence(sequences, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        logits = self.classifier(self.dropout(outputs))
        return logits


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = df["secondary_structure"].value_counts()
    total = df.shape[0]
    weights = []
    for label in LABELS:
        count = counts.get(label, 1)
        weights.append(total / (len(LABELS) * count))
    weights = np.asarray(weights, dtype=np.float32)
    weights /= weights.mean()
    return torch.from_numpy(weights)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    for sequences, labels, lengths in tqdm(dataloader, desc="Training", leave=False):
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(sequences, lengths)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            mask = labels != IGNORE_INDEX
            predictions = logits.argmax(dim=-1)
            correct_tokens += ((predictions == labels) & mask).sum().item()
            total_tokens += mask.sum().item()
        total_loss += loss.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct_tokens / max(total_tokens, 1)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for sequences, labels, lengths in tqdm(dataloader, desc="Validation", leave=False):
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(sequences, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            mask = labels != IGNORE_INDEX
            predictions = logits.argmax(dim=-1)
            correct_tokens += ((predictions == labels) & mask).sum().item()
            total_tokens += mask.sum().item()
            total_loss += loss.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct_tokens / max(total_tokens, 1)
    return avg_loss, accuracy


def protein_level_split(df: pd.DataFrame, val_ratio: float = 0.1):
    proteins = df["protein"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(proteins)
    split_idx = int(len(proteins) * (1 - val_ratio))
    train_proteins = set(proteins[:split_idx])
    val_proteins = set(proteins[split_idx:])
    train_df = df[df["protein"].isin(train_proteins)].copy()
    val_df = df[df["protein"].isin(val_proteins)].copy()
    return train_df, val_df


def predict_test(model, test_df: pd.DataFrame, metadata: Dict[str, Dict[str, int]], device):
    predictions = np.full(len(test_df), ".", dtype="<U1")
    grouped = test_df.groupby("protein")
    model.eval()

    with torch.no_grad():
        for protein, group in tqdm(grouped, desc="Predicting"):
            meta = metadata.get(protein)
            if not meta:
                continue
            protein_len = meta.get("length", 0)
            emb_path = os.path.join(PROTEIN_DIR, f"{protein}.npy")
            if not os.path.exists(emb_path):
                continue

            sorted_group = group.sort_values("position")
            positions = sorted_group["position"].to_numpy(dtype=np.int32)
            mask = positions < protein_len
            if not mask.any():
                continue

            valid_positions = positions[mask]
            emb = np.load(emb_path, mmap_mode="r")
            seq_emb = emb[valid_positions]
            del emb
            seq_tensor = torch.from_numpy(np.asarray(seq_emb, dtype=np.float32)).unsqueeze(0).to(device)
            lengths = torch.tensor([seq_tensor.size(1)], dtype=torch.long, device=device)

            logits = model(seq_tensor, lengths)
            pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            pred_labels = [INDEX_TO_LABEL[int(idx)] for idx in pred_ids]

            valid_indices = sorted_group.index.values[mask]
            predictions[valid_indices] = pred_labels

    return predictions


def main():
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df = pd.read_csv(TRAIN_PATH, sep="\t")
    test_df = pd.read_csv(TEST_PATH, sep="\t")
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    for df in (train_df, test_df):
        split_cols = df["id"].str.rsplit("_", n=2, expand=True)
        split_cols.columns = ["protein", "amino_acid", "position"]
        df[["protein", "amino_acid"]] = split_cols[["protein", "amino_acid"]]
        df["position"] = split_cols["position"].astype(np.int32) - 1

    metadata = load_metadata()
    train_df = filter_df_by_embeddings(train_df, metadata, "Train")
    train_split_df, val_split_df = protein_level_split(train_df, val_ratio=0.1)
    print(f"Proteins - Train: {train_split_df['protein'].nunique()}, Val: {val_split_df['protein'].nunique()}")

    class_weights = compute_class_weights(train_split_df).to(device)

    train_dataset = ProteinDataset(train_split_df, PROTEIN_DIR, LABEL_TO_INDEX)
    val_dataset = ProteinDataset(val_split_df, PROTEIN_DIR, LABEL_TO_INDEX)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    input_dim = None
    for file_name in os.listdir(PROTEIN_DIR):
        if not file_name.endswith(".npy"):
            continue
        arr = np.load(os.path.join(PROTEIN_DIR, file_name), mmap_mode="r")
        input_dim = arr.shape[1]
        del arr
        break
    if input_dim is None:
        raise RuntimeError(f"No embedding files found under {PROTEIN_DIR}.")

    model = BiLSTMTagger(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=len(LABELS),
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=1, min_lr=MIN_LR
    )

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_val_acc = val_acc
            patience_counter = 0
            print(f"  ✅ New best model saved (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("  Early stopping triggered.")
                break

    if not os.path.exists(MODEL_SAVE_PATH):
        raise RuntimeError("Training did not produce a saved model. Check logs for issues.")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    # Full-train evaluation (optional) – skipped to keep runtime reasonable

    test_predictions = predict_test(model, test_df, metadata, device)

    result_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "secondary_structure": test_predictions,
        }
    )
    result_df["secondary_structure"].fillna(".", inplace=True)
    result_df.to_csv(PREDICTION_CSV, sep="\t", index=False, lineterminator="\n")
    with zipfile.ZipFile(PREDICTION_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(PREDICTION_CSV)

    print("\nSubmission files created:")
    print(f"  - {PREDICTION_CSV}")
    print(f"  - {PREDICTION_ZIP}")
    print(f"Validation accuracy achieved: {best_val_acc*100:.2f}%")
    print("Upload predictions.zip to Codabench to obtain the final score.")


if __name__ == "__main__":
    main()

