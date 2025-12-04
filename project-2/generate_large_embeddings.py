#!/usr/bin/env python3
"""
Generate high-quality protein embeddings with a large ESM2 model.

Why this script exists:
    - The earlier 35M-parameter embeddings are not expressive enough for >50% accuracy.
    - We now upgrade to facebook/esm2_t33_650M_UR50D and store per-protein embeddings.
    - Embeddings are saved in float16 to keep disk usage manageable and can be re-used
      by any downstream classifier (BiLSTM, CRF, etc.).

Usage:
    python generate_large_embeddings.py
        - Loads train/test TSVs to figure out which proteins we actually need.
        - Extracts embeddings per protein (with automatic chunking for long sequences).
        - Saves them under cache/esm2_t33_650M/proteins/<protein_id>.npy
        - Writes/updates cache/esm2_t33_650M/metadata.json with sequence lengths.

Notes:
    * Runs comfortably within 12 GB RAM by processing one protein at a time.
    * Safe to re-run; already-generated proteins are skipped automatically.
"""

import json
import os
import sys
import gc
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
OUTPUT_ROOT = os.path.join("cache", "esm2_t33_650M")
PROTEIN_DIR = os.path.join(OUTPUT_ROOT, "proteins")
METADATA_PATH = os.path.join(OUTPUT_ROOT, "metadata.json")
PROJECT_DIR = "project_1"
TRAIN_PATH = os.path.join(PROJECT_DIR, "train.tsv")
TEST_PATH = os.path.join(PROJECT_DIR, "test.tsv")
FASTA_PATH = os.path.join(PROJECT_DIR, "sequences.fasta")
DTYPE = np.float16  # keeps disk footprint manageable

# ESM2 models cap sequences at 1024 tokens (including <cls>/<eos>),
# so we keep a comfortable margin for chunking.
CHUNK_RESIDUES = 1000


def load_fasta(filepath: str) -> Dict[str, str]:
    """Load the FASTA file into a dict of {protein_id: sequence}."""
    sequences: Dict[str, str] = {}
    current_id = None
    with open(filepath, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].strip()
                sequences[current_id] = ""
            elif current_id:
                sequences[current_id] += line.strip().upper()
    return sequences


def parse_id(identifier: str):
    """Split an id like 3KVH_LYS_6 into (protein, amino_acid, 0-index position)."""
    protein, amino_acid, position = identifier.split("_")
    return protein, amino_acid, int(position) - 1


def collect_required_proteins(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    """Return sorted list of unique proteins appearing in train or test."""
    proteins: Set[str] = set(train_df["protein"].unique()).union(test_df["protein"].unique())
    return sorted(proteins)


def load_metadata() -> Dict[str, Dict[str, int]]:
    """Load metadata.json if it exists."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_metadata(metadata: Dict[str, Dict[str, int]]):
    """Persist metadata.json (atomically to avoid corruption)."""
    tmp_path = METADATA_PATH + ".tmp"
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, METADATA_PATH)


def embed_sequence(sequence: str, tokenizer, model, device) -> np.ndarray:
    """
    Compute embeddings for every residue in `sequence`.
    The model max length is 1024 (incl. CLS/EOS), so we slide over the sequence
    in CHUNK_RESIDUES windows and stitch the results back together.
    """
    sequence = sequence.strip().upper()
    if not sequence:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)

    embed_dim = model.config.hidden_size
    seq_len = len(sequence)
    embeddings = np.zeros((seq_len, embed_dim), dtype=np.float32)

    for start in range(0, seq_len, CHUNK_RESIDUES):
        end = min(seq_len, start + CHUNK_RESIDUES)
        chunk = sequence[start:end]
        # Tokenizer expects space-separated residues
        tokenized = tokenizer(
            " ".join(list(chunk)),
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = model(**tokenized)
            # Remove CLS at index 0; keep only actual residue embeddings
            chunk_embeddings = outputs.last_hidden_state[0, 1 : len(chunk) + 1, :]

        embeddings[start:end] = chunk_embeddings.cpu().numpy()
        del tokenized
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return embeddings


def ensure_embeddings(proteins: List[str], sequences: Dict[str, str], tokenizer, model, device):
    """Generate embeddings for each protein, skipping files that already exist."""
    os.makedirs(PROTEIN_DIR, exist_ok=True)
    metadata = load_metadata()

    missing_sequences = []
    generated = 0

    for protein in tqdm(proteins, desc="Embedding proteins"):
        out_path = os.path.join(PROTEIN_DIR, f"{protein}.npy")

        if os.path.exists(out_path):
            # Ensure metadata has the length recorded
            if protein not in metadata:
                arr = np.load(out_path, mmap_mode="r")
                metadata[protein] = {"length": int(arr.shape[0])}
                del arr
                save_metadata(metadata)
            continue

        sequence = sequences.get(protein)
        if not sequence:
            missing_sequences.append(protein)
            continue

        embeddings = embed_sequence(sequence, tokenizer, model, device)
        np.save(out_path, embeddings.astype(DTYPE))
        metadata[protein] = {
            "length": int(embeddings.shape[0]),
            "dtype": str(DTYPE),
            "model": MODEL_NAME,
        }
        save_metadata(metadata)
        generated += 1

        # Free up RAM aggressively
        del embeddings
        gc.collect()

    if missing_sequences:
        print("\n⚠️  Warning: Missing sequences for the following proteins (skipped):")
        print(", ".join(sorted(missing_sequences)))

    print(f"\nGenerated {generated} new protein embedding files.")
    print(f"Metadata stored at {METADATA_PATH}")


def main():
    print("=" * 90)
    print("Generating high-quality ESM2 embeddings (facebook/esm2_t33_650M_UR50D)")
    print("=" * 90)

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("Error: train.tsv or test.tsv not found under project_1/.")
        sys.exit(1)

    train_df = pd.read_csv(TRAIN_PATH, sep="\t")
    test_df = pd.read_csv(TEST_PATH, sep="\t")

    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")

    # Parse IDs once so we know which proteins we need embeddings for
    for df in (train_df, test_df):
        split_cols = df["id"].str.rsplit("_", n=2, expand=True)
        split_cols.columns = ["protein", "amino_acid", "position"]
        df[["protein", "amino_acid"]] = split_cols[["protein", "amino_acid"]]
        df["position"] = split_cols["position"].astype(np.int32) - 1

    required_proteins = collect_required_proteins(train_df, test_df)
    print(f"Unique proteins needed: {len(required_proteins):,}")

    # Load FASTA sequences
    print(f"\nLoading FASTA sequences from {FASTA_PATH} ...")
    sequences = load_fasta(FASTA_PATH)
    print(f"Total sequences available: {len(sequences):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.\n")

    ensure_embeddings(required_proteins, sequences, tokenizer, model, device)

    print("\nAll requested embeddings are now available under:")
    print(f"  {PROTEIN_DIR}")
    print("You can proceed to train the high-accuracy model (bilstm_predict.py).")
    print("=" * 90)


if __name__ == "__main__":
    main()

