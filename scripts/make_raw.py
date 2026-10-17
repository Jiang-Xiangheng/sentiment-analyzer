"""Download a deterministic IMDB subset -> data/raw/imdb_raw.csv.

Run ONCE (or to mint a new dataset version) BEFORE `dvc repro`.
Requires: pip install datasets pandas
"""
from __future__ import annotations

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path

import pandas as pd
from datasets import load_dataset

SEED = 42
N_RAW = 7000  # after prepare this yields ~5000/1000/1000
RAW_PATH = Path("data/raw/imdb_raw.csv")


def main() -> None:
    ds = load_dataset("stanfordnlp/imdb")
    rows = [
        {"text": ex["text"], "label": int(ex["label"])}
        for split in ("train", "test")
        for ex in ds[split]
    ]
    df = pd.DataFrame(rows).sample(n=N_RAW, random_state=SEED).reset_index(drop=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"Wrote {len(df)} raw rows -> {RAW_PATH}")


if __name__ == "__main__":
    main()
