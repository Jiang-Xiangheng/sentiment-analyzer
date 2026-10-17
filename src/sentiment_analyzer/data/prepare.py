
"""Data preparation stage: raw -> cleaned, label-encoded, split CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils import get_logger

log = get_logger(__name__)

SEED = 42
RAW_FILENAME = "imdb_raw.csv"
N_TRAIN, N_VAL, N_TEST = 5000, 1000, 1000


def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("<br />", " ").split())


def prepare_data(raw_dir: Path | str, processed_dir: Path | str) -> None:
    raw, out = Path(raw_dir), Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = raw / RAW_FILENAME
    if not raw_path.exists():
        raise FileNotFoundError(f"No raw at {raw_path}; run `python scripts/make_raw.py` first.")

    log.info("prepare.start", raw=str(raw_path))
    df = pd.read_csv(raw_path)[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).map(_clean)
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"]).reset_index(drop=True)

    tv, test = train_test_split(df, test_size=N_TEST, stratify=df["label"], random_state=SEED)
    train, val = train_test_split(tv, test_size=N_VAL, stratify=tv["label"], random_state=SEED)

    for name, split in (("train", train), ("val", val), ("test", test)):
        split.to_csv(out / f"{name}.csv", index=False)
        log.info("prepare.wrote", name=name, rows=len(split))

    (out / "report.json").write_text(json.dumps({
        "seed": SEED,
        "splits": {k: len(v) for k, v in (("train", train), ("val", val), ("test", test))},
        "label_dist": {k: v["label"].value_counts().to_dict() for k, v in (("train", train), ("val", val), ("test", test))},
    }, indent=2))
    log.info("prepare.done")


if __name__ ==  "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    prepare_data(a.raw, a.out)

