"""Download a deterministic IMDB subset -> data/raw/imdb_raw.csv.

Run to mint a new dataset version BEFORE `dvc repro`.
Requires: pip install datasets pandas (nltk optional, for synonym augmentation)

Dataset versions
----------------
v1: SEED=42, N_BASE=7000, no augmentation.
v2: SEED=1337, N_BASE=10000, + deterministic light augmentation
    (character dropout, WordNet synonym replacement, ending-punctuation
    variation). Augmented rows are label-preserving and reproducible
    (each row seeded by (SEED, row_index)).
"""

from __future__ import annotations

import json
import os
import random
import re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path

import pandas as pd
from datasets import load_dataset

SEED = 1337
N_BASE = 10000  # real IMDB rows sampled with the new seed
AUG_FRACTION = 0.4  # fraction of base rows that get an augmented copy
RAW_PATH = Path("data/raw/imdb_raw.csv")
MANIFEST_PATH = Path("data/raw/raw_manifest.json")

# Common English stopwords (lowercased) — never replaced during synonym aug.
STOPWORDS = set(
    """
    a an the and or but if while of to in on at by for with without from into as is are
    was were be been being this that these those it its he she they we you i him her them
    his their our your my me us not no do does did have has had will would can could should
    may might must about up down out off over under again more most very just only also too
    """.split()
)


def _replace_core(tok: str, rng: random.Random) -> str:
    """Replace the alphabetic core of a token with a WordNet synonym, preserving
    any surrounding punctuation. Falls back to the original token when nltk/wordnet
    is unavailable or no synonym is found."""
    try:
        from nltk.corpus import wordnet

        i, j = 0, len(tok)
        while i < j and not tok[i].isalnum():
            i += 1
        while j > i and not tok[j - 1].isalnum():
            j -= 1
        core = tok[i:j]
        if len(core) <= 3 or core.lower() in STOPWORDS:
            return tok
        if rng.random() >= 0.12:
            return tok

        lemmas: set[str] = set()
        for syn in wordnet.synsets(core):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != core.lower():
                    lemmas.add(name)
        if not lemmas:
            return tok
        repl = rng.choice(sorted(lemmas))
        return tok[:i] + repl + tok[j:]
    except Exception:
        # nltk/wordnet unavailable (e.g. offline) — skip synonym step.
        return tok


def _augment_text(text: str, rng: random.Random) -> str:
    """Deterministic, label-preserving light augmentation.

    Note: prepare.py's `_clean` normalises all whitespace, so we avoid
    whitespace-only noise and instead perturb characters / words / ending
    punctuation — those survive cleaning.
    """
    tokens = text.split()
    if not tokens:
        return text

    # 1) character dropout (low probability, only on longer tokens)
    out = []
    for tok in tokens:
        if len(tok) > 3 and rng.random() < 0.03:
            idx = rng.randrange(len(tok))
            tok = tok[:idx] + tok[idx + 1 :]
        out.append(tok)
    text = " ".join(out)

    # 2) ending-punctuation variation (survives _clean)
    stripped = text.rstrip(".!?")
    if stripped != text and rng.random() < 0.5:
        text = stripped + rng.choice([".", "!", "?"])
    elif not text.endswith((".", "!", "?")) and rng.random() < 0.2:
        text = text + rng.choice([".", "!", "?"])

    # 3) WordNet synonym replacement
    text = " ".join(_replace_core(tok, rng) for tok in text.split())
    return text


def main() -> None:
    # Best-effort download of WordNet corpora; if it fails we simply skip
    # the synonym step (graceful fallback).
    try:
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception:
        pass

    ds = load_dataset("stanfordnlp/imdb")
    rows = [
        {"text": ex["text"], "label": int(ex["label"])}
        for split in ("train", "test")
        for ex in ds[split]
    ]
    base = pd.DataFrame(rows).sample(n=N_BASE, random_state=SEED).reset_index(drop=True)

    n_to_aug = int(N_BASE * AUG_FRACTION)
    idx_rng = random.Random(SEED)
    chosen = set(idx_rng.sample(range(len(base)), n_to_aug))

    aug_rows = []
    for i in chosen:
        src = base.iloc[i]
        rng = random.Random(SEED * 1_000_003 + i)
        aug_text = _augment_text(str(src["text"]), rng)
        if aug_text and aug_text.strip():
            aug_rows.append({"text": aug_text, "label": int(src["label"])})

    df = pd.concat([base, pd.DataFrame(aug_rows)], ignore_index=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)

    manifest = {
        "dataset_version": "v2",
        "seed": SEED,
        "n_base": len(base),
        "n_augmented": len(aug_rows),
        "techniques": [
            "character_dropout",
            "wordnet_synonym_replacement",
            "ending_punctuation_variation",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(
        f"Wrote {len(df)} raw rows (base={len(base)}, aug={len(aug_rows)}) -> {RAW_PATH}"
    )
    print(f"Manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
