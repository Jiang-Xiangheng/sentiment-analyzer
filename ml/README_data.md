# Data Documentation

This document describes the training data used by the sentiment analysis
application, how it is versioned with DVC, and how code commits map to
dataset versions.

## 1. Source

| Field | Value |
|-------|-------|
| Dataset | `<e.g. IMDB Movie Reviews>` |
| Source URL | `<https://...>` |
| License | `<license>` |
| Raw location | `data/raw/` |
| Storage | DVC remote (Azure Blob Storage) |
| Task | Binary sentiment classification |
| Schema | `text` (string), `label` (int: `0` = negative, `1` = positive) |

> Raw data is **not** committed to Git. Only the small `.dvc` pointer files
> live in the repository; the actual files live in the DVC Azure remote.

## 2. Dataset versions

At least two versions are maintained.

| Version | Git tag | Commit | DVC pointer | Description |
|---------|---------|--------|-------------|-------------|
| v1 | `data-v1` | `<commit-sha>` | `data/raw.dvc` | First usable dataset used for the baseline model. |
| v2 | `data-v2` | `<commit-sha>` | `data/raw.dvc` | Improved version (see changes below). |

## 3. v1 → v2 changes

Describe what changed between the two versions:

- `<removed N duplicate / outlier samples>`
- `<balanced the classes by ...>`
- `<added synthetic / augmented samples ...>`
- `<changed the train/val/test split ratio to ...>`
- `<fixed label noise in ...>`

## 4. Reproducing a dataset version

```bash
# Restore the tracked pointer for a given code commit
git checkout <commit-sha>
dvc pull

# Create and publish a new version
dvc add data/raw
git add data/raw.dvc .dvc/config
git commit -m "data: add v2 dataset"
git tag data-v2
dvc push
```

## 5. Processed data

The `prepare` DVC stage turns the raw data into the processed splits used by
training:

```
data/raw/            ->  data/processed/train.csv
                         data/processed/val.csv
                         data/processed/test.csv
```

Run it with:

```bash
dvc repro prepare
# or
python -m sentiment_analyzer.data.prepare --raw data/raw --out data/processed
```
