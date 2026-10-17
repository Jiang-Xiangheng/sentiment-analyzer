# Provenance

This file links **code ↔ data ↔ model** so that any deployed model can be
traced back to the exact code and dataset that produced it.

## Traceability table

| Model / run | Training commit | Dataset version | DVC pointer | MLflow run | Artifact |
|-------------|-----------------|-----------------|-------------|------------|----------|
| Baseline | `<commit-sha>` | `v1` (`data-v1`) | `data/raw.dvc` | `<run-id>` | `artifacts/checkpoints/best.ckpt` |
| Production | `<commit-sha>` | `v2` (`data-v2`) | `data/raw.dvc` | `<run-id>` | `artifacts/checkpoints/best.ckpt` |

## How to answer "which dataset produced this model?"

1. Find the model's MLflow run and read the `git_sha` tag and
   `dataset_version` param.
2. Check out that commit:
   ```bash
   git checkout <commit-sha>
   ```
3. Restore the matching data:
   ```bash
   dvc pull
   ```
4. The `.dvc` file and `dvc.lock` at that commit record the exact content
   hashes of the inputs and outputs.

## How to reproduce a model

```bash
git checkout <commit-sha>
dvc pull
dvc repro train
```
