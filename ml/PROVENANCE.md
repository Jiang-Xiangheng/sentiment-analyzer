# Provenance

This file links **code ↔ data ↔ model** so that any deployed model can be
traced back to the exact code and dataset that produced it.

## Traceability table

| Model / run | Training commit | Dataset version | DVC pointer | MLflow run | Artifact |
|-------------|-----------------|-----------------|-------------|------------|----------|
| Baseline (`distilbert-base-uncased`) | `be08c21f60086efb693ca956713bae72b7c99ee8` | `v2` | `data/raw/imdb_raw.csv.dvc` (`md5:8fc4498fad8451c4091cca0a4ff595de`) | `c2c55b733e804815ae404edd2e7d4a8a` | `best.ckpt` |
| Production (`bert-base-uncased`) | `be08c21f60086efb693ca956713bae72b7c99ee8` | `v2` | `data/raw/imdb_raw.csv.dvc` (`md5:8fc4498fad8451c4091cca0a4ff595de`) | `675f0d90823040ce9d667b03a2c9f69e` | `best.ckpt` |

> The two runs were produced from the working tree based on commit `be08c21`
> plus the MLflow wiring change in this branch. The commit SHA is logged as the
> `git_sha` tag on every run, and the full Hydra config is stored as the
> `config.yaml` artifact.

## How to answer "which dataset produced this model?"

1. Find the model's MLflow run and read the `git_sha` tag and
   `dataset_version` / `dvc_raw_md5` params.
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
