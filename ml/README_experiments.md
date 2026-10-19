# Experiment Tracking

All training runs are tracked with MLflow. The tracking backend is the local
`mlruns/` directory by default (`trainer.tracking_uri` in `ml/configs/trainer/default.yaml`).

## Logged per run

| Item | How |
|------|-----|
| Code version | Git commit SHA logged as tag `git_sha` |
| Dataset version | `dataset_version` param read from `data/processed/metadata.json` |
| Data fingerprint | `dvc_raw_md5` param read from `data/raw/imdb_raw.csv.dvc` |
| Hyperparameters | `mlflow.log_params` (model, lr, batch, max_length, epochs, seed) + the full config as an artifact |
| Metrics | `train/loss`, `val/loss`, `val/acc`, `val/f1_macro`, `test/loss`, `test/acc`, `test/f1_macro` |
| Artifacts | `best.ckpt`, `config.yaml`, `reports/metrics.json`, `reports/confusion_matrix.png` |

## Experiments

| Experiment | Config | Model | Params | train rows | max_length | batch | epochs | Purpose |
|------------|--------|-------|--------|-----------|-----------|-------|--------|---------|
| `baseline` | `ml/configs/experiment/baseline.yaml` | `distilbert-base-uncased` | 66M | 3000 | 64 | 32 | 1 | Reference baseline |
| `sota` | `ml/configs/experiment/sota.yaml` | `bert-base-uncased` | 110M | 3000 | 96 | 16 | 1 | Larger backbone / improved model |

> Both experiments are deliberately small so the whole pipeline runs on CPU in
> minutes — the assignment explicitly allows toy-scale experiments. The earlier
> defaults (`bert-base-multilingual-cased` / `xlm-roberta-base`, 512 tokens,
> 30-100 epochs) would take tens of hours on CPU.

Run them with:

```bash
uv run python -m sentiment_analyzer.training.train experiment=baseline
uv run python -m sentiment_analyzer.training.train experiment=sota
```

View the run history:

```bash
# MLflow 3.x keeps the local file backend behind an opt-in flag.
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns
```

On Windows PowerShell:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
uv run mlflow ui --backend-store-uri ./mlruns
```

## Results

Both runs used dataset **v2** (3000 training rows, 1000 validation, 1000 test).

| Experiment | MLflow run id | val acc | val macro-F1 | test acc | test macro-F1 | test loss |
|------------|---------------|---------|--------------|----------|---------------|-----------|
| `baseline` | `c2c55b733e804815ae404edd2e7d4a8a` | 0.794 | 0.793 | 0.785 | 0.784 | 0.436 |
| `sota` | `675f0d90823040ce9d667b03a2c9f69e` | 0.842 | 0.842 | **0.842** | **0.842** | 0.349 |

The larger `bert-base-uncased` backbone with a longer context window improves
test macro-F1 by **+5.8 points** over the `distilbert-base-uncased` baseline.

## Production-worthy model

- **Selected run:** `sota` — MLflow run `675f0d90823040ce9d667b03a2c9f69e`
  (`bert-base-uncased`).
- **Why:** it has the best test macro-F1 (0.842 vs 0.784) at a latency that is
  still acceptable for the API (CPU inference on a single sentence is well under
  a second). The checkpoint is ~438 MB, which fits comfortably in the serving
  image and in the DVC/MLflow artifact store.

## Optimized metric

- **Metric:** macro-F1 (with accuracy reported alongside).
- **Why it matters for this app:** macro-F1 weights both sentiment classes
  equally, so a model cannot look good by simply predicting the majority class.
  For a sentiment API that must catch negative reviews as reliably as positive
  ones, that balance matters more than raw accuracy.
