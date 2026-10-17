# Experiment Tracking

All training runs are tracked with MLflow. The tracking backend is the local
`mlruns/` directory by default (`trainer.tracking_uri` in `ml/configs/trainer/default.yaml`).

## Logged per run

| Item | How |
|------|-----|
| Code version | Git commit SHA logged as tag `git_sha` |
| Dataset version | DVC version / tag logged as param `dataset_version` |
| Hyperparameters | Logged automatically by the Lightning `MLFlowLogger` |
| Metrics | Accuracy / F1 / loss logged by the Lightning module |
| Artifacts | Model checkpoint (`artifacts/checkpoints/best.ckpt`), plots, metrics |

## Experiments

| Experiment | Config | Model | Purpose |
|------------|--------|-------|---------|
| `baseline` | `ml/configs/experiment/baseline.yaml` | `bert-base-multilingual-cased` | Reference baseline |
| `sota` | `ml/configs/experiment/sota.yaml` | `xlm-roberta-base` | Improved / tuned model |

Run them with:

```bash
python -m sentiment_analyzer.training.train experiment=baseline
python -m sentiment_analyzer.training.train experiment=sota
```

View the run history:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Production-worthy model

<!-- TODO: fill in after running both experiments. -->

- **Selected run:** `<experiment name + MLflow run id>`
- **Why:** `<e.g. higher macro-F1 on the validation/test split while keeping
  inference latency acceptable for the API>`

## Optimized metric

<!-- TODO -->

- **Metric:** `<accuracy / macro-F1 / ...>`
- **Why it matters for this app:** `<e.g. macro-F1 treats both sentiment
  classes equally, avoiding a model that simply predicts the majority class>`
