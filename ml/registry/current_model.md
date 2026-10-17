# Current Production Model

<!-- Update this file whenever a new model is promoted. -->

| Field | Value |
|-------|-------|
| Model | `<e.g. xlm-roberta-base fine-tuned for sentiment>` |
| MLflow experiment | `sota` |
| MLflow run ID | `<run-id>` |
| Dataset version | `v2` (`data-v2`) |
| Training commit | `<git-sha>` |
| Artifact | `artifacts/checkpoints/best.ckpt` |
| Serving endpoint | `POST /classify` |
| Promoted at | `<YYYY-MM-DD>` |

## Validation metrics

| Metric | Value |
|--------|-------|
| Accuracy | `<...>` |
| Macro F1 | `<...>` |
| Loss | `<...>` |

## Why this model is in production

`<Short explanation: best validation metric, acceptable latency, stable
behaviour on edge cases.>`

## How to load

```bash
MODEL_PATH=artifacts/checkpoints/best.ckpt \
  uvicorn sentiment_analyzer.serving.main:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this movie"}'
```
