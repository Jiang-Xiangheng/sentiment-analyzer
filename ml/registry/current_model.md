# Current Production Model

<!-- Update this file whenever a new model is promoted. -->

| Field | Value |
|-------|-------|
| Model | `bert-base-uncased` fine-tuned for binary sentiment |
| MLflow experiment | `sota` |
| MLflow run ID | `675f0d90823040ce9d667b03a2c9f69e` |
| Dataset version | `v2` (DVC raw md5 `8fc4498fad8451c4091cca0a4ff595de`) |
| Training commit | `be08c21f60086efb693ca956713bae72b7c99ee8` (plus the MLflow wiring change) |
| Artifact | MLflow artifact `best.ckpt` of run `675f0d90823040ce9d667b03a2c9f69e` (`artifacts/checkpoints/best.ckpt` when reproduced locally) |
| Serving endpoint | `POST /classify` |
| Promoted at | 2026-10-18 |

## Validation metrics

| Metric | Value |
|--------|-------|
| Test accuracy | 0.842 |
| Test macro F1 | 0.842 |
| Test loss | 0.349 |

## Why this model is in production

Best test macro-F1 of the two tracked experiments (0.842 vs 0.784 for the
`distilbert-base-uncased` baseline), with CPU inference latency well under one
second per request and a checkpoint size that fits comfortably in the serving
image.

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
