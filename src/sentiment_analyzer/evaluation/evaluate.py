"""Evaluation - run a checkpoint on the test split and write metrics + confusion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger

log = get_logger(__name__)


def collect_predictions(model, dataloader) -> tuple[list[int], list[int]]:
    """Run ``model`` over ``dataloader`` and return ``(y_true, y_pred)``."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("text", "label")}
            logits = model(**inputs).logits
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
    return y_true, y_pred


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Accuracy + macro-F1 without requiring scikit-learn at runtime."""
    n = len(y_true)
    if n == 0:
        return {"n_test": 0, "accuracy": 0.0, "f1_macro": 0.0}

    correct = sum(int(t == p) for t, p in zip(y_true, y_pred, strict=True))
    labels = sorted(set(y_true) | set(y_pred))
    f1_scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return {
        "n_test": n,
        "accuracy": correct / n,
        "f1_macro": sum(f1_scores) / len(f1_scores),
    }


def save_confusion_matrix(y_true: list[int], y_pred: list[int], out_path: Path) -> bool:
    """Write a confusion-matrix PNG. Returns False when matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    labels = sorted(set(y_true) | set(y_pred))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred, strict=True):
        matrix[index[true], index[pred]] += 1

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Directory containing test.csv")
    parser.add_argument("--out", default="reports/metrics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    configure_logging()
    log.info("evaluate.start", checkpoint=args.checkpoint, data=args.data)

    from torch.utils.data import DataLoader

    from ..data import TextDataset
    from ..inference.predict import load_model

    loaded = load_model(args.checkpoint)
    test_csv = Path(args.data) / "test.csv"
    dataset = TextDataset(test_csv, tokenizer=loaded.tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    y_true, y_pred = collect_predictions(loaded.backbone, loader)
    metrics = compute_metrics(y_true, y_pred)
    metrics["checkpoint"] = str(args.checkpoint)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))

    cm_path = out.parent / "confusion_matrix.png"
    saved = save_confusion_matrix(y_true, y_pred, cm_path)

    log.info(
        "evaluate.done",
        metrics=metrics,
        confusion_matrix=str(cm_path) if saved else None,
    )


if __name__ == "__main__":
    main()
