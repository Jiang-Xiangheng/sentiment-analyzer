"""Inference CLI - load a checkpoint and predict on input(s)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger

log = get_logger(__name__)

MAX_LENGTH = 512
FALLBACK_ID2LABEL = {0: "negative", 1: "positive"}


def load_model(checkpoint_path: str | Path):
    """Load a Lightning module from checkpoint, rebuilding the backbone from hparams."""
    import torch

    from ..models import NLPModule, build_model

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    model_name = hp.get("model_name")
    num_labels = hp.get("num_labels")
    if model_name is None or num_labels is None:
        raise ValueError(
            "Checkpoint missing model_name/num_labels hparams - re-train after upgrading NLPModule."
        )
    backbone = build_model(model_name, num_labels=num_labels)
    return NLPModule.load_from_checkpoint(str(checkpoint_path), model=backbone)


def predict(model, text: str) -> dict:
    """Classify the sentiment of a single piece of text.

    Returns a dict matching ``ClassificationResponse``: ``{"pred": int,
    "top_k": [{"label": str, "score": float}, ...]}``.
    """
    import torch
    from transformers import AutoTokenizer

    model_name = getattr(model.hparams, "model_name", None)
    if not model_name:
        raise ValueError("Checkpoint missing model_name hparam")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    encoded = tokenizer(
        str(text),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model.model(**encoded).logits
        probs = logits.softmax(-1).squeeze(0)

    pred = int(probs.argmax())
    id2label = getattr(model.model.config, "id2label", None) or {}

    top_k: list[dict] = []
    for idx in probs.argsort(descending=True).tolist()[: min(2, probs.numel())]:
        label = id2label.get(int(idx))
        if not label or str(label).startswith("LABEL_"):
            label = FALLBACK_ID2LABEL.get(int(idx), str(idx))
        top_k.append({"label": str(label), "score": float(probs[int(idx)])})

    return {"pred": pred, "top_k": top_k}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Text, or a path to a text file")
    args = parser.parse_args()
    configure_logging()
    model = load_model(args.checkpoint)
    try:
        path = Path(args.input)
        text = path.read_text(encoding="utf-8") if path.is_file() else args.input
    except OSError:
        text = args.input
    result = predict(model, text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
