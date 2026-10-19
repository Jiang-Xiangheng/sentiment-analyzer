"""Inference - load a checkpoint and classify raw text."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils import configure_logging, get_logger

# Use the CN mirror unless the caller overrides it.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

log = get_logger(__name__)

#: Maximum token length. Must match ``ml/configs/data/default.yaml``.
MAX_LENGTH = 512

#: Fallback labels for a binary sentiment model (0 = negative, 1 = positive),
#: matching the convention used when the training CSV was built.
_BINARY_LABELS = {0: "negative", 1: "positive"}


@dataclass
class LoadedModel:
    """A checkpoint plus everything needed to run inference on raw text."""

    model: Any
    tokenizer: Any
    num_labels: int
    model_name: str

    @property
    def backbone(self) -> Any:
        """The underlying Hugging Face model (unwraps the Lightning module)."""
        inner = getattr(self.model, "model", None)
        return inner if inner is not None else self.model

    @property
    def id2label(self) -> dict[int, str]:
        """Map class index to a human-readable label.

        Prefers the label names stored in the backbone config; falls back to
        ``negative``/``positive`` for binary models and ``label_<i>`` otherwise.
        """
        config = getattr(self.backbone, "config", None)
        raw = getattr(config, "id2label", None)
        if raw and len(raw) == self.num_labels:
            mapping = {int(k): str(v) for k, v in raw.items()}
            # transformers fills in "LABEL_0", "LABEL_1", ... when the checkpoint
            # was trained without explicit label names.
            if not all(value.startswith("LABEL_") for value in mapping.values()):
                return mapping
        if self.num_labels == 2:
            return dict(_BINARY_LABELS)
        return {index: f"label_{index}" for index in range(self.num_labels)}


def load_model(checkpoint_path: str | Path) -> LoadedModel:
    """Load a Lightning checkpoint and rebuild the backbone from its hparams."""
    import torch
    from transformers import AutoTokenizer

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
    model = NLPModule.load_from_checkpoint(str(checkpoint_path), model=backbone)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info("model.loaded", model=model_name, num_labels=num_labels)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        num_labels=int(num_labels),
        model_name=str(model_name),
    )


def predict(model: LoadedModel, text: str, top_k: int | None = None) -> dict[str, Any]:
    """Classify one piece of text.

    Args:
        model: Object returned by :func:`load_model`.
        text: Raw input text.
        top_k: If given, keep only the ``top_k`` highest-scoring classes.

    Returns:
        ``{"pred": <class index>, "top_k": [{"label": ..., "score": ...}, ...]}``
        with scores sorted from most to least likely.
    """
    import torch

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    encoded = model.tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model.backbone(**encoded).logits

    probabilities = torch.softmax(logits, dim=-1)[0].tolist()

    id2label = model.id2label
    items = [
        {"label": id2label.get(index, str(index)), "score": float(score)}
        for index, score in enumerate(probabilities)
    ]
    items.sort(key=lambda item: item["score"], reverse=True)
    if top_k is not None:
        items = items[:top_k]

    pred = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    return {"pred": pred, "top_k": items}


def main() -> None:
    """CLI entrypoint: classify ``--text`` or the contents of ``--input``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", help="Text to classify.")
    parser.add_argument("--input", help="Path to a UTF-8 text file to classify.")
    args = parser.parse_args()

    if args.text is None and args.input is None:
        parser.error("one of --text or --input is required")

    configure_logging()
    model = load_model(args.checkpoint)
    text = args.text if args.text is not None else Path(args.input).read_text(encoding="utf-8")
    print(json.dumps(predict(model, text), indent=2))


if __name__ == "__main__":
    main()
