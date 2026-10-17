"""Inference CLI - load a checkpoint and predict on input(s)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger

log = get_logger(__name__)


def load_model(checkpoint_path: str | Path):
    """Load a Lightning module from checkpoint, rebuilding the backbone from hparams."""
    import torch

    from ..models import build_model
    from ..models import NLPModule

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    model_name = hp.get("model_name")
    num_labels = hp.get("num_labels")
    if model_name is None or num_labels is None:
        raise ValueError(
            "Checkpoint missing model_name/num_labels hparams - "
            "re-train after upgrading NLPModule."
        )
    backbone = build_model(model_name, num_labels=num_labels)
    return NLPModule.load_from_checkpoint(str(checkpoint_path), model=backbone)


def predict(model, input_path: str | Path):
    """Run a single prediction. Returns a task-specific result dict."""
    raise NotImplementedError("Override predict() per project")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    configure_logging()
    model = load_model(args.checkpoint)
    result = predict(model, args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
