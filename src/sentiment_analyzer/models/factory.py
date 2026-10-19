"""Model factory - returns a torch.nn.Module by name."""

from __future__ import annotations

import os

from torch import nn

# Use the CN mirror unless the caller overrides it. Must be set before
# transformers/huggingface_hub is imported.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def build_model(name: str, num_labels: int) -> nn.Module:
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
