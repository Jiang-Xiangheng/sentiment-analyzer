"""Model factory - returns a torch.nn.Module by name."""
from __future__ import annotations

from torch import nn

def build_model(name: str, num_labels: int) -> nn.Module:
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
