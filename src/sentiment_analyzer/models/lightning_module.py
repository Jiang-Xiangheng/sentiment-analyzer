"""Lightning module wrappers."""
from __future__ import annotations

import lightning as L
import torch
from torch import nn, optim

from torchmetrics.classification import MulticlassF1Score


class NLPModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_labels: int,
        lr: float = 2e-5,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.val_f1 = MulticlassF1Score(num_classes=num_labels, average="macro")
        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs = {k: v for k, v in batch.items() if k not in ("text", "label")}
        out = self.model(**inputs, labels=batch["label"])
        self.log("train/loss", out.loss, prog_bar=True, on_epoch=True)
        return out.loss

    def validation_step(self, batch, batch_idx: int) -> None:
        inputs = {k: v for k, v in batch.items() if k not in ("text", "label")}
        out = self.model(**inputs, labels=batch["label"])
        self.val_f1(out.logits, batch["label"])
        self.log("val/loss", out.loss, prog_bar=True)
        self.log("val/f1_macro", self.val_f1, prog_bar=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)  # type: ignore[attr-defined]
