"""Lightning module wrappers."""

from __future__ import annotations

import lightning as L
import torch
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class NLPModule(L.LightningModule):
    """Thin Lightning wrapper around a Hugging Face sequence classifier."""

    def __init__(
        self,
        model: nn.Module,
        num_labels: int,
        lr: float = 2e-5,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.val_acc = MulticlassAccuracy(num_classes=num_labels)
        self.val_f1 = MulticlassF1Score(num_classes=num_labels, average="macro")
        self.test_acc = MulticlassAccuracy(num_classes=num_labels)
        self.test_f1 = MulticlassF1Score(num_classes=num_labels, average="macro")
        self.save_hyperparameters(ignore=["model"])

    def _forward(self, batch: dict):
        inputs = {k: v for k, v in batch.items() if k not in ("text", "label")}
        return self.model(**inputs, labels=batch["label"])

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self._forward(batch)
        self.log("train/loss", out.loss, prog_bar=True, on_epoch=True)
        return out.loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        out = self._forward(batch)
        self.val_acc(out.logits, batch["label"])
        self.val_f1(out.logits, batch["label"])
        self.log("val/loss", out.loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/f1_macro", self.val_f1, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        out = self._forward(batch)
        self.test_acc(out.logits, batch["label"])
        self.test_f1(out.logits, batch["label"])
        self.log("test/loss", out.loss)
        self.log("test/acc", self.test_acc, prog_bar=True)
        self.log("test/f1_macro", self.test_f1, prog_bar=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)  # type: ignore[attr-defined]
