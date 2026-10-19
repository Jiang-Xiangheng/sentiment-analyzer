"""Dataset implementations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset


class TextDataset(Dataset):
    """CSV-backed text classification dataset.

    Tokenisation is cached per item, so long reviews are only tokenised once
    instead of once per epoch. This is the difference between minutes and
    hours when training on CPU.
    """

    def __init__(
        self,
        csv_path: Path | str,
        text_col: str = "text",
        label_col: str = "label",
        tokenizer: Callable | None = None,
        max_length: int = 512,
    ) -> None:
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        self.text_col = text_col
        self.label_col = label_col
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        if idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        item = {"text": str(row[self.text_col]), "label": int(row[self.label_col])}
        if self.tokenizer is not None:
            enc = self.tokenizer(
                item["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item.update({k: v.squeeze(0) for k, v in enc.items()})
            self._cache[idx] = item
        return item
