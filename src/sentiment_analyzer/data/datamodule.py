"""Lightning DataModule."""
from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, random_split

