"""Training entrypoint (Hydra-powered, MLflow-tracked)."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ..utils import configure_logging, get_logger, seed_everything

# Windows consoles default to GBK; rich (used by Lightning's progress bar)
# writes characters such as "•" that GBK cannot encode. Force UTF-8 on the
# standard streams so a long run does not crash while tearing the bar down.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Model weights are pulled from Hugging Face; use the CN mirror unless the
# caller explicitly overrides it.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# MLflow 3.x puts the local file store behind an opt-in flag. The course
# project explicitly allows a local file backend, so enable it.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

log = get_logger(__name__)

REPORTS_DIR = Path("reports")
METADATA_PATH = Path("data/processed/metadata.json")
RAW_DVC_PATH = Path("data/raw/imdb_raw.csv.dvc")


def _git_sha() -> str:
    """Return the current commit SHA, or 'unknown' outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_dvc_md5(path: Path) -> str:
    """Extract the md5 recorded in a ``.dvc`` pointer file."""
    try:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        outs = data.get("outs", []) if isinstance(data, dict) else []
        return str(outs[0].get("md5", "unknown")) if outs else "unknown"
    except Exception:
        return "unknown"


@hydra.main(version_base=None, config_path="../../../ml/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.get("log_level", "INFO"))
    seed_everything(cfg.get("seed", 42))
    log.info("train.start", config=OmegaConf.to_container(cfg, resolve=True))

    import lightning as L
    import mlflow
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoTokenizer

    from ..data import TextDataset
    from ..evaluation.evaluate import (
        collect_predictions,
        compute_metrics,
        save_confusion_matrix,
    )
    from ..models import NLPModule, build_model

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    train_ds = TextDataset(cfg.data.train_csv, tokenizer=tok, max_length=cfg.data.max_length)
    val_ds = TextDataset(cfg.data.val_csv, tokenizer=tok, max_length=cfg.data.max_length)
    test_ds = TextDataset(cfg.data.test_csv, tokenizer=tok, max_length=cfg.data.max_length)

    # Optional toy-scale subsample, to keep CPU experiments in the minutes range.
    max_train = cfg.data.get("max_train_samples")
    if max_train and int(max_train) < len(train_ds):
        indices = random.Random(cfg.seed).sample(range(len(train_ds)), int(max_train))
        train_ds = Subset(train_ds, indices)

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size)

    net = build_model(cfg.model.name, num_labels=cfg.model.num_labels)
    lit = NLPModule(
        net,
        num_labels=cfg.model.num_labels,
        lr=cfg.model.lr,
        model_name=cfg.model.name,
    )

    out_dir = Path(cfg.trainer.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Drop checkpoints from a previous experiment so `best.ckpt` always
    # belongs to the current run (Lightning would otherwise write `best-v1.ckpt`).
    for stale in ckpt_dir.glob("best*.ckpt"):
        stale.unlink()

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor=cfg.trainer.monitor,
            mode=cfg.trainer.monitor_mode,
            save_top_k=1,
            save_weights_only=True,
        ),
        EarlyStopping(
            monitor=cfg.trainer.monitor,
            mode=cfg.trainer.monitor_mode,
            patience=cfg.trainer.patience,
        ),
    ]

    metadata = _read_json(METADATA_PATH)

    mlflow.set_tracking_uri(cfg.trainer.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=f"{cfg.experiment_name}-{cfg.model.name}") as run:
        # --- provenance: code + data -------------------------------------
        mlflow.set_tag("git_sha", _git_sha())
        mlflow.set_tag("python", platform.python_version())
        mlflow.set_tag("torch", torch.__version__)
        mlflow.log_param("dataset_version", metadata.get("dataset_version", "unknown"))
        mlflow.log_param("dvc_raw_md5", _read_dvc_md5(RAW_DVC_PATH))
        mlflow.log_param("train_rows", len(train_ds))
        mlflow.log_param("val_rows", len(val_ds))
        mlflow.log_param("test_rows", len(test_ds))

        # --- hyperparameters ---------------------------------------------
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "num_labels": cfg.model.num_labels,
                "lr": cfg.model.lr,
                "batch_size": cfg.data.batch_size,
                "max_length": cfg.data.max_length,
                "max_epochs": cfg.trainer.max_epochs,
                "seed": cfg.seed,
            }
        )
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")

        # --- train, reusing the run created above ------------------------
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.experiment_name,
            tracking_uri=cfg.trainer.tracking_uri,
            run_id=run.info.run_id,
        )
        trainer = L.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            callbacks=callbacks,
            logger=mlflow_logger,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            deterministic="warn",
        )
        trainer.fit(lit, train_loader, val_loader)
        best_path = Path(getattr(trainer.checkpoint_callback, "best_model_path", "") or "")
        ckpt_path = str(best_path) if best_path and best_path.exists() else "best"
        trainer.test(lit, test_loader, ckpt_path=ckpt_path)

        # --- artifacts ----------------------------------------------------
        y_true, y_pred = collect_predictions(lit.model, test_loader)
        metrics = compute_metrics(y_true, y_pred)
        metrics_path = REPORTS_DIR / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        cm_path = REPORTS_DIR / "confusion_matrix.png"
        has_cm = save_confusion_matrix(y_true, y_pred, cm_path)

        if best_path.exists():
            mlflow.log_artifact(str(best_path))
        mlflow.log_artifact(str(metrics_path))
        if has_cm:
            mlflow.log_artifact(str(cm_path))

        log.info(
            "train.done",
            ckpt=str(best_path),
            metrics=metrics,
            run_id=run.info.run_id,
        )


if __name__ == "__main__":
    main()
