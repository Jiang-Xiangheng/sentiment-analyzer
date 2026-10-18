"""Training entrypoint (Hydra-powered)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ..utils import configure_logging, get_logger, seed_everything

log = get_logger(__name__)


def _git_sha() -> str:
    """Return the current git commit SHA, or ``"unknown"`` when unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _dataset_version(processed_dir: str | Path) -> str:
    """Read the dataset version written by the prepare stage."""
    metadata = Path(processed_dir) / "metadata.json"
    if not metadata.exists():
        return "unknown"
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        return str(payload.get("dataset_version", "unknown"))
    except (OSError, ValueError):
        return "unknown"


def _collect_metrics(trainer, cfg: DictConfig) -> dict:
    """Collect scalar metrics from the Lightning trainer."""
    metrics: dict = {}
    for key, value in trainer.callback_metrics.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    metrics["model_name"] = str(cfg.model.name)
    metrics["num_labels"] = int(cfg.model.num_labels)
    return metrics


def _write_metrics(metrics: dict, reports_dir: Path) -> Path:
    """Write metrics to ``reports/metrics.json`` (declared in dvc.yaml)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _log_mlflow(
    lightning_logger,
    run_id: str | None,
    git_sha: str,
    dataset_version: str,
    artifacts: list[Path],
) -> None:
    """Attach provenance and artifacts to the active MLflow run."""
    if not run_id:
        return
    client = lightning_logger.experiment
    client.set_tag(run_id, "git_sha", git_sha)
    client.log_param(run_id, "dataset_version", dataset_version)
    for artifact in artifacts:
        if artifact.exists():
            client.log_artifact(run_id, str(artifact))


@hydra.main(version_base=None, config_path="../../../ml/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.get("log_level", "INFO"))
    seed_everything(cfg.get("seed", 42))
    log.info("train.start", config=OmegaConf.to_container(cfg, resolve=True))

    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from ..data import TextDataset
    from ..models import NLPModule, build_model

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    train_ds = TextDataset(cfg.data.train_csv, tokenizer=tok, max_length=cfg.data.max_length)
    val_ds = TextDataset(cfg.data.val_csv, tokenizer=tok, max_length=cfg.data.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size)
    net = build_model(cfg.model.name, num_labels=cfg.model.num_labels)
    lit = NLPModule(
        net,
        num_labels=cfg.model.num_labels,
        lr=cfg.model.lr,
        model_name=cfg.model.name,
    )

    out_dir = Path(cfg.trainer.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir / "checkpoints",
            filename="best",
            monitor=cfg.trainer.monitor,
            mode=cfg.trainer.monitor_mode,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor=cfg.trainer.monitor,
            mode=cfg.trainer.monitor_mode,
            patience=cfg.trainer.patience,
        ),
    ]
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.experiment_name, tracking_uri=cfg.trainer.tracking_uri
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

    git_sha = _git_sha()
    dataset_version = _dataset_version(Path(cfg.data.train_csv).parent)
    metrics = _collect_metrics(trainer, cfg)
    metrics_path = _write_metrics(metrics, Path("reports"))
    ckpt_path = out_dir / "checkpoints" / "best.ckpt"

    _log_mlflow(
        mlflow_logger,
        mlflow_logger.run_id,
        git_sha,
        dataset_version,
        [metrics_path, ckpt_path],
    )
    log.info(
        "train.done",
        ckpt=str(ckpt_path),
        metrics=str(metrics_path),
        git_sha=git_sha,
        dataset_version=dataset_version,
    )


if __name__ == "__main__":
    main()
