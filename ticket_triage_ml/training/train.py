"""Training orchestration module."""

from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from ticket_triage_ml.training.datamodule import TicketDataModule
from ticket_triage_ml.training.model import MultiTaskTicketClassifier
from ticket_triage_ml.utils.git import get_git_commit_id
from ticket_triage_ml.utils.logging import create_plots, log_config_to_mlflow, setup_mlflow
from ticket_triage_ml.utils.paths import get_project_root
from ticket_triage_ml.utils.seeding import set_seed


class MetricsCallback(pl.Callback):
    """Callback to collect metrics for plotting."""

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()
        self.train_losses: list = []
        self.val_losses: list = []
        self.train_f1_scores: list = []
        self.val_f1_scores: list = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Collect training metrics at epoch end.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module.
        """
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.train_losses.append(float(metrics["train_loss"]))
        if "train_f1_macro" in metrics:
            self.train_f1_scores.append(float(metrics["train_f1_macro"]))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Collect validation metrics at epoch end.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module.
        """
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.val_losses.append(float(metrics["val_loss"]))
        if "val_f1_macro" in metrics:
            self.val_f1_scores.append(float(metrics["val_f1_macro"]))


def train_model(cfg: DictConfig) -> Dict[str, Any]:
    """Train the multi-task ticket classifier.

    Args:
        cfg: Hydra configuration.

    Returns:
        Dictionary with training results and paths.
    """
    set_seed(cfg.seed)

    project_root = get_project_root()
    checkpoint_dir = project_root / cfg.train.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mlflow_run = setup_mlflow(cfg)
    git_commit = get_git_commit_id()

    if mlflow_run:
        log_config_to_mlflow(cfg, git_commit)

    logger.info("Setting up data module...")
    datamodule = TicketDataModule(cfg)
    datamodule.setup("fit")

    logger.info(f"Number of topics: {datamodule.num_topics}")
    logger.info(f"Number of priorities: {datamodule.num_priorities}")

    logger.info("Creating model...")
    model = MultiTaskTicketClassifier(
        cfg=cfg,
        num_topics=datamodule.num_topics,
        num_priorities=datamodule.num_priorities,
        class_weights_topic=datamodule.class_weights_topic,
        class_weights_priority=datamodule.class_weights_priority,
    )

    callbacks = _create_callbacks(cfg, checkpoint_dir)
    metrics_callback = MetricsCallback()
    callbacks.append(metrics_callback)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    best_checkpoint_path = _get_best_checkpoint(callbacks)
    logger.info(f"Best checkpoint: {best_checkpoint_path}")

    tokenizer_path = project_root / cfg.data.artifacts_dir / "tokenizer"
    datamodule.save_tokenizer(tokenizer_path)
    logger.info(f"Saved tokenizer to: {tokenizer_path}")

    _save_best_checkpoint_copy(best_checkpoint_path, checkpoint_dir / "best.ckpt")

    plots_dir = project_root / cfg.logging.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_predictions = _run_test_evaluation(trainer, model, datamodule)

    create_plots(
        train_losses=metrics_callback.train_losses,
        val_losses=metrics_callback.val_losses,
        train_f1_scores=metrics_callback.train_f1_scores,
        val_f1_scores=metrics_callback.val_f1_scores,
        test_predictions=test_predictions,
        label_maps=datamodule.label_maps,
        plots_dir=plots_dir,
        cfg=cfg,
    )

    if mlflow_run:
        import mlflow

        mlflow.log_artifacts(str(plots_dir), artifact_path="plots")

        final_metrics = trainer.callback_metrics
        for key, value in final_metrics.items():
            if isinstance(value, torch.Tensor):
                mlflow.log_metric(key, float(value))

        mlflow.end_run()

    return {
        "best_checkpoint": str(best_checkpoint_path),
        "tokenizer_path": str(tokenizer_path),
        "plots_dir": str(plots_dir),
    }


def _create_callbacks(cfg: DictConfig, checkpoint_dir: Path) -> list:
    """Create training callbacks.

    Args:
        cfg: Configuration.
        checkpoint_dir: Directory for checkpoints.

    Returns:
        List of callbacks.
    """
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}-{val_f1_macro:.4f}",
        monitor=cfg.train.checkpoint_metric,
        mode=cfg.train.checkpoint_mode,
        save_top_k=cfg.train.save_top_k,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    if cfg.train.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=cfg.train.early_stopping_metric,
            patience=cfg.train.early_stopping_patience,
            mode=cfg.train.early_stopping_mode,
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    return callbacks


def _get_best_checkpoint(callbacks: list) -> Optional[Path]:
    """Get path to best checkpoint.

    Args:
        callbacks: List of training callbacks.

    Returns:
        Path to best checkpoint or None.
    """
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_path = callback.best_model_path
            if best_path:
                return Path(best_path)
    return None


def _save_best_checkpoint_copy(source: Optional[Path], destination: Path) -> None:
    """Copy best checkpoint to standard location.

    Args:
        source: Source checkpoint path.
        destination: Destination path.
    """
    if source and source.exists():
        import shutil

        shutil.copy(source, destination)
        logger.info(f"Copied best checkpoint to: {destination}")


def _run_test_evaluation(
    trainer: pl.Trainer,
    model: MultiTaskTicketClassifier,
    datamodule: TicketDataModule,
) -> Optional[Dict]:
    """Run evaluation on test set.

    Args:
        trainer: PyTorch Lightning trainer.
        model: Trained model.
        datamodule: Data module with test data.

    Returns:
        Dictionary with test predictions or None.
    """
    try:
        datamodule.setup("test")

        model.eval()
        all_topic_preds = []
        all_topic_labels = []
        all_priority_preds = []
        all_priority_labels = []

        with torch.no_grad():
            for batch in datamodule.test_dataloader():
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                if next(model.parameters()).is_cuda:
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()

                topic_logits, priority_logits = model(input_ids, attention_mask)

                topic_preds = torch.argmax(topic_logits, dim=1).cpu()
                priority_preds = torch.argmax(priority_logits, dim=1).cpu()

                all_topic_preds.append(topic_preds)
                all_topic_labels.append(batch["topic_label"])
                all_priority_preds.append(priority_preds)
                all_priority_labels.append(batch["priority_label"])

        return {
            "topic_preds": torch.cat(all_topic_preds).numpy(),
            "topic_labels": torch.cat(all_topic_labels).numpy(),
            "priority_preds": torch.cat(all_priority_preds).numpy(),
            "priority_labels": torch.cat(all_priority_labels).numpy(),
        }

    except Exception as exc:
        logger.warning(f"Test evaluation failed: {exc}")
        return None
