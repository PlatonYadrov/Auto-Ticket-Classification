"""Training API endpoints with progress tracking."""

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from pydantic import BaseModel

router = APIRouter(prefix="/train", tags=["training"])


class TrainingStatus(str, Enum):
    """Training job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingConfig(BaseModel):
    """Training configuration from API."""

    max_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5


class TrainingProgress(BaseModel):
    """Training progress information."""

    job_id: str
    status: TrainingStatus
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_f1_macro: Optional[float] = None
    progress_percent: float = 0.0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    logs: List[str] = []


@dataclass
class TrainingJob:
    """Internal training job state."""

    job_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_f1_macro: Optional[float] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    _thread: Optional[threading.Thread] = None

    def add_log(self, message: str) -> None:
        """Add log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_epochs == 0:
            return 0.0
        epoch_progress = (self.current_epoch / self.total_epochs) * 100
        if self.total_steps > 0:
            step_progress = (self.current_step / self.total_steps) * (100 / self.total_epochs)
            return min(epoch_progress + step_progress, 100.0)
        return epoch_progress

    def to_progress(self) -> TrainingProgress:
        """Convert to API response model."""
        return TrainingProgress(
            job_id=self.job_id,
            status=self.status,
            current_epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            current_step=self.current_step,
            total_steps=self.total_steps,
            train_loss=self.train_loss,
            val_loss=self.val_loss,
            val_f1_macro=self.val_f1_macro,
            progress_percent=self.progress_percent,
            started_at=self.started_at.isoformat() if self.started_at else None,
            finished_at=self.finished_at.isoformat() if self.finished_at else None,
            error_message=self.error_message,
            logs=self.logs[-20:],
        )


# In-memory storage for training jobs
_training_jobs: Dict[str, TrainingJob] = {}
_current_job_id: Optional[str] = None


def _run_training(job: TrainingJob) -> None:
    """Run training in background thread."""
    global _current_job_id

    try:
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now()
        job.total_epochs = job.config.max_epochs
        job.add_log(f"Начало обучения: {job.config.max_epochs} эпох, batch_size={job.config.batch_size}")

        import pytorch_lightning as pl
        import torch
        from omegaconf import OmegaConf

        from ticket_triage_ml.training.datamodule import TicketDataModule
        from ticket_triage_ml.training.model import MultiTaskTicketClassifier
        from ticket_triage_ml.utils.paths import get_project_root

        project_root = get_project_root()
        config_path = project_root / "configs" / "config.yaml"
        cfg = OmegaConf.load(config_path)

        cfg.train.max_epochs = job.config.max_epochs
        cfg.train.batch_size = job.config.batch_size
        cfg.train.learning_rate = job.config.learning_rate

        job.add_log("Загрузка данных...")
        datamodule = TicketDataModule(cfg)
        datamodule.setup("fit")

        job.add_log(f"Темы: {datamodule.num_topics}, Приоритеты: {datamodule.num_priorities}")

        job.add_log("Создание модели...")
        model = MultiTaskTicketClassifier(
            cfg=cfg,
            num_topics=datamodule.num_topics,
            num_priorities=datamodule.num_priorities,
            class_weights_topic=datamodule.class_weights_topic,
            class_weights_priority=datamodule.class_weights_priority,
        )

        class ProgressCallback(pl.Callback):
            """Callback to update training progress."""

            def __init__(self, training_job: TrainingJob):
                self.job = training_job

            def on_train_epoch_start(self, trainer, pl_module):
                self.job.current_epoch = trainer.current_epoch
                self.job.add_log(f"Эпоха {trainer.current_epoch + 1}/{self.job.total_epochs}")

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.job.current_step = batch_idx + 1
                self.job.total_steps = trainer.num_training_batches
                if "loss" in outputs:
                    self.job.train_loss = float(outputs["loss"])

            def on_validation_epoch_end(self, trainer, pl_module):
                metrics = trainer.callback_metrics
                if "val_loss" in metrics:
                    self.job.val_loss = float(metrics["val_loss"])
                if "val_f1_macro" in metrics:
                    self.job.val_f1_macro = float(metrics["val_f1_macro"])
                    self.job.add_log(
                        f"  val_loss={self.job.val_loss:.4f}, val_f1={self.job.val_f1_macro:.4f}"
                    )

        checkpoint_dir = project_root / cfg.train.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        from pytorch_lightning.callbacks import ModelCheckpoint

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val_f1_macro:.4f}",
            monitor="val_f1_macro",
            mode="max",
            save_top_k=1,
        )

        trainer = pl.Trainer(
            max_epochs=job.config.max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[ProgressCallback(job), checkpoint_callback],
            enable_progress_bar=False,
            logger=False,
        )

        job.add_log("Запуск обучения...")
        trainer.fit(model, datamodule=datamodule)

        job.current_epoch = job.total_epochs
        job.add_log("Сохранение модели...")

        best_path = checkpoint_callback.best_model_path
        if best_path:
            import shutil
            shutil.copy(best_path, checkpoint_dir / "best.ckpt")
            job.add_log(f"Лучшая модель: val_f1={job.val_f1_macro:.4f}")

        job.status = TrainingStatus.COMPLETED
        job.finished_at = datetime.now()
        job.add_log("✅ Обучение завершено!")

    except Exception as exc:
        job.status = TrainingStatus.FAILED
        job.finished_at = datetime.now()
        job.error_message = str(exc)
        job.add_log(f"❌ Ошибка: {exc}")
        logger.exception(f"Training failed: {exc}")

    finally:
        _current_job_id = None


@router.post("/start", response_model=TrainingProgress)
async def start_training(
    config: TrainingConfig = TrainingConfig(),
    background_tasks: BackgroundTasks = None,
) -> TrainingProgress:
    """Start a new training job.

    Args:
        config: Training configuration.

    Returns:
        Training progress with job ID.
    """
    global _current_job_id

    if _current_job_id is not None:
        job = _training_jobs.get(_current_job_id)
        if job and job.status == TrainingStatus.RUNNING:
            raise HTTPException(
                status_code=409,
                detail=f"Training already in progress: {_current_job_id}",
            )

    job_id = str(uuid.uuid4())[:8]
    job = TrainingJob(job_id=job_id, config=config)
    _training_jobs[job_id] = job
    _current_job_id = job_id

    thread = threading.Thread(target=_run_training, args=(job,), daemon=True)
    job._thread = thread
    thread.start()

    job.add_log("Задача обучения создана")
    return job.to_progress()


@router.get("/status/{job_id}", response_model=TrainingProgress)
async def get_training_status(job_id: str) -> TrainingProgress:
    """Get training job status.

    Args:
        job_id: Training job ID.

    Returns:
        Current training progress.
    """
    job = _training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_progress()


@router.get("/status", response_model=Optional[TrainingProgress])
async def get_current_training_status() -> Optional[TrainingProgress]:
    """Get current training job status.

    Returns:
        Current training progress or None if no training is running.
    """
    if _current_job_id is None:
        return None
    job = _training_jobs.get(_current_job_id)
    if not job:
        return None
    return job.to_progress()


@router.get("/jobs", response_model=List[TrainingProgress])
async def list_training_jobs() -> List[TrainingProgress]:
    """List all training jobs.

    Returns:
        List of all training jobs.
    """
    return [job.to_progress() for job in _training_jobs.values()]


@router.post("/cancel/{job_id}")
async def cancel_training(job_id: str) -> Dict[str, str]:
    """Cancel a training job.

    Args:
        job_id: Training job ID.

    Returns:
        Cancellation status.
    """
    job = _training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status != TrainingStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not running: {job.status}",
        )

    job.status = TrainingStatus.CANCELLED
    job.add_log("⚠️ Обучение отменено")
    return {"status": "cancelled", "job_id": job_id}

