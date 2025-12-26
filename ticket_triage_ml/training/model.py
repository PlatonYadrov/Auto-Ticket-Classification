"""Multi-task transformer model for ticket classification."""

from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import Accuracy, F1Score
from transformers import AutoModel


class ClassificationHead(nn.Module):
    """Classification head with optional hidden layers.

    Attributes:
        layers: Sequential layers for classification.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """Initialize classification head.

        Args:
            input_dim: Input feature dimension.
            num_classes: Number of output classes.
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout probability.
            activation: Activation function name.
        """
        super().__init__()

        activation_fn = self._get_activation(activation)
        hidden_dims = hidden_dims or []

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    activation_fn,
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name.

        Args:
            name: Activation function name.

        Returns:
            Activation module.
        """
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }
        return activations.get(name.lower(), nn.GELU())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head.

        Args:
            features: Input features of shape (batch_size, input_dim).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        return self.layers(features)


class MultiTaskTicketClassifier(pl.LightningModule):
    """Multi-task classifier for topic and priority prediction.

    Uses a shared transformer encoder with separate classification heads
    for topic and priority prediction.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_topics: int,
        num_priorities: int = 3,
        class_weights_topic: Optional[torch.Tensor] = None,
        class_weights_priority: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the multi-task classifier.

        Args:
            cfg: Hydra configuration.
            num_topics: Number of topic classes.
            num_priorities: Number of priority classes.
            class_weights_topic: Class weights for topic loss.
            class_weights_priority: Class weights for priority loss.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights_topic", "class_weights_priority"])

        self.cfg = cfg
        self.num_topics = num_topics
        self.num_priorities = num_priorities

        self.encoder = AutoModel.from_pretrained(cfg.model.encoder_name)

        if cfg.model.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = cfg.model.hidden_size

        self.topic_head = ClassificationHead(
            input_dim=hidden_size,
            num_classes=num_topics,
            hidden_dims=list(cfg.model.topic_head.hidden_dims),
            dropout=cfg.model.topic_head.dropout,
            activation=cfg.model.topic_head.activation,
        )

        self.priority_head = ClassificationHead(
            input_dim=hidden_size,
            num_classes=num_priorities,
            hidden_dims=list(cfg.model.priority_head.hidden_dims),
            dropout=cfg.model.priority_head.dropout,
            activation=cfg.model.priority_head.activation,
        )

        self.dropout = nn.Dropout(cfg.model.classifier_dropout)

        self.register_buffer("class_weights_topic", class_weights_topic)
        self.register_buffer("class_weights_priority", class_weights_priority)

        self._init_metrics()

        self.train_losses = []
        self.val_losses = []
        self.train_f1_scores = []
        self.val_f1_scores = []

    def _init_metrics(self) -> None:
        """Initialize metrics for tracking."""
        self.train_acc_topic = Accuracy(task="multiclass", num_classes=self.num_topics)
        self.train_acc_priority = Accuracy(task="multiclass", num_classes=self.num_priorities)
        self.val_acc_topic = Accuracy(task="multiclass", num_classes=self.num_topics)
        self.val_acc_priority = Accuracy(task="multiclass", num_classes=self.num_priorities)

        self.train_f1_topic = F1Score(
            task="multiclass", num_classes=self.num_topics, average="macro"
        )
        self.train_f1_priority = F1Score(
            task="multiclass", num_classes=self.num_priorities, average="macro"
        )
        self.val_f1_topic = F1Score(task="multiclass", num_classes=self.num_topics, average="macro")
        self.val_f1_priority = F1Score(
            task="multiclass", num_classes=self.num_priorities, average="macro"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length).
            attention_mask: Attention mask of shape (batch_size, seq_length).

        Returns:
            Tuple of (topic_logits, priority_logits).
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        topic_logits = self.topic_head(pooled_output)
        priority_logits = self.priority_head(pooled_output)

        return topic_logits, priority_logits

    def _compute_loss(
        self,
        topic_logits: torch.Tensor,
        priority_logits: torch.Tensor,
        topic_labels: torch.Tensor,
        priority_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute multi-task loss.

        Args:
            topic_logits: Topic predictions.
            priority_logits: Priority predictions.
            topic_labels: Ground truth topic labels.
            priority_labels: Ground truth priority labels.

        Returns:
            Tuple of (total_loss, topic_loss, priority_loss).
        """
        topic_loss = F.cross_entropy(
            topic_logits,
            topic_labels,
            weight=self.class_weights_topic,
        )

        priority_loss = F.cross_entropy(
            priority_logits,
            priority_labels,
            weight=self.class_weights_priority,
        )

        total_loss = (
            self.cfg.train.topic_loss_weight * topic_loss
            + self.cfg.train.priority_loss_weight * priority_loss
        )

        return total_loss, topic_loss, priority_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch dictionary with inputs and labels.
            batch_idx: Batch index.

        Returns:
            Total loss.
        """
        topic_logits, priority_logits = self(
            batch["input_ids"],
            batch["attention_mask"],
        )

        total_loss, topic_loss, priority_loss = self._compute_loss(
            topic_logits,
            priority_logits,
            batch["topic_label"],
            batch["priority_label"],
        )

        topic_preds = torch.argmax(topic_logits, dim=1)
        priority_preds = torch.argmax(priority_logits, dim=1)

        self.train_acc_topic(topic_preds, batch["topic_label"])
        self.train_acc_priority(priority_preds, batch["priority_label"])
        self.train_f1_topic(topic_preds, batch["topic_label"])
        self.train_f1_priority(priority_preds, batch["priority_label"])

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_topic_loss", topic_loss)
        self.log("train_priority_loss", priority_loss)

        return total_loss

    def on_train_epoch_end(self) -> None:
        """Log metrics at end of training epoch."""
        acc_topic = self.train_acc_topic.compute()
        acc_priority = self.train_acc_priority.compute()
        f1_topic = self.train_f1_topic.compute()
        f1_priority = self.train_f1_priority.compute()

        self.log("train_acc_topic", acc_topic)
        self.log("train_acc_priority", acc_priority)
        self.log("train_f1_topic", f1_topic)
        self.log("train_f1_priority", f1_priority)

        f1_macro = (f1_topic + f1_priority) / 2
        self.log("train_f1_macro", f1_macro)

        logger.info(
            f"Epoch {self.current_epoch} [TRAIN] "
            f"acc_topic={acc_topic:.4f} acc_priority={acc_priority:.4f} "
            f"f1_macro={f1_macro:.4f}"
        )

        self.train_acc_topic.reset()
        self.train_acc_priority.reset()
        self.train_f1_topic.reset()
        self.train_f1_priority.reset()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Batch dictionary with inputs and labels.
            batch_idx: Batch index.

        Returns:
            Total loss.
        """
        topic_logits, priority_logits = self(
            batch["input_ids"],
            batch["attention_mask"],
        )

        total_loss, topic_loss, priority_loss = self._compute_loss(
            topic_logits,
            priority_logits,
            batch["topic_label"],
            batch["priority_label"],
        )

        topic_preds = torch.argmax(topic_logits, dim=1)
        priority_preds = torch.argmax(priority_logits, dim=1)

        self.val_acc_topic(topic_preds, batch["topic_label"])
        self.val_acc_priority(priority_preds, batch["priority_label"])
        self.val_f1_topic(topic_preds, batch["topic_label"])
        self.val_f1_priority(priority_preds, batch["priority_label"])

        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_topic_loss", topic_loss)
        self.log("val_priority_loss", priority_loss)

        return total_loss

    def on_validation_epoch_end(self) -> None:
        """Log metrics at end of validation epoch."""
        acc_topic = self.val_acc_topic.compute()
        acc_priority = self.val_acc_priority.compute()
        f1_topic = self.val_f1_topic.compute()
        f1_priority = self.val_f1_priority.compute()

        self.log("val_acc_topic", acc_topic)
        self.log("val_acc_priority", acc_priority)
        self.log("val_f1_topic", f1_topic)
        self.log("val_f1_priority", f1_priority)

        f1_macro = (f1_topic + f1_priority) / 2
        self.log("val_f1_macro", f1_macro, prog_bar=True)

        val_loss = self.trainer.callback_metrics.get("val_loss", 0)
        logger.info(
            f"Epoch {self.current_epoch} [VAL] "
            f"loss={val_loss:.4f} acc_topic={acc_topic:.4f} acc_priority={acc_priority:.4f} "
            f"f1_macro={f1_macro:.4f}"
        )

        self.val_acc_topic.reset()
        self.val_acc_priority.reset()
        self.val_f1_topic.reset()
        self.val_f1_priority.reset()

    def configure_optimizers(self) -> Dict:
        """Configure optimizer and scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.cfg.train.warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
            )

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
