"""PyTorch Lightning DataModule for ticket classification."""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ticket_triage_ml.data.io import read_data
from ticket_triage_ml.data.preprocess import load_label_maps
from ticket_triage_ml.utils.paths import get_project_root


class TicketDataset(Dataset):
    """PyTorch Dataset for ticket classification.

    Attributes:
        texts: List of ticket texts.
        topic_labels: List of topic label IDs.
        priority_labels: List of priority label IDs.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        padding: Padding strategy.
        truncation: Whether to truncate sequences.
    """

    def __init__(
        self,
        texts: List[str],
        topic_labels: List[int],
        priority_labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            texts: List of ticket texts.
            topic_labels: List of topic label IDs.
            priority_labels: List of priority label IDs.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            padding: Padding strategy.
            truncation: Whether to truncate sequences.
        """
        self.texts = texts
        self.topic_labels = topic_labels
        self.priority_labels = priority_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with input tensors and labels.
        """
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "topic_label": torch.tensor(self.topic_labels[idx], dtype=torch.long),
            "priority_label": torch.tensor(self.priority_labels[idx], dtype=torch.long),
        }


class TicketDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for ticket classification.

    Handles data loading, tokenization, and batching for training,
    validation, and testing.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the DataModule.

        Args:
            cfg: Hydra configuration.
        """
        super().__init__()
        self.cfg = cfg
        self.project_root = get_project_root()

        self.tokenizer: Optional[AutoTokenizer] = None
        self.label_maps: Optional[Dict] = None

        self.train_dataset: Optional[TicketDataset] = None
        self.val_dataset: Optional[TicketDataset] = None
        self.test_dataset: Optional[TicketDataset] = None

        self._class_weights_topic: Optional[torch.Tensor] = None
        self._class_weights_priority: Optional[torch.Tensor] = None

    def prepare_data(self) -> None:
        """Download tokenizer if needed (called on single GPU)."""
        AutoTokenizer.from_pretrained(self.cfg.model.encoder_name)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training/validation/testing.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or None for all).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.encoder_name)

        label_maps_path = (
            self.project_root / self.cfg.data.artifacts_dir / self.cfg.data.label_maps_file
        )
        self.label_maps = load_label_maps(label_maps_path)

        processed_dir = self.project_root / self.cfg.data.processed_dir

        if stage == "fit" or stage is None:
            train_df = read_data(processed_dir / self.cfg.data.train_file)
            val_df = read_data(processed_dir / self.cfg.data.val_file)

            self.train_dataset = self._create_dataset(train_df)
            self.val_dataset = self._create_dataset(val_df)

            if self.cfg.train.use_class_weights:
                self._compute_class_weights(train_df)

        if stage == "test" or stage is None:
            test_df = read_data(processed_dir / self.cfg.data.test_file)
            self.test_dataset = self._create_dataset(test_df)

    def _create_dataset(self, dataframe: pd.DataFrame) -> TicketDataset:
        """Create a TicketDataset from DataFrame.

        Args:
            dataframe: DataFrame with text and labels.

        Returns:
            TicketDataset instance.
        """
        texts = dataframe[self.cfg.data.text_column].tolist()

        topic_labels = [
            self.label_maps["topic_to_id"][topic] for topic in dataframe[self.cfg.data.topic_column]
        ]

        priority_labels = [
            self.label_maps["priority_to_id"][priority]
            for priority in dataframe[self.cfg.data.priority_column]
        ]

        return TicketDataset(
            texts=texts,
            topic_labels=topic_labels,
            priority_labels=priority_labels,
            tokenizer=self.tokenizer,
            max_length=self.cfg.train.max_length,
            padding=self.cfg.train.padding,
            truncation=self.cfg.train.truncation,
        )

    def _compute_class_weights(self, train_df: pd.DataFrame) -> None:
        """Compute class weights for handling imbalanced data.

        Args:
            train_df: Training DataFrame.
        """
        topic_counts = train_df[self.cfg.data.topic_column].value_counts()
        num_topics = len(self.label_maps["topic_to_id"])
        topic_weights = torch.zeros(num_topics)

        for topic, topic_id in self.label_maps["topic_to_id"].items():
            count = topic_counts.get(topic, 1)
            topic_weights[topic_id] = 1.0 / count

        topic_weights = topic_weights / topic_weights.sum() * num_topics
        self._class_weights_topic = topic_weights

        priority_counts = train_df[self.cfg.data.priority_column].value_counts()
        num_priorities = len(self.label_maps["priority_to_id"])
        priority_weights = torch.zeros(num_priorities)

        for priority, priority_id in self.label_maps["priority_to_id"].items():
            count = priority_counts.get(priority, 1)
            priority_weights[priority_id] = 1.0 / count

        priority_weights = priority_weights / priority_weights.sum() * num_priorities
        self._class_weights_priority = priority_weights

    @property
    def class_weights_topic(self) -> Optional[torch.Tensor]:
        """Get class weights for topic classification."""
        return self._class_weights_topic

    @property
    def class_weights_priority(self) -> Optional[torch.Tensor]:
        """Get class weights for priority classification."""
        return self._class_weights_priority

    @property
    def num_topics(self) -> int:
        """Get number of topic classes."""
        return len(self.label_maps["topic_to_id"])

    @property
    def num_priorities(self) -> int:
        """Get number of priority classes."""
        return len(self.label_maps["priority_to_id"])

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=self.cfg.train.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=self.cfg.train.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=self.cfg.train.pin_memory,
        )

    def save_tokenizer(self, output_dir: Path) -> None:
        """Save tokenizer to directory.

        Args:
            output_dir: Directory to save tokenizer.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
