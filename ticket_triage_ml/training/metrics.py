"""Metrics computation utilities."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        predictions: Predicted class labels.
        labels: Ground truth labels.
        average: Averaging method for F1 score.

    Returns:
        Dictionary with accuracy and F1 scores.
    """
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average=average, zero_division=0)

    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
    }


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Compute confusion matrix.

    Args:
        predictions: Predicted class labels.
        labels: Ground truth labels.
        class_names: List of class names for labeling.

    Returns:
        Tuple of (confusion_matrix, class_names).
    """
    conf_matrix = confusion_matrix(labels, predictions)
    return conf_matrix, class_names


def get_class_distribution(labels: np.ndarray, class_names: List[str]) -> Dict[str, int]:
    """Get class distribution from labels.

    Args:
        labels: Array of label indices.
        class_names: List of class names.

    Returns:
        Dictionary mapping class names to counts.
    """
    unique, counts = np.unique(labels, return_counts=True)
    distribution = {}

    for idx, count in zip(unique, counts):
        if idx < len(class_names):
            distribution[class_names[idx]] = int(count)

    return distribution


def aggregate_epoch_metrics(
    all_topic_preds: List[torch.Tensor],
    all_topic_labels: List[torch.Tensor],
    all_priority_preds: List[torch.Tensor],
    all_priority_labels: List[torch.Tensor],
) -> Dict[str, float]:
    """Aggregate predictions across batches and compute metrics.

    Args:
        all_topic_preds: List of topic predictions per batch.
        all_topic_labels: List of topic labels per batch.
        all_priority_preds: List of priority predictions per batch.
        all_priority_labels: List of priority labels per batch.

    Returns:
        Dictionary with aggregated metrics.
    """
    topic_preds = torch.cat(all_topic_preds).cpu().numpy()
    topic_labels = torch.cat(all_topic_labels).cpu().numpy()
    priority_preds = torch.cat(all_priority_preds).cpu().numpy()
    priority_labels = torch.cat(all_priority_labels).cpu().numpy()

    topic_metrics = compute_metrics(topic_preds, topic_labels)
    priority_metrics = compute_metrics(priority_preds, priority_labels)

    return {
        "topic_accuracy": topic_metrics["accuracy"],
        "topic_f1": topic_metrics["f1_score"],
        "priority_accuracy": priority_metrics["accuracy"],
        "priority_f1": priority_metrics["f1_score"],
        "f1_macro": (topic_metrics["f1_score"] + priority_metrics["f1_score"]) / 2,
    }
