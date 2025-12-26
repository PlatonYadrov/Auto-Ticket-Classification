"""Baseline model using TF-IDF and Logistic Regression."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from ticket_triage_ml.data.io import read_data
from ticket_triage_ml.data.preprocess import load_label_maps
from ticket_triage_ml.data.schema import InferenceOutput
from ticket_triage_ml.utils.paths import get_project_root


class BaselineModel:
    """Baseline classifier using TF-IDF + Logistic Regression.

    Uses separate classifiers for topic and priority prediction.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        max_iter: int = 1000,
    ) -> None:
        """Initialize the baseline model.

        Args:
            max_features: Maximum number of TF-IDF features.
            ngram_range: N-gram range for TF-IDF.
            max_iter: Maximum iterations for logistic regression.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            strip_accents="unicode",
        )

        self.topic_classifier = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=42,
        )

        self.priority_classifier = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=42,
        )

        self.label_maps: Optional[Dict] = None
        self._is_fitted = False

    def fit(
        self,
        texts: List[str],
        topic_labels: List[int],
        priority_labels: List[int],
    ) -> "BaselineModel":
        """Fit the baseline model.

        Args:
            texts: List of ticket texts.
            topic_labels: Topic label IDs.
            priority_labels: Priority label IDs.

        Returns:
            Self for chaining.
        """
        logger.info("Fitting TF-IDF vectorizer...")
        features = self.vectorizer.fit_transform(texts)

        logger.info("Training topic classifier...")
        self.topic_classifier.fit(features, topic_labels)

        logger.info("Training priority classifier...")
        self.priority_classifier.fit(features, priority_labels)

        self._is_fitted = True
        return self

    def predict(
        self,
        texts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict topic and priority.

        Args:
            texts: List of ticket texts.

        Returns:
            Tuple of (topic_preds, priority_preds, topic_probs, priority_probs).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        features = self.vectorizer.transform(texts)

        topic_preds = self.topic_classifier.predict(features)
        topic_probs = self.topic_classifier.predict_proba(features)

        priority_preds = self.priority_classifier.predict(features)
        priority_probs = self.priority_classifier.predict_proba(features)

        return topic_preds, priority_preds, topic_probs, priority_probs

    def predict_single(self, text: str) -> InferenceOutput:
        """Predict for a single text.

        Args:
            text: Ticket text.

        Returns:
            InferenceOutput with predictions and scores.
        """
        if self.label_maps is None:
            raise RuntimeError("Label maps not set. Load model properly.")

        topic_preds, priority_preds, topic_probs, priority_probs = self.predict([text])

        topic_id = int(topic_preds[0])
        priority_id = int(priority_preds[0])

        topic = self.label_maps["id_to_topic"][topic_id]
        priority = self.label_maps["id_to_priority"][priority_id]

        topic_classes = self.topic_classifier.classes_
        topic_scores = {
            self.label_maps["id_to_topic"][int(cls)]: round(float(prob), 4)
            for cls, prob in zip(topic_classes, topic_probs[0])
        }

        priority_classes = self.priority_classifier.classes_
        priority_scores = {
            self.label_maps["id_to_priority"][int(cls)]: round(float(prob), 4)
            for cls, prob in zip(priority_classes, priority_probs[0])
        }

        return InferenceOutput(
            topic=topic,
            priority=priority,
            topic_scores=topic_scores,
            priority_scores=priority_scores,
        )

    def evaluate(
        self,
        texts: List[str],
        topic_labels: List[int],
        priority_labels: List[int],
    ) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            texts: List of ticket texts.
            topic_labels: True topic labels.
            priority_labels: True priority labels.

        Returns:
            Dictionary with evaluation metrics.
        """
        topic_preds, priority_preds, _, _ = self.predict(texts)

        metrics = {
            "topic_accuracy": accuracy_score(topic_labels, topic_preds),
            "topic_f1_macro": f1_score(topic_labels, topic_preds, average="macro"),
            "priority_accuracy": accuracy_score(priority_labels, priority_preds),
            "priority_f1_macro": f1_score(priority_labels, priority_preds, average="macro"),
        }

        return metrics

    def save(self, output_dir: Path) -> None:
        """Save model to directory.

        Args:
            output_dir: Directory to save model files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

        with open(output_dir / "topic_classifier.pkl", "wb") as f:
            pickle.dump(self.topic_classifier, f)

        with open(output_dir / "priority_classifier.pkl", "wb") as f:
            pickle.dump(self.priority_classifier, f)

        logger.info(f"Baseline model saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: Path, label_maps: Dict) -> "BaselineModel":
        """Load model from directory.

        Args:
            model_dir: Directory with saved model files.
            label_maps: Label encoding maps.

        Returns:
            Loaded BaselineModel.
        """
        model = cls()

        with open(model_dir / "vectorizer.pkl", "rb") as f:
            model.vectorizer = pickle.load(f)

        with open(model_dir / "topic_classifier.pkl", "rb") as f:
            model.topic_classifier = pickle.load(f)

        with open(model_dir / "priority_classifier.pkl", "rb") as f:
            model.priority_classifier = pickle.load(f)

        model.label_maps = label_maps
        model._is_fitted = True

        logger.info(f"Baseline model loaded from {model_dir}")
        return model


def train_baseline(cfg: DictConfig) -> Dict[str, float]:
    """Train baseline model and evaluate.

    Args:
        cfg: Hydra configuration.

    Returns:
        Dictionary with evaluation metrics.
    """
    project_root = get_project_root()
    processed_dir = project_root / cfg.data.processed_dir
    artifacts_dir = project_root / cfg.data.artifacts_dir

    label_maps_path = artifacts_dir / cfg.data.label_maps_file
    label_maps = load_label_maps(label_maps_path)

    logger.info("Loading training data...")
    train_df = read_data(processed_dir / cfg.data.train_file)
    val_df = read_data(processed_dir / cfg.data.val_file)
    test_df = read_data(processed_dir / cfg.data.test_file)

    text_col = cfg.data.text_column
    topic_col = cfg.data.topic_column
    priority_col = cfg.data.priority_column

    train_texts = train_df[text_col].tolist()
    train_topic_labels = [label_maps["topic_to_id"][t] for t in train_df[topic_col]]
    train_priority_labels = [label_maps["priority_to_id"][p] for p in train_df[priority_col]]

    logger.info("Training baseline model...")
    model = BaselineModel()
    model.fit(train_texts, train_topic_labels, train_priority_labels)
    model.label_maps = label_maps

    val_texts = val_df[text_col].tolist()
    val_topic_labels = [label_maps["topic_to_id"][t] for t in val_df[topic_col]]
    val_priority_labels = [label_maps["priority_to_id"][p] for p in val_df[priority_col]]

    logger.info("Evaluating on validation set...")
    val_metrics = model.evaluate(val_texts, val_topic_labels, val_priority_labels)

    for key, value in val_metrics.items():
        logger.info(f"  val_{key}: {value:.4f}")

    test_texts = test_df[text_col].tolist()
    test_topic_labels = [label_maps["topic_to_id"][t] for t in test_df[topic_col]]
    test_priority_labels = [label_maps["priority_to_id"][p] for p in test_df[priority_col]]

    logger.info("Evaluating on test set...")
    test_metrics = model.evaluate(test_texts, test_topic_labels, test_priority_labels)

    for key, value in test_metrics.items():
        logger.info(f"  test_{key}: {value:.4f}")

    baseline_dir = artifacts_dir / "baseline"
    model.save(baseline_dir)

    all_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
    all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    return all_metrics
