"""Data preprocessing and splitting utilities."""

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from ticket_triage_ml.data.io import read_data, write_data
from ticket_triage_ml.utils.paths import get_project_root


def preprocess_data(cfg: DictConfig) -> Tuple[Path, Path, Path, Path]:
    """Preprocess raw data: clean, split, and create label maps.

    Args:
        cfg: Hydra configuration with preprocessing settings.

    Returns:
        Tuple of paths: (train_path, val_path, test_path, label_maps_path)
    """
    project_root = get_project_root()
    raw_file = project_root / cfg.data.raw_dir / cfg.data.raw_file
    processed_dir = project_root / cfg.data.processed_dir
    artifacts_dir = project_root / cfg.data.artifacts_dir

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw data from: {raw_file}")
    dataframe = read_data(raw_file)
    logger.info(f"Loaded {len(dataframe)} records")

    dataframe = _clean_data(dataframe, cfg)
    logger.info(f"After cleaning: {len(dataframe)} records")

    label_maps = _create_label_maps(dataframe, cfg)
    label_maps_path = artifacts_dir / cfg.data.label_maps_file
    _save_label_maps(label_maps, label_maps_path)

    train_df, val_df, test_df = _split_data(dataframe, cfg)

    train_path = processed_dir / cfg.data.train_file
    val_path = processed_dir / cfg.data.val_file
    test_path = processed_dir / cfg.data.test_file

    write_data(train_df, train_path)
    write_data(val_df, val_path)
    write_data(test_df, test_path)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_path, val_path, test_path, label_maps_path


def _clean_data(dataframe: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Apply cleaning steps to the data.

    Args:
        dataframe: Raw DataFrame.
        cfg: Configuration with preprocessing settings.

    Returns:
        Cleaned DataFrame with standardized columns.
    """
    preprocess_cfg = cfg.preprocess
    dataframe = dataframe.copy()

    # Handle different dataset formats
    if "text_columns" in cfg.data and cfg.data.text_columns:
        # Kaggle dataset: combine multiple text columns
        text_cols = list(cfg.data.text_columns)
        existing_cols = [c for c in text_cols if c in dataframe.columns]
        if existing_cols:
            dataframe["text"] = dataframe[existing_cols].fillna("").astype(str).agg(" ".join, axis=1)
            logger.info(f"Combined columns {existing_cols} into 'text'")
    
    # Map topic column from raw data
    raw_topic_col = cfg.data.get("raw_topic_column", cfg.data.topic_column)
    if raw_topic_col in dataframe.columns:
        dataframe["topic"] = dataframe[raw_topic_col]
        if raw_topic_col != "topic":
            logger.info(f"Mapped '{raw_topic_col}' to 'topic'")
    
    # Map priority column from raw data
    raw_priority_col = cfg.data.get("raw_priority_column", cfg.data.priority_column)
    if raw_priority_col in dataframe.columns:
        dataframe["priority"] = dataframe[raw_priority_col]
        if raw_priority_col != "priority":
            logger.info(f"Mapped '{raw_priority_col}' to 'priority'")

    # Standardize priority values to low/medium/high
    if "priority" in dataframe.columns:
        priority_mapping = {
            "Low": "low", "low": "low",
            "Medium": "medium", "medium": "medium", 
            "High": "high", "high": "high",
            "Critical": "high", "critical": "high",
            "Urgent": "high", "urgent": "high",
        }
        dataframe["priority"] = dataframe["priority"].map(
            lambda x: priority_mapping.get(str(x), "medium")
        )

    text_col = "text"
    dataframe = dataframe.dropna(subset=[text_col])

    if preprocess_cfg.strip_whitespace:
        dataframe[text_col] = dataframe[text_col].astype(str).str.strip()

    if preprocess_cfg.lowercase:
        dataframe[text_col] = dataframe[text_col].str.lower()

    if preprocess_cfg.remove_urls:
        url_pattern = (
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        dataframe[text_col] = dataframe[text_col].str.replace(url_pattern, "", regex=True)

    if preprocess_cfg.remove_emails:
        email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
        dataframe[text_col] = dataframe[text_col].str.replace(email_pattern, "", regex=True)

    if preprocess_cfg.remove_signatures:
        for pattern in preprocess_cfg.signature_patterns:
            dataframe[text_col] = dataframe[text_col].str.replace(
                pattern, "", regex=True, flags=re.IGNORECASE
            )

    dataframe[text_col] = dataframe[text_col].str.strip()

    min_length = preprocess_cfg.min_text_length
    dataframe = dataframe[dataframe[text_col].str.len() >= min_length]

    # Use standardized column names after mapping
    dataframe["topic"] = dataframe["topic"].astype(str).str.strip().str.lower()
    dataframe["priority"] = dataframe["priority"].astype(str).str.strip().str.lower()

    valid_priorities = ["low", "medium", "high"]
    priority_mask = dataframe["priority"].isin(valid_priorities)

    if not priority_mask.all():
        invalid_count = (~priority_mask).sum()
        logger.warning(f"Found {invalid_count} records with invalid priority, mapping to 'medium'")
        dataframe.loc[~priority_mask, "priority"] = "medium"

    # Keep only necessary columns
    dataframe = dataframe[["text", "topic", "priority"]]

    return dataframe.reset_index(drop=True)


def _create_label_maps(dataframe: pd.DataFrame, cfg: DictConfig) -> Dict[str, Dict[str, int]]:
    """Create label to ID mappings for topics and priorities.

    Args:
        dataframe: Cleaned DataFrame with standardized columns.
        cfg: Configuration (unused, kept for compatibility).

    Returns:
        Dictionary with label mappings.
    """
    unique_topics = sorted(dataframe["topic"].unique().tolist())
    unique_priorities = ["low", "medium", "high"]

    topic_to_id = {topic: idx for idx, topic in enumerate(unique_topics)}
    id_to_topic = {idx: topic for topic, idx in topic_to_id.items()}

    priority_to_id = {priority: idx for idx, priority in enumerate(unique_priorities)}
    id_to_priority = {idx: priority for priority, idx in priority_to_id.items()}

    label_maps = {
        "topic_to_id": topic_to_id,
        "id_to_topic": id_to_topic,
        "priority_to_id": priority_to_id,
        "id_to_priority": id_to_priority,
    }

    logger.info(f"Created label maps: {len(topic_to_id)} topics, {len(priority_to_id)} priorities")

    return label_maps


def _save_label_maps(label_maps: Dict, output_path: Path) -> None:
    """Save label maps to JSON file.

    Args:
        label_maps: Dictionary with label mappings.
        output_path: Path to save the JSON file.
    """
    serializable_maps = {
        "topic_to_id": label_maps["topic_to_id"],
        "id_to_topic": {str(k): v for k, v in label_maps["id_to_topic"].items()},
        "priority_to_id": label_maps["priority_to_id"],
        "id_to_priority": {str(k): v for k, v in label_maps["id_to_priority"].items()},
    }

    with open(output_path, "w") as file_handle:
        json.dump(serializable_maps, file_handle, indent=2)

    logger.info(f"Saved label maps to: {output_path}")


def load_label_maps(label_maps_path: Path) -> Dict:
    """Load label maps from JSON file.

    Args:
        label_maps_path: Path to the label maps JSON file.

    Returns:
        Dictionary with label mappings.
    """
    with open(label_maps_path) as file_handle:
        label_maps = json.load(file_handle)

    label_maps["id_to_topic"] = {int(k): v for k, v in label_maps["id_to_topic"].items()}
    label_maps["id_to_priority"] = {int(k): v for k, v in label_maps["id_to_priority"].items()}

    return label_maps


def _split_data(
    dataframe: pd.DataFrame, cfg: DictConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.

    Args:
        dataframe: Cleaned DataFrame.
        cfg: Configuration with split ratios.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    preprocess_cfg = cfg.preprocess
    stratify_col = preprocess_cfg.stratify_column

    train_ratio = preprocess_cfg.train_ratio
    val_ratio = preprocess_cfg.val_ratio
    test_ratio = preprocess_cfg.test_ratio

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.warning(f"Split ratios sum to {total_ratio}, normalizing...")
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio

    stratify = dataframe[stratify_col] if stratify_col in dataframe.columns else None

    test_val_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=test_val_ratio,
        stratify=stratify,
        random_state=42,
    )

    if stratify is not None:
        temp_stratify = temp_df[stratify_col]
    else:
        temp_stratify = None

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_stratify,
        random_state=42,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
