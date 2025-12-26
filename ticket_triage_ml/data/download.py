"""Data download and availability utilities."""

import random
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from loguru import logger
from omegaconf import DictConfig

from ticket_triage_ml.data.dvc_utils import dvc_pull
from ticket_triage_ml.data.io import write_data
from ticket_triage_ml.utils.paths import get_project_root


def download_data(cfg: DictConfig) -> Path:
    """Download raw data from configured sources.

    Attempts data acquisition in the following order:
    1. DVC pull (if configured)
    2. Fallback URL download
    3. Synthetic data generation (last resort)

    Args:
        cfg: Hydra configuration with data settings.

    Returns:
        Path to the downloaded/generated raw data file.
    """
    project_root = get_project_root()
    raw_dir = project_root / cfg.data.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / cfg.data.raw_file

    if raw_file.exists():
        logger.info(f"Raw data already exists: {raw_file}")
        return raw_file

    logger.info("Attempting to download data...")

    if _try_dvc_pull(raw_file):
        return raw_file

    if _try_fallback_download(cfg, raw_file):
        return raw_file

    if cfg.data.synthetic_fallback:
        logger.warning("SYNTHETIC FALLBACK: Generating synthetic dataset for smoke-run only")
        return _generate_synthetic_data(cfg, raw_file)

    raise RuntimeError("Failed to obtain data from any source")


def ensure_data(cfg: DictConfig) -> Path:
    """Ensure raw data is available, downloading if necessary.

    Args:
        cfg: Hydra configuration with data settings.

    Returns:
        Path to the raw data file.
    """
    project_root = get_project_root()
    raw_file = project_root / cfg.data.raw_dir / cfg.data.raw_file

    if raw_file.exists():
        logger.debug(f"Data available at: {raw_file}")
        return raw_file

    return download_data(cfg)


def _try_dvc_pull(target_file: Path) -> bool:
    """Attempt to pull data via DVC.

    Args:
        target_file: Expected file path after DVC pull.

    Returns:
        True if DVC pull succeeded and file exists.
    """
    dvc_file = Path(str(target_file) + ".dvc")

    if not dvc_file.exists():
        logger.debug("No DVC file found, skipping DVC pull")
        return False

    success = dvc_pull([str(dvc_file)])

    if success and target_file.exists():
        logger.info(f"Successfully pulled data via DVC: {target_file}")
        return True

    return False


def _try_fallback_download(cfg: DictConfig, output_path: Path) -> bool:
    """Attempt to download from fallback URL.

    Args:
        cfg: Configuration with fallback_url setting.
        output_path: Path to save downloaded file.

    Returns:
        True if download succeeded.
    """
    fallback_url = cfg.data.get("fallback_url")

    if not fallback_url:
        logger.debug("No fallback URL configured")
        return False

    logger.info(f"Attempting fallback download from: {fallback_url}")

    try:
        response = requests.get(fallback_url, timeout=60)
        response.raise_for_status()

        dataframe = pd.read_csv(pd.io.common.StringIO(response.text))
        dataframe = _normalize_fallback_data(dataframe, cfg)

        if dataframe is not None:
            write_data(dataframe, output_path)
            logger.info(f"Downloaded and normalized data: {output_path}")
            return True

    except requests.RequestException as exc:
        logger.warning(f"Fallback download failed: {exc}")

    return False


def _normalize_fallback_data(dataframe: pd.DataFrame, cfg: DictConfig) -> Optional[pd.DataFrame]:
    """Normalize fallback data to expected schema.

    Args:
        dataframe: Downloaded DataFrame.
        cfg: Configuration with column names.

    Returns:
        Normalized DataFrame or None if normalization fails.
    """
    text_col = cfg.data.text_column
    topic_col = cfg.data.topic_column
    priority_col = cfg.data.priority_column

    if text_col in dataframe.columns and topic_col in dataframe.columns:
        if priority_col not in dataframe.columns:
            dataframe[priority_col] = _assign_random_priority(len(dataframe))
        return dataframe[[text_col, topic_col, priority_col]]

    text_candidates = ["text", "question", "body", "content", "description", "message"]
    topic_candidates = ["topic", "category", "label", "class", "type"]

    text_found = _find_column(dataframe, text_candidates)
    topic_found = _find_column(dataframe, topic_candidates)

    if text_found and topic_found:
        result = pd.DataFrame(
            {
                text_col: dataframe[text_found],
                topic_col: dataframe[topic_found],
                priority_col: _assign_random_priority(len(dataframe)),
            }
        )
        return result

    if len(dataframe.columns) >= 2:
        first_text_col = None
        for col in dataframe.columns:
            if dataframe[col].dtype == object:
                avg_len = dataframe[col].astype(str).str.len().mean()
                if avg_len > 20:
                    first_text_col = col
                    break

        if first_text_col:
            other_cols = [col for col in dataframe.columns if col != first_text_col]
            topic_source = other_cols[0] if other_cols else None

            if topic_source:
                result = pd.DataFrame(
                    {
                        text_col: dataframe[first_text_col],
                        topic_col: dataframe[topic_source].astype(str),
                        priority_col: _assign_random_priority(len(dataframe)),
                    }
                )
                return result

    logger.warning("Could not normalize fallback data to expected schema")
    return None


def _find_column(dataframe: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column from candidates list.

    Args:
        dataframe: DataFrame to search.
        candidates: List of candidate column names.

    Returns:
        First matching column name or None.
    """
    df_cols_lower = {col.lower(): col for col in dataframe.columns}

    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]

    return None


def _assign_random_priority(num_samples: int) -> List[str]:
    """Assign random priority levels with realistic distribution.

    Args:
        num_samples: Number of samples to generate priorities for.

    Returns:
        List of priority strings.
    """
    priorities = ["low", "medium", "high"]
    weights = [0.5, 0.35, 0.15]
    return random.choices(priorities, weights=weights, k=num_samples)


def _generate_synthetic_data(cfg: DictConfig, output_path: Path) -> Path:
    """Generate synthetic ticket data for testing.

    Args:
        cfg: Configuration with synthetic data settings.
        output_path: Path to save generated data.

    Returns:
        Path to the generated data file.
    """
    num_samples = cfg.data.synthetic_samples

    topics = [
        "Technical Support",
        "Billing",
        "Account Access",
        "Feature Request",
        "Bug Report",
    ]

    priorities = ["low", "medium", "high"]
    priority_weights = [0.5, 0.35, 0.15]

    templates = {
        "Technical Support": [
            "I'm having trouble connecting to the VPN from my laptop",
            "The application keeps crashing when I try to export data",
            "Cannot install the software on Windows 11",
            "Slow performance when loading large files",
            "Error message appears when syncing data",
        ],
        "Billing": [
            "I was charged twice for my subscription",
            "Need to update my payment method",
            "Question about the pricing tiers",
            "Invoice not received for last month",
            "Want to cancel my subscription",
        ],
        "Account Access": [
            "Forgot my password and cannot reset it",
            "Two-factor authentication not working",
            "Account locked after multiple attempts",
            "Cannot change my email address",
            "Need to merge two accounts",
        ],
        "Feature Request": [
            "Would love to have dark mode option",
            "Please add export to PDF feature",
            "Suggestion for keyboard shortcuts",
            "Integration with Slack would be helpful",
            "Mobile app version needed",
        ],
        "Bug Report": [
            "Button not working on the settings page",
            "Data not saving properly in forms",
            "Display issue on mobile browsers",
            "Search function returns wrong results",
            "Notification emails not being sent",
        ],
    }

    variations = [
        "",
        " Please help urgently.",
        " This is affecting my work.",
        " I've tried restarting but issue persists.",
        " Happening since yesterday.",
        " Multiple users affected.",
    ]

    data = []
    for _ in range(num_samples):
        topic = random.choice(topics)
        priority = random.choices(priorities, weights=priority_weights, k=1)[0]
        base_text = random.choice(templates[topic])
        variation = random.choice(variations)
        text = base_text + variation

        data.append(
            {
                cfg.data.text_column: text,
                cfg.data.topic_column: topic,
                cfg.data.priority_column: priority,
            }
        )

    dataframe = pd.DataFrame(data)
    write_data(dataframe, output_path)
    logger.warning(f"Generated {num_samples} synthetic samples at {output_path}")

    return output_path
