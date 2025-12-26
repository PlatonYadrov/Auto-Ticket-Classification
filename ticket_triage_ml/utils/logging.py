"""Logging and MLflow utilities."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def get_logger(name: str = __name__) -> "logger":
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured loguru logger.
    """
    return logger


def configure_logging(cfg: DictConfig) -> None:
    """Configure logging based on configuration.

    Args:
        cfg: Hydra configuration with logging settings.
    """
    logger.remove()

    log_format = cfg.logging.console.format
    log_level = cfg.logging.console.level

    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
    )


def setup_mlflow(cfg: DictConfig) -> Optional[Any]:
    """Set up MLflow tracking.

    Args:
        cfg: Hydra configuration with MLflow settings.

    Returns:
        MLflow run object if successful, None otherwise.
    """
    try:
        import mlflow

        tracking_uri = cfg.logging.mlflow.tracking_uri
        experiment_name = cfg.logging.mlflow.experiment_name

        mlflow.set_tracking_uri(tracking_uri)

        try:
            mlflow.set_experiment(experiment_name)
        except Exception as exc:
            logger.warning(f"Could not set MLflow experiment: {exc}")
            logger.warning("Continuing without MLflow tracking")
            return None

        run_name = cfg.logging.mlflow.run_name

        run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run.info.run_id}")

        return run

    except Exception as exc:
        logger.warning(f"MLflow setup failed: {exc}")
        logger.warning("Continuing without MLflow tracking")
        return None


def log_config_to_mlflow(cfg: DictConfig, git_commit: Optional[str] = None) -> None:
    """Log configuration to MLflow.

    Args:
        cfg: Hydra configuration.
        git_commit: Optional git commit hash.
    """
    try:
        import mlflow

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        _flatten_and_log_params(config_dict)

        if git_commit:
            mlflow.log_param("git_commit", git_commit)

    except Exception as exc:
        logger.warning(f"Failed to log config to MLflow: {exc}")


def _flatten_and_log_params(
    config_dict: Dict,
    parent_key: str = "",
    max_length: int = 250,
) -> None:
    """Flatten nested config and log to MLflow.

    Args:
        config_dict: Configuration dictionary.
        parent_key: Parent key for nested values.
        max_length: Maximum value length for MLflow.
    """
    import mlflow

    for key, value in config_dict.items():
        full_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict):
            _flatten_and_log_params(value, full_key, max_length)
        else:
            str_value = str(value)
            if len(str_value) > max_length:
                str_value = str_value[:max_length] + "..."

            try:
                mlflow.log_param(full_key, str_value)
            except Exception:
                pass


def create_plots(
    train_losses: List[float],
    val_losses: List[float],
    train_f1_scores: List[float],
    val_f1_scores: List[float],
    test_predictions: Optional[Dict],
    label_maps: Dict,
    plots_dir: Path,
    cfg: DictConfig,
) -> List[Path]:
    """Create training visualization plots.

    Args:
        train_losses: Training losses per epoch.
        val_losses: Validation losses per epoch.
        train_f1_scores: Training F1 scores per epoch.
        val_f1_scores: Validation F1 scores per epoch.
        test_predictions: Optional test set predictions.
        label_maps: Label encoding maps.
        plots_dir: Directory to save plots.
        cfg: Configuration.

    Returns:
        List of paths to created plot files.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    created_plots = []

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    figsize = tuple(cfg.logging.figure.figsize)
    dpi = cfg.logging.figure.dpi

    if train_losses or val_losses:
        loss_plot_path = plots_dir / "loss_curve.png"
        _create_loss_curve(train_losses or [], val_losses or [], loss_plot_path, figsize, dpi)
        created_plots.append(loss_plot_path)

    if train_f1_scores or val_f1_scores:
        f1_plot_path = plots_dir / "f1_curve.png"
        _create_f1_curve(train_f1_scores or [], val_f1_scores or [], f1_plot_path, figsize, dpi)
        created_plots.append(f1_plot_path)

    if test_predictions:
        topic_cm_path = plots_dir / "confusion_matrix_topic.png"
        _create_confusion_matrix(
            test_predictions["topic_preds"],
            test_predictions["topic_labels"],
            label_maps["id_to_topic"],
            "Topic",
            topic_cm_path,
            figsize,
            dpi,
        )
        created_plots.append(topic_cm_path)

        priority_cm_path = plots_dir / "confusion_matrix_priority.png"
        _create_confusion_matrix(
            test_predictions["priority_preds"],
            test_predictions["priority_labels"],
            label_maps["id_to_priority"],
            "Priority",
            priority_cm_path,
            figsize,
            dpi,
        )
        created_plots.append(priority_cm_path)

    logger.info(f"Created {len(created_plots)} plots in {plots_dir}")

    return created_plots


def _create_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    output_path: Path,
    figsize: tuple,
    dpi: int,
) -> None:
    """Create loss curve plot.

    Args:
        train_losses: Training losses per epoch.
        val_losses: Validation losses per epoch.
        output_path: Path to save the plot.
        figsize: Figure size.
        dpi: Figure DPI.
    """
    plt.figure(figsize=figsize)

    if train_losses:
        train_epochs = range(1, len(train_losses) + 1)
        plt.plot(train_epochs, train_losses, "b-", label="Training Loss", linewidth=2)

    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    logger.debug(f"Saved loss curve: {output_path}")


def _create_f1_curve(
    train_f1_scores: List[float],
    val_f1_scores: List[float],
    output_path: Path,
    figsize: tuple,
    dpi: int,
) -> None:
    """Create F1 score curve plot.

    Args:
        train_f1_scores: Training F1 scores per epoch.
        val_f1_scores: Validation F1 scores per epoch.
        output_path: Path to save the plot.
        figsize: Figure size.
        dpi: Figure DPI.
    """
    plt.figure(figsize=figsize)

    if train_f1_scores:
        train_epochs = range(1, len(train_f1_scores) + 1)
        plt.plot(train_epochs, train_f1_scores, "b-", label="Training F1 (Macro)", linewidth=2)

    if val_f1_scores:
        val_epochs = range(1, len(val_f1_scores) + 1)
        plt.plot(val_epochs, val_f1_scores, "r-", label="Validation F1 (Macro)", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("Training and Validation F1 Score (Macro)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    logger.debug(f"Saved F1 curve: {output_path}")


def _create_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    id_to_label: Dict[int, str],
    task_name: str,
    output_path: Path,
    figsize: tuple,
    dpi: int,
) -> None:
    """Create confusion matrix plot.

    Args:
        predictions: Predicted labels.
        labels: Ground truth labels.
        id_to_label: Mapping from ID to label string.
        task_name: Name of the task for title.
        output_path: Path to save the plot.
        figsize: Figure size.
        dpi: Figure DPI.
    """
    from sklearn.metrics import confusion_matrix

    class_names = [id_to_label[idx] for idx in sorted(id_to_label.keys())]
    conf_matrix = confusion_matrix(labels, predictions)

    plt.figure(figsize=figsize)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(f"Confusion Matrix - {task_name}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    logger.debug(f"Saved confusion matrix: {output_path}")
