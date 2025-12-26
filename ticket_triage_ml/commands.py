"""CLI commands using Python Fire.

This module provides the main command-line interface for the ticket-triage-ml
package. All commands use Hydra compose API for configuration management.
"""

import json
from pathlib import Path
from typing import List, Optional

import fire
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import DictConfig

from ticket_triage_ml.utils.paths import get_project_root


def _load_config(overrides: Optional[List[str]] = None) -> DictConfig:
    """Load Hydra configuration with optional overrides.

    Args:
        overrides: List of Hydra override strings (e.g., ["train.epochs=5"]).

    Returns:
        Resolved Hydra configuration.
    """
    project_root = get_project_root()
    config_dir = str(project_root / "configs")

    GlobalHydra.instance().clear()

    initialize_config_dir(config_dir=config_dir, version_base=None)

    overrides = overrides or []
    cfg = compose(config_name="config", overrides=overrides)

    return cfg


class TicketTriageCLI:
    """Command-line interface for Auto Ticket Classification.

    Provides commands for data download, preprocessing, training,
    ONNX export, and inference.
    """

    def download_data(self, overrides: Optional[List[str]] = None) -> None:
        """Download raw data from configured sources.

        Attempts data acquisition in order:
        1. DVC pull
        2. Fallback URL download
        3. Synthetic data generation (last resort)

        Args:
            overrides: Hydra config overrides.

        Example:
            poetry run python -m ticket_triage_ml.commands download_data
        """
        cfg = _load_config(overrides)

        from ticket_triage_ml.data.download import download_data as _download_data
        from ticket_triage_ml.utils.logging import configure_logging

        configure_logging(cfg)

        logger.info("Starting data download...")
        raw_file = _download_data(cfg)
        logger.info(f"Data available at: {raw_file}")

    def preprocess(self, overrides: Optional[List[str]] = None) -> None:
        r"""Preprocess raw data: clean, split, and create label maps.

        Creates train/val/test splits and saves them as Parquet files.
        Also creates label_maps.json for encoding/decoding labels.

        Args:
            overrides: Hydra config overrides.

        Example:
            poetry run python -m ticket_triage_ml.commands preprocess
            poetry run python -m ticket_triage_ml.commands preprocess \
                --overrides='["preprocess.train_ratio=0.8"]'
        """
        cfg = _load_config(overrides)

        from ticket_triage_ml.data.download import ensure_data
        from ticket_triage_ml.data.preprocess import preprocess_data
        from ticket_triage_ml.utils.logging import configure_logging

        configure_logging(cfg)

        logger.info("Ensuring raw data is available...")
        ensure_data(cfg)

        logger.info("Starting preprocessing...")
        train_path, val_path, test_path, label_maps_path = preprocess_data(cfg)

        logger.info("Preprocessing complete!")
        logger.info(f"  Train: {train_path}")
        logger.info(f"  Val: {val_path}")
        logger.info(f"  Test: {test_path}")
        logger.info(f"  Label maps: {label_maps_path}")

    def train(self, overrides: Optional[List[str]] = None) -> None:
        r"""Train the multi-task ticket classifier.

        Trains a transformer-based model with two classification heads
        for topic and priority prediction. Logs metrics to MLflow and
        saves plots to the plots directory.

        Args:
            overrides: Hydra config overrides.

        Example:
            poetry run python -m ticket_triage_ml.commands train
            poetry run python -m ticket_triage_ml.commands train \
                --overrides='["train.max_epochs=5", "train.batch_size=32"]'
        """
        cfg = _load_config(overrides)

        from ticket_triage_ml.data.download import ensure_data
        from ticket_triage_ml.data.preprocess import preprocess_data
        from ticket_triage_ml.training.train import train_model
        from ticket_triage_ml.utils.logging import configure_logging

        configure_logging(cfg)

        project_root = get_project_root()
        processed_dir = project_root / cfg.data.processed_dir

        if not (processed_dir / cfg.data.train_file).exists():
            logger.info("Processed data not found, running preprocessing...")
            ensure_data(cfg)
            preprocess_data(cfg)

        logger.info("Starting training...")
        results = train_model(cfg)

        logger.info("Training complete!")
        logger.info(f"  Best checkpoint: {results['best_checkpoint']}")
        logger.info(f"  Tokenizer: {results['tokenizer_path']}")
        logger.info(f"  Plots: {results['plots_dir']}")

    def export_onnx(self, overrides: Optional[List[str]] = None) -> None:
        """Export trained model to ONNX format.

        Converts the PyTorch checkpoint to ONNX for production inference.
        The ONNX model is saved to artifacts/model.onnx.

        Args:
            overrides: Hydra config overrides.

        Example:
            poetry run python -m ticket_triage_ml.commands export_onnx
        """
        cfg = _load_config(overrides)

        from ticket_triage_ml.production.export_onnx import export_to_onnx
        from ticket_triage_ml.utils.logging import configure_logging

        configure_logging(cfg)

        logger.info("Starting ONNX export...")
        onnx_path = export_to_onnx(cfg)

        logger.info(f"ONNX model exported to: {onnx_path}")

    def infer(
        self,
        text: Optional[str] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        overrides: Optional[List[str]] = None,
    ) -> Optional[str]:
        r"""Run inference on single text or batch file.

        For single text inference, returns JSON with predictions.
        For batch inference, saves predictions to output file.

        Args:
            text: Single ticket text to classify.
            input_path: Path to input CSV/Parquet for batch inference.
            output_path: Path to save batch predictions.
            overrides: Hydra config overrides.

        Returns:
            JSON string for single text inference, None for batch.

        Example:
            # Single text
            poetry run python -m ticket_triage_ml.commands infer \
                --text="Cannot access VPN"

            # Batch inference
            poetry run python -m ticket_triage_ml.commands infer \
                --input_path=data.csv --output_path=predictions.csv
        """
        cfg = _load_config(overrides)

        from ticket_triage_ml.production.infer_onnx import infer_batch, infer_text
        from ticket_triage_ml.utils.logging import configure_logging

        configure_logging(cfg)

        if text is not None:
            logger.info("Running single-text inference...")
            result = infer_text(text, cfg=cfg)
            return json.dumps(result, indent=2)

        elif input_path is not None:
            if output_path is None:
                input_file = Path(input_path)
                output_path = str(input_file.parent / f"{input_file.stem}_predictions.parquet")

            logger.info(f"Running batch inference: {input_path} -> {output_path}")
            result_path = infer_batch(
                input_path=input_path,
                output_path=output_path,
                cfg=cfg,
            )
            logger.info(f"Predictions saved to: {result_path}")
            return None

        else:
            logger.error("Either --text or --input_path must be provided")
            raise ValueError("Either --text or --input_path must be provided")

    def baseline(self, overrides: Optional[List[str]] = None) -> None:
        """Train and evaluate baseline TF-IDF + LogisticRegression model.

        Trains a simple baseline for comparison with the neural model.
        Results are saved to artifacts/baseline/.

        Args:
            overrides: Hydra config overrides.

        Example:
            poetry run python -m ticket_triage_ml.commands baseline
        """
        cfg = _load_config(overrides)

        from ticket_triage_ml.baseline.model import train_baseline
        from ticket_triage_ml.data.download import ensure_data
        from ticket_triage_ml.data.preprocess import preprocess_data
        from ticket_triage_ml.utils.logging import configure_logging

        configure_logging(cfg)

        project_root = get_project_root()
        processed_dir = project_root / cfg.data.processed_dir

        if not (processed_dir / cfg.data.train_file).exists():
            logger.info("Processed data not found, running preprocessing...")
            ensure_data(cfg)
            preprocess_data(cfg)

        logger.info("Training baseline model...")
        metrics = train_baseline(cfg)

        logger.info("Baseline training complete!")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the FastAPI REST API server.

        Launches production inference server with /predict endpoint.

        Args:
            host: Host address to bind to.
            port: Port number.

        Example:
            poetry run python -m ticket_triage_ml.commands serve
            poetry run python -m ticket_triage_ml.commands serve --port=8080
        """
        import uvicorn

        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(
            "ticket_triage_ml.api.app:app",
            host=host,
            port=port,
            reload=False,
        )

    def version(self) -> None:
        """Print package version."""
        from ticket_triage_ml import __version__

        print(f"ticket-triage-ml version {__version__}")


def main() -> None:
    """Main entry point for the CLI."""
    fire.Fire(TicketTriageCLI)


if __name__ == "__main__":
    main()
