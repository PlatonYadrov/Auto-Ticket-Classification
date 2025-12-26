"""MLflow Model Serving wrapper.

This module provides utilities for serving the model using MLflow's built-in
serving capabilities.
"""

import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger

from ticket_triage_ml.utils.paths import get_project_root


def register_model_to_mlflow(model_path: Optional[Path] = None) -> str:
    """Register ONNX model to MLflow model registry.

    Args:
        model_path: Path to the ONNX model. If None, uses default path.

    Returns:
        The model URI that can be used for serving.
    """
    import mlflow
    import mlflow.onnx

    project_root = get_project_root()

    if model_path is None:
        model_path = project_root / "artifacts" / "model.onnx"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Set tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Log model to MLflow
    with mlflow.start_run(run_name="model_registration"):
        model_uri = mlflow.onnx.log_model(
            onnx_model=str(model_path),
            artifact_path="model",
            registered_model_name="ticket-triage-model",
        )
        logger.info(f"Model registered at: {model_uri}")

    return model_uri


def serve_with_mlflow(
    model_uri: str = "models:/ticket-triage-model/latest",
    host: str = "0.0.0.0",
    port: int = 5001,
) -> None:
    """Serve model using MLflow serve command.

    Args:
        model_uri: MLflow model URI.
        host: Host to bind to.
        port: Port to serve on.
    """
    logger.info(f"Starting MLflow model server on {host}:{port}")
    logger.info(f"Model URI: {model_uri}")

    cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        model_uri,
        "-h",
        host,
        "-p",
        str(port),
        "--env-manager",
        "local",
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start MLflow server: {e}")
        raise
    except KeyboardInterrupt:
        logger.info("MLflow server stopped by user")


def main() -> None:
    """Main entry point for MLflow serving."""
    import fire

    fire.Fire(
        {
            "register": register_model_to_mlflow,
            "serve": serve_with_mlflow,
        }
    )


if __name__ == "__main__":
    main()
