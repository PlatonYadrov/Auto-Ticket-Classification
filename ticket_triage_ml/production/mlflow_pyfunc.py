"""MLflow PyFunc wrapper for model serving (optional)."""

from pathlib import Path
from typing import Optional

import mlflow.pyfunc
import pandas as pd

from ticket_triage_ml.production.infer_onnx import ONNXInferenceEngine


class TicketTriageModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc model wrapper for ticket triage inference.

    This class wraps the ONNX inference engine for use with
    MLflow model serving.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load model artifacts.

        Args:
            context: MLflow model context with artifact paths.
        """
        artifacts_dir = Path(context.artifacts["artifacts_dir"])

        self.engine = ONNXInferenceEngine(
            onnx_model_path=artifacts_dir / "model.onnx",
            tokenizer_path=artifacts_dir / "tokenizer",
            label_maps_path=artifacts_dir / "label_maps.json",
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run inference on input DataFrame.

        Args:
            context: MLflow model context.
            model_input: DataFrame with 'text' column.

        Returns:
            DataFrame with predictions.
        """
        if "text" not in model_input.columns:
            raise ValueError("Input DataFrame must have a 'text' column")

        texts = model_input["text"].astype(str).tolist()
        results = self.engine.predict_batch(texts)

        return pd.DataFrame(
            {
                "topic": [r.topic for r in results],
                "priority": [r.priority for r in results],
                "topic_scores": [r.topic_scores for r in results],
                "priority_scores": [r.priority_scores for r in results],
            }
        )


def log_model_to_mlflow(artifacts_dir: Path, registered_model_name: Optional[str] = None) -> str:
    """Log the model to MLflow.

    Args:
        artifacts_dir: Directory containing model artifacts.
        registered_model_name: Optional name for model registration.

    Returns:
        Model URI.
    """
    model = TicketTriageModel()

    return mlflow.pyfunc.log_model(
        artifact_path="ticket_triage_model",
        python_model=model,
        artifacts={"artifacts_dir": str(artifacts_dir)},
        registered_model_name=registered_model_name,
    )
