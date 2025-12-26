"""FastAPI application for ticket classification."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from loguru import logger

from ticket_triage_ml.data.schema import InferenceInput, InferenceOutput
from ticket_triage_ml.production.infer_onnx import ONNXInferenceEngine
from ticket_triage_ml.utils.paths import get_project_root

_engine: Optional[ONNXInferenceEngine] = None


def get_engine() -> ONNXInferenceEngine:
    """Get or create the inference engine singleton.

    Returns:
        Initialized ONNXInferenceEngine.

    Raises:
        HTTPException: If model artifacts are not found.
    """
    global _engine

    if _engine is not None:
        return _engine

    project_root = get_project_root()

    onnx_path = project_root / "artifacts" / "model.onnx"
    tokenizer_path = project_root / "artifacts" / "tokenizer"
    label_maps_path = project_root / "artifacts" / "label_maps.json"

    if not onnx_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run training and export first.",
        )

    try:
        _engine = ONNXInferenceEngine(
            onnx_model_path=onnx_path,
            tokenizer_path=tokenizer_path,
            label_maps_path=label_maps_path,
        )
        logger.info("Inference engine initialized")
        return _engine
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load model: {exc}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app.
    """
    from ticket_triage_ml.api.training import router as training_router

    app = FastAPI(
        title="Auto Ticket Classification API",
        description="Автоматическая классификация и приоритизация тикетов техподдержки",
        version="0.1.0",
    )

    app.include_router(training_router)

    @app.get("/health")
    def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/ready")
    def readiness_check():
        """Readiness check - verifies model is loaded."""
        try:
            get_engine()
            return {"status": "ready"}
        except HTTPException:
            return {"status": "not ready", "reason": "Model not loaded"}

    @app.post("/predict", response_model=InferenceOutput)
    def predict(request: InferenceInput) -> InferenceOutput:
        """Predict topic and priority for a support ticket.

        Args:
            request: Input with ticket text.

        Returns:
            Predictions with topic, priority, and probability scores.
        """
        engine = get_engine()
        result = engine.predict_single(request.text)
        return result

    return app


app = create_app()


def main() -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "ticket_triage_ml.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
