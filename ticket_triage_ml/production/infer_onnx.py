"""ONNX Runtime inference without PyTorch dependency."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
from loguru import logger
from omegaconf import DictConfig
from transformers import AutoTokenizer

from ticket_triage_ml.data.io import read_data, write_data
from ticket_triage_ml.data.schema import InferenceOutput
from ticket_triage_ml.utils.paths import get_project_root


class ONNXInferenceEngine:
    """ONNX Runtime inference engine for ticket classification.

    This class provides inference capabilities using ONNX Runtime,
    without requiring PyTorch at inference time.
    """

    def __init__(
        self,
        onnx_model_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        label_maps_path: Union[str, Path],
        max_length: int = 256,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> None:
        """Initialize the inference engine.

        Args:
            onnx_model_path: Path to the ONNX model file.
            tokenizer_path: Path to the saved tokenizer directory.
            label_maps_path: Path to the label maps JSON file.
            max_length: Maximum sequence length for tokenization.
            padding: Padding strategy.
            truncation: Whether to truncate sequences.
        """
        self.onnx_model_path = Path(onnx_model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.label_maps_path = Path(label_maps_path)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        self._validate_paths()

        logger.info(f"Loading tokenizer from: {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))

        logger.info(f"Loading label maps from: {self.label_maps_path}")
        self.label_maps = self._load_label_maps()

        logger.info(f"Loading ONNX model from: {self.onnx_model_path}")
        self.session = self._create_session()

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        logger.info("ONNX inference engine initialized successfully")

    def _validate_paths(self) -> None:
        """Validate that all required files exist."""
        if not self.onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")
        if not self.label_maps_path.exists():
            raise FileNotFoundError(f"Label maps not found: {self.label_maps_path}")

    def _load_label_maps(self) -> Dict:
        """Load label maps from JSON file."""
        with open(self.label_maps_path) as file_handle:
            label_maps = json.load(file_handle)

        label_maps["id_to_topic"] = {int(k): v for k, v in label_maps["id_to_topic"].items()}
        label_maps["id_to_priority"] = {int(k): v for k, v in label_maps["id_to_priority"].items()}

        return label_maps

    def _create_session(self) -> ort.InferenceSession:
        """Create ONNX Runtime inference session."""
        providers = ["CPUExecutionProvider"]

        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
            logger.info("Using CUDA for ONNX Runtime inference")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return ort.InferenceSession(
            str(self.onnx_model_path),
            sess_options=session_options,
            providers=providers,
        )

    def _tokenize(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Tokenize input texts.

        Args:
            texts: List of text strings to tokenize.

        Returns:
            Dictionary with input_ids and attention_mask arrays.
        """
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="np",
        )

        return {
            "input_ids": encodings["input_ids"].astype(np.int64),
            "attention_mask": encodings["attention_mask"].astype(np.int64),
        }

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits.

        Args:
            logits: Raw model outputs.

        Returns:
            Probability distribution.
        """
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def predict_single(self, text: str) -> InferenceOutput:
        """Predict topic and priority for a single text.

        Args:
            text: Support ticket text.

        Returns:
            InferenceOutput with predictions and scores.
        """
        inputs = self._tokenize([text])

        outputs = self.session.run(
            self.output_names,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )

        topic_logits = outputs[0][0]
        priority_logits = outputs[1][0]

        topic_probs = self._softmax(topic_logits)
        priority_probs = self._softmax(priority_logits)

        topic_id = int(np.argmax(topic_probs))
        priority_id = int(np.argmax(priority_probs))

        topic = self.label_maps["id_to_topic"][topic_id]
        priority = self.label_maps["id_to_priority"][priority_id]

        topic_scores = {
            self.label_maps["id_to_topic"][idx]: round(float(prob), 4)
            for idx, prob in enumerate(topic_probs)
        }

        priority_scores = {
            self.label_maps["id_to_priority"][idx]: round(float(prob), 4)
            for idx, prob in enumerate(priority_probs)
        }

        return InferenceOutput(
            topic=topic,
            priority=priority,
            topic_scores=topic_scores,
            priority_scores=priority_scores,
        )

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[InferenceOutput]:
        """Predict topic and priority for multiple texts.

        Args:
            texts: List of support ticket texts.
            batch_size: Batch size for inference.

        Returns:
            List of InferenceOutput objects.
        """
        results = []

        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            inputs = self._tokenize(batch_texts)

            outputs = self.session.run(
                self.output_names,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
            )

            topic_logits = outputs[0]
            priority_logits = outputs[1]

            topic_probs = self._softmax(topic_logits)
            priority_probs = self._softmax(priority_logits)

            for idx in range(len(batch_texts)):
                topic_id = int(np.argmax(topic_probs[idx]))
                priority_id = int(np.argmax(priority_probs[idx]))

                topic = self.label_maps["id_to_topic"][topic_id]
                priority = self.label_maps["id_to_priority"][priority_id]

                topic_scores = {
                    self.label_maps["id_to_topic"][tid]: round(float(prob), 4)
                    for tid, prob in enumerate(topic_probs[idx])
                }

                priority_scores = {
                    self.label_maps["id_to_priority"][pid]: round(float(prob), 4)
                    for pid, prob in enumerate(priority_probs[idx])
                }

                results.append(
                    InferenceOutput(
                        topic=topic,
                        priority=priority,
                        topic_scores=topic_scores,
                        priority_scores=priority_scores,
                    )
                )

        return results


def infer_text(
    text: str,
    cfg: Optional[DictConfig] = None,
    engine: Optional[ONNXInferenceEngine] = None,
) -> Dict:
    """Run inference on a single text and return JSON-compatible output.

    Args:
        text: Support ticket text to classify.
        cfg: Optional Hydra configuration.
        engine: Optional pre-initialized inference engine.

    Returns:
        Dictionary with predictions matching the output JSON spec.
    """
    if engine is None:
        if cfg is None:
            raise ValueError("Either cfg or engine must be provided")
        engine = _create_engine_from_config(cfg)

    result = engine.predict_single(text)

    return result.model_dump()


def infer_batch(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    cfg: Optional[DictConfig] = None,
    engine: Optional[ONNXInferenceEngine] = None,
    text_column: str = "text",
) -> Path:
    """Run batch inference on a CSV/Parquet file.

    Args:
        input_path: Path to input file (CSV or Parquet).
        output_path: Path to save predictions.
        cfg: Optional Hydra configuration.
        engine: Optional pre-initialized inference engine.
        text_column: Name of the text column in input file.

    Returns:
        Path to the output file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if engine is None:
        if cfg is None:
            raise ValueError("Either cfg or engine must be provided")
        engine = _create_engine_from_config(cfg)

    logger.info(f"Loading input data from: {input_path}")
    dataframe = read_data(input_path)

    if text_column not in dataframe.columns:
        raise ValueError(f"Text column '{text_column}' not found in input file")

    texts = dataframe[text_column].astype(str).tolist()
    batch_size = cfg.infer.batch_size if cfg else 32

    logger.info(f"Running inference on {len(texts)} samples...")
    results = engine.predict_batch(texts, batch_size=batch_size)

    dataframe["predicted_topic"] = [r.topic for r in results]
    dataframe["predicted_priority"] = [r.priority for r in results]
    dataframe["topic_scores"] = [json.dumps(r.topic_scores) for r in results]
    dataframe["priority_scores"] = [json.dumps(r.priority_scores) for r in results]

    write_data(dataframe, output_path)
    logger.info(f"Saved predictions to: {output_path}")

    return output_path


def _create_engine_from_config(cfg: DictConfig) -> ONNXInferenceEngine:
    """Create inference engine from configuration.

    Args:
        cfg: Hydra configuration.

    Returns:
        Initialized ONNXInferenceEngine.
    """
    project_root = get_project_root()

    return ONNXInferenceEngine(
        onnx_model_path=project_root / cfg.infer.onnx_model_path,
        tokenizer_path=project_root / cfg.infer.tokenizer_path,
        label_maps_path=project_root / cfg.infer.label_maps_path,
        max_length=cfg.infer.max_length,
        padding=cfg.infer.padding,
        truncation=cfg.infer.truncation,
    )
