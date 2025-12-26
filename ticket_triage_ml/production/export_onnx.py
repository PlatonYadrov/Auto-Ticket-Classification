"""ONNX model export functionality."""

from pathlib import Path
from typing import Tuple

import torch
from loguru import logger
from omegaconf import DictConfig

from ticket_triage_ml.data.preprocess import load_label_maps
from ticket_triage_ml.training.model import MultiTaskTicketClassifier
from ticket_triage_ml.utils.paths import get_project_root


def export_to_onnx(cfg: DictConfig) -> Path:
    """Export trained PyTorch model to ONNX format.

    Args:
        cfg: Hydra configuration with export settings.

    Returns:
        Path to the exported ONNX model.
    """
    project_root = get_project_root()

    checkpoint_path = project_root / cfg.export.checkpoint_path
    onnx_output_path = project_root / cfg.export.onnx_output_path

    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    label_maps_path = project_root / cfg.data.artifacts_dir / cfg.data.label_maps_file
    label_maps = load_label_maps(label_maps_path)

    num_topics = len(label_maps["topic_to_id"])
    num_priorities = len(label_maps["priority_to_id"])

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = MultiTaskTicketClassifier(
        cfg=cfg,
        num_topics=num_topics,
        num_priorities=num_priorities,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.eval()
    model.cpu()

    dummy_input_ids, dummy_attention_mask = _create_dummy_input(cfg)

    logger.info(f"Exporting model to ONNX: {onnx_output_path}")

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_output_path),
        export_params=True,
        opset_version=cfg.export.onnx_opset_version,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["topic_logits", "priority_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "topic_logits": {0: "batch_size"},
            "priority_logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    _verify_onnx_model(onnx_output_path)

    logger.info(f"ONNX model exported successfully: {onnx_output_path}")

    return onnx_output_path


def _create_dummy_input(cfg: DictConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create dummy input tensors for ONNX export.

    Args:
        cfg: Configuration with model settings.

    Returns:
        Tuple of (input_ids, attention_mask) tensors.
    """
    batch_size = 1
    seq_length = cfg.train.max_length

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

    return dummy_input_ids, dummy_attention_mask


def _verify_onnx_model(onnx_path: Path) -> None:
    """Verify the exported ONNX model.

    Args:
        onnx_path: Path to the ONNX model file.

    Raises:
        ValueError: If model verification fails.
    """
    import onnx

    logger.info("Verifying ONNX model...")

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    logger.info("ONNX model verification passed")
