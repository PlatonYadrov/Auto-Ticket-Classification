"""Pydantic schemas for data validation."""

from typing import Dict

from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    """Schema for single-text inference input.

    Attributes:
        text: The support ticket text to classify.
    """

    text: str = Field(..., description="Support ticket text to classify", min_length=1)


class InferenceOutput(BaseModel):
    """Schema for inference output with predictions and scores.

    Attributes:
        topic: Predicted topic class.
        priority: Predicted priority level.
        topic_scores: Probability scores for each topic class.
        priority_scores: Probability scores for each priority level.
    """

    topic: str = Field(..., description="Predicted topic class")
    priority: str = Field(..., description="Predicted priority level (low/medium/high)")
    topic_scores: Dict[str, float] = Field(
        ..., description="Probability scores for each topic class"
    )
    priority_scores: Dict[str, float] = Field(
        ..., description="Probability scores for each priority level"
    )


class TicketRecord(BaseModel):
    """Schema for a single ticket record in the dataset.

    Attributes:
        text: The ticket text content.
        topic: The topic category label.
        priority: The priority level label.
    """

    text: str = Field(..., description="Ticket text content")
    topic: str = Field(..., description="Topic category label")
    priority: str = Field(..., description="Priority level label")


class LabelMaps(BaseModel):
    """Schema for label encoding maps.

    Attributes:
        topic_to_id: Mapping from topic string to integer ID.
        id_to_topic: Mapping from integer ID to topic string.
        priority_to_id: Mapping from priority string to integer ID.
        id_to_priority: Mapping from integer ID to priority string.
    """

    topic_to_id: Dict[str, int] = Field(..., description="Topic to integer ID mapping")
    id_to_topic: Dict[int, str] = Field(..., description="Integer ID to topic mapping")
    priority_to_id: Dict[str, int] = Field(..., description="Priority to integer ID mapping")
    id_to_priority: Dict[int, str] = Field(..., description="Integer ID to priority mapping")
