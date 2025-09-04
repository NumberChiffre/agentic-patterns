from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field


class JudgeScores(BaseModel):
    """Per-candidate evaluation metrics for preview quality used to select a winner."""

    index: int = Field(
        ..., ge=0, description="Zero-based candidate index corresponding to the input ordering"
    )
    relevance: float = Field(
        ..., ge=0, le=1, description="How directly the preview addresses the user query (0..1)"
    )
    coverage: float = Field(
        ..., ge=0, le=1, description="Breadth/depth of sections and evidence plan (0..1)"
    )
    faithfulness: float = Field(
        ..., ge=0, le=1, description="Likelihood that planned content will be accurate (0..1)"
    )
    overall: float = Field(
        ..., ge=0, le=1, description="Holistic preview quality, not an average (0..1)"
    )


class JudgeVerdict(BaseModel):
    """Judge output: a single winner and per-candidate scores."""

    winner_index: int = Field(
        ..., ge=0, description="Index of the winning candidate (must be a single winner)"
    )
    scores: list[JudgeScores] = Field(
        description="List of scores for each candidate preview, by index"
    )


@dataclass
class PreviewOutcome:
    name: str
    text: str
    tokens: int


# ==== Enums and shared constants for the parallel agents package ====

class Strategy(StrEnum):
    """Selection strategy for choosing/ordering candidate agents."""

    BASELINE = "baseline"
    BANDIT = "bandit"


class StreamEventType(StrEnum):
    """Event types observed from the Agents SDK streaming API."""

    TEXT_DELTA_EVENT = "text_delta_event"
    RAW_RESPONSE_EVENT = "raw_response_event"
    MESSAGE_OUTPUT_ITEM = "message_output_item"


class RawDataCategory(StrEnum):
    """Substrings observed in raw response event data type fields."""

    RESPONSE_OUTPUT_TEXT_DELTA = "response.output_text.delta"
    OUTPUT_TEXT_DELTA = "output_text.delta"
    TEXT_DELTA = "text.delta"
    DELTA = "delta"
    WEB_SEARCH_CALL = "web_search_call"
    COMPLETED = "completed"
    ANNOTATION_ADDED = "annotation.added"


class AnnotationType(StrEnum):
    """Annotation types surfaced by the Agents SDK for citations, etc."""

    URL_CITATION = "url_citation"


# Convenience tuple used to detect text delta-like raw data types via substring match
DELTA_DATA_TYPE_SUBSTRINGS: tuple[str, ...] = (
    RawDataCategory.RESPONSE_OUTPUT_TEXT_DELTA,
    RawDataCategory.OUTPUT_TEXT_DELTA,
    RawDataCategory.TEXT_DELTA,
    RawDataCategory.DELTA,
)


__all__ = [
    "JudgeScores",
    "JudgeVerdict",
    "PreviewOutcome",
    "Strategy",
    "StreamEventType",
    "RawDataCategory",
    "AnnotationType",
    "DELTA_DATA_TYPE_SUBSTRINGS",
]


