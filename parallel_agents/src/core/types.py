from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field
from typing import Protocol, runtime_checkable


class JudgeScores(BaseModel):
    """Per-candidate evaluation metrics for preview quality used to select a winner."""

    index: int = Field(
        ...,
        ge=0,
        description="Zero-based candidate index corresponding to the input ordering",
    )
    relevance: float = Field(
        ...,
        ge=0,
        le=1,
        description="How directly the preview addresses the user query (0..1)",
    )
    coverage: float = Field(
        ...,
        ge=0,
        le=1,
        description="Breadth/depth of sections and evidence plan (0..1)",
    )
    faithfulness: float = Field(
        ...,
        ge=0,
        le=1,
        description="Likelihood that planned content will be accurate (0..1)",
    )
    overall: float = Field(
        ..., ge=0, le=1, description="Holistic preview quality, not an average (0..1)"
    )


class JudgeVerdict(BaseModel):
    """Judge output: a single winner and per-candidate scores."""

    winner_index: int = Field(
        ...,
        ge=0,
        description="Index of the winning candidate (must be a single winner)",
    )
    scores: list[JudgeScores] = Field(
        description="List of scores for each candidate preview, by index"
    )


class PreviewOutcome(BaseModel):
    name: str
    text: str
    tokens: int
    latency_s: float | None = None


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
    "Router",
    "FeatureExtractor",
    "RewardPolicy",
]


class RaceTuning(BaseModel):
    """Tuning parameters to make `race_with_judge_and_stream` production-ready and explicit.

    All time-like values are seconds; all ratios are in [0,1].
    """

    adaptive_min_tokens_min_scale: float = Field(
        0.75,
        ge=0.1,
        le=4.0,
        description="Lower multiplier bound for adaptive preview tokens",
    )
    adaptive_min_tokens_max_scale: float = Field(
        1.50,
        ge=0.1,
        le=8.0,
        description="Upper multiplier bound for adaptive preview tokens",
    )
    latency_bias_scale: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Magnitude of negative bias added to slower arms during selection",
    )
    speculative_min_query_length: int = Field(
        2000,
        ge=1,
        description="Query length threshold to enable speculative top-2 full stage",
    )
    log_every_preview_tokens: int = Field(
        20, ge=1, description="Logging cadence during preview streaming"
    )
    log_every_full_tokens: int = Field(
        50, ge=1, description="Logging cadence during full streaming"
    )


# === Tier 0 Protocols ===


@runtime_checkable
class Router(Protocol):
    def select(self, x: list[float], arms: list[str], k: int = 1) -> list[str]: ...

    def bulk_update(self, x: list[float], rewards: dict[str, float]) -> None: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    def compute(self, query: str) -> list[float]: ...


@runtime_checkable
class RewardPolicy(Protocol):
    def compose(
        self,
        preview_tokens: list[int],
        judge_overall: list[float],
        cost_terms: list[float] | None = None,
    ) -> list[float]: ...
