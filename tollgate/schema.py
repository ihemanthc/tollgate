"""Pydantic schemas shared by collection, training, evaluation and serving.

The single source of truth for what a *label* means in Tollgate: the cheapest
tier whose answer an LLM judge rated equivalent or acceptable against the
frontier answer. No difficulty scores, no heuristics -- only the minimum
sufficient tier.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class Tier(StrEnum):
    """Model tiers, declared cheapest-first."""

    LOCAL_SMALL = "local_small"
    MID_TIER = "mid_tier"
    FRONTIER = "frontier"

    @property
    def rank(self) -> int:
        """Position in the cost ordering; 0 is cheapest."""
        return TIER_ORDER.index(self)


TIER_ORDER: tuple[Tier, ...] = (Tier.LOCAL_SMALL, Tier.MID_TIER, Tier.FRONTIER)


class Split(StrEnum):
    """Dataset partitions. Temperature scaling is fitted on CALIBRATION only."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class TollgateModel(BaseModel):
    """Base config: strict, immutable records that round-trip to JSONL."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class QueryRecord(TollgateModel):
    """One input query, before any model has been run against it."""

    query_id: str = Field(description="Stable id; sha256 of the prompt text.")
    prompt: str
    source: str = Field(description="Upstream dataset or collection script name.")
    domain: str | None = None
    token_estimate: int | None = Field(default=None, ge=0)
    collected_at: datetime | None = None


class TierRun(TollgateModel):
    """The result of answering one query at one tier."""

    query_id: str
    tier: Tier
    model: str = Field(description="Concrete litellm model id used for this tier.")
    completion: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_s: float = Field(ge=0.0)
    cached: bool = Field(default=False, description="Served from data/cache/, cost_usd is 0.")
    error: str | None = None


Verdict = Literal["equivalent", "acceptable", "worse"]
SUFFICIENT_VERDICTS: frozenset[str] = frozenset({"equivalent", "acceptable"})


class JudgeVerdict(TollgateModel):
    """An LLM judge comparing one candidate tier's answer to the frontier answer."""

    query_id: str
    candidate_tier: Tier
    reference_tier: Tier = Tier.FRONTIER
    verdict: Verdict
    reason: str = Field(description="One line from the judge explaining the verdict.")
    judge_model: str

    @property
    def sufficient(self) -> bool:
        """True when the candidate could stand in for the frontier answer."""
        return self.verdict in SUFFICIENT_VERDICTS


class LabeledExample(TollgateModel):
    """A training row: query text plus the three Laya targets and its split."""

    query_id: str
    prompt: str
    conversation_summary: str | None = Field(
        default=None, description="Prior-turn summary for multi-turn queries; None if single-turn."
    )
    source: str = Field(description="Seed source; a stratification key alongside the label.")
    split: Split
    tier: Tier = Field(
        description="Minimum sufficient tier (cheapest judged equivalent/acceptable)."
    )
    needs_tools: bool
    needs_rag: bool
    judged_tiers: tuple[Tier, ...] = Field(
        default=(),
        description="Tiers that were actually run and judged for this query.",
    )
    frontier_cost_usd: float | None = Field(default=None, ge=0.0)
    label_cost_usd: float | None = Field(default=None, ge=0.0)


class RouteDecision(TollgateModel):
    """What the router returns for a live query: a tier plus calibrated beliefs."""

    query_id: str
    tier: Tier = Field(description="Tier the router selected after applying its policy.")
    predicted_tier: Tier = Field(description="Argmax tier before policy adjustments.")
    tier_probs: dict[Tier, Probability] = Field(description="Calibrated, sums to 1.")
    confidence: Probability = Field(description="Calibrated probability of the selected tier.")
    p_needs_tools: Probability
    p_needs_rag: Probability
    temperature: float = Field(gt=0.0, description="Scaling temperature fitted on calibration.")
    escalated: bool = Field(
        default=False,
        description="True when policy overrode the argmax upward, e.g. low confidence.",
    )
    reason: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
