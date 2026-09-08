"""Turn judge verdicts into Laya targets: minimum sufficient tier, needs_tools, needs_rag.

The tier label comes only from judge verdicts. needs_tools / needs_rag currently come from
`heuristic_needs`, a placeholder meant to be swapped for hand labels via `needs_fn`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from tollgate.schema import TIER_ORDER, JudgeVerdict, Tier, TierRun, TollgateModel


class Needs(TollgateModel):
    needs_tools: bool
    needs_rag: bool


class QueryLabels(TollgateModel):
    """Everything label.py decides for one query; split assignment happens elsewhere."""

    query_id: str
    tier: Tier
    needs_tools: bool
    needs_rag: bool
    judged_tiers: tuple[Tier, ...]

    @property
    def determined(self) -> bool:
        """True when every tier cheaper than the label has a verdict.

        If a cheaper tier went unjudged (its run or the judge failed) it might have been
        sufficient, so the label could be too expensive: not a trustworthy training row.
        """
        return all(t in self.judged_tiers for t in TIER_ORDER[: self.tier.rank])


NeedsFn = Callable[[TierRun], Needs]


def minimum_sufficient_tier(verdicts: Iterable[JudgeVerdict]) -> Tier:
    """Cheapest tier judged equivalent or acceptable vs. frontier; frontier if none.

    Tiers with no verdict (run or judge failed) count as not sufficient. A cheap tier that
    passes still wins even if a pricier tier was judged worse -- the label is the cheapest
    sufficient tier, not the start of a monotone run.
    """
    by_tier: dict[Tier, JudgeVerdict] = {}
    for verdict in verdicts:
        if verdict.reference_tier is not Tier.FRONTIER:
            raise ValueError(f"verdict must be against frontier, got {verdict.reference_tier}")
        if verdict.candidate_tier is Tier.FRONTIER:
            raise ValueError("frontier cannot be a judged candidate")
        if verdict.candidate_tier in by_tier:
            raise ValueError(f"duplicate verdict for {verdict.candidate_tier}")
        by_tier[verdict.candidate_tier] = verdict
    if len({v.query_id for v in by_tier.values()}) > 1:
        raise ValueError("verdicts span more than one query")
    for tier in TIER_ORDER:
        verdict = by_tier.get(tier)
        if verdict is not None and verdict.sufficient:
            return tier
    return Tier.FRONTIER


# =============================================================================================
# HEURISTIC needs_tools / needs_rag -- PLACEHOLDER, replace with hand labels.
#
# Regexes over the frontier completion. Runner does not offer tools to the model, so there are
# never structured tool_calls; "tool-call presence" means tool-call markup leaking into the
# text or the model saying it cannot act. To replace: pass needs_fn= to label_query().
# Nothing else in Tollgate reads these patterns.
# =============================================================================================

_APOS = "['\N{RIGHT SINGLE QUOTATION MARK}]"
# "I can't", "I cannot", "I'm unable to", "I am not able to", "I don't have the ability to" ...
_I_CANT = (
    rf"\bI\s*(?:can(?:not|{_APOS}t)|can\s+not|(?:am|{_APOS}m)\s+(?:unable|not\s+able)\s+to"
    rf"|(?:do\s+not|don{_APOS}t)\s+have\s+(?:the\s+)?(?:ability|capability|access)\s+to)\s+"
)
_TOOL_MARKUP = re.compile(
    r"<tool_call>|<function_call>|\"(?:tool_calls|function_call)\"\s*:|```tool_code", re.I
)
_TOOL_INABILITY = re.compile(
    _I_CANT + r"(?:run|execute|send|book|schedule|place|interact\s+with"
    rf"|make\s+(?:api|http|network|phone)|access\s+(?:your|local|the\s+user{_APOS}s))\b",
    re.I,
)
_RAG_INABILITY = re.compile(
    _I_CANT + r"(?:browse|search|look\s+up|retrieve|access\s+(?:the\s+)?"
    r"(?:internet|web|online|real[-\s]time|current|live|up[-\s]to[-\s]date|external))\b",
    re.I,
)
_NO_CURRENT_INFO = re.compile(
    rf"\b(?:do\s+not|don{_APOS}t)\s+have\s+(?:access\s+to\s+)?"
    r"(?:real[-\s]time|current|up[-\s]to[-\s]date|live|the\s+latest|recent)\s+"
    r"(?:[\w-]+\s+){0,2}(?:information|data|news|prices|updates|events)\b",
    re.I,
)
_KNOWLEDGE_CUTOFF = re.compile(
    r"\b(?:as\s+of|since)\s+my\s+(?:last\s+)?(?:knowledge|training)\s+(?:cut-?off|update)"
    r"|\bmy\s+(?:knowledge|training\s+data)\s+(?:cut-?off|only\s+goes|is\s+limited|ends)",
    re.I,
)


def heuristic_needs(frontier: TierRun) -> Needs:
    """PLACEHOLDER: guess needs_tools / needs_rag from the frontier answer's text."""
    text = frontier.completion
    return Needs(
        needs_tools=bool(_TOOL_MARKUP.search(text) or _TOOL_INABILITY.search(text)),
        needs_rag=bool(
            _RAG_INABILITY.search(text)
            or _NO_CURRENT_INFO.search(text)
            or _KNOWLEDGE_CUTOFF.search(text)
        ),
    )


# ======================================= end heuristics =======================================


def label_query(
    frontier: TierRun,
    verdicts: Iterable[JudgeVerdict],
    needs_fn: NeedsFn = heuristic_needs,
) -> QueryLabels:
    if frontier.tier is not Tier.FRONTIER or frontier.error is not None:
        raise ValueError("label_query needs a successful frontier run")
    verdicts = list(verdicts)
    if any(v.query_id != frontier.query_id for v in verdicts):
        raise ValueError("verdicts are for a different query than the frontier run")
    needs = needs_fn(frontier)
    return QueryLabels(
        query_id=frontier.query_id,
        tier=minimum_sufficient_tier(verdicts),
        needs_tools=needs.needs_tools,
        needs_rag=needs.needs_rag,
        judged_tiers=tuple(t for t in TIER_ORDER if any(v.candidate_tier is t for v in verdicts)),
    )
