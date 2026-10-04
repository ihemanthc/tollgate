"""minimum_sufficient_tier ordering on fabricated verdicts; heuristic needs; label_query."""

from __future__ import annotations

from itertools import permutations

import pytest

from tollgate.collect.label import (
    Needs,
    QueryLabels,
    heuristic_needs,
    label_query,
    minimum_sufficient_tier,
)
from tollgate.schema import JudgeVerdict, Tier, TierRun, Verdict

L, M, F = Tier.LOCAL_SMALL, Tier.MID_TIER, Tier.FRONTIER


def _v(tier: Tier, verdict: Verdict, query_id: str = "q1") -> JudgeVerdict:
    return JudgeVerdict(
        query_id=query_id, candidate_tier=tier, verdict=verdict, reason="r", judge_model="j"
    )


def _frontier(completion: str = "Paris.", query_id: str = "q1") -> TierRun:
    return TierRun(
        query_id=query_id,
        tier=F,
        model="frontier-model",
        completion=completion,
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=0.0,
        latency_s=0.0,
    )


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        ([_v(L, "equivalent"), _v(M, "equivalent")], L),
        ([_v(L, "acceptable"), _v(M, "equivalent")], L),
        ([_v(L, "worse"), _v(M, "acceptable")], M),
        ([_v(L, "worse"), _v(M, "equivalent")], M),
        ([_v(L, "worse"), _v(M, "worse")], F),
        # non-monotone: cheapest sufficient wins even though the pricier tier was worse
        ([_v(L, "acceptable"), _v(M, "worse")], L),
        # missing verdicts count as not sufficient
        ([_v(M, "acceptable")], M),
        ([_v(L, "worse")], F),
        ([], F),
    ],
)
def test_minimum_sufficient_tier(verdicts: list[JudgeVerdict], expected: Tier) -> None:
    assert minimum_sufficient_tier(verdicts) is expected


def test_input_order_does_not_matter() -> None:
    verdicts = [_v(M, "equivalent"), _v(L, "worse")]
    for perm in permutations(verdicts):
        assert minimum_sufficient_tier(perm) is M


@pytest.mark.parametrize(
    "verdicts",
    [
        [_v(L, "worse"), _v(L, "equivalent")],  # duplicate tier
        [_v(F, "equivalent")],  # frontier as candidate
        [_v(L, "worse", "q1"), _v(M, "worse", "q2")],  # mixed queries
        [_v(L, "worse").model_copy(update={"reference_tier": M})],  # wrong reference
    ],
)
def test_rejects_malformed_verdict_sets(verdicts: list[JudgeVerdict]) -> None:
    with pytest.raises(ValueError):
        minimum_sufficient_tier(verdicts)


@pytest.mark.parametrize(
    ("text", "tools", "rag"),
    [
        ("The capital of France is Paris.", False, False),
        ("I can't browse the internet, but as of 2023 the CEO was ...", False, True),
        ("I don\N{RIGHT SINGLE QUOTATION MARK}t have real-time stock prices.", False, True),
        ("As of my last knowledge update, the record was 9.58s.", False, True),
        ("I'm unable to execute code, but running it would print 42.", True, False),
        ("I cannot send emails on your behalf.", True, False),
        ('<tool_call>{"name": "search"}</tool_call>', True, False),
        ("You can run this script yourself: python main.py", False, False),
    ],
)
def test_heuristic_needs(text: str, tools: bool, rag: bool) -> None:
    assert heuristic_needs(_frontier(text)) == Needs(needs_tools=tools, needs_rag=rag)


def test_label_query_uses_injected_needs_fn() -> None:
    labels = label_query(
        _frontier("I cannot browse the web."),
        [_v(M, "acceptable"), _v(L, "worse")],
        needs_fn=lambda _: Needs(needs_tools=True, needs_rag=False),
    )
    assert labels.tier is M
    assert (labels.needs_tools, labels.needs_rag) == (True, False)
    assert labels.judged_tiers == (L, M)


@pytest.mark.parametrize(
    ("tier", "judged", "determined"),
    [
        (L, (L,), True),
        (M, (L, M), True),
        (M, (M,), False),  # local never judged: it might have been sufficient
        (F, (L, M), True),
        (F, (L,), False),
        (F, (), False),
    ],
)
def test_determined(tier: Tier, judged: tuple[Tier, ...], determined: bool) -> None:
    labels = QueryLabels(
        query_id="q", tier=tier, needs_tools=False, needs_rag=False, judged_tiers=judged
    )
    assert labels.determined is determined


def test_label_query_rejects_mismatched_query() -> None:
    with pytest.raises(ValueError):
        label_query(_frontier(query_id="q1"), [_v(L, "worse", "q2")])
