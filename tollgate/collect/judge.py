"""LLM judge: can a cheaper tier's answer stand in for the frontier answer?

Verdicts are equivalent | acceptable | worse plus a one-line reason. Calls go through
Runner.complete, so they are cached and ledgered like tier runs (ledger tier = "judge").
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from tollgate.collect.runner import Progress, Runner
from tollgate.config import ModelConfig
from tollgate.schema import JudgeVerdict, QueryRecord, Tier, TierRun, Verdict

log = logging.getLogger(__name__)

JUDGE_TAG = "judge"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")

JUDGE_TEMPLATE = """\
You are grading whether a CANDIDATE answer can replace a REFERENCE answer to a user's question.
The reference comes from a strong model; judge the candidate against the question itself,
using the reference as a guide to what a good answer contains.

Verdicts:
- "equivalent": same substance as the reference (same final answer, same key facts or steps);
  differences are only wording, length or formatting.
- "acceptable": differs from the reference or is less thorough, but still answers the question
  correctly and completely enough that the user would be satisfied.
- "worse": wrong, missing something the user needs, unsafe, off-topic, or a refusal the
  reference did not make.

Reply with ONLY a JSON object, no prose, no code fences:
{{"verdict": "equivalent" | "acceptable" | "worse", "reason": "<one sentence>"}}

<question>
{question}
</question>

<reference>
{reference}
</reference>

<candidate>
{candidate}
</candidate>"""

RETRY_TEMPLATE = """\
{prompt}

Your previous reply was:
<previous>
{previous}
</previous>
It was rejected because: {error}
Reply again with ONLY the JSON object."""


class JudgeOutput(BaseModel):
    """What the judge model must return. Extra keys are ignored, not a parse failure."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    verdict: Verdict
    reason: str

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize_verdict(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("reason")
    @classmethod
    def _one_line(cls, value: str) -> str:
        lines = [line.strip() for line in value.strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError("reason is empty")
        return lines[0]


class JudgeError(RuntimeError):
    """The judge reply could not be parsed, even after the one retry."""


def build_prompt(question: str, reference: str, candidate: str) -> str:
    return JUDGE_TEMPLATE.format(question=question, reference=reference, candidate=candidate)


def parse_judge_output(text: str) -> JudgeOutput:
    """Validate a judge reply; tolerates code fences and prose around one JSON object."""
    body = _FENCE.sub("", text.strip())
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in reply")
    return JudgeOutput.model_validate_json(body[start : end + 1])


def _describe(exc: ValueError) -> str:
    # Stable across runs (no pydantic doc URLs) so the retry prompt's cache key is stable.
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'reply'}: {e['msg']}" for e in exc.errors()
        )
    return str(exc)


async def judge(
    runner: Runner,
    judge_model: ModelConfig,
    record: QueryRecord,
    frontier: TierRun,
    candidate: TierRun,
) -> JudgeVerdict:
    """Judge one candidate run against the frontier run. Raises JudgeError on bad output.

    The retry is a different prompt (it quotes the bad reply), so it gets its own cache entry:
    reruns replay both cached replies and never re-call the API.
    """
    _check_pair(record, frontier, candidate)
    prompt = build_prompt(record.prompt, frontier.completion, candidate.completion)
    entry, _ = await runner.complete(prompt, judge_model, tag=JUDGE_TAG, json_mode=True)
    try:
        output = parse_judge_output(entry.completion)
    except ValueError as exc:
        log.warning("judge reply for %s unparseable (%s); retrying once", record.query_id[:12], exc)
        retry_prompt = RETRY_TEMPLATE.format(
            prompt=prompt, previous=entry.completion[:2000], error=_describe(exc)
        )
        entry, _ = await runner.complete(retry_prompt, judge_model, tag=JUDGE_TAG, json_mode=True)
        try:
            output = parse_judge_output(entry.completion)
        except ValueError as retry_exc:
            raise JudgeError(
                f"{record.query_id} {candidate.tier}: {_describe(retry_exc)}; "
                f"reply={json.dumps(entry.completion[:200])}"
            ) from retry_exc
    return JudgeVerdict(
        query_id=record.query_id,
        candidate_tier=candidate.tier,
        verdict=output.verdict,
        reason=output.reason,
        judge_model=judge_model.model,
    )


async def judge_query(
    runner: Runner, judge_model: ModelConfig, record: QueryRecord, runs: Sequence[TierRun]
) -> list[JudgeVerdict]:
    """Judge every successful cheaper-tier run of one query. Failures are logged and skipped,
    so a missing verdict means "not shown sufficient", never a fabricated one."""
    frontier = next((r for r in runs if r.tier is Tier.FRONTIER and r.error is None), None)
    if frontier is None:
        log.error("%s: no successful frontier run; cannot judge", record.query_id[:12])
        return []
    candidates = [r for r in runs if r.tier is not Tier.FRONTIER and r.error is None]
    results = await asyncio.gather(
        *(judge(runner, judge_model, record, frontier, c) for c in candidates),
        return_exceptions=True,
    )
    verdicts: list[JudgeVerdict] = []
    for candidate, result in zip(candidates, results, strict=True):
        if isinstance(result, BaseException):
            log.error("judge %s %s failed: %r", record.query_id[:12], candidate.tier, result)
        else:
            verdicts.append(result)
    return verdicts


def judgeable_pairs(runs: Sequence[TierRun]) -> int:
    """How many judge calls `judge_query` makes for one query's runs."""
    if not any(r.tier is Tier.FRONTIER and r.error is None for r in runs):
        return 0
    return sum(r.tier is not Tier.FRONTIER and r.error is None for r in runs)


async def judge_all(
    runner: Runner,
    judge_model: ModelConfig,
    records: Sequence[QueryRecord],
    runs: Sequence[TierRun],
) -> tuple[list[JudgeVerdict], int]:
    """Judge every query's cheaper runs. Returns (verdicts, number of judge calls that failed)."""
    by_query: dict[str, list[TierRun]] = defaultdict(list)
    for run in runs:
        by_query[run.query_id].append(run)
    progress = Progress("judged queries", len(records))

    async def one(record: QueryRecord) -> list[JudgeVerdict]:
        verdicts = await judge_query(runner, judge_model, record, by_query.get(record.query_id, []))
        progress.step()
        return verdicts

    per_query = await asyncio.gather(*(one(r) for r in records))
    verdicts = [v for vs in per_query for v in vs]
    expected = sum(judgeable_pairs(by_query.get(r.query_id, [])) for r in records)
    return verdicts, expected - len(verdicts)


def read_verdicts(path: Path) -> list[JudgeVerdict]:
    """data/verdicts.jsonl: one JudgeVerdict per line."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [JudgeVerdict.model_validate_json(line) for line in lines if line.strip()]


def write_verdicts(verdicts: Sequence[JudgeVerdict], path: Path) -> Path:
    """Upsert by (query_id, candidate_tier): reruns replace rows, never duplicate them."""
    merged = {(v.query_id, v.candidate_tier): v for v in read_verdicts(path)}
    merged.update({(v.query_id, v.candidate_tier): v for v in verdicts})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.writelines(v.model_dump_json() + "\n" for v in merged.values())
    os.replace(tmp, path)
    return path


def _check_pair(record: QueryRecord, frontier: TierRun, candidate: TierRun) -> None:
    if frontier.tier is not Tier.FRONTIER:
        raise ValueError(f"reference must be the frontier run, got {frontier.tier}")
    if candidate.tier is Tier.FRONTIER:
        raise ValueError("candidate must be a cheaper tier than frontier")
    if not frontier.query_id == candidate.query_id == record.query_id:
        raise ValueError("record, frontier and candidate are for different queries")
    for run in (frontier, candidate):
        if run.error is not None:
            raise ValueError(f"{run.tier} run errored, nothing to judge: {run.error}")
