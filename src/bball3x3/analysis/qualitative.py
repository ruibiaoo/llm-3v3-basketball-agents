"""Readable possession transcripts generated entirely from saved game records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from bball3x3.schemas.log import (
    CaptainCallRecord,
    DecisionRecord,
    MessageRecord,
    PossessionEndRecord,
    RefereeRecord,
    ResolutionRecord,
    StateRecord,
)

from .log_reader import GameLogIndex, LogRecord, record_event_idx, record_type
from .possessions import reconstruct_possessions


class TranscriptError(ValueError):
    """A requested possession cannot be located in the saved timeline."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _block(label: str, value: str) -> list[str]:
    return [f"  {label}:", *(f"    {line}" for line in value.split("\n"))]


def _action(action: str, target: str | None) -> str:
    return f"{action} target={target if target is not None else 'none'}"


def _render_record(record: LogRecord) -> list[str]:
    if isinstance(record, DecisionRecord):
        parsed = record.parsed_output
        lines = [f"Decision {record.player} (team {record.team})"]
        observation = record.observation_text
        lines.extend(_block(
            "Observation", "[not recorded]" if observation is None else observation
        ))
        lines.append(f"  Observation hash: {record.observation_hash or '[not recorded]'}")
        proposed = _action(parsed.action, parsed.target) if parsed else "[parse failed]"
        lines.extend([
            f"  Proposed: {proposed}",
            f"  Executed: {_action(record.executed_action, record.executed_target)}",
            f"  Parse OK: {record.parse_ok}; invalid reason: {record.invalid_reason or 'none'}",
        ])
        refs = record.memory_ref
        if refs is None and parsed is not None:
            refs = parsed.memory_ref
        lines.extend(_block("Memory references", _json(refs)))
        rationale = parsed.rationale if parsed is not None else None
        lines.extend(_block("Rationale", "[not recorded]" if rationale is None else rationale))
        lines.extend(_block("Raw output", record.raw_output))
        return lines

    if isinstance(record, MessageRecord):
        delivery = (
            str(record.delivered_at) if record.delivered_at is not None
            else f"{record.event_idx + 1} (inferred t+1)"
        )
        if record.suppressed:
            delivery = "none (suppressed)"
        lines = [
            f"Message {record.sender} (team {record.team}, {record.channel}/{record.kind})",
            f"  Emitted: {record.event_idx}; delivery: {delivery}",
        ]
        lines.extend(_block("Text", record.text))
        directive = record.directive.model_dump(mode="json") if record.directive else None
        lines.extend(_block("Directive", _json(directive)))
        lines.append(f"  CS key: {record.cs or 'none'}")
        return lines

    if isinstance(record, CaptainCallRecord):
        return [
            f"Captain call (team {record.team}, emitted {record.event_idx})",
            f"  Maintain previous call: {record.maintain}",
            *_block("Tactic", record.tactic if record.tactic is not None else "[not supplied]"),
            f"  CS key: {record.cs or 'none'}",
            *_block("Directives", _json(record.directives)),
        ]

    if isinstance(record, RefereeRecord):
        return [
            "Referee",
            *_block("Query", record.query),
            *_block("Retrieved chunks", _json(record.retrieved_chunk_ids)),
            *_block("Decision", _json(record.decision.model_dump(mode="json"))),
        ]

    if isinstance(record, StateRecord):
        state = record.state.model_dump(mode="json") if record.state else record.raw_state
        return [
            "State after transition",
            *_block("Snapshot", _json(state)),
            *_block("State validation errors", _json(record.state_validation_errors)),
            f"  Invariant checks completed: {record.invariant_checks_completed}",
            f"  Constraints checked: {record.constraints_checked}",
        ]

    # Preserve all outcome, mutation, invariant, and future-record fields.
    data = record.model_dump(mode="json") if isinstance(record, BaseModel) else dict(record)
    kind = record_type(record)
    data.pop("record_type", None)
    data.pop("event_idx", None)
    return [kind, *_block("Details", _json(data))]


def _prior_context(index: GameLogIndex, first: int, last: int) -> list[LogRecord]:
    """Earlier delivered messages and last captain replacement plus maintains.

    This is contextual evidence, not a reconstruction of each player's inbox.
    Exact visibility is supplied by the decision's recorded observation.
    """
    messages: list[tuple[int, LogRecord]] = []
    calls: dict[str, list[tuple[int, LogRecord]]] = {}
    for position, record in enumerate(index.records):
        event = record_event_idx(record)
        if event is None or event >= first:
            continue
        if isinstance(record, MessageRecord) and not record.suppressed:
            delivery = record.delivered_at or record.event_idx + 1
            if first <= delivery <= last:
                messages.append((position, record))
        elif isinstance(record, CaptainCallRecord):
            if not record.maintain:
                calls[record.team] = []
            calls.setdefault(record.team, []).append((position, record))
    contextual = messages + [item for team_calls in calls.values() for item in team_calls]
    return [record for _, record in sorted(contextual, key=lambda item: item[0])]


def render_possession_transcript(
    records: GameLogIndex | Iterable[LogRecord], possession_id: str
) -> str:
    """Render one possession, including both teams and shared boundary events.

    The selector is the exact possession ID (not its ordinal). Boundaries are
    reconstructed from resolutions; an unfinished prefix can show an open
    possession and pending decisions. Malformed possession timelines raise
    PossessionError. Missing observations remain explicitly unavailable.

    Event spans are inclusive: a transfer can close one possession and open
    another at the same event. Full boundary events are shown as context;
    decisions belong to possession_before and scores retain their recorded
    possession IDs. Original record order is preserved within the span.
    """
    index = records if isinstance(records, GameLogIndex) else GameLogIndex.from_records(records)
    summary = reconstruct_possessions(index)
    start = index.game_start
    if start is None:
        raise TranscriptError("game_start is required for a transcript")
    closure: PossessionEndRecord | None = next(
        (record for record in summary.closed if record.possession_id == possession_id), None
    )
    if closure is not None:
        first, last, team = closure.start_event_idx, closure.end_event_idx, closure.team
    elif summary.active is not None and summary.active.possession_id == possession_id:
        team = summary.active.team
        first = 1
        for record in index.records:
            if isinstance(record, ResolutionRecord) and any(
                change.to_possession_id == possession_id for change in record.possession_changes
            ):
                first = record.event_idx
        last = max((record_event_idx(record) or 0 for record in index.records), default=0)
        last = max(first, last)
    else:
        known = [record.possession_id for record in summary.closed]
        if summary.active is not None:
            known.append(summary.active.possession_id)
        raise TranscriptError(
            f"Unknown possession {possession_id!r}. Available IDs: {', '.join(known)}"
        )

    lines = [
        f"Game {start.game_id} | possession {possession_id} | team {team}",
        f"Game log: {'complete' if index.is_complete else 'unfinished'} | "
        f"possession: {'closed' if closure else 'open'} | events {first}–{last}",
        "Conditions: " + ", ".join(
            f"{key}={value}" for key, value in start.team_conditions.items()
        ),
    ]
    if closure is not None:
        lines.append(
            f"Possession points: A={closure.points_by_team['A']}, "
            f"B={closure.points_by_team['B']} | end reason: {closure.end_reason}"
        )
    lines.extend([
        "Boundary events may be shared with adjacent possessions.",
        "Decisions belong to possession_before; scoring entries retain their possession IDs.",
        "Recorded observations are authoritative; missing observation content is not inferred.",
    ])
    prior = _prior_context(index, first, last)
    if prior:
        lines.extend([
            "", "Earlier context (not new actions in this possession)",
            "Messages: delivery falls within this span. Captain calls: last replacement and",
            "subsequent maintains; t+1 visibility assumed, not proof of an agent's inbox.",
        ])
        for record in prior:
            lines.extend(_render_record(record))
            lines.append("")

    current_event = None
    for record in index.records:
        event = record_event_idx(record)
        if event is None or not first <= event <= last:
            continue
        if event != current_event:
            lines.extend(["", f"Event {event}"])
            current_event = event
        if isinstance(record, ResolutionRecord):
            lines.append(f"  Possession before decisions: {record.possession_before.possession_id}")
        lines.extend(_render_record(record))
    return "\n".join(lines) + "\n"
