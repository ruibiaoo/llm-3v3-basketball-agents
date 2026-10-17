"""Build and check possessions from explicit engine resolution outcomes.

The runner can feed each resolution to ``PossessionTracker`` and log the
returned ``possession_end`` records after the event's state record. Offline
analysis can reconstruct the same records and verify the saved game log.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from bball3x3.schemas.log import (
    GameEndRecord,
    GameStartRecord,
    PossessionEndRecord,
    PossessionRef,
    ResolutionRecord,
    StateRecord,
    Team,
)

from .log_reader import GameLogIndex, LogRecord, record_event_idx


class PossessionError(ValueError):
    """Resolution or saved possession records contradict the game timeline."""


@dataclass
class _OpenPossession:
    possession_id: str
    team: Team
    start_event_idx: int
    points_by_team: dict[Team, int]

    @property
    def ref(self) -> PossessionRef:
        return PossessionRef(possession_id=self.possession_id, team=self.team)

    def close(self, event_idx: int, reason: str) -> PossessionEndRecord:
        return PossessionEndRecord(
            possession_id=self.possession_id,
            team=self.team,
            points=self.points_by_team[self.team],
            points_by_team=self.points_by_team.copy(),
            end_reason=reason,
            start_event_idx=self.start_event_idx,
            end_event_idx=event_idx,
        )


class PossessionTracker:
    """Track explicit control, including free throws and retained possession.

    Each ``apply`` call is atomic: an invalid resolution leaves the tracker at
    its previous event. A transfer closes the outgoing possession at that
    event and opens the incoming one at the same event. The final active
    possession closes at game termination, even if it has zero points.
    """

    def __init__(self, start: GameStartRecord) -> None:
        self._active: _OpenPossession | None = _OpenPossession(
            start.initial_possession_id, start.first_possession, 1, {"A": 0, "B": 0}
        )
        self._seen_ids = {start.initial_possession_id}
        self._last_event_idx = 0
        self._ended = False
        self._closed: list[PossessionEndRecord] = []

    @property
    def active(self) -> PossessionRef | None:
        return self._active.ref if self._active is not None else None

    @property
    def last_event_idx(self) -> int:
        return self._last_event_idx

    @property
    def closed(self) -> tuple[PossessionEndRecord, ...]:
        return tuple(self._closed)

    def apply(self, resolution: ResolutionRecord) -> tuple[PossessionEndRecord, ...]:
        event_idx = resolution.event_idx
        if self._ended or self._active is None:
            raise PossessionError(f"event {event_idx}: game already ended")
        if event_idx != self._last_event_idx + 1:
            raise PossessionError(
                f"event {event_idx}: expected contiguous event {self._last_event_idx + 1}"
            )
        if resolution.possession_before != self._active.ref:
            raise PossessionError(f"event {event_idx}: possession_before does not match control")

        current = _OpenPossession(
            self._active.possession_id,
            self._active.team,
            self._active.start_event_idx,
            self._active.points_by_team.copy(),
        )
        seen_ids = self._seen_ids.copy()
        segments = {current.possession_id: current}
        closures: list[tuple[_OpenPossession, str]] = []
        for change in resolution.possession_changes:
            if change.from_possession_id != current.possession_id:
                raise PossessionError(f"event {event_idx}: broken possession change chain")
            if change.to_possession_id in seen_ids:
                raise PossessionError(
                    f"event {event_idx}: reused possession ID {change.to_possession_id}"
                )
            if change.kind == "CONTROL_TRANSFER" and change.to_team == current.team:
                raise PossessionError(f"event {event_idx}: control transfer must change team")
            closures.append((current, change.reason))
            seen_ids.add(change.to_possession_id)
            current = _OpenPossession(
                change.to_possession_id, change.to_team, event_idx, {"A": 0, "B": 0}
            )
            segments[current.possession_id] = current

        if resolution.game_over:
            if resolution.possession_after is not None or resolution.game_end_reason is None:
                raise PossessionError(f"event {event_idx}: invalid terminal resolution")
        elif resolution.possession_after != current.ref:
            raise PossessionError(f"event {event_idx}: possession_after does not match control")

        for score in resolution.scoring_events:
            segment = segments.get(score.possession_id)
            if segment is None:
                raise PossessionError(
                    f"event {event_idx}: scoring refers to an inactive possession"
                )
            segment.points_by_team[score.team] += score.points

        if resolution.game_over:
            assert resolution.game_end_reason is not None
            closures.append((current, resolution.game_end_reason))
        emitted = tuple(segment.close(event_idx, reason) for segment, reason in closures)
        self._closed.extend(emitted)
        self._active = None if resolution.game_over else current
        self._seen_ids = seen_ids
        self._last_event_idx = event_idx
        self._ended = resolution.game_over
        return emitted


@dataclass(frozen=True)
class PossessionSummary:
    """Reconstructed closures and any still-active possession in an unfinished log."""

    closed: tuple[PossessionEndRecord, ...]
    active: PossessionRef | None
    last_event_idx: int


def _items(records: GameLogIndex | Iterable[LogRecord]) -> list[LogRecord]:
    return list(records.records if isinstance(records, GameLogIndex) else records)


def reconstruct_possessions(records: GameLogIndex | Iterable[LogRecord]) -> PossessionSummary:
    """Reconstruct closures from the start and contiguous resolution records."""
    items = _items(records)
    if not items or not isinstance(items[0], GameStartRecord):
        raise PossessionError("game_start must be the first record")
    tracker = PossessionTracker(items[0])
    for record in items[1:]:
        if isinstance(record, GameStartRecord):
            raise PossessionError("duplicate game_start")
        if isinstance(record, ResolutionRecord):
            tracker.apply(record)
    return PossessionSummary(tracker.closed, tracker.active, tracker.last_event_idx)


def _score_from_state(state: StateRecord) -> dict[str, int]:
    score: Any = state.state.score if state.state is not None else state.raw_state.get("score")
    if not isinstance(score, dict) or set(score) != {"A", "B"}:
        raise PossessionError(f"event {state.event_idx}: state score is unavailable")
    if any(type(score[team]) is not int or score[team] < 0 for team in ("A", "B")):
        raise PossessionError(f"event {state.event_idx}: state score is invalid")
    return score


def validate_possessions(records: GameLogIndex | Iterable[LogRecord]) -> PossessionSummary:
    """Check saved closures, score snapshots and game totals against resolutions.

    An unfinished prefix may retain one active possession. For a complete log,
    every resolution must have a state and the terminal resolution must match
    ``game_end``. Raises ``PossessionError`` on the first contradiction.
    """
    items = _items(records)
    summary = reconstruct_possessions(items)
    actual = [record for record in items if isinstance(record, PossessionEndRecord)]
    if actual != list(summary.closed):
        raise PossessionError("possession_end records disagree with resolution outcomes")
    latest_event = 0
    states_seen: set[int] = set()
    for record in items:
        if isinstance(record, PossessionEndRecord):
            if record.end_event_idx not in states_seen or latest_event > record.end_event_idx:
                raise PossessionError(
                    f"event {record.end_event_idx}: possession_end is outside its event"
                )
        else:
            if isinstance(record, StateRecord):
                states_seen.add(record.event_idx)
            event_idx = record_event_idx(record)
            if event_idx is not None:
                latest_event = max(latest_event, event_idx)

    ends = [record for record in items if isinstance(record, GameEndRecord)]
    if len(ends) > 1 or (ends and items[-1] is not ends[0]):
        raise PossessionError("game_end must occur exactly once as the final record")
    end = ends[0] if ends else None
    if end is not None:
        if summary.active is not None or summary.last_event_idx != end.last_event_idx:
            raise PossessionError("game_end does not match the terminal resolution")
        terminal = next(
            (record for record in reversed(items) if isinstance(record, ResolutionRecord)), None
        )
        if terminal is None or not terminal.game_over or terminal.game_end_reason != end.end_reason:
            raise PossessionError("game_end reason does not match the terminal resolution")
        for team in ("A", "B"):
            count = sum(record.team == team for record in summary.closed)
            if count != end.totals.possessions_by_team[team]:
                raise PossessionError(f"team {team}: possession count disagrees with game_end")

    states = [record for record in items if isinstance(record, StateRecord)]
    state_by_event = {record.event_idx: record for record in states}
    if len(state_by_event) != len(states):
        raise PossessionError("duplicate state for a transition")
    if end is not None and set(state_by_event) != set(range(1, end.last_event_idx + 1)):
        raise PossessionError("complete game requires one state per resolution")
    scores = {"A": 0, "B": 0}
    for record in items:
        if not isinstance(record, ResolutionRecord):
            continue
        for scoring in record.scoring_events:
            scores[scoring.team] += scoring.points
        state = state_by_event.get(record.event_idx)
        if state is not None and _score_from_state(state) != scores:
            raise PossessionError(f"event {record.event_idx}: score differs from scoring events")
    if end is not None:
        if scores != end.final_score:
            raise PossessionError("final score differs from scoring events")
        if sum(record.points_by_team["A"] for record in summary.closed) != scores["A"]:
            raise PossessionError("team A possession points differ from final score")
        if sum(record.points_by_team["B"] for record in summary.closed) != scores["B"]:
            raise PossessionError("team B possession points differ from final score")
    return summary
