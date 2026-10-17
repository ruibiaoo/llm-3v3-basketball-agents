"""Per-game metrics computed from event-log records."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from bball3x3.schemas.log import (
    CaptainCallRecord,
    CSProposalRecord,
    CSWriteRecord,
    DecisionRecord,
    GameEndRecord,
    GameStartRecord,
    InvariantViolationRecord,
    MemoryProposalRecord,
    MemoryWriteRecord,
    MessageRecord,
    PossessionEndRecord,
    RefereeRecord,
    ResolutionRecord,
    StateRecord,
)

from .log_reader import GameLogIndex, LogRecord, record_event_idx, record_type
from .possessions import PossessionError, validate_possessions

Team = Literal["A", "B"]
MetricStatus = Literal[
    "ok",
    "undefined",
    "incomplete",
    "insufficient_evidence",
    "not_applicable",
]
_TEAMS: tuple[Team, Team] = ("A", "B")


@dataclass(frozen=True)
class RateMetric:
    """A numerator and denominator with an explicit evidence status."""

    numerator: int
    denominator: int
    status: MetricStatus = "ok"

    @property
    def rate(self) -> float | None:
        if self.status != "ok" or self.denominator == 0:
            return None
        return self.numerator / self.denominator

    def to_dict(self) -> dict[str, int | float | str | None]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.rate,
            "status": self.status,
        }


@dataclass(frozen=True)
class MemoryGrowthMetric:
    """Number of stored entries at game start and after each event."""

    initial_count: int | None
    final_count: int | None
    net_change: int | None
    peak_count: int | None
    by_event: dict[int, int]
    status: MetricStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_count": self.initial_count,
            "final_count": self.final_count,
            "net_change": self.net_change,
            "peak_count": self.peak_count,
            "by_event": {str(event): count for event, count in self.by_event.items()},
            "status": self.status,
        }


@dataclass(frozen=True)
class AdaptationMetrics:
    """Five fixed regulation bins plus an overtime bin for three rates."""

    regulation_duration_s: float | None
    ppp: dict[str, dict[str, RateMetric]]
    maur: dict[str, dict[str, RateMetric]]
    cssr: dict[str, dict[str, RateMetric]]

    def to_dict(self) -> dict[str, Any]:
        def rates(values: Mapping[str, Mapping[str, RateMetric]]) -> dict[str, Any]:
            return {
                team: {label: metric.to_dict() for label, metric in bins.items()}
                for team, bins in values.items()
            }

        return {
            "regulation_duration_s": self.regulation_duration_s,
            "binning": "five equal elapsed-regulation-time bins; OT separate",
            "regulation_bin_edges_fraction": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "ppp_attribution": "possession start",
            "maur_cssr_attribution": "event start",
            "ppp": rates(self.ppp),
            "maur": rates(self.maur),
            "cssr": rates(self.cssr),
        }


def _rate(
    numerator: int,
    denominator: int,
    *,
    complete: bool,
    evidence_complete: bool = True,
    applicable: bool = True,
) -> RateMetric:
    if not applicable:
        status: MetricStatus = "not_applicable"
    elif not complete:
        status = "incomplete"
    elif not evidence_complete:
        status = "insufficient_evidence"
    elif denominator == 0:
        status = "undefined"
    else:
        status = "ok"
    return RateMetric(numerator, denominator, status)


@dataclass
class GameMetrics:
    """Counts and rates for one game, separated by physical team where relevant."""

    game_complete: bool
    team_conditions: dict[str, str]
    ppp: dict[str, RateMetric]
    maar: dict[str, RateMetric]
    ccar: dict[str, RateMetric]
    cssr: dict[str, RateMetric]
    maur: dict[str, RateMetric]
    grounded_maur: dict[str, RateMetric]
    iar: dict[str, RateMetric]
    iar_game: RateMetric
    iar_by_reason: dict[str, dict[str, RateMetric]]
    sir: RateMetric
    sir_by_constraint: dict[str, RateMetric]
    referee_call_rate: RateMetric
    assist_rate: dict[str, RateMetric]
    turnover_rate: dict[str, RateMetric]
    screen_to_shot_conversion: dict[str, RateMetric]
    coordination_conflict_rate: dict[str, RateMetric]
    memory_growth: dict[str, MemoryGrowthMetric]
    adaptation: AdaptationMetrics
    referee_calls_by_type: dict[str, int] = field(default_factory=dict)
    memory_validation_complete: dict[str, bool] = field(default_factory=dict)
    grounding_context_complete: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary without discarding denominator status."""

        def team_rates(values: Mapping[str, RateMetric]) -> dict[str, Any]:
            return {team: metric.to_dict() for team, metric in values.items()}

        return {
            "game_complete": self.game_complete,
            "team_conditions": dict(self.team_conditions),
            "ppp": team_rates(self.ppp),
            "maar": team_rates(self.maar),
            "ccar": team_rates(self.ccar),
            "cssr": team_rates(self.cssr),
            "maur": team_rates(self.maur),
            "grounded_maur": team_rates(self.grounded_maur),
            "iar": team_rates(self.iar),
            "iar_game": self.iar_game.to_dict(),
            "iar_by_reason": {
                team: team_rates(metrics) for team, metrics in self.iar_by_reason.items()
            },
            "sir": self.sir.to_dict(),
            "sir_by_constraint": team_rates(self.sir_by_constraint),
            "referee_call_rate": self.referee_call_rate.to_dict(),
            "assist_rate": team_rates(self.assist_rate),
            "turnover_rate": team_rates(self.turnover_rate),
            "screen_to_shot_conversion": team_rates(self.screen_to_shot_conversion),
            "coordination_conflict_rate": team_rates(self.coordination_conflict_rate),
            "memory_growth": {
                team: metric.to_dict() for team, metric in self.memory_growth.items()
            },
            "adaptation": self.adaptation.to_dict(),
            "referee_calls_by_type": dict(self.referee_calls_by_type),
            "memory_validation_complete": dict(self.memory_validation_complete),
            "grounding_context_complete": dict(self.grounding_context_complete),
        }


@dataclass(frozen=True)
class _DirectiveGroup:
    team: Team
    delivery_event: int
    recipient: str
    expected: tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class _TacticCall:
    team: Team
    event_idx: int
    cs_key: str | None


def _data(record: LogRecord) -> Mapping[str, Any]:
    if isinstance(record, BaseModel):
        return record.model_dump(mode="python")
    return record


def _nested(value: Any) -> Mapping[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return value if isinstance(value, Mapping) else {}


def _records(records: GameLogIndex | Iterable[LogRecord]) -> list[LogRecord]:
    return list(records.records if isinstance(records, GameLogIndex) else records)


def _team(value: Any) -> Team | None:
    return value if value in _TEAMS else None


def _game_records(
    records: Sequence[LogRecord],
) -> tuple[GameStartRecord | None, GameEndRecord | None, bool]:
    start = records[0] if records and isinstance(records[0], GameStartRecord) else None
    end = records[-1] if records and isinstance(records[-1], GameEndRecord) else None
    return start, end, end is not None


def _feature_enabled(start: GameStartRecord | None, team: Team, feature: str) -> bool:
    if start is None:
        return True
    return bool(getattr(start.coordination[team], feature))


def _decision_map(
    records: Sequence[LogRecord],
) -> tuple[dict[tuple[int, str], DecisionRecord], dict[Team, list[DecisionRecord]]]:
    by_key: dict[tuple[int, str], DecisionRecord] = {}
    by_team: dict[Team, list[DecisionRecord]] = {team: [] for team in _TEAMS}
    for record in records:
        if not isinstance(record, DecisionRecord):
            continue
        by_key[(record.event_idx, record.player)] = record
        by_team[record.team].append(record)
    return by_key, by_team


def _directive_groups(records: Sequence[LogRecord], channel: str) -> list[_DirectiveGroup]:
    grouped: defaultdict[tuple[Team, int, str], list[tuple[str, str | None]]] = defaultdict(list)
    captain_messages: defaultdict[
        tuple[int, Team], dict[str, list[tuple[str, str | None]]]
    ] = defaultdict(lambda: defaultdict(list))

    for record in records:
        if not isinstance(record, MessageRecord):
            continue
        if record.channel != channel or record.suppressed or record.directive is None:
            continue
        delivery_event = record.delivered_at or record.event_idx + 1
        key = (record.team, delivery_event, record.directive.to)
        grouped[key].append((record.directive.act.upper(), record.directive.target))
        if channel == "captain":
            captain_messages[(record.event_idx, record.team)][record.directive.to].append(
                (record.directive.act.upper(), record.directive.target)
            )

    if channel == "captain":
        current: dict[Team, dict[str, list[tuple[str, str | None]]]] = {
            team: {} for team in _TEAMS
        }
        calls = sorted(
            (record for record in records if isinstance(record, CaptainCallRecord)),
            key=lambda record: record.event_idx,
        )
        for record in calls:
            message_directives = captain_messages.get((record.event_idx, record.team))
            parsed: defaultdict[str, list[tuple[str, str | None]]] = defaultdict(list)
            for raw in record.directives:
                entry = _nested(raw)
                directive = _nested(entry.get("directive")) or entry
                recipient = directive.get("to")
                action = directive.get("act")
                target = directive.get("target")
                if isinstance(recipient, str) and isinstance(action, str):
                    parsed[recipient].append(
                        (action.upper(), target if isinstance(target, str) else None)
                    )
            if message_directives:
                current[record.team] = dict(message_directives)
            elif not record.maintain:
                current[record.team] = dict(parsed)

            if not message_directives:
                for recipient, expected in current[record.team].items():
                    grouped[(record.team, record.event_idx + 1, recipient)].extend(expected)

    return [
        _DirectiveGroup(team, delivery, recipient, tuple(expected))
        for (team, delivery, recipient), expected in grouped.items()
    ]


def _instruction_adherence(
    records: Sequence[LogRecord],
    channel: str,
    window: int,
    complete: bool,
    start: GameStartRecord | None,
    decision_evidence: Mapping[Team, bool],
) -> dict[str, RateMetric]:
    decisions, _ = _decision_map(records)
    matched: defaultdict[Team, int] = defaultdict(int)
    addressed: defaultdict[Team, int] = defaultdict(int)

    for group in _directive_groups(records, channel):
        responses = [
            decisions[(event_idx, group.recipient)]
            for event_idx in range(group.delivery_event, group.delivery_event + window)
            if (event_idx, group.recipient) in decisions
        ]
        if not responses:
            continue
        addressed[group.team] += 1
        if any(
            response.parsed_output is not None
            and any(
                response.parsed_output.action.upper() == action
                and (target is None or response.parsed_output.target == target)
                for action, target in group.expected
            )
            for response in responses
        ):
            matched[group.team] += 1

    feature = "comms" if channel == "teammate" else "captain"
    return {
        team: _rate(
            matched[team],
            addressed[team],
            complete=complete,
            evidence_complete=decision_evidence[team],
            applicable=_feature_enabled(start, team, feature),
        )
        for team in _TEAMS
    }


def _tactic_calls(records: Sequence[LogRecord]) -> list[_TacticCall]:
    calls: list[_TacticCall] = []
    captain_message_groups: set[tuple[int, Team]] = set()
    for record in records:
        if not isinstance(record, MessageRecord):
            continue
        if record.kind != "tactic" or record.suppressed:
            continue
        calls.append(_TacticCall(record.team, record.event_idx, record.cs))
        if record.channel == "captain":
            captain_message_groups.add((record.event_idx, record.team))

    for record in records:
        if not isinstance(record, CaptainCallRecord):
            continue
        if (record.event_idx, record.team) in captain_message_groups:
            continue
        if record.tactic is not None:
            calls.append(_TacticCall(record.team, record.event_idx, record.cs))
    return calls


def _visible_at(data: Mapping[str, Any], event_idx: int) -> int:
    value = data.get("visible_at", data.get("delivered_at"))
    return value if isinstance(value, int) and not isinstance(value, bool) else event_idx + 1


def _cs_availability(records: Sequence[LogRecord]) -> dict[Team, dict[str, int]]:
    available: dict[Team, dict[str, int]] = {team: {} for team in _TEAMS}
    for record in records:
        if isinstance(record, CSWriteRecord):
            available[record.team].setdefault(record.key, record.visible_at)
            continue
        if record_type(record) != "cs_write":
            continue
        data = _data(record)
        team = _team(data.get("team"))
        key = data.get("key")
        event_idx = record_event_idx(record)
        if team is not None and isinstance(key, str) and key and event_idx is not None:
            available[team].setdefault(key, _visible_at(data, event_idx))

    # Proposal records predate explicit write records. Use them as a fallback
    # only when the actual store mutation is absent from this log.
    for record in records:
        if isinstance(record, CSProposalRecord) and record.accepted and record.key:
            available[record.team].setdefault(record.key, record.event_idx + 1)
    return available


def _cssr(
    records: Sequence[LogRecord],
    complete: bool,
    start: GameStartRecord | None,
) -> dict[str, RateMetric]:
    availability = _cs_availability(records)
    valid: defaultdict[Team, int] = defaultdict(int)
    total: defaultdict[Team, int] = defaultdict(int)
    for call in _tactic_calls(records):
        total[call.team] += 1
        if call.cs_key is not None:
            visible_at = availability[call.team].get(call.cs_key)
            if visible_at is not None and visible_at <= call.event_idx:
                valid[call.team] += 1
    return {
        team: _rate(
            valid[team],
            total[team],
            complete=complete,
            applicable=_feature_enabled(start, team, "captain"),
        )
        for team in _TEAMS
    }


def _normalise_initial_memory(
    known_memory_ids: Mapping[str, Iterable[str]] | None,
    memory_opponents: Mapping[str, Mapping[str, str]] | None,
    start: GameStartRecord | None,
) -> tuple[
    dict[Team, dict[str, int]],
    dict[Team, dict[str, int]],
    dict[Team, dict[str, str]],
    dict[Team, bool],
]:
    availability: dict[Team, dict[str, int]] = {team: {} for team in _TEAMS}
    unavailable_from: dict[Team, dict[str, int]] = {team: {} for team in _TEAMS}
    opponents: dict[Team, dict[str, str]] = {team: {} for team in _TEAMS}
    complete = {team: False for team in _TEAMS}

    if start is not None and start.initial_memory is not None:
        for team in _TEAMS:
            complete[team] = True
            for entry in start.initial_memory[team]:
                availability[team][entry.id] = 0
                opponents[team][entry.id] = entry.opponent

    if known_memory_ids is not None:
        for team in _TEAMS:
            if team in known_memory_ids:
                complete[team] = True
            for memory_id in known_memory_ids.get(team, ()):
                if isinstance(memory_id, str) and memory_id:
                    availability[team].setdefault(memory_id, 0)
    if memory_opponents is not None:
        for team in _TEAMS:
            for memory_id, opponent in memory_opponents.get(team, {}).items():
                if isinstance(memory_id, str) and isinstance(opponent, str):
                    opponents[team][memory_id] = opponent
    return availability, unavailable_from, opponents, complete


def _add_memory_writes(
    records: Sequence[LogRecord],
    availability: dict[Team, dict[str, int]],
    unavailable_from: dict[Team, dict[str, int]],
    opponents: dict[Team, dict[str, str]],
) -> None:
    explicit_ids: set[tuple[Team, str]] = set()
    for record in records:
        if isinstance(record, MemoryWriteRecord):
            explicit_ids.add((record.team, record.memory_id))
            availability[record.team].setdefault(record.memory_id, record.visible_at)
            opponents[record.team][record.memory_id] = record.opponent
            for evicted_id in record.evicted_ids:
                unavailable_from[record.team][evicted_id] = record.visible_at
            continue
        if record_type(record) != "memory_write":
            continue
        data = _data(record)
        team = _team(data.get("team"))
        event_idx = record_event_idx(record)
        memory_id = data.get("id", data.get("memory_id"))
        opponent = data.get("opponent")
        if team is None or event_idx is None or not isinstance(memory_id, str) or not memory_id:
            continue
        explicit_ids.add((team, memory_id))
        availability[team].setdefault(memory_id, _visible_at(data, event_idx))
        if isinstance(opponent, str):
            opponents[team][memory_id] = opponent
        evicted = data.get("evicted_ids", data.get("evicted_id", ()))
        if isinstance(evicted, str):
            evicted = (evicted,)
        if isinstance(evicted, (list, tuple, set)):
            for evicted_id in evicted:
                if isinstance(evicted_id, str) and evicted_id:
                    unavailable_from[team][evicted_id] = _visible_at(data, event_idx)

    # Accepted proposals are used as a compatibility fallback for older logs
    # that recorded the mutation only on the proposal record.
    for record in records:
        if not isinstance(record, MemoryProposalRecord) or not record.accepted:
            continue
        for memory_id in (record.new_id, record.merged_into):
            if memory_id and (record.team, memory_id) not in explicit_ids:
                availability[record.team].setdefault(memory_id, record.event_idx + 1)
                opponents[record.team][memory_id] = record.proposal.opponent


def _explicit_involved_opponents(
    decision: DecisionRecord,
    resolution: ResolutionRecord | None,
) -> tuple[set[str], bool]:
    if resolution is None:
        return set(), False
    if resolution.involved_opponents is not None:
        # The resolver's record describes the action that actually ran, so it
        # takes precedence over the agent's possibly remapped proposal.
        return set(resolution.involved_opponents[decision.player]), True

    involved: set[str] = set()
    own_team = decision.team
    target = decision.parsed_output.target if decision.parsed_output is not None else None
    if isinstance(target, str) and target[:1] in _TEAMS and target[:1] != own_team:
        involved.add(target)
    player_outcome = resolution.outcomes.get(decision.player)
    if not isinstance(player_outcome, Mapping):
        return involved, False
    explicit = player_outcome.get("involved_opponents")
    if not isinstance(explicit, (list, tuple, set)):
        return involved, False
    involved.update(
        value
        for value in explicit
        if isinstance(value, str) and value[:1] in _TEAMS and value[:1] != own_team
    )
    return involved, True


def _valid_memory_refs(
    decision: DecisionRecord,
    availability: Mapping[Team, Mapping[str, int]],
    unavailable_from: Mapping[Team, Mapping[str, int]],
) -> list[str]:
    refs = decision.memory_ref
    if refs is None and decision.parsed_output is not None:
        refs = decision.parsed_output.memory_ref
    return [
        ref
        for ref in refs or ()
        if ref in availability[decision.team]
        and availability[decision.team][ref] <= decision.event_idx
        and (
            ref not in unavailable_from[decision.team]
            or decision.event_idx < unavailable_from[decision.team][ref]
        )
    ]


def _memory_metrics(
    records: Sequence[LogRecord],
    decisions_by_team: Mapping[Team, Sequence[DecisionRecord]],
    complete: bool,
    decision_evidence: Mapping[Team, bool],
    known_memory_ids: Mapping[str, Iterable[str]] | None,
    memory_opponents: Mapping[str, Mapping[str, str]] | None,
    start: GameStartRecord | None,
) -> tuple[dict[str, RateMetric], dict[str, RateMetric], dict[str, bool], dict[str, bool]]:
    availability, unavailable_from, opponents, validation_complete = (
        _normalise_initial_memory(known_memory_ids, memory_opponents, start)
    )
    _add_memory_writes(records, availability, unavailable_from, opponents)
    resolutions = {
        record.event_idx: record for record in records if isinstance(record, ResolutionRecord)
    }
    used: defaultdict[Team, int] = defaultdict(int)
    grounded: defaultdict[Team, int] = defaultdict(int)
    grounding_complete = {
        team: validation_complete[team]
        and all(memory_id in opponents[team] for memory_id in availability[team])
        for team in _TEAMS
    }

    for team in _TEAMS:
        for decision in decisions_by_team[team]:
            valid_refs = _valid_memory_refs(decision, availability, unavailable_from)
            if not valid_refs:
                continue
            used[team] += 1
            involved, involvement_complete = _explicit_involved_opponents(
                decision, resolutions.get(decision.event_idx)
            )
            if any(opponents[team].get(ref) in involved for ref in valid_refs):
                grounded[team] += 1
            elif not involvement_complete:
                grounding_complete[team] = False

    maur = {
        team: _rate(
            used[team],
            len(decisions_by_team[team]),
            complete=complete,
            evidence_complete=validation_complete[team] and decision_evidence[team],
        )
        for team in _TEAMS
    }
    grounded_maur = {
        team: _rate(
            grounded[team],
            len(decisions_by_team[team]),
            complete=complete,
            evidence_complete=grounding_complete[team] and decision_evidence[team],
        )
        for team in _TEAMS
    }
    return maur, grounded_maur, validation_complete, grounding_complete


_ZONES = frozenset(
    {
        "TOP", "LEFT_WING", "RIGHT_WING", "LEFT_CORNER", "RIGHT_CORNER",
        "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_BLOCK", "RIGHT_BLOCK", "PAINT",
    }
)
_TIME_BINS = ("q1", "q2", "q3", "q4", "q5", "ot")
_TURNOVER_END_REASONS = frozenset(
    {"TURNOVER", "STEAL", "INTERCEPTION", "SHOT_CLOCK_VIOLATION", "OFFENSIVE_FOUL"}
)


def _descriptive_outcome_rates(
    records: Sequence[LogRecord],
    end: GameEndRecord | None,
    complete: bool,
    ppp: Mapping[str, RateMetric],
) -> tuple[dict[str, RateMetric], dict[str, RateMetric], dict[str, RateMetric]]:
    resolutions = [record for record in records if isinstance(record, ResolutionRecord)]
    resolution_evidence = (
        end is not None
        and len(resolutions) == end.last_event_idx
        and {record.event_idx for record in resolutions} == set(range(1, end.last_event_idx + 1))
    )
    shot_evidence = resolution_evidence
    screen_evidence = resolution_evidence
    turnover_evidence = resolution_evidence
    made: defaultdict[Team, int] = defaultdict(int)
    assisted: defaultdict[Team, int] = defaultdict(int)
    screens: defaultdict[Team, int] = defaultdict(int)
    screens_by_id: dict[str, tuple[int, Team, str]] = {}
    linked_shots: list[tuple[int, Team, str, str]] = []
    turnovers: dict[Team, set[str]] = {team: set() for team in _TEAMS}
    closed = {
        record.possession_id: record
        for record in records
        if isinstance(record, PossessionEndRecord)
    }
    player_team = {
        (record.event_idx, record.player): record.team
        for record in records
        if isinstance(record, DecisionRecord)
    }

    for resolution in resolutions:
        if resolution.shots is None:
            shot_evidence = False
            screen_evidence = False
        else:
            for shot in resolution.shots:
                if player_team.get((resolution.event_idx, shot.shooter)) != shot.team:
                    shot_evidence = False
                if shot.assist_by is not None and (
                    player_team.get((resolution.event_idx, shot.assist_by)) != shot.team
                ):
                    shot_evidence = False
                if shot.made:
                    made[shot.team] += 1
                    if shot.assist_by is not None:
                        assisted[shot.team] += 1
                if shot.screen_id is not None:
                    linked_shots.append(
                        (resolution.event_idx, shot.team, shot.shooter, shot.screen_id)
                    )
            for team in _TEAMS:
                scored = sum(
                    scoring.team == team and scoring.source == "FIELD_GOAL"
                    for scoring in resolution.scoring_events
                )
                made_here = sum(shot.team == team and shot.made for shot in resolution.shots)
                if scored != made_here:
                    shot_evidence = False

        if resolution.screens is None:
            screen_evidence = False
        else:
            for screen in resolution.screens:
                screens[screen.team] += 1
                if (
                    player_team.get((resolution.event_idx, screen.screener)) != screen.team
                    or player_team.get((resolution.event_idx, screen.freed_player)) != screen.team
                    or resolution.actions.get(screen.screener) != "SCREEN"
                ):
                    screen_evidence = False
                if screen.screen_id in screens_by_id:
                    screen_evidence = False
                screens_by_id[screen.screen_id] = (
                    resolution.event_idx, screen.team, screen.freed_player
                )

        if resolution.turnovers is None:
            turnover_evidence = False
        else:
            for turnover in resolution.turnovers:
                if turnover.player is not None and (
                    player_team.get((resolution.event_idx, turnover.player)) != turnover.team
                ):
                    turnover_evidence = False
                possession = closed.get(turnover.possession_id)
                if (
                    possession is None
                    or possession.team != turnover.team
                    or possession.end_event_idx != resolution.event_idx
                    or turnover.possession_id in turnovers[turnover.team]
                ):
                    turnover_evidence = False
                turnovers[turnover.team].add(turnover.possession_id)

    for possession in closed.values():
        if (
            possession.end_reason.upper() in _TURNOVER_END_REASONS
            and possession.possession_id not in turnovers[possession.team]
        ):
            turnover_evidence = False

    converted: set[str] = set()
    for event_idx, team, shooter, screen_id in linked_shots:
        screen = screens_by_id.get(screen_id)
        if screen is None or screen[1:] != (team, shooter):
            screen_evidence = False
        elif 0 <= event_idx - screen[0] <= 1:
            converted.add(screen_id)

    assist_rate = {
        team: _rate(
            assisted[team], made[team], complete=complete, evidence_complete=shot_evidence
        )
        for team in _TEAMS
    }
    turnover_rate = {
        team: _rate(
            len(turnovers[team]),
            ppp[team].denominator,
            complete=complete,
            evidence_complete=turnover_evidence and ppp[team].status in ("ok", "undefined"),
        )
        for team in _TEAMS
    }
    screen_conversion = {
        team: _rate(
            sum(screens_by_id[screen_id][1] == team for screen_id in converted),
            screens[team],
            complete=complete,
            evidence_complete=screen_evidence,
        )
        for team in _TEAMS
    }
    return assist_rate, turnover_rate, screen_conversion


def _coordination_conflicts(
    records: Sequence[LogRecord],
    end: GameEndRecord | None,
    complete: bool,
    decision_evidence: Mapping[Team, bool],
) -> dict[str, RateMetric]:
    by_event: defaultdict[tuple[int, Team], dict[str, str]] = defaultdict(dict)
    unique = {team: True for team in _TEAMS}
    for record in records:
        if not isinstance(record, DecisionRecord):
            continue
        key = (record.event_idx, record.team)
        if record.player in by_event[key]:
            unique[record.team] = False
        by_event[key][record.player] = record.executed_target or ""

    eligible: defaultdict[Team, int] = defaultdict(int)
    conflicts: defaultdict[Team, int] = defaultdict(int)
    for (_, team), targets in by_event.items():
        zone_targets = [target for target in targets.values() if target in _ZONES]
        if len(zone_targets) < 2:
            continue
        eligible[team] += 1
        if len(set(zone_targets)) < len(zone_targets):
            conflicts[team] += 1
    if end is not None:
        for team in _TEAMS:
            if any(
                len(by_event.get((event_idx, team), {})) != 3
                for event_idx in range(1, end.last_event_idx + 1)
            ):
                unique[team] = False
    return {
        team: _rate(
            conflicts[team],
            eligible[team],
            complete=complete,
            evidence_complete=decision_evidence[team] and unique[team],
        )
        for team in _TEAMS
    }


def _memory_growth(
    records: Sequence[LogRecord],
    start: GameStartRecord | None,
    end: GameEndRecord | None,
    complete: bool,
    known_memory_ids: Mapping[str, Iterable[str]] | None,
) -> dict[str, MemoryGrowthMetric]:
    event_limit = end.last_event_idx if end is not None else max(
        (record.event_idx for record in records if isinstance(record, ResolutionRecord)),
        default=0,
    )
    writes: defaultdict[tuple[int, Team], list[MemoryWriteRecord]] = defaultdict(list)
    written_ids: set[tuple[int, Team, str]] = set()
    for record in records:
        if isinstance(record, MemoryWriteRecord):
            writes[(record.event_idx, record.team)].append(record)
            written_ids.add((record.event_idx, record.team, record.memory_id))

    result: dict[str, MemoryGrowthMetric] = {}
    for team in _TEAMS:
        if start is not None and start.initial_memory is not None:
            initial = {entry.id for entry in start.initial_memory[team]}
        elif known_memory_ids is not None and team in known_memory_ids:
            initial = set(known_memory_ids[team])
        else:
            result[team] = MemoryGrowthMetric(None, None, None, None, {},
                "incomplete" if not complete else "insufficient_evidence")
            continue

        active = initial.copy()
        counts = {0: len(initial)}
        evidence = True
        for event_idx in range(1, event_limit + 1):
            for write in writes[(event_idx, team)]:
                for evicted_id in write.evicted_ids:
                    if evicted_id not in active:
                        evidence = False
                    active.discard(evicted_id)
                active.add(write.memory_id)
            counts[event_idx] = len(active)
        for record in records:
            if isinstance(record, MemoryWriteRecord) and record.team == team:
                if record.event_idx > event_limit:
                    evidence = False
            elif isinstance(record, MemoryProposalRecord) and record.team == team:
                if record.accepted:
                    memory_id = record.new_id or record.merged_into
                    if not memory_id or (record.event_idx, team, memory_id) not in written_ids:
                        evidence = False
        status: MetricStatus = (
            "incomplete" if not complete else "ok" if evidence else "insufficient_evidence"
        )
        result[team] = MemoryGrowthMetric(
            len(initial), len(active), len(active) - len(initial), max(counts.values()),
            counts, status,
        )
    return result


def _state_clock_and_phase(state: StateRecord | None) -> tuple[float, str] | None:
    if state is None:
        return None
    if state.state is not None:
        return state.state.game_clock_s, state.state.phase
    raw = state.raw_state or {}
    clock, phase = raw.get("game_clock_s"), raw.get("phase")
    if isinstance(clock, (int, float)) and not isinstance(clock, bool) and isinstance(phase, str):
        return float(clock), phase
    return None


def _adaptation_metrics(
    records: Sequence[LogRecord],
    start: GameStartRecord | None,
    end: GameEndRecord | None,
    complete: bool,
    ppp: Mapping[str, RateMetric],
    maur: Mapping[str, RateMetric],
    cssr: Mapping[str, RateMetric],
    known_memory_ids: Mapping[str, Iterable[str]] | None,
    memory_opponents: Mapping[str, Mapping[str, str]] | None,
) -> AdaptationMetrics:
    duration = start.initial_state.game_clock_s if start is not None else None
    valid_duration = (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(duration)
        and duration > 0
    )
    configured_duration = (
        start.config_snapshot.get("rules", {}).get("game_length_s")
        if start is not None and isinstance(start.config_snapshot.get("rules"), Mapping)
        else None
    )
    states = {record.event_idx: record for record in records if isinstance(record, StateRecord)}
    state_count = sum(isinstance(record, StateRecord) for record in records)
    resolutions = [record for record in records if isinstance(record, ResolutionRecord)]
    last_event = end.last_event_idx if end is not None else max(states, default=0)
    timing_evidence = (
        valid_duration
        and start is not None
        and end is not None
        and len(states) == state_count
        and set(states) == set(range(1, last_event + 1))
        and len(resolutions) == last_event
        and {record.event_idx for record in resolutions} == set(range(1, last_event + 1))
    )
    for state in states.values():
        timing = _state_clock_and_phase(state)
        if (
            timing is None
            or not valid_duration
            or not math.isfinite(timing[0])
            or timing[0] < 0
            or timing[0] > duration
            or timing[1] not in ("LIVE", "DEAD", "FT", "OT")
        ):
            timing_evidence = False
    if configured_duration is not None and configured_duration != duration:
        timing_evidence = False
    event_bins: dict[int, str] = {}
    for event_idx in range(1, last_event + 1):
        previous = (
            (start.initial_state.game_clock_s, start.initial_state.phase)
            if event_idx == 1 and start is not None else
            _state_clock_and_phase(states.get(event_idx - 1))
        )
        if previous is None or not valid_duration:
            timing_evidence = False
            continue
        clock, phase = previous
        if not math.isfinite(clock) or clock < 0 or clock > duration:
            timing_evidence = False
            continue
        if phase == "OT":
            event_bins[event_idx] = "ot"
        elif phase in ("LIVE", "DEAD", "FT"):
            elapsed_fraction = (duration - clock) / duration
            event_bins[event_idx] = f"q{min(5, int(elapsed_fraction * 5) + 1)}"
        else:
            timing_evidence = False

    ot_ids: set[str] = set()
    ot_started = False
    for resolution in (record for record in records if isinstance(record, ResolutionRecord)):
        for change in resolution.possession_changes:
            if change.kind == "PERIOD_BOUNDARY":
                ot_started = True
            if ot_started:
                ot_ids.add(change.to_possession_id)
    if end is not None and (
        end.ot_flag != ot_started or ("ot" in event_bins.values() and not end.ot_flag)
    ):
        timing_evidence = False

    counts: dict[str, dict[Team, dict[str, list[int]]]] = {
        metric: {
            team: {label: [0, 0] for label in _TIME_BINS} for team in _TEAMS
        }
        for metric in ("ppp", "maur", "cssr")
    }
    for record in records:
        if not isinstance(record, PossessionEndRecord):
            continue
        label = "ot" if record.possession_id in ot_ids else event_bins.get(record.start_event_idx)
        if label is None:
            timing_evidence = False
            continue
        counts["ppp"][record.team][label][1] += 1
        for team in _TEAMS:
            counts["ppp"][team][label][0] += record.points_by_team[team]

    availability, unavailable_from, opponents, _ = _normalise_initial_memory(
        known_memory_ids, memory_opponents, start
    )
    _add_memory_writes(records, availability, unavailable_from, opponents)
    for record in records:
        if not isinstance(record, DecisionRecord):
            continue
        label = event_bins.get(record.event_idx)
        if label is None:
            timing_evidence = False
            continue
        slot = counts["maur"][record.team][label]
        slot[1] += 1
        if _valid_memory_refs(record, availability, unavailable_from):
            slot[0] += 1

    cs_availability = _cs_availability(records)
    for call in _tactic_calls(records):
        label = event_bins.get(call.event_idx)
        if label is None:
            timing_evidence = False
            continue
        slot = counts["cssr"][call.team][label]
        slot[1] += 1
        if call.cs_key is not None:
            visible_at = cs_availability[call.team].get(call.cs_key)
            if visible_at is not None and visible_at <= call.event_idx:
                slot[0] += 1

    def rates(metric: str, overall: Mapping[str, RateMetric]) -> dict[str, dict[str, RateMetric]]:
        return {
            team: {
                label: _rate(
                    *counts[metric][team][label],
                    complete=complete,
                    evidence_complete=timing_evidence
                    and overall[team].status in ("ok", "undefined", "not_applicable"),
                    applicable=overall[team].status != "not_applicable",
                )
                for label in _TIME_BINS
            }
            for team in _TEAMS
        }

    return AdaptationMetrics(
        float(duration) if valid_duration else None,
        rates("ppp", ppp), rates("maur", maur), rates("cssr", cssr),
    )


def compute_game_metrics(
    records: GameLogIndex | Iterable[LogRecord],
    *,
    window: int = 1,
    known_memory_ids: Mapping[str, Iterable[str]] | None = None,
    memory_opponents: Mapping[str, Mapping[str, str]] | None = None,
) -> GameMetrics:
    """Compute per-team coordination/adaptation metrics and game validity metrics.

    ``window=1`` checks the decision at message delivery. ``window=2`` counts
    an addressed opportunity as aligned when either of the next two decisions
    matches at least one directive. Initial memory is read from game_start when
    present; team-scoped mappings can supply it for older logs.
    """
    if window not in (1, 2):
        raise ValueError("window must be 1 or 2")
    items = _records(records)
    memory_ids = (
        {team: tuple(ids) for team, ids in known_memory_ids.items()}
        if known_memory_ids is not None else None
    )
    start, end, complete = _game_records(items)
    team_conditions = dict(start.team_conditions) if start is not None else {}
    _, decisions_by_team = _decision_map(items)
    decision_evidence = {
        team: end is not None
        and len(decisions_by_team[team]) == end.totals.decisions_by_team[team]
        for team in _TEAMS
    }

    points: defaultdict[Team, int] = defaultdict(int)
    possessions: defaultdict[Team, int] = defaultdict(int)
    for record in items:
        if not isinstance(record, PossessionEndRecord):
            continue
        possessions[record.team] += 1
        for team in _TEAMS:
            points[team] += record.points_by_team[team]
    possession_evidence = False
    if complete:
        try:
            validate_possessions(items)
        except PossessionError:
            pass
        else:
            possession_evidence = True
    ppp = {}
    for team in _TEAMS:
        ppp[team] = _rate(
            points[team],
            possessions[team],
            complete=complete,
            evidence_complete=possession_evidence,
        )

    maar = _instruction_adherence(
        items, "teammate", window, complete, start, decision_evidence
    )
    ccar = _instruction_adherence(
        items, "captain", window, complete, start, decision_evidence
    )
    cssr = _cssr(items, complete, start)
    maur, grounded_maur, memory_complete, grounding_complete = _memory_metrics(
        items,
        decisions_by_team,
        complete,
        decision_evidence,
        memory_ids,
        memory_opponents,
        start,
    )

    assist_rate, turnover_rate, screen_conversion = _descriptive_outcome_rates(
        items, end, complete, ppp
    )
    coordination_conflict_rate = _coordination_conflicts(
        items, end, complete, decision_evidence
    )
    memory_growth = _memory_growth(items, start, end, complete, memory_ids)
    adaptation = _adaptation_metrics(
        items, start, end, complete, ppp, maur, cssr, memory_ids, memory_opponents
    )

    invalid_counts: defaultdict[Team, int] = defaultdict(int)
    invalid_by_reason: dict[Team, defaultdict[str, int]] = {
        team: defaultdict(int) for team in _TEAMS
    }
    for team in _TEAMS:
        for decision in decisions_by_team[team]:
            if decision.invalid_reason is not None:
                invalid_counts[team] += 1
                invalid_by_reason[team][decision.invalid_reason] += 1
    iar = {
        team: _rate(
            invalid_counts[team],
            len(decisions_by_team[team]),
            complete=complete,
            evidence_complete=decision_evidence[team],
        )
        for team in _TEAMS
    }
    total_decisions = sum(len(decisions_by_team[team]) for team in _TEAMS)
    total_invalid = sum(invalid_counts.values())
    iar_game = _rate(
        total_invalid,
        total_decisions,
        complete=complete,
        evidence_complete=end is not None and total_decisions == end.totals.decisions,
    )
    iar_by_reason = {
        team: {
            reason: _rate(
                count,
                len(decisions_by_team[team]),
                complete=complete,
                evidence_complete=decision_evidence[team],
            )
            for reason, count in sorted(invalid_by_reason[team].items())
        }
        for team in _TEAMS
    }

    state_records = [record for record in items if isinstance(record, StateRecord)]
    states = {record.event_idx: record for record in state_records}
    violations: defaultdict[str, set[int]] = defaultdict(set)
    violated_events: set[int] = set()
    violation_events_valid = True
    for record in items:
        if not isinstance(record, InvariantViolationRecord):
            continue
        if record.event_idx not in states:
            violation_events_valid = False
            continue
        violated_events.add(record.event_idx)
        violations[record.constraint_name].add(record.event_idx)

    expected_events = set(range(1, end.last_event_idx + 1)) if end is not None else set(states)
    sir_evidence = (
        end is not None
        and set(states) == expected_events
        and len(states) == len(state_records)
        and violation_events_valid
        and all(
            state.invariant_checks_completed and set(state.constraints_checked) == set(range(1, 21))
            for state in states.values()
        )
    )
    sir = _rate(
        len(violated_events), len(states), complete=complete, evidence_complete=sir_evidence
    )
    sir_by_constraint = {
        name: _rate(len(events), len(states), complete=complete, evidence_complete=sir_evidence)
        for name, events in sorted(violations.items())
    }

    referee_by_event: defaultdict[int, list[RefereeRecord]] = defaultdict(list)
    calls_by_type: defaultdict[str, int] = defaultdict(int)
    for record in items:
        if isinstance(record, RefereeRecord):
            referee_by_event[record.event_idx].append(record)
            if record.decision.call != "NONE":
                calls_by_type[record.decision.call] += 1
    referee_evidence = (
        end is not None
        and set(referee_by_event) == expected_events
        and all(len(records_at_event) == 1 for records_at_event in referee_by_event.values())
    )
    referee_call_rate = _rate(
        sum(calls_by_type.values()),
        len(states),
        complete=complete,
        evidence_complete=referee_evidence,
    )

    return GameMetrics(
        game_complete=complete,
        team_conditions=team_conditions,
        ppp=ppp,
        maar=maar,
        ccar=ccar,
        cssr=cssr,
        maur=maur,
        grounded_maur=grounded_maur,
        iar=iar,
        iar_game=iar_game,
        iar_by_reason=iar_by_reason,
        sir=sir,
        sir_by_constraint=sir_by_constraint,
        referee_call_rate=referee_call_rate,
        assist_rate=assist_rate,
        turnover_rate=turnover_rate,
        screen_to_shot_conversion=screen_conversion,
        coordination_conflict_rate=coordination_conflict_rate,
        memory_growth=memory_growth,
        adaptation=adaptation,
        referee_calls_by_type=dict(sorted(calls_by_type.items())),
        memory_validation_complete=memory_complete,
        grounding_context_complete=grounding_complete,
    )
