"""Reproducible game, team, and possession tables from saved JSONL logs."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bball3x3.schemas.log import (
    GameStartRecord,
    PossessionEndRecord,
    ResolutionRecord,
    StateRecord,
    Team,
)

from .log_reader import GameLogIndex, read_game_log
from .metrics import GameMetrics, RateMetric, compute_game_metrics
from .possessions import PossessionError, PossessionSummary, validate_possessions

_TEAMS: tuple[Team, Team] = ("A", "B")
_TEAM_RATE_NAMES = (
    "ppp", "maar", "ccar", "cssr", "maur", "grounded_maur", "iar",
    "assist_rate", "turnover_rate", "screen_to_shot_conversion",
    "coordination_conflict_rate",
)
_RATE_PARTS = ("numerator", "denominator", "rate", "status")
_COMMON_FIELDS = (
    "experiment_id", "game_id", "replicate_id", "seed", "first_possession",
    "team_A_condition", "team_B_condition", "config_hash", "code_commit",
    "model_slug", "schema_version", "source", "game_complete", "possession_validation",
    "possession_validation_error",
)
GAME_FIELDS = (
    *_COMMON_FIELDS, "config_snapshot_json", "end_reason", "ot_flag",
    "final_score_A", "final_score_B", "total_cost_usd", "transitions",
    "decision_total", "possessions_A", "possessions_B",
    *(f"{name}_{part}" for name in ("iar_game", "sir", "referee_call_rate")
      for part in _RATE_PARTS),
    "sir_by_constraint_json", "referee_calls_by_type_json",
)
TEAM_FIELDS = (
    *_COMMON_FIELDS, "physical_team", "condition", "opponent_condition",
    "comms_enabled", "captain_enabled", "memory_write_enabled",
    *(f"{name}_{part}" for name in _TEAM_RATE_NAMES for part in _RATE_PARTS),
    "iar_by_reason_json", "memory_growth_json", "adaptation_json",
    "memory_validation_complete", "grounding_context_complete",
)
POSSESSION_FIELDS = (
    *_COMMON_FIELDS, "possession_id", "physical_team", "condition", "closed",
    "start_event_idx", "end_event_idx", "end_reason", "points", "points_A",
    "points_B", "score_A_before", "score_B_before", "score_diff_before",
    "start_clock_s", "time_bin", "time_bin_status", "model_eligible",
)


class ExportError(ValueError):
    """Saved logs cannot be assembled into one unambiguous experiment export."""


@dataclass
class ExperimentTables:
    """Native Python rows suitable for CSV writing or later statistical analysis."""

    games: list[dict[str, Any]]
    teams: list[dict[str, Any]]
    possessions: list[dict[str, Any]]
    manifest: dict[str, Any]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _rate_fields(name: str, metric: RateMetric) -> dict[str, Any]:
    return {f"{name}_{part}": value for part, value in metric.to_dict().items()}


def _identity(start: GameStartRecord, index: GameLogIndex, status: str,
              error: str | None) -> dict[str, Any]:
    return {
        "experiment_id": start.experiment_id,
        "game_id": start.game_id,
        "replicate_id": start.replicate_id,
        "seed": start.seed,
        "first_possession": start.first_possession,
        "team_A_condition": start.team_conditions["A"],
        "team_B_condition": start.team_conditions["B"],
        "config_hash": start.config_hash,
        "code_commit": start.code_commit,
        "model_slug": start.model_slug,
        "schema_version": start.schema_version,
        "source": index.source,
        "game_complete": index.is_complete,
        "possession_validation": status,
        "possession_validation_error": error,
    }


def _game_row(identity: dict[str, Any], start: GameStartRecord,
              index: GameLogIndex, metrics: GameMetrics) -> dict[str, Any]:
    end = index.game_end
    row = {
        **identity,
        "config_snapshot_json": _json(start.config_snapshot),
        "end_reason": end.end_reason if end else None,
        "ot_flag": end.ot_flag if end else None,
        "final_score_A": end.final_score["A"] if end else None,
        "final_score_B": end.final_score["B"] if end else None,
        "total_cost_usd": end.total_cost_usd if end else None,
        "transitions": end.totals.transitions if end else None,
        "decision_total": end.totals.decisions if end else None,
        "possessions_A": end.totals.possessions_by_team["A"] if end else None,
        "possessions_B": end.totals.possessions_by_team["B"] if end else None,
        "sir_by_constraint_json": _json({
            name: rate.to_dict() for name, rate in metrics.sir_by_constraint.items()
        }),
        "referee_calls_by_type_json": _json(metrics.referee_calls_by_type),
    }
    for name in ("iar_game", "sir", "referee_call_rate"):
        row.update(_rate_fields(name, getattr(metrics, name)))
    return row


def _team_row(identity: dict[str, Any], start: GameStartRecord,
              metrics: GameMetrics, team: Team) -> dict[str, Any]:
    other: Team = "B" if team == "A" else "A"
    flags = start.coordination[team]
    row = {
        **identity,
        "physical_team": team,
        "condition": start.team_conditions[team],
        "opponent_condition": start.team_conditions[other],
        "comms_enabled": flags.comms,
        "captain_enabled": flags.captain,
        "memory_write_enabled": flags.memory_write,
        "iar_by_reason_json": _json({
            name: rate.to_dict() for name, rate in metrics.iar_by_reason[team].items()
        }),
        "memory_growth_json": _json(metrics.memory_growth[team].to_dict()),
        "adaptation_json": _json({
            name: {bin_name: rate.to_dict() for bin_name, rate in
                   getattr(metrics.adaptation, name)[team].items()}
            for name in ("ppp", "maur", "cssr")
        }),
        "memory_validation_complete": metrics.memory_validation_complete[team],
        "grounding_context_complete": metrics.grounding_context_complete[team],
    }
    for name in _TEAM_RATE_NAMES:
        row.update(_rate_fields(name, getattr(metrics, name)[team]))
    return row


def _openings(index: GameLogIndex, start: GameStartRecord) -> dict[str, tuple[int, dict[str, int]]]:
    """Score at each opening, applying the outgoing segment's points first."""
    score = dict(start.initial_state.score)
    openings = {start.initial_possession_id: (1, score.copy())}
    for record in index.records:
        if not isinstance(record, ResolutionRecord):
            continue
        by_possession: dict[str, list[tuple[Team, int]]] = {}
        for event in record.scoring_events:
            by_possession.setdefault(event.possession_id, []).append((event.team, event.points))
        current = record.possession_before.possession_id
        for change in record.possession_changes:
            for team, points in by_possession.get(current, ()):
                score[team] += points
            current = change.to_possession_id
            openings[current] = (record.event_idx, score.copy())
        for team, points in by_possession.get(current, ()):
            score[team] += points
    return openings


def _timing(index: GameLogIndex, start: GameStartRecord,
            event_idx: int, overtime: bool) -> tuple[float | None, str | None]:
    initial = start.initial_state
    duration = initial.game_clock_s
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        return None, None
    rules = start.config_snapshot.get("rules")
    if isinstance(rules, Mapping) and rules.get("game_length_s", duration) != duration:
        return None, None
    if event_idx == 1:
        clock, phase = initial.game_clock_s, initial.phase
    else:
        states = [record for record in index.records_of_type("state")
                  if isinstance(record, StateRecord) and record.event_idx == event_idx - 1]
        if len(states) != 1:
            return None, None
        state = states[0]
        if state.state is not None:
            clock, phase = state.state.game_clock_s, state.state.phase
        else:
            raw = state.raw_state or {}
            clock, phase = raw.get("game_clock_s"), raw.get("phase")
    if (isinstance(clock, bool) or not isinstance(clock, (int, float))
            or not math.isfinite(clock) or not 0 <= clock <= duration):
        return None, None
    if phase == "OT" or overtime:
        return float(clock), "ot"
    if phase not in ("LIVE", "DEAD", "FT"):
        return None, None
    return float(clock), f"q{min(5, int(((duration - clock) / duration) * 5) + 1)}"


def _overtime_ids(index: GameLogIndex) -> set[str]:
    result: set[str] = set()
    started = False
    for record in index.records:
        if not isinstance(record, ResolutionRecord):
            continue
        for change in record.possession_changes:
            if change.kind == "PERIOD_BOUNDARY":
                started = True
            if started:
                result.add(change.to_possession_id)
    return result


def _possession_rows(identity: dict[str, Any], start: GameStartRecord,
                     index: GameLogIndex, summary: PossessionSummary,
                     metrics: GameMetrics) -> list[dict[str, Any]]:
    openings = _openings(index, start)
    overtime_ids = _overtime_ids(index)
    closures: list[PossessionEndRecord | None] = list(summary.closed)
    if summary.active is not None:
        closures.append(None)
    rows = []
    for closure in closures:
        possession_id = (closure.possession_id if closure is not None
                         else summary.active.possession_id)
        team: Team = closure.team if closure is not None else summary.active.team
        first, before = openings[possession_id]
        clock, bin_name = _timing(index, start, first, possession_id in overtime_ids)
        other: Team = "B" if team == "A" else "A"
        bin_status = (
            metrics.adaptation.ppp[team][bin_name].status if bin_name
            else "insufficient_evidence"
        )
        rows.append({
            **identity,
            "possession_id": possession_id,
            "physical_team": team,
            "condition": start.team_conditions[team],
            "closed": closure is not None,
            "start_event_idx": first,
            "end_event_idx": closure.end_event_idx if closure else None,
            "end_reason": closure.end_reason if closure else None,
            "points": closure.points if closure else None,
            "points_A": closure.points_by_team["A"] if closure else None,
            "points_B": closure.points_by_team["B"] if closure else None,
            "score_A_before": before["A"],
            "score_B_before": before["B"],
            "score_diff_before": before[team] - before[other],
            "start_clock_s": clock,
            "time_bin": bin_name,
            "time_bin_status": bin_status,
            "model_eligible": index.is_complete and closure is not None
            and bin_status in ("ok", "undefined"),
        })
    return rows


def build_experiment_tables(
    paths: Iterable[str | Path], *, include_incomplete: bool = False, window: int = 1
) -> ExperimentTables:
    """Read logs and build three joinable tables, preserving evidence statuses.

    By default unfinished games are listed in the manifest and excluded. A
    complete game with contradictory possession records remains in game/team
    tables, but has no possession rows. All files must have unique game IDs.
    """
    if window not in (1, 2):
        raise ValueError("window must be 1 or 2")
    games: list[dict[str, Any]] = []
    teams: list[dict[str, Any]] = []
    possessions: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    omitted_possessions: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    seen_ids: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in seen_paths:
            raise ExportError(f"Duplicate input path: {path}")
        seen_paths.add(path)
        index = read_game_log(path)
        start = index.game_start
        if start is None:
            raise ExportError(f"Missing game_start in {path}")
        if start.game_id in seen_ids:
            raise ExportError(f"Duplicate game_id {start.game_id!r} in {path}")
        seen_ids.add(start.game_id)
        if not index.is_complete and not include_incomplete:
            excluded.append({"game_id": start.game_id, "source": str(path),
                             "reason": "missing_game_end"})
            continue
        metrics = compute_game_metrics(index, window=window)
        try:
            summary = validate_possessions(index)
        except PossessionError as exc:
            summary = None
            validation = "failed"
            error = str(exc)
            omitted_possessions.append({"game_id": start.game_id, "source": str(path),
                                        "reason": error})
        else:
            validation, error = "ok", None
        identity = _identity(start, index, validation, error)
        games.append(_game_row(identity, start, index, metrics))
        teams.extend(_team_row(identity, start, metrics, team) for team in _TEAMS)
        if summary is not None:
            possessions.extend(_possession_rows(identity, start, index, summary, metrics))
    if not seen_paths:
        raise ExportError("Provide at least one game log")
    if not games:
        raise ExportError("No eligible games; pass include_incomplete=True to inspect prefixes")
    manifest = {
        "schema_version": 1,
        "include_incomplete": include_incomplete,
        "message_window": window,
        "game_count": len(games),
        "team_row_count": len(teams),
        "possession_row_count": len(possessions),
        "excluded_games": excluded,
        "omitted_possessions": omitted_possessions,
        "rate_columns": ["numerator", "denominator", "rate", "status"],
        "score_diff_before": "owning team score minus opponent score at possession opening",
        "same_event_transfer": "apply points attributed to outgoing segments before opening next",
        "time_bin": "five equal elapsed-regulation-clock bins q1–q5; OT is ot",
        "time_bin_source": "clock/phase before the possession's start event",
        "possession_points": "owner points; points_A and points_B include exceptional scoring",
        "model_eligible": "complete game, validated closed possession, known time bin",
    }
    return ExperimentTables(games, teams, possessions, manifest)


def write_experiment_tables(tables: ExperimentTables, directory: str | Path) -> Path:
    """Write CSV tables and a JSON manifest into a new output directory."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    for name, fields, rows in (
        ("games", GAME_FIELDS, tables.games),
        ("teams", TEAM_FIELDS, tables.teams),
        ("possessions", POSSESSION_FIELDS, tables.possessions),
    ):
        with (destination / f"{name}.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(tables.manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return destination
