"""Hand-calculated tests for the WS6 descriptive metrics."""

import json

import pytest
from pydantic import ValidationError

from bball3x3.analysis import PossessionTracker, compute_game_metrics, read_game_log
from bball3x3.logging import EventLogger
from bball3x3.schemas import (
    AgentOutput,
    BallState,
    CoordinationConfig,
    DecisionRecord,
    GameEndRecord,
    GameStartRecord,
    GameState,
    InitialMemoryEntry,
    MemoryWriteRecord,
    MessageRecord,
    ResolutionRecord,
    ScreenEvent,
    ShotAttempt,
    StateRecord,
    TurnoverEvent,
)
from bball3x3.schemas.log import CSWriteRecord, PossessionChange, ScoringEvent


def _start() -> GameStartRecord:
    return GameStartRecord(
        schema_version=1,
        game_id="descriptive_1",
        experiment_id="descriptive_tests",
        replicate_id="A_starts",
        seed=1,
        first_possession="A",
        team_conditions={"A": "A_full", "B": "B_full"},
        matchup={},
        coordination={"A": CoordinationConfig(), "B": CoordinationConfig()},
        roster={},
        config_hash="fixture",
        config_snapshot={"rules": {"game_length_s": 600}},
        prompt_hashes={},
        model_slug="synthetic",
        code_commit=None,
        initial_state=GameState(ball=BallState(pos=(7, 1), holder="A1")),
        initial_possession_id="p1",
        initial_memory={
            "A": [InitialMemoryEntry(
                id="M0", opponent="B1", category="general", tendency="helps late"
            )],
            "B": [],
        },
    )


def _decision(event_idx: int, player: str, action: str, target=None, refs=None):
    return DecisionRecord(
        event_idx=event_idx,
        player=player,
        team=player[0],
        raw_output="fixture",
        parsed_output=AgentOutput(action=action, target=target, memory_ref=refs),
        parse_ok=True,
        invalid_reason=None,
        executed_action=action,
        executed_target=target,
        memory_ref=refs,
    )


def _resolution(event_idx: int, decisions: list[DecisionRecord]) -> ResolutionRecord:
    before = ("p1", "A") if event_idx <= 2 else ("p2", "B") if event_idx == 3 else ("p3", "A")
    after = ("p1", "A") if event_idx == 1 else ("p2", "B") if event_idx == 2 else ("p3", "A")
    scores = (
        [ScoringEvent(team="A", points=2, source="FIELD_GOAL", possession_id="p1")]
        if event_idx == 2 else
        [ScoringEvent(team="A", points=1, source="FIELD_GOAL", possession_id="p3")]
        if event_idx == 4 else []
    )
    changes = (
        [PossessionChange(
            from_possession_id="p1", to_possession_id="p2", to_team="B",
            reason="MADE_BASKET", kind="CONTROL_TRANSFER",
        )]
        if event_idx == 2 else
        [PossessionChange(
            from_possession_id="p2", to_possession_id="p3", to_team="A",
            reason="TURNOVER", kind="CONTROL_TRANSFER",
        )]
        if event_idx == 3 else []
    )
    return ResolutionRecord(
        event_idx=event_idx,
        actions={decision.player: decision.executed_action for decision in decisions},
        action_targets={decision.player: decision.executed_target for decision in decisions},
        possession_before={"possession_id": before[0], "team": before[1]},
        possession_after=(
            None if event_idx == 4 else {"possession_id": after[0], "team": after[1]}
        ),
        scoring_events=scores,
        possession_changes=changes,
        shots=(
            [ShotAttempt(
                team="A", shooter="A1", made=True, assist_by="A2", screen_id="s1"
            )]
            if event_idx == 2 else
            [ShotAttempt(team="A", shooter="A1", made=True)]
            if event_idx == 4 else []
        ),
        screens=(
            [ScreenEvent(screen_id="s1", team="A", screener="A2", freed_player="A1")]
            if event_idx == 1 else []
        ),
        turnovers=(
            [TurnoverEvent(team="B", possession_id="p2", reason="INTERCEPTION", player="B1")]
            if event_idx == 3 else []
        ),
        game_over=event_idx == 4,
        game_end_reason="CLOCK_EXPIRED" if event_idx == 4 else None,
    )


def _records():
    start = _start()
    tracker = PossessionTracker(start)
    records = [start]
    clocks = {1: 480, 2: 360, 3: 240, 4: 120}
    scores = {1: 0, 2: 2, 3: 2, 4: 3}
    for event_idx in range(1, 5):
        a1 = (
            ("MOVE", "LEFT_WING") if event_idx == 1 else
            ("SHOOT", None) if event_idx in (2, 4) else ("HOLD", None)
        )
        a2 = (
            ("SCREEN", "B1") if event_idx == 1 else
            ("SPOT_UP", "RIGHT_CORNER") if event_idx in (2, 4) else ("HOLD", None)
        )
        a3 = (
            ("MOVE", "LEFT_WING") if event_idx == 1 else
            ("SPOT_UP", "LEFT_CORNER") if event_idx in (2, 4) else ("HOLD", None)
        )
        decisions = [
            _decision(event_idx, "A1", *a1, refs=["M0"] if event_idx <= 2 else ["M1"]),
            _decision(event_idx, "A2", *a2),
            _decision(event_idx, "A3", *a3),
            *(_decision(event_idx, player, "HOLD") for player in ("B1", "B2", "B3")),
        ]
        records.extend(decisions)
        if event_idx <= 3:
            records.append(MessageRecord(
                event_idx=event_idx, sender="A1", team="A", channel="teammate",
                kind="tactic", text="counter", cs="c1",
            ))
        if event_idx == 2:
            records.extend([
                MemoryWriteRecord(
                    event_idx=2, team="A", memory_id="M1", opponent="B1",
                    category="general", tendency="overhelps", visible_at=3,
                ),
                CSWriteRecord(
                    event_idx=2, team="A", key="c1", side="offense",
                    opponent_strategy="switch", counter_strategy="slip", visible_at=3,
                ),
            ])
        if event_idx == 3:
            records.append(MemoryWriteRecord(
                event_idx=3, team="A", memory_id="M2", opponent="B2",
                category="general", tendency="drops", visible_at=4, evicted_ids=["M0"],
            ))
        resolution = _resolution(event_idx, decisions)
        records.append(resolution)
        records.append(StateRecord(
            event_idx=event_idx,
            state=GameState(
                event_idx=event_idx,
                game_clock_s=clocks[event_idx],
                score={"A": scores[event_idx], "B": 0},
                ball=BallState(pos=(7, 1), holder="A1"),
            ),
            invariant_checks_completed=True,
            constraints_checked=list(range(1, 21)),
        ))
        records.extend(tracker.apply(resolution))
    records.append(GameEndRecord(
        last_event_idx=4, end_reason="CLOCK_EXPIRED", final_score={"A": 3, "B": 0},
        ot_flag=False,
        totals={
            "decisions": 24, "decisions_by_team": {"A": 12, "B": 12},
            "transitions": 4, "possessions_by_team": {"A": 2, "B": 1},
        },
    ))
    return records


def test_descriptive_metrics_round_trip_and_expected_rates(tmp_path):
    path = tmp_path / "descriptive.jsonl"
    with EventLogger(path) as logger:
        for record in _records():
            logger.write(record)
    metrics = compute_game_metrics(read_game_log(path))

    assert metrics.assist_rate["A"].to_dict() == {
        "numerator": 1, "denominator": 2, "rate": 0.5, "status": "ok"
    }
    assert metrics.assist_rate["B"].status == "undefined"
    assert metrics.turnover_rate["B"].rate == 1.0
    assert metrics.turnover_rate["A"].rate == 0.0
    assert metrics.screen_to_shot_conversion["A"].rate == 1.0
    assert metrics.coordination_conflict_rate["A"].rate == pytest.approx(1 / 3)
    assert metrics.coordination_conflict_rate["B"].status == "undefined"
    assert metrics.memory_growth["A"].to_dict() == {
        "initial_count": 1, "final_count": 2, "net_change": 1, "peak_count": 2,
        "by_event": {"0": 1, "1": 1, "2": 2, "3": 2, "4": 2}, "status": "ok",
    }
    assert metrics.memory_growth["B"].final_count == 0
    assert metrics.adaptation.ppp["A"]["q1"].rate == 2.0
    assert metrics.adaptation.ppp["A"]["q3"].rate == 1.0
    assert metrics.adaptation.ppp["B"]["q2"].rate == 0.0
    assert metrics.adaptation.ppp["A"]["q5"].status == "undefined"
    assert metrics.adaptation.maur["A"]["q4"].rate == pytest.approx(1 / 3)
    assert metrics.adaptation.cssr["A"]["q1"].rate == 0.0
    assert metrics.adaptation.cssr["A"]["q3"].rate == 1.0
    for name in ("ppp", "maur", "cssr"):
        overall = getattr(metrics, name)["A"]
        bins = getattr(metrics.adaptation, name)["A"].values()
        assert sum(metric.numerator for metric in bins) == overall.numerator
        assert sum(metric.denominator for metric in bins) == overall.denominator
    json.dumps(metrics.to_dict())


def test_missing_descriptive_evidence_is_not_treated_as_zero():
    records = _records()
    resolution = next(
        record for record in records
        if isinstance(record, ResolutionRecord) and record.event_idx == 2
    )
    resolution.shots = None
    resolution.screens = None
    resolution.turnovers = None
    metrics = compute_game_metrics(records)
    assert metrics.assist_rate["A"].status == "insufficient_evidence"
    assert metrics.screen_to_shot_conversion["A"].status == "insufficient_evidence"
    assert metrics.turnover_rate["A"].status == "insufficient_evidence"


def test_missing_timing_and_initial_memory_are_reported():
    records = _records()
    start = records[0]
    assert isinstance(start, GameStartRecord)
    start.initial_memory = None
    state = next(
        record for record in records
        if isinstance(record, StateRecord) and record.event_idx == 2
    )
    records.remove(state)
    metrics = compute_game_metrics(records)
    assert metrics.memory_growth["A"].status == "insufficient_evidence"
    assert metrics.adaptation.ppp["A"]["q1"].status == "insufficient_evidence"


def test_unfinished_game_marks_descriptive_metrics_incomplete():
    metrics = compute_game_metrics(_records()[:-1])
    assert metrics.assist_rate["A"].status == "incomplete"
    assert metrics.memory_growth["A"].status == "incomplete"
    assert metrics.adaptation.maur["A"]["q1"].status == "incomplete"


@pytest.mark.parametrize("bad_shot", [
    {"team": "A", "shooter": "A1", "made": False, "assist_by": "A2"},
    {"team": "A", "shooter": "A1", "made": True, "assist_by": "A1"},
])
def test_invalid_assist_evidence_is_rejected_by_schema(bad_shot):
    with pytest.raises(ValidationError):
        ShotAttempt(**bad_shot)


def test_screen_link_outside_one_event_window_does_not_convert():
    records = _records()
    first = next(
        record for record in records
        if isinstance(record, ResolutionRecord) and record.event_idx == 1
    )
    second = next(
        record for record in records
        if isinstance(record, ResolutionRecord) and record.event_idx == 2
    )
    third = next(
        record for record in records
        if isinstance(record, ResolutionRecord) and record.event_idx == 3
    )
    second.shots[0].screen_id = None
    third.shots = [ShotAttempt(team="A", shooter="A1", made=False, screen_id="s1")]
    assert first.screens[0].screen_id == "s1"
    metrics = compute_game_metrics(records)
    assert metrics.screen_to_shot_conversion["A"].rate == 0.0


def test_unknown_screen_link_is_insufficient_evidence():
    records = _records()
    shot = next(
        record.shots[0] for record in records
        if isinstance(record, ResolutionRecord) and record.event_idx == 2
    )
    shot.screen_id = "missing"
    metrics = compute_game_metrics(records)
    assert metrics.screen_to_shot_conversion["A"].status == "insufficient_evidence"


def test_overtime_possession_and_decisions_use_separate_ot_bin():
    records = _records()
    third_resolution = next(
        record for record in records
        if isinstance(record, ResolutionRecord) and record.event_idx == 3
    )
    third_resolution.possession_changes[0].kind = "PERIOD_BOUNDARY"
    third_resolution.possession_changes[0].reason = "REGULATION_END"
    third_resolution.turnovers = []
    third_closure = next(
        record for record in records
        if record.record_type == "possession_end" and record.possession_id == "p2"
    )
    third_closure.end_reason = "REGULATION_END"
    for record in records:
        if isinstance(record, StateRecord) and record.event_idx >= 3:
            record.state.phase = "OT"
    end = records[-1]
    assert isinstance(end, GameEndRecord)
    end.ot_flag = True

    metrics = compute_game_metrics(records)
    assert metrics.ppp["A"].rate == 1.5
    assert metrics.adaptation.ppp["A"]["ot"].rate == 1.0
    assert metrics.adaptation.maur["A"]["ot"].rate == pytest.approx(1 / 3)
    assert metrics.adaptation.ppp["A"]["q4"].status == "undefined"


def test_memory_growth_needs_committed_write_for_accepted_proposal():
    from bball3x3.schemas import MemoryProposal, MemoryProposalRecord

    records = _records()
    records.insert(-1, MemoryProposalRecord(
        event_idx=4, team="A", player="A1",
        proposal=MemoryProposal(opponent="B1", category="general", tendency="new"),
        accepted=True, new_id="M3",
    ))
    metrics = compute_game_metrics(records)
    assert metrics.memory_growth["A"].status == "insufficient_evidence"
