"""End-to-end tests for indexed log reading and WS6 metrics."""

import json

import pytest

from bball3x3.analysis import LogReadError, compute_game_metrics, read_game_log
from bball3x3.logging import LogReadError as LoggingReadError
from bball3x3.schemas.agent import AgentOutput, CoordinationConfig
from bball3x3.schemas.log import (
    CSProposalRecord,
    DecisionRecord,
    GameEndRecord,
    GameStartRecord,
    InvariantViolationRecord,
    MemoryProposalRecord,
    MessageRecord,
    PossessionChange,
    PossessionEndRecord,
    RefereeRecord,
    ResolutionRecord,
    ScoringEvent,
    StateRecord,
)
from bball3x3.schemas.memory import CSProposal, MemoryProposal
from bball3x3.schemas.messages import Directive
from bball3x3.schemas.referee import RefereeDecision
from bball3x3.schemas.state import BallState, GameState


def _start() -> GameStartRecord:
    return GameStartRecord(
        schema_version=1,
        game_id="analysis_001",
        experiment_id="analysis_tests",
        replicate_id="A_starts",
        seed=101,
        first_possession="A",
        team_conditions={"A": "A_full", "B": "B_no_comm"},
        matchup={"home": "A_full", "away": "B_no_comm"},
        coordination={"A": CoordinationConfig(), "B": CoordinationConfig(comms=False)},
        roster={},
        config_hash="fixture",
        config_snapshot={},
        prompt_hashes={},
        model_slug="synthetic",
        code_commit=None,
        initial_state=GameState(ball=BallState(pos=(7, 1), holder="A1")),
        initial_possession_id="p0001",
    )


def _decision(
    event_idx: int,
    player: str,
    action: str,
    *,
    target: str | None = None,
    memory_ref: list[str] | None = None,
    invalid_reason: str | None = None,
) -> DecisionRecord:
    team = player[0]
    output = AgentOutput(action=action, target=target, memory_ref=memory_ref)
    if invalid_reason is None:
        executed_action, executed_target = action, target
        remapped_action = remapped_target = None
    else:
        executed_action, executed_target = "HOLD", None
        remapped_action, remapped_target = "HOLD", None
    return DecisionRecord(
        event_idx=event_idx,
        player=player,
        team=team,
        raw_output=f'<{player} action="{action}"/>',
        parsed_output=output,
        parse_ok=True,
        invalid_reason=invalid_reason,
        remapped_action=remapped_action,
        remapped_target=remapped_target,
        executed_action=executed_action,
        executed_target=executed_target,
        memory_ref=memory_ref,
    )


def _resolution(event_idx: int, *, game_over: bool = False) -> ResolutionRecord:
    outcomes = {
        "A1": {"involved_opponents": ["B1"]},
        "B1": {"involved_opponents": ["A1"]},
    }
    return ResolutionRecord(
        event_idx=event_idx,
        actions={"A1": "HOLD", "B1": "HOLD"},
        action_targets={"A1": None, "B1": None},
        outcomes=outcomes,
        possession_before={
            "possession_id": "p0002" if event_idx == 3 else "p0001",
            "team": "B" if event_idx == 3 else "A",
        },
        possession_after=None
        if game_over
        else {
            "possession_id": "p0002" if event_idx == 2 else "p0001",
            "team": "B" if event_idx == 2 else "A",
        },
        scoring_events=(
            [ScoringEvent(team="A", points=2, source="FIELD_GOAL", possession_id="p0001")]
            if event_idx == 2
            else [ScoringEvent(team="B", points=1, source="FIELD_GOAL", possession_id="p0002")]
            if event_idx == 3
            else []
        ),
        possession_changes=(
            [PossessionChange(
                from_possession_id="p0001",
                to_possession_id="p0002",
                to_team="B",
                reason="MADE_BASKET",
                kind="CONTROL_TRANSFER",
            )]
            if event_idx == 2
            else []
        ),
        game_over=game_over,
        game_end_reason="CLOCK_EXPIRED" if game_over else None,
    )


def _state(event_idx: int, *, checks_complete: bool = True) -> StateRecord:
    state = GameState(
        event_idx=event_idx,
        game_clock_s=3 - event_idx,
        score={"A": 2 if event_idx >= 2 else 0, "B": 1 if event_idx >= 3 else 0},
        ball=BallState(pos=(7, 1), holder="A1"),
    )
    return StateRecord(
        event_idx=event_idx,
        state=state,
        invariant_checks_completed=checks_complete,
        constraints_checked=list(range(1, 21)) if checks_complete else [1, 2],
    )


def _complete_records() -> list:
    records = [_start()]
    records.extend(
        [
            _decision(1, "A1", "PASS", target="A2", memory_ref=["M1"], invalid_reason="BAD_TARGET"),
            _decision(1, "B1", "HOLD"),
            MessageRecord(
                event_idx=1,
                sender="A2",
                team="A",
                channel="teammate",
                kind="action",
                text="Shoot next",
                directive=Directive(to="A1", act="SHOOT"),
                delivered_at=2,
            ),
            MessageRecord(
                event_idx=1,
                sender="A3",
                team="A",
                channel="teammate",
                kind="action",
                text="Or pass",
                directive=Directive(to="A1", act="PASS", target="A3"),
                delivered_at=2,
            ),
            MessageRecord(
                event_idx=1,
                sender="B1",
                team="B",
                channel="captain",
                kind="action",
                text="Hold",
                directive=Directive(to="B1", act="HOLD"),
                delivered_at=2,
            ),
            MessageRecord(
                event_idx=1,
                sender="A1",
                team="A",
                channel="teammate",
                kind="tactic",
                text="Too early",
                cs="01",
            ),
            _resolution(1),
            RefereeRecord(event_idx=1, query="event 1", decision=RefereeDecision()),
            _state(1),
            _decision(2, "A1", "HOLD", memory_ref=["M0"]),
            _decision(2, "B1", "HOLD"),
            CSProposalRecord(
                event_idx=2,
                team="A",
                player="A1",
                proposal=CSProposal(
                    opponent_strategy="switch", counter_strategy="slip", side="offense"
                ),
                accepted=True,
                key="01",
            ),
            MemoryProposalRecord(
                event_idx=2,
                team="A",
                player="A1",
                proposal=MemoryProposal(
                    opponent="B1", category="general", tendency="helps late"
                ),
                accepted=True,
                new_id="M1",
            ),
            _resolution(2),
            RefereeRecord(
                event_idx=2,
                query="event 2",
                decision=RefereeDecision(
                    call="FOUL",
                    offender="B1",
                    fouled_player="A1",
                    foul_type="DEFENSIVE_FOUL",
                ),
            ),
            InvariantViolationRecord(
                event_idx=2,
                constraint_id=2,
                constraint_name="ball_possessor_consistency",
                details="synthetic violation",
            ),
            _state(2),
            PossessionEndRecord(
                possession_id="p0001",
                team="A",
                points=2,
                points_by_team={"A": 2, "B": 0},
                end_reason="MADE_BASKET",
                start_event_idx=1,
                end_event_idx=2,
            ),
            _decision(3, "A1", "SHOOT", memory_ref=["M1"]),
            _decision(3, "B1", "HOLD"),
            MessageRecord(
                event_idx=3,
                sender="A1",
                team="A",
                channel="teammate",
                kind="tactic",
                text="Use the learned counter",
                cs="01",
            ),
            _resolution(3, game_over=True),
            RefereeRecord(event_idx=3, query="event 3", decision=RefereeDecision()),
            _state(3),
            PossessionEndRecord(
                possession_id="p0002",
                team="B",
                points=1,
                points_by_team={"A": 0, "B": 1},
                end_reason="CLOCK_EXPIRED",
                start_event_idx=2,
                end_event_idx=3,
            ),
            GameEndRecord(
                last_event_idx=3,
                end_reason="CLOCK_EXPIRED",
                final_score={"A": 2, "B": 1},
                ot_flag=False,
                totals={
                    "decisions": 6,
                    "decisions_by_team": {"A": 3, "B": 3},
                    "transitions": 3,
                    "possessions_by_team": {"A": 1, "B": 1},
                },
            ),
        ]
    )
    return records


def test_reader_uses_logging_error_and_preserves_unknown_records(tmp_path):
    assert LogReadError is LoggingReadError
    start = _start()
    unknown = {
        "record_type": "future_annotation",
        "event_idx": 1,
        "team": "A",
        "memory_id": "M02",
        "opponent": "B1",
    }
    path = tmp_path / "unfinished.jsonl"
    path.write_text(
        start.model_dump_json() + "\n" + json.dumps(unknown) + "\n",
        encoding="utf-8",
    )
    index = read_game_log(path)
    assert not index.is_complete
    assert index.records_of_type("future_annotation") == (unknown,)
    assert index.records_at_event(1) == (unknown,)


@pytest.mark.parametrize(
    "bad_line",
    [
        "",
        '{"record_type":"memory_write","event_idx":0}',
        '{"record_type":"memory_write","event_idx":1,"event_idx":2}',
        '{"record_type":"memory_write","event_idx":1,"value":NaN}',
    ],
)
def test_reader_rejects_invalid_jsonl_with_line_number(tmp_path, bad_line):
    path = tmp_path / "bad.jsonl"
    path.write_text(_start().model_dump_json() + "\n" + bad_line + "\n", encoding="utf-8")
    with pytest.raises(LogReadError) as error:
        read_game_log(path)
    assert error.value.line_number == 2
    assert f"{path}:2:" in str(error.value)


def test_reader_rejects_records_after_game_end(tmp_path):
    records = _complete_records()
    path = tmp_path / "bad_order.jsonl"
    path.write_text(
        "\n".join(record.model_dump_json() for record in records)
        + "\n"
        + json.dumps({"record_type": "future_annotation", "event_idx": 4}),
        encoding="utf-8",
    )
    with pytest.raises(LogReadError, match="after game_end"):
        read_game_log(path)


def test_metrics_are_per_team_time_scoped_and_windowed():
    records = _complete_records()
    initial_ids = {"A": {"M0"}, "B": set()}
    opponents = {"A": {"M0": "B1"}, "B": {}}

    primary = compute_game_metrics(
        records,
        window=1,
        known_memory_ids=initial_ids,
        memory_opponents=opponents,
    )
    robust = compute_game_metrics(
        records,
        window=2,
        known_memory_ids=initial_ids,
        memory_opponents=opponents,
    )

    assert primary.game_complete
    assert primary.team_conditions == {"A": "A_full", "B": "B_no_comm"}
    assert primary.ppp["A"].rate == 2.0
    assert primary.ppp["B"].rate == 1.0

    # Two messages addressing A1 at the same delivery event form one eligible
    # decision. The second-event robustness window sees A1's later shot.
    assert primary.maar["A"].to_dict() == {
        "numerator": 0,
        "denominator": 1,
        "rate": 0.0,
        "status": "ok",
    }
    assert robust.maar["A"].rate == 1.0
    assert primary.maar["B"].status == "not_applicable"
    assert primary.ccar["B"].rate == 1.0

    # Key 01 is not valid at event 1; the event-2 write becomes visible at 3.
    assert primary.cssr["A"].numerator == 1
    assert primary.cssr["A"].denominator == 2
    assert primary.cssr["A"].rate == 0.5

    # M1 is invalid before its event-2 write and valid at event 3.
    assert primary.maur["A"].numerator == 2
    assert primary.maur["A"].denominator == 3
    assert primary.grounded_maur["A"].rate == pytest.approx(2 / 3)
    assert primary.maur["B"].rate == 0.0

    assert primary.iar["A"].rate == pytest.approx(1 / 3)
    assert primary.iar["B"].rate == 0.0
    assert primary.iar_game.rate == pytest.approx(1 / 6)
    assert primary.sir.rate == pytest.approx(1 / 3)
    assert primary.referee_call_rate.rate == pytest.approx(1 / 3)
    assert primary.to_dict()["maur"]["A"]["status"] == "ok"


def test_metrics_mark_unfinished_or_incomplete_evidence():
    complete_records = _complete_records()
    unfinished = compute_game_metrics(complete_records[:-1])
    assert not unfinished.game_complete
    assert unfinished.ppp["A"].status == "incomplete"
    assert unfinished.iar_game.status == "incomplete"
    assert unfinished.sir.status == "incomplete"

    incomplete_checks = _complete_records()
    incomplete_checks[incomplete_checks.index(next(
        record
        for record in incomplete_checks
        if isinstance(record, StateRecord) and record.event_idx == 2
    ))] = _state(2, checks_complete=False)
    metrics = compute_game_metrics(incomplete_checks)
    assert metrics.sir.status == "insufficient_evidence"
    assert metrics.sir.rate is None


def test_memory_metrics_require_team_scoped_initial_evidence():
    metrics = compute_game_metrics(_complete_records())
    assert metrics.maur["A"].status == "insufficient_evidence"
    assert metrics.grounded_maur["A"].status == "insufficient_evidence"


def test_ppp_uses_points_by_team_for_exceptional_scoring():
    records = _complete_records()
    possessions = [record for record in records if isinstance(record, PossessionEndRecord)]
    possessions[0].points = 1
    possessions[0].points_by_team = {"A": 1, "B": 1}
    possessions[1].points = 0
    possessions[1].points_by_team = {"A": 0, "B": 0}
    end = records[-1]
    assert isinstance(end, GameEndRecord)
    end.final_score = {"A": 1, "B": 1}
    second = next(
        record for record in records
        if isinstance(record, ResolutionRecord) and record.event_idx == 2
    )
    second.scoring_events = [
        ScoringEvent(team="A", points=1, source="FIELD_GOAL", possession_id="p0001"),
        ScoringEvent(team="B", points=1, source="FREE_THROW", possession_id="p0001"),
    ]
    third = next(
        record for record in records
        if isinstance(record, ResolutionRecord) and record.event_idx == 3
    )
    third.scoring_events = []
    for record in records:
        if isinstance(record, StateRecord) and record.state is not None and record.event_idx >= 2:
            record.state.score = {"A": 1, "B": 1}

    metrics = compute_game_metrics(records)
    assert metrics.ppp["A"].rate == 1.0
    assert metrics.ppp["B"].rate == 1.0
