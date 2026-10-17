"""Possession construction and JSONL/PPP integration tests."""

import pytest

from bball3x3.analysis import (
    PossessionError,
    PossessionTracker,
    compute_game_metrics,
    read_game_log,
    reconstruct_possessions,
    validate_possessions,
)
from bball3x3.logging import EventLogger
from bball3x3.schemas.agent import CoordinationConfig
from bball3x3.schemas.log import (
    GameEndRecord,
    GameStartRecord,
    PossessionChange,
    ResolutionRecord,
    ScoringEvent,
    StateRecord,
)
from bball3x3.schemas.state import BallState, GameState


def _start() -> GameStartRecord:
    return GameStartRecord(
        schema_version=1,
        game_id="possession_test",
        experiment_id="test",
        replicate_id="test_1",
        seed=1,
        first_possession="A",
        team_conditions={"A": "A_full", "B": "B_full"},
        matchup={},
        coordination={"A": CoordinationConfig(), "B": CoordinationConfig()},
        roster={},
        config_hash="test",
        config_snapshot={},
        prompt_hashes={},
        model_slug="synthetic",
        code_commit=None,
        initial_state=GameState(ball=BallState(pos=(7, 1), holder="A1")),
        initial_possession_id="p1",
    )


def _resolution(
    event_idx: int,
    before: tuple[str, str],
    *,
    after: tuple[str, str] | None = None,
    scores: tuple[tuple[str, int, str, str], ...] = (),
    changes: tuple[tuple[str, str, str, str, str], ...] = (),
    game_over: bool = False,
) -> ResolutionRecord:
    return ResolutionRecord(
        event_idx=event_idx,
        actions={},
        action_targets={},
        possession_before={"possession_id": before[0], "team": before[1]},
        possession_after=(
            None if after is None else {"possession_id": after[0], "team": after[1]}
        ),
        scoring_events=[
            ScoringEvent(team=team, points=points, source=source, possession_id=pid)
            for team, points, source, pid in scores
        ],
        possession_changes=[
            PossessionChange(
                from_possession_id=source_id,
                to_possession_id=target_id,
                to_team=team,
                reason=reason,
                kind=kind,
            )
            for source_id, target_id, team, reason, kind in changes
        ],
        game_over=game_over,
        game_end_reason="CLOCK_EXPIRED" if game_over else None,
    )


def _state(event_idx: int, a: int, b: int) -> StateRecord:
    return StateRecord(
        event_idx=event_idx,
        state=GameState(
            event_idx=event_idx,
            score={"A": a, "B": b},
            ball=BallState(pos=(7, 1), holder="A1"),
        ),
        invariant_checks_completed=True,
        constraints_checked=list(range(1, 21)),
    )


def _end(event_idx: int, a: int, b: int, a_possessions: int, b_possessions: int):
    return GameEndRecord(
        last_event_idx=event_idx,
        end_reason="CLOCK_EXPIRED",
        final_score={"A": a, "B": b},
        ot_flag=False,
        totals={
            "decisions": 0,
            "decisions_by_team": {"A": 0, "B": 0},
            "transitions": event_idx,
            "possessions_by_team": {"A": a_possessions, "B": b_possessions},
        },
    )


def test_offensive_rebound_and_one_retained_possession_and_terminal_zero(tmp_path):
    start = _start()
    tracker = PossessionTracker(start)
    events = [
        (_resolution(1, ("p1", "A"), after=("p1", "A")), _state(1, 0, 0)),
        (
            _resolution(
                2,
                ("p1", "A"),
                after=("p1", "A"),
                scores=(("A", 2, "FIELD_GOAL", "p1"), ("A", 1, "FREE_THROW", "p1")),
            ),
            _state(2, 3, 0),
        ),
        (
            _resolution(
                3,
                ("p1", "A"),
                after=("p2", "B"),
                changes=(("p1", "p2", "B", "TURNOVER", "CONTROL_TRANSFER"),),
            ),
            _state(3, 3, 0),
        ),
        (_resolution(4, ("p2", "B"), game_over=True), _state(4, 3, 0)),
    ]
    path = tmp_path / "complete.jsonl"
    with EventLogger(path) as logger:
        logger.write(start)
        for resolution, state in events:
            closures = tracker.apply(resolution)
            logger.write(resolution)
            logger.write(state)
            for closure in closures:
                logger.write(closure)
        logger.write(_end(4, 3, 0, 1, 1))

    index = read_game_log(path)
    summary = validate_possessions(index)
    assert summary.active is None
    spans = [(p.possession_id, p.start_event_idx, p.end_event_idx, p.points)
             for p in summary.closed]
    assert spans == [
        ("p1", 1, 3, 3),
        ("p2", 3, 4, 0),
    ]
    metrics = compute_game_metrics(index)
    assert metrics.ppp["A"].rate == 3.0
    assert metrics.ppp["B"].rate == 0.0


def test_overtime_boundary_can_start_same_team_possession():
    tracker = PossessionTracker(_start())
    closed = tracker.apply(
        _resolution(
            1,
            ("p1", "A"),
            after=("ot1", "A"),
            changes=(("p1", "ot1", "A", "REGULATION_END", "PERIOD_BOUNDARY"),),
        )
    )
    assert len(closed) == 1
    assert closed[0].end_reason == "REGULATION_END"
    assert tracker.active.possession_id == "ot1"
    assert tracker.apply(_resolution(2, ("ot1", "A"), game_over=True))[0].possession_id == "ot1"


def test_transfer_at_terminal_event_closes_both_real_possessions():
    tracker = PossessionTracker(_start())
    closed = tracker.apply(
        _resolution(
            1,
            ("p1", "A"),
            game_over=True,
            scores=(("A", 1, "FREE_THROW", "p1"),),
            changes=(("p1", "p2", "B", "TURNOVER", "CONTROL_TRANSFER"),),
        )
    )
    assert [(record.possession_id, record.points, record.end_event_idx) for record in closed] == [
        ("p1", 1, 1),
        ("p2", 0, 1),
    ]
    assert tracker.active is None


def test_invalid_transition_is_atomic_and_reused_id_is_rejected():
    tracker = PossessionTracker(_start())
    with pytest.raises(PossessionError, match="must change team"):
        tracker.apply(
            _resolution(
                1,
                ("p1", "A"),
                after=("p2", "A"),
                changes=(("p1", "p2", "A", "TURNOVER", "CONTROL_TRANSFER"),),
            )
        )
    assert tracker.last_event_idx == 0
    assert tracker.closed == ()
    tracker.apply(
        _resolution(
            1,
            ("p1", "A"),
            after=("p2", "B"),
            changes=(("p1", "p2", "B", "TURNOVER", "CONTROL_TRANSFER"),),
        )
    )
    with pytest.raises(PossessionError, match="reused possession ID"):
        tracker.apply(
            _resolution(
                2,
                ("p2", "B"),
                after=("p1", "A"),
                changes=(("p2", "p1", "A", "TURNOVER", "CONTROL_TRANSFER"),),
            )
        )
    assert tracker.last_event_idx == 1


def test_invalid_scoring_reference_does_not_change_tracker():
    tracker = PossessionTracker(_start())
    with pytest.raises(PossessionError, match="inactive possession"):
        tracker.apply(
            _resolution(
                1,
                ("p1", "A"),
                after=("p1", "A"),
                scores=(("A", 1, "FREE_THROW", "unknown"),),
            )
        )
    assert tracker.last_event_idx == 0
    assert tracker.active.possession_id == "p1"


def test_missing_or_wrong_saved_closure_is_rejected():
    start = _start()
    terminal = _resolution(1, ("p1", "A"), game_over=True)
    closed = PossessionTracker(start).apply(terminal)[0]
    records = [start, terminal, _state(1, 0, 0), closed, _end(1, 0, 0, 1, 0)]
    assert validate_possessions(records).closed == (closed,)
    with pytest.raises(PossessionError, match="possession_end"):
        validate_possessions([start, terminal, _state(1, 0, 0), records[-1]])
    closed.end_reason = "WRONG_REASON"
    with pytest.raises(PossessionError, match="possession_end"):
        validate_possessions(records)


def test_closure_must_be_logged_after_its_state_before_next_event():
    start = _start()
    tracker = PossessionTracker(start)
    first = _resolution(
        1,
        ("p1", "A"),
        after=("p2", "B"),
        changes=(("p1", "p2", "B", "TURNOVER", "CONTROL_TRANSFER"),),
    )
    first_close = tracker.apply(first)[0]
    last = _resolution(2, ("p2", "B"), game_over=True)
    last_close = tracker.apply(last)[0]
    end = _end(2, 0, 0, 1, 1)
    with pytest.raises(PossessionError, match="outside its event"):
        validate_possessions(
            [start, first, _state(1, 0, 0), last, _state(2, 0, 0), first_close, last_close, end]
        )


def test_score_and_total_disagreements_are_rejected():
    start = _start()
    terminal = _resolution(1, ("p1", "A"), game_over=True, scores=(("A", 1, "FREE_THROW", "p1"),))
    closed = PossessionTracker(start).apply(terminal)[0]
    end = _end(1, 1, 0, 1, 0)
    with pytest.raises(PossessionError, match="score differs"):
        validate_possessions([start, terminal, _state(1, 0, 0), closed, end])
    assert compute_game_metrics([start, terminal, _state(1, 0, 0), closed, end]).ppp[
        "A"
    ].status == "insufficient_evidence"
    with pytest.raises(PossessionError, match="possession count"):
        validate_possessions([start, terminal, _state(1, 1, 0), closed, _end(1, 1, 0, 0, 0)])


def test_unfinished_game_keeps_active_possession():
    start = _start()
    resolution = _resolution(1, ("p1", "A"), after=("p1", "A"))
    records = [start, resolution, _state(1, 0, 0)]
    summary = reconstruct_possessions(records)
    assert summary.closed == ()
    assert summary.active.possession_id == "p1"
    assert validate_possessions(records) == summary
    assert compute_game_metrics(records).ppp["A"].status == "incomplete"


def test_ppp_rejects_missing_possession_closure():
    start = _start()
    terminal = _resolution(1, ("p1", "A"), game_over=True)
    records = [start, terminal, _state(1, 0, 0), _end(1, 0, 0, 1, 0)]
    assert compute_game_metrics(records).ppp["A"].status == "insufficient_evidence"
