"""Persistence tests independent of the engine and any LLM service."""

import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from bball3x3.logging import EventLogger, IncompleteLogError, LogReadError, read_events
from bball3x3.schemas.agent import AgentOutput, CoordinationConfig
from bball3x3.schemas.log import (
    CaptainCallRecord,
    CSProposalRecord,
    DecisionRecord,
    GameEndRecord,
    GameStartRecord,
    InvariantViolationRecord,
    MemoryProposalRecord,
    MessageRecord,
    PossessionEndRecord,
    RefereeRecord,
    ResolutionRecord,
    StateRecord,
)
from bball3x3.schemas.memory import CSProposal, MemoryProposal
from bball3x3.schemas.referee import RefereeDecision
from bball3x3.schemas.state import BallState, GameState, PlayerState


@pytest.fixture
def start():
    players = {
        f"{team}{slot}": PlayerState(
            id=f"{team}{slot}",
            team=team,
            pos=(slot, 5 if team == "A" else 6),
            prev_pos=(slot, 5 if team == "A" else 6),
            has_ball=team == "A" and slot == 1,
        )
        for team in ("A", "B")
        for slot in range(1, 4)
    }
    return GameStartRecord(
        schema_version=1,
        game_id="synthetic_001",
        experiment_id="logger_tests",
        replicate_id="A_starts",
        seed=101,
        first_possession="A",
        team_conditions={"A": "A_full", "B": "B_no_comm"},
        matchup={"home": "A_full", "away": "B_no_comm"},
        coordination={"A": CoordinationConfig(), "B": CoordinationConfig(comms=False)},
        roster={pid: {"team": player.team} for pid, player in players.items()},
        config_hash="synthetic-fixture",
        config_snapshot={"rules": {"game_length_s": 1}},
        prompt_hashes={},
        model_slug="synthetic",
        code_commit=None,
        initial_state=GameState(
            game_clock_s=1,
            ball=BallState(pos=(1, 5), holder="A1"),
            players=players,
        ),
        initial_possession_id="p0001",
    )


@pytest.fixture
def decision():
    return DecisionRecord(
        event_idx=1,
        player="A1",
        team="A",
        raw_output='<A1 action="SHOOT"/>\n看篮筐 🏀',
        parsed_output=AgentOutput(action="SHOOT", rationale="Open shot\nTake it."),
        parse_ok=True,
        invalid_reason=None,
        executed_action="SHOOT",
        executed_target=None,
    )


@pytest.fixture
def end():
    return GameEndRecord(
        last_event_idx=1,
        end_reason="CLOCK_EXPIRED",
        final_score={"A": 1, "B": 0},
        ot_flag=False,
        totals={
            "decisions": 6,
            "decisions_by_team": {"A": 3, "B": 3},
            "transitions": 1,
            "possessions_by_team": {"A": 1, "B": 0},
        },
    )


@pytest.fixture
def records(start, decision, end):
    """One short synthetic game exercising every currently supported type."""
    decisions = [decision]
    for player in start.initial_state.players.values():
        if player.id != "A1":
            decisions.append(
                DecisionRecord(
                    event_idx=1,
                    player=player.id,
                    team=player.team,
                    raw_output=f'<{player.id} action="HOLD"/>',
                    parsed_output=AgentOutput(action="HOLD"),
                    parse_ok=True,
                    invalid_reason=None,
                    executed_action="HOLD",
                    executed_target=None,
                )
            )
    state = start.initial_state.model_copy(deep=True)
    state.event_idx = 1
    state.score = {"A": 1, "B": 0}
    state.game_clock_s = 0
    state.phase = "DEAD"
    state.ball.status = "DEAD"
    state.ball.holder = None
    state.players["A1"].has_ball = False
    # Preserve a reported engine inconsistency rather than filtering it out.
    state.matchups = {"B2": "B3"}
    return [
        start,
        *decisions,
        MessageRecord(
            event_idx=1, sender="A2", team="A", channel="teammate", kind="info", text="Open!"
        ),
        CaptainCallRecord(event_idx=1, team="A", maintain=False, tactic="Spread out"),
        MemoryProposalRecord(
            event_idx=1,
            team="A",
            player="A1",
            proposal=MemoryProposal(opponent="B1", category="general", tendency="Helps late"),
            accepted=False,
        ),
        CSProposalRecord(
            event_idx=1,
            team="A",
            player="A1",
            proposal=CSProposal(
                opponent_strategy="Switch", counter_strategy="Slip", side="offense"
            ),
            accepted=False,
        ),
        RefereeRecord(event_idx=1, query="shot; contact=false", decision=RefereeDecision()),
        ResolutionRecord(
            event_idx=1,
            actions={d.player: d.executed_action for d in decisions},
            action_targets={d.player: d.executed_target for d in decisions},
            possession_before={"possession_id": "p0001", "team": "A"},
            possession_after=None,
            scoring_events=[
                {"team": "A", "points": 1, "source": "FIELD_GOAL", "possession_id": "p0001"}
            ],
            possession_changes=[],
            game_over=True,
            game_end_reason="CLOCK_EXPIRED",
        ),
        InvariantViolationRecord(
            event_idx=1,
            constraint_id=17,
            constraint_name="defensive_matchup_validity",
            details="Synthetic fault: B2 guards teammate B3.",
        ),
        StateRecord(
            event_idx=1,
            state=state,
            invariant_checks_completed=True,
            constraints_checked=list(range(1, 21)),
        ),
        PossessionEndRecord(
            possession_id="p0001",
            team="A",
            points=1,
            points_by_team={"A": 1, "B": 0},
            end_reason="CLOCK_EXPIRED",
            start_event_idx=1,
            end_event_idx=1,
        ),
        end,
    ]


def test_all_record_types_round_trip_as_utf8_jsonl(tmp_path, records):
    path = tmp_path / "experiment" / "game" / "events.jsonl"
    with EventLogger(path) as logger:
        for record in records:
            logger.write(record)
    restored = list(read_events(path))
    assert restored == records
    assert [type(r) for r in restored] == [type(r) for r in records]
    assert len({r.record_type for r in restored}) == 12
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw
    assert "看篮筐 🏀".encode() in raw
    assert len(raw.splitlines()) == len(records)
    assert all(isinstance(json.loads(line), dict) for line in raw.splitlines())


def test_existing_file_is_never_overwritten(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"keep this history")
    with pytest.raises(FileExistsError):
        EventLogger(path)
    assert path.read_bytes() == b"keep this history"


def test_open_writer_is_readable_and_close_does_not_finish_game(tmp_path, start):
    path = tmp_path / "events.jsonl"
    with EventLogger(path) as logger:
        logger.write(start)
        assert list(read_events(path, require_complete=False)) == [start]
        logger.flush()
    assert logger.closed
    assert list(read_events(path, require_complete=False)) == [start]
    with pytest.raises(IncompleteLogError, match="Missing game_end") as error:
        list(read_events(path))
    assert error.value.line_number == 2


def test_context_manager_preserves_exception_and_closes(tmp_path, start):
    path = tmp_path / "events.jsonl"
    with pytest.raises(RuntimeError, match="engine stopped"):
        with EventLogger(path) as logger:
            logger.write(start)
            raise RuntimeError("engine stopped")
    assert logger.closed
    assert list(read_events(path, require_complete=False)) == [start]


def test_closed_logger_cannot_be_reused(tmp_path, start):
    logger = EventLogger(tmp_path / "events.jsonl")
    logger.close()
    logger.close()
    for operation in (lambda: logger.write(start), logger.flush, logger.__enter__):
        with pytest.raises(ValueError, match="closed"):
            operation()


def test_writer_enforces_game_boundaries_before_appending(tmp_path, start, decision, end):
    path = tmp_path / "events.jsonl"
    with EventLogger(path) as logger:
        with pytest.raises(ValueError, match="first record"):
            logger.write(decision)
        assert path.read_bytes() == b""
        logger.write(start)
        with pytest.raises(ValueError, match="Only one game_start"):
            logger.write(start)
        logger.write(end)
        for record in (decision, start, end):
            with pytest.raises(ValueError, match="after game_end"):
                logger.write(record)
    # Boundary validation does not claim six decisions/score checks occurred.
    assert list(read_events(path)) == [start, end]


@pytest.mark.parametrize("mutation", ["event", "nested_target", "nested_parse_flag"])
def test_mutated_models_are_revalidated_before_write(tmp_path, start, decision, mutation):
    path = tmp_path / "events.jsonl"
    if mutation == "event":
        decision.event_idx = 0
    elif mutation == "nested_target":
        decision.parsed_output.target = "B1"
    else:
        decision.parsed_output.parse_ok = False
    with EventLogger(path) as logger:
        logger.write(start)
        before = path.read_bytes()
        with pytest.raises((ValueError, TypeError)):
            logger.write(decision)
        assert path.read_bytes() == before


def test_writer_rejects_dictionary_input(tmp_path, start):
    with EventLogger(tmp_path / "events.jsonl") as logger:
        with pytest.raises(TypeError, match="typed log record"):
            logger.write(start.model_dump())


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_writer_rejects_nonfinite_values_even_inside_metadata(tmp_path, start, number):
    path = tmp_path / "events.jsonl"
    start.config_snapshot["bad_number"] = number
    with EventLogger(path) as logger:
        with pytest.raises(ValueError):
            logger.write(start)
        assert path.read_bytes() == b""


def test_writer_propagates_io_failure(tmp_path, start, monkeypatch):
    with EventLogger(tmp_path / "events.jsonl") as logger:
        broken_stream = Mock(wraps=logger._file)
        broken_stream.closed = False
        broken_stream.write.side_effect = OSError("disk full")
        monkeypatch.setattr(logger, "_file", broken_stream)
        with pytest.raises(OSError, match="disk full"):
            logger.write(start)


@pytest.mark.parametrize(
    "bad_line",
    [
        b"\n",
        b'{"record_type":',
        b'{"record_type":"unknown"}\n',
        b'{"record_type":"decision","event_idx":0}\n',
        b'{"record_type":"state","record_type":"game_end"}\n',
        b'{"record_type":"state","metadata":{"x":1,"x":2}}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'{"value":1e9999}\n',
        b'[]\n',
        b'null\n',
        b'\xff\n',
    ],
)
def test_reader_reports_path_and_line_for_bad_input(tmp_path, start, bad_line):
    path = tmp_path / "events.jsonl"
    path.write_bytes(start.model_dump_json().encode() + b"\n" + bad_line)
    with pytest.raises(LogReadError) as error:
        list(read_events(path, require_complete=False))
    assert error.value.path == path
    assert error.value.line_number == 2
    assert f"{path}:2:" in str(error.value)


@pytest.mark.parametrize("case", ["first_decision", "duplicate_start", "after_end"])
def test_reader_enforces_boundaries(tmp_path, start, decision, end, case):
    bad_records = {
        "first_decision": [decision],
        "duplicate_start": [start, start],
        "after_end": [start, end, decision],
    }[case]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(r.model_dump_json() for r in bad_records), encoding="utf-8")
    with pytest.raises(LogReadError):
        list(read_events(path, require_complete=False))


def test_missing_trailing_newline_is_allowed(tmp_path, start, end):
    path = tmp_path / "events.jsonl"
    path.write_text(start.model_dump_json() + "\n" + end.model_dump_json(), encoding="utf-8")
    assert list(read_events(path)) == [start, end]


def test_reader_rejects_empty_file_and_preserves_missing_file_error(tmp_path):
    path = tmp_path / "events.jsonl"
    with pytest.raises(FileNotFoundError):
        list(read_events(path))
    path.touch()
    with pytest.raises(IncompleteLogError, match="Empty log"):
        list(read_events(path, require_complete=False))


@pytest.mark.parametrize("version", [None, 2])
def test_reader_rejects_legacy_or_unsupported_version(tmp_path, start, version):
    data = start.model_dump(mode="json")
    if version is None:
        del data["schema_version"]
    else:
        data["schema_version"] = version
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(LogReadError, match="schema_version"):
        list(read_events(path, require_complete=False))


def test_diagnostic_invalid_state_is_preserved(tmp_path, start):
    raw_state = start.initial_state.model_dump(mode="json")
    raw_state["event_idx"] = 1
    raw_state["players"]["A1"]["fatigue"] = 99
    with pytest.raises(ValidationError):
        GameState.model_validate(raw_state)
    record = StateRecord(
        event_idx=1,
        state=None,
        raw_state=raw_state,
        state_validation_errors=["A1 fatigue exceeds 10"],
        invariant_checks_completed=True,
        constraints_checked=list(range(1, 21)),
    )
    path = tmp_path / "events.jsonl"
    with EventLogger(path) as logger:
        logger.write(start)
        logger.write(record)
    restored = list(read_events(path, require_complete=False))[-1]
    assert restored == record
    assert restored.raw_state["players"]["A1"]["fatigue"] == 99
