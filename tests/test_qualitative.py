"""Observation integrity and saved-log transcript integration tests."""

import hashlib
import json

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from bball3x3.analysis import (
    TranscriptError,
    read_game_log,
    render_possession_transcript,
)
from bball3x3.cli import app
from bball3x3.logging import EventLogger, LogReadError
from bball3x3.schemas.log import (
    CaptainCallRecord,
    DecisionRecord,
    MemoryWriteRecord,
    MessageRecord,
    StateRecord,
)
from tests.test_analysis import _complete_records, _decision, _start

runner = CliRunner()


@pytest.fixture
def transcript_path(tmp_path):
    records = _complete_records()
    decision = records[1]
    data = decision.model_dump()
    data["observation_text"] = 'Score 0–0\nCoach: "screen & roll" 🏀'
    data["parsed_output"]["rationale"] = "Pass to the open teammate."
    records[1] = DecisionRecord.model_validate(data)
    # Same synthetic possession timeline as the analysis tests, now with
    # captain persistence and a committed mutation for qualitative review.
    records.insert(3, CaptainCallRecord(
        event_idx=1, team="A", maintain=False, tactic="Spread the floor", directives=[]
    ))
    event2 = next(i for i, record in enumerate(records)
                  if isinstance(record, DecisionRecord) and record.event_idx == 2)
    records.insert(event2 + 2, CaptainCallRecord(
        event_idx=2, team="A", maintain=True, directives=[]
    ))
    records.insert(event2 + 3, MemoryWriteRecord(
        event_idx=2, team="A", memory_id="M1", opponent="B1", category="general",
        tendency="helps late", visible_at=3,
    ))
    path = tmp_path / "game.jsonl"
    with EventLogger(path) as logger:
        for record in records:
            logger.write(record)
    return path


@pytest.mark.parametrize("observation", [None, "", "line one\nline two 🏀"])
def test_observation_round_trip_and_hash(tmp_path, observation):
    data = _decision(1, "A1", "HOLD").model_dump()
    data["observation_text"] = observation
    decision = DecisionRecord.model_validate(data)
    expected = (
        hashlib.sha256(observation.encode("utf-8")).hexdigest()
        if observation is not None else ""
    )
    assert decision.observation_hash == expected
    path = tmp_path / "prefix.jsonl"
    with EventLogger(path) as logger:
        logger.write(_start())
        logger.write(decision)
    saved = read_game_log(path).records_of_type("decision")[0]
    assert saved.observation_text == observation
    assert saved.observation_hash == expected
    # Embedded newlines must remain within one JSONL record.
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_mismatched_observation_hash_rejected_by_reader_and_writer(tmp_path):
    data = _decision(1, "A1", "HOLD").model_dump()
    data.update(observation_text="actual observation", observation_hash="incorrect")
    with pytest.raises(ValidationError, match="SHA-256"):
        DecisionRecord.model_validate(data)
    path = tmp_path / "bad.jsonl"
    path.write_text(_start().model_dump_json() + "\n" + json.dumps(data), encoding="utf-8")
    with pytest.raises(LogReadError, match=r"bad.jsonl:2:.*"):
        read_game_log(path)

    data["observation_hash"] = ""
    decision = DecisionRecord.model_validate(data)
    decision.observation_text = "mutated after hashing"
    path = tmp_path / "writer.jsonl"
    with EventLogger(path) as logger:
        logger.write(_start())
        with pytest.raises(ValidationError, match="SHA-256"):
            logger.write(decision)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_transcript_round_trip_preserves_content_and_order(transcript_path):
    transcript = render_possession_transcript(read_game_log(transcript_path), "p0001")
    assert "possession p0001 | team A" in transcript
    assert "Possession points: A=2, B=0" in transcript
    assert 'Coach: "screen & roll" 🏀' in transcript
    assert "Proposed: PASS target=A2" in transcript
    assert "Executed: HOLD target=none" in transcript
    assert "BAD_TARGET" in transcript
    assert "Pass to the open teammate." in transcript
    assert '<A1 action="PASS"/>' in transcript
    assert '"M1"' in transcript
    assert "DEFENSIVE_FOUL" in transcript
    assert "scoring_events" in transcript
    assert "memory_write" in transcript
    assert "helps late" in transcript
    assert "synthetic violation" in transcript
    assert transcript.index("Decision A1") < transcript.index("Captain call")
    assert transcript.index("Captain call") < transcript.index("Message A2")
    assert "Event 3" not in transcript
    assert "Use the learned counter" not in transcript


def test_shared_boundary_and_prior_delivery_context(transcript_path):
    transcript = render_possession_transcript(read_game_log(transcript_path), "p0002")
    assert "events 2–3" in transcript
    assert "Possession points: A=0, B=1" in transcript
    assert "Earlier context" in transcript
    assert "Shoot next" in transcript
    assert "Spread the floor" in transcript
    assert "Maintain previous call: True" in transcript
    assert "Possession before decisions: p0001" in transcript
    assert "Possession before decisions: p0002" in transcript
    assert "Boundary events may be shared" in transcript
    assert "Event 1" not in transcript
    assert transcript.index("Shoot next") < transcript.index("Event 2")


def test_open_possession_and_pending_decision_are_inspectable():
    records = [_start(), _decision(1, "A1", "HOLD")]
    transcript = render_possession_transcript(records, "p0001")
    assert "Game log: unfinished | possession: open" in transcript
    assert "Decision A1" in transcript
    assert "[not recorded]" in transcript
    assert "Possession points:" not in transcript


def test_closed_possession_in_unfinished_game(transcript_path):
    index = read_game_log(transcript_path)
    prefix = index.records[:next(i for i, record in enumerate(index.records)
                                if isinstance(record, DecisionRecord) and record.event_idx == 3)]
    transcript = render_possession_transcript(prefix, "p0001")
    assert "Game log: unfinished | possession: closed" in transcript


def test_parse_failure_suppression_unknown_records_and_invalid_state():
    data = _decision(1, "A1", "HOLD").model_dump()
    data.update(parsed_output=None, parse_ok=False, invalid_reason="PARSE_FAIL",
                remapped_action="HOLD", raw_output="malformed <ACTION")
    records = [
        _start(), DecisionRecord.model_validate(data),
        MessageRecord(event_idx=1, sender="A1", team="A", channel="teammate",
                      kind="info", text="suppressed message", suppressed=True),
        {"record_type": "future_annotation", "event_idx": 1, "note": "unrecognized 🏀"},
        StateRecord(event_idx=1, state=None,
                    raw_state={"event_idx": 1, "score": {"A": -1, "B": 0}},
                    state_validation_errors=["ball is missing"],
                    invariant_checks_completed=False, constraints_checked=[]),
    ]
    transcript = render_possession_transcript(records, "p0001")
    assert "Proposed: [parse failed]" in transcript
    assert "malformed <ACTION" in transcript
    assert "delivery: none (suppressed)" in transcript
    assert "future_annotation" in transcript
    assert "unrecognized 🏀" in transcript
    assert "ball is missing" in transcript
    assert '"A": -1' in transcript


def test_unknown_possession_is_actionable(transcript_path):
    with pytest.raises(TranscriptError, match="Available IDs: p0001, p0002"):
        render_possession_transcript(read_game_log(transcript_path), "missing")


def test_cli_prints_saved_transcript(transcript_path):
    result = runner.invoke(app, ["transcript", str(transcript_path), "p0001"])
    assert result.exit_code == 0, result.output
    assert result.stdout == render_possession_transcript(read_game_log(transcript_path), "p0001")


def test_cli_reports_unknown_possession_and_malformed_log(transcript_path, tmp_path):
    result = runner.invoke(app, ["transcript", str(transcript_path), "missing"])
    assert result.exit_code == 1
    assert "Unknown possession" in result.stderr
    path = tmp_path / "broken.jsonl"
    path.write_text("{invalid JSON", encoding="utf-8")
    result = runner.invoke(app, ["transcript", str(path), "p0001"])
    assert result.exit_code == 1
    assert "broken.jsonl:1:" in result.stderr
    assert "Traceback" not in result.output
