"""Saved-log experiment tables: pairing, covariates, exclusions, and CSV CLI."""

import csv
import json

import pytest
from typer.testing import CliRunner

from bball3x3.analysis import (
    ExportError,
    build_experiment_tables,
    write_experiment_tables,
)
from bball3x3.cli import app
from bball3x3.logging import EventLogger
from bball3x3.schemas.log import (
    DecisionRecord,
    GameEndRecord,
    PossessionChange,
    PossessionEndRecord,
    ResolutionRecord,
    ScoringEvent,
    StateRecord,
)
from tests.test_analysis import _complete_records
from tests.test_descriptive_metrics import _records

runner = CliRunner()


def _saved_game(tmp_path, name, *, seed=17, replicate="A_starts", other="B_no_comm",
                complete=True, invalid_closure=False):
    records = _records()
    start = records[0]
    start.game_id = name
    start.experiment_id = "ablation_comms"
    start.seed = seed
    start.replicate_id = replicate
    start.team_conditions = {"A": "A_full", "B": other}
    if other == "C_no_captain":
        start.coordination["B"].captain = False
    if invalid_closure:
        closure = next(record for record in records if isinstance(record, PossessionEndRecord))
        closure.points = 3
        closure.points_by_team = {"A": 3, "B": 0}
    if not complete:
        last = next(i for i, record in enumerate(records)
                    if isinstance(record, DecisionRecord) and record.event_idx == 4)
        records = records[:last]
    path = tmp_path / f"{name}.jsonl"
    with EventLogger(path) as logger:
        for record in records:
            logger.write(record)
    return path


def test_batch_identity_metrics_and_possession_covariates(tmp_path):
    first = _saved_game(tmp_path, "game_a", replicate="A_starts")
    second = _saved_game(tmp_path, "game_b", replicate="B_starts", other="C_no_captain")
    tables = build_experiment_tables([first, second])

    assert [row["game_id"] for row in tables.games] == ["game_a", "game_b"]
    assert [row["replicate_id"] for row in tables.games] == ["A_starts", "B_starts"]
    assert all(row["seed"] == 17 for row in tables.games)
    assert all(row["possession_validation"] == "ok" for row in tables.games)
    assert len(tables.teams) == 4
    assert len(tables.possessions) == 6
    assert tables.teams[0]["physical_team"] == "A"
    assert tables.teams[1]["physical_team"] == "B"
    assert tables.teams[0]["ppp_numerator"] == 3
    assert tables.teams[0]["ppp_denominator"] == 2
    assert tables.teams[0]["ppp_rate"] == 1.5
    assert tables.teams[0]["ppp_status"] == "ok"
    assert tables.teams[3]["ccar_status"] == "not_applicable"
    assert tables.games[0]["config_hash"] == "fixture"
    assert tables.games[0]["final_score_A"] == 3
    assert tables.games[0]["sir_status"] == "ok"
    assert json.loads(tables.teams[0]["memory_growth_json"])["final_count"] == 2
    bins = json.loads(tables.teams[0]["adaptation_json"])
    assert bins["ppp"]["q1"]["numerator"] == 2
    assert bins["maur"]["q4"]["status"] == "ok"

    possessions = {row["possession_id"]: row for row in tables.possessions[:3]}
    assert list(possessions) == ["p1", "p2", "p3"]
    assert possessions["p1"]["score_diff_before"] == 0
    assert possessions["p1"]["points"] == 2
    assert possessions["p1"]["time_bin"] == "q1"
    assert possessions["p2"]["physical_team"] == "B"
    assert possessions["p2"]["score_A_before"] == 2
    assert possessions["p2"]["score_diff_before"] == -2
    assert possessions["p2"]["time_bin"] == "q2"
    assert possessions["p2"]["points"] == 0
    assert possessions["p3"]["score_diff_before"] == 2
    assert possessions["p3"]["time_bin"] == "q3"
    assert all(row["model_eligible"] for row in possessions.values())


def test_csv_manifest_and_cli_round_trip(tmp_path):
    path = _saved_game(tmp_path, "unicode_🏀")
    target = tmp_path / "exports"
    result = runner.invoke(app, ["export", str(target), str(path), "--window", "2"])
    assert result.exit_code == 0, result.output
    assert "Exported 1 games, 2 teams, 3 possessions" in result.stdout
    with (target / "games.csv").open(encoding="utf-8", newline="") as stream:
        games = list(csv.DictReader(stream))
    with (target / "teams.csv").open(encoding="utf-8", newline="") as stream:
        teams = list(csv.DictReader(stream))
    with (target / "possessions.csv").open(encoding="utf-8", newline="") as stream:
        possessions = list(csv.DictReader(stream))
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert len(games) == 1 and len(teams) == 2 and len(possessions) == 3
    assert games[0]["game_id"] == "unicode_🏀"
    assert games[0]["config_hash"] == "fixture"
    assert teams[0]["ppp_rate"] == "1.5"
    assert teams[0]["ppp_status"] == "ok"
    assert json.loads(teams[0]["adaptation_json"])["ppp"]["q1"]["rate"] == 2.0
    assert possessions[1]["score_diff_before"] == "-2"
    assert manifest["message_window"] == 2
    assert manifest["game_count"] == 1
    assert manifest["possession_row_count"] == 3
    assert manifest["excluded_games"] == []


def test_incomplete_games_excluded_by_default_and_open_rows_opt_in(tmp_path):
    complete = _saved_game(tmp_path, "complete")
    partial = _saved_game(tmp_path, "partial", complete=False)
    tables = build_experiment_tables([complete, partial])
    assert len(tables.games) == 1
    assert tables.manifest["excluded_games"] == [{
        "game_id": "partial", "source": str(partial.resolve()),
        "reason": "missing_game_end",
    }]

    included = build_experiment_tables([complete, partial], include_incomplete=True)
    assert len(included.games) == 2
    assert len(included.teams) == 4
    open_row = next(row for row in included.possessions
                    if row["game_id"] == "partial" and row["possession_id"] == "p3")
    assert open_row["closed"] is False
    assert open_row["points"] is None
    assert open_row["end_event_idx"] is None
    assert open_row["score_diff_before"] == 2
    assert open_row["model_eligible"] is False
    assert included.games[1]["game_complete"] is False
    assert included.teams[2]["ppp_status"] == "incomplete"


def test_invalid_complete_game_keeps_metric_rows_but_omits_possessions(tmp_path):
    path = _saved_game(tmp_path, "invalid", invalid_closure=True)
    tables = build_experiment_tables([path])
    assert len(tables.games) == 1 and len(tables.teams) == 2
    assert tables.possessions == []
    assert tables.games[0]["game_complete"] is True
    assert tables.games[0]["possession_validation"] == "failed"
    assert "possession_end records disagree" in tables.games[0]["possession_validation_error"]
    assert tables.teams[0]["ppp_status"] == "insufficient_evidence"
    assert tables.manifest["omitted_possessions"][0]["game_id"] == "invalid"


def test_exceptional_other_team_points_stay_with_accounting_possession(tmp_path):
    records = _complete_records()
    first = next(record for record in records if isinstance(record, PossessionEndRecord))
    first.points = 1
    first.points_by_team = {"A": 1, "B": 1}
    last = [record for record in records if isinstance(record, PossessionEndRecord)][-1]
    last.points = 0
    last.points_by_team = {"A": 0, "B": 0}
    end = records[-1]
    assert isinstance(end, GameEndRecord)
    end.final_score = {"A": 1, "B": 1}
    second_event = next(
        record for record in records
        if isinstance(record, ResolutionRecord) and record.event_idx == 2
    )
    second_event.scoring_events = [
        ScoringEvent(team="A", points=1, source="FIELD_GOAL", possession_id="p0001"),
        ScoringEvent(team="B", points=1, source="FREE_THROW", possession_id="p0001"),
    ]
    third_event = next(
        record for record in records
        if isinstance(record, ResolutionRecord) and record.event_idx == 3
    )
    third_event.scoring_events = []
    for record in records:
        if isinstance(record, StateRecord) and record.state and record.event_idx >= 2:
            record.state.score = {"A": 1, "B": 1}
    path = tmp_path / "exceptional.jsonl"
    with EventLogger(path) as logger:
        for record in records:
            logger.write(record)
    tables = build_experiment_tables([path])
    assert tables.games[0]["possession_validation"] == "ok"
    assert tables.possessions[0]["points"] == 1
    assert tables.possessions[0]["points_B"] == 1
    assert tables.possessions[1]["physical_team"] == "B"
    assert tables.possessions[1]["score_A_before"] == 1
    assert tables.possessions[1]["score_B_before"] == 1
    assert tables.possessions[1]["score_diff_before"] == 0


def test_period_boundary_marks_new_possession_as_overtime(tmp_path):
    records = _records()
    resolution = next(
        record for record in records
        if isinstance(record, ResolutionRecord) and record.event_idx == 3
    )
    boundary: PossessionChange = resolution.possession_changes[0]
    boundary.kind = "PERIOD_BOUNDARY"
    boundary.reason = "REGULATION_END"
    resolution.turnovers = []
    closure = next(
        record for record in records
        if isinstance(record, PossessionEndRecord) and record.possession_id == "p2"
    )
    closure.end_reason = "REGULATION_END"
    for record in records:
        if isinstance(record, StateRecord) and record.event_idx >= 3:
            record.state.phase = "OT"
    end = records[-1]
    assert isinstance(end, GameEndRecord)
    end.ot_flag = True
    path = tmp_path / "overtime.jsonl"
    with EventLogger(path) as logger:
        for record in records:
            logger.write(record)
    tables = build_experiment_tables([path])
    possessions = {row["possession_id"]: row for row in tables.possessions}
    assert possessions["p2"]["time_bin"] == "q2"
    assert possessions["p3"]["time_bin"] == "ot"
    assert possessions["p3"]["time_bin_status"] == "ok"
    assert possessions["p3"]["model_eligible"] is True
    assert json.loads(tables.teams[0]["adaptation_json"])["ppp"]["ot"]["rate"] == 1.0


def test_duplicate_identity_and_existing_output_are_rejected(tmp_path):
    path = _saved_game(tmp_path, "single")
    with pytest.raises(ExportError, match="Duplicate input path"):
        build_experiment_tables([path, path])
    other = _saved_game(tmp_path, "other")
    # Distinct source files cannot claim the same saved game identity.
    second = tmp_path / "second.jsonl"
    second.write_bytes(path.read_bytes())
    with pytest.raises(ExportError, match="Duplicate game_id"):
        build_experiment_tables([path, second])
    tables = build_experiment_tables([path, other])
    output = tmp_path / "result"
    write_experiment_tables(tables, output)
    with pytest.raises(FileExistsError):
        write_experiment_tables(tables, output)
    result = runner.invoke(app, ["export", str(output), str(path)])
    assert result.exit_code == 1
    assert "File exists" in result.stderr


def test_empty_and_only_incomplete_batch_explains_selection(tmp_path):
    with pytest.raises(ExportError, match="at least one game log"):
        build_experiment_tables([])
    partial = _saved_game(tmp_path, "partial", complete=False)
    with pytest.raises(ExportError, match="No eligible games"):
        build_experiment_tables([partial])
    with pytest.raises(ValueError, match="window must be 1 or 2"):
        build_experiment_tables([partial], window=3)
