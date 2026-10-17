"""Adaptation curves: count pooling, evidence gaps, OT and exported CLI output."""

import csv
import json

import pytest
from typer.testing import CliRunner

from bball3x3.analysis import (
    ExperimentTables,
    PlotError,
    build_adaptation_plot_data,
    write_adaptation_plots,
    write_experiment_tables,
)
from bball3x3.cli import app

_BINS = ("q1", "q2", "q3", "q4", "q5", "ot")


def _rate(numerator=0, denominator=0, status=None):
    status = status or ("ok" if denominator else "undefined")
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if status == "ok" else None,
            "status": status}


def _tables(*, comparison="B_no_comm", missing_side=None):
    games, teams = [], []
    for seed in (1, 2):
        for side in ("A", "B"):
            if (seed, side) == missing_side:
                continue
            game_id = f"g-{seed}-{side}"
            games.append({"experiment_id": "ablation", "game_id": game_id,
                          "seed": seed, "first_possession": side,
                          "team_A_condition": "A_full", "team_B_condition": comparison,
                          "game_complete": True, "possession_validation": "ok"})
            for condition in ("A_full", comparison):
                data = {metric: {bin_name: _rate() for bin_name in _BINS}
                        for metric in ("ppp", "maur", "cssr")}
                data["ppp"]["q1"] = _rate(2 if seed == 1 else 0, 1) if (
                    condition == "A_full"
                ) else _rate(1, 1)
                data["maur"]["q1"] = _rate(seed, 2)
                data["cssr"]["q1"] = _rate(1, 2)
                if seed == 1:
                    data["ppp"]["q2"] = _rate(1, 1, "insufficient_evidence")
                else:
                    data["ppp"]["q2"] = _rate(1, 2)
                if seed == 1 and side == "A" and condition == "A_full":
                    data["ppp"]["ot"] = _rate(2, 1)
                if condition == "C_no_captain":
                    data["cssr"] = {bin_name: _rate(status="not_applicable")
                                    for bin_name in _BINS}
                teams.append({"game_id": game_id, "condition": condition,
                              "adaptation_json": json.dumps(data)})
    manifest = {"schema_version": 1, "excluded_games": [{"game_id": "interrupted"}]}
    return ExperimentTables(games, teams, [], manifest)


def _point(data, metric, condition, bin_name):
    return next(row for row in data.rows if row["metric"] == metric
                and row["condition"] == condition and row["time_bin"] == bin_name)


def test_pooled_counts_missing_bins_and_separate_ot():
    data = build_adaptation_plot_data(_tables())
    first = _point(data, "ppp", "A_full", "q1")
    assert (first["numerator"], first["denominator"], first["rate"]) == (4, 4, 1)
    assert first["n_games_with_exposure"] == 4
    gap = _point(data, "ppp", "A_full", "q5")
    assert gap["status"] == "no_exposure"
    assert gap["rate"] is None
    assert gap["denominator"] == 0
    second = _point(data, "ppp", "A_full", "q2")
    assert second["status"] == "partial_evidence"
    assert second["n_games_without_evidence"] == 2
    assert (second["numerator"], second["denominator"], second["rate"]) == (2, 4, 0.5)
    overtime = _point(data, "ppp", "A_full", "ot")
    assert (overtime["numerator"], overtime["denominator"], overtime["rate"]) == (2, 1, 2)
    assert data.manifest["overtime_display"].startswith("separate marker")
    assert data.manifest["ppp_attribution"] == "possession start"
    assert data.manifest["source_excluded_games"] == [{"game_id": "interrupted"}]


def test_nonapplicable_and_missing_pair_are_explicit():
    data = build_adaptation_plot_data(_tables(comparison="C_no_captain",
                                              missing_side=(2, "B")))
    assert data.manifest["groups"][0]["n_paired_seeds"] == 1
    assert data.manifest["groups"][0]["excluded_seeds"][0]["reason"] == (
        "missing_or_duplicate_start_side"
    )
    cssr = _point(data, "cssr", "C_no_captain", "q1")
    assert cssr["status"] == "not_applicable"
    assert cssr["rate"] is None
    assert cssr["n_games_not_applicable"] == 2


def test_plot_cli_exports_csv_png_and_manifest(tmp_path):
    source = tmp_path / "export"
    write_experiment_tables(_tables(), source)
    target = tmp_path / "plots"
    result = CliRunner().invoke(app, ["plots", str(source), str(target)])
    assert result.exit_code == 0, result.output
    assert "Wrote 3 plots" in result.stdout
    with (target / "adaptation.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 36
    assert next(row for row in rows if row["metric"] == "ppp" and
                row["condition"] == "A_full" and row["time_bin"] == "q5")["rate"] == ""
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["plots"]) == 3
    for plot in manifest["plots"]:
        assert (target / plot["file"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert CliRunner().invoke(app, ["plots", str(source), str(target)]).exit_code == 1


def test_render_is_deterministic_and_never_overwrites(tmp_path):
    data = build_adaptation_plot_data(_tables())
    first, second = tmp_path / "first", tmp_path / "second"
    write_adaptation_plots(data, first)
    write_adaptation_plots(data, second)
    assert (first / "adaptation.csv").read_bytes() == (second / "adaptation.csv").read_bytes()
    assert (first / "comparison-001-ppp.png").read_bytes() == (
        second / "comparison-001-ppp.png"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        write_adaptation_plots(data, first)


def test_malformed_counts_fail_before_output_directory_created(tmp_path):
    tables = _tables()
    tables.teams[0]["adaptation_json"] = '{"ppp": {}}'
    with pytest.raises(PlotError, match="missing adaptation"):
        data = build_adaptation_plot_data(tables)
        write_adaptation_plots(data, tmp_path / "plots")
    assert not (tmp_path / "plots").exists()
