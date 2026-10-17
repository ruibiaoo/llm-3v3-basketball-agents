"""WS6 paired seed analysis from exported table evidence."""

import json

import pytest
from typer.testing import CliRunner

from bball3x3.analysis import (
    ExperimentTables,
    StatsError,
    analyse_experiment,
    read_experiment_tables,
    write_experiment_stats,
    write_experiment_tables,
)
from bball3x3.cli import app


def _tables(*, seeds=range(6), missing_side=None, all_zero=False):
    games, teams, possessions = [], [], []
    for seed in seeds:
        for side in ("A", "B"):
            if (seed, side) == missing_side:
                continue
            game_id = f"s{seed}_{side}"
            games.append({"experiment_id": "ablation_comms", "game_id": game_id,
                          "seed": seed, "first_possession": side,
                          "team_A_condition": "A_full", "team_B_condition": "B_no_comm",
                          "game_complete": True, "possession_validation": "ok"})
            effect = 0 if all_zero else seed + 1
            for condition, rate in (("A_full", effect + 1), ("B_no_comm", 1)):
                teams.append({"game_id": game_id, "condition": condition,
                              "ppp_status": "ok", "ppp_rate": rate})
                for bin_number in range(5):
                    for repeat in range(2):
                        points = (1 if condition == "A_full" else 0) + bin_number % 2
                        points += seed % 3 + (repeat + seed + bin_number) % 2
                        possessions.append({
                            "game_id": game_id, "condition": condition,
                            "model_eligible": True, "time_bin": f"q{bin_number + 1}",
                            "points": points, "score_diff_before": (seed + repeat) % 5 - 2,
                        })
    return ExperimentTables(games, teams, possessions, {"schema_version": 1,
                                                        "excluded_games": []})


def test_seed_pairs_bootstrap_and_wilcoxon_are_reproducible():
    tables = _tables()
    first = analyse_experiment(tables, metrics=("ppp",), bootstrap_replicates=200)
    second = analyse_experiment(tables, metrics=("ppp",), bootstrap_replicates=200)
    assert first == second
    result = first["results"][0]["metrics"]["ppp"]
    assert result["n_seeds"] == 6
    assert result["n_games"] == 12
    assert result["effect_mean"] == 3.5
    assert result["positive_seeds"] == 6
    assert result["sign_agreement"] == 1
    assert result["paired_seeds"][2]["game_differences"] == [3, 3]
    assert result["bootstrap_95_ci"][0] < 3.5 < result["bootstrap_95_ci"][1]
    assert result["wilcoxon"]["p_value"] == pytest.approx(0.03125)


def test_missing_side_and_undefined_metric_exclude_seed():
    tables = _tables(seeds=range(3), missing_side=(1, "B"))
    tables.teams[0]["ppp_status"] = "undefined"
    result = analyse_experiment(tables, metrics=("ppp",), bootstrap_replicates=100)
    ppp = result["results"][0]["metrics"]["ppp"]
    assert ppp["status"] == "insufficient_pairs"
    assert ppp["n_seeds"] == 1
    assert ppp["bootstrap_95_ci"] is None
    assert {row["reason"] for row in ppp["excluded_seeds"]} == {
        "A_full_ppp_undefined", "missing_or_duplicate_start_side",
    }


def test_zero_effects_have_explicit_test_status():
    result = analyse_experiment(_tables(seeds=range(2), all_zero=True), metrics=("ppp",),
                                bootstrap_replicates=100)
    ppp = result["results"][0]["metrics"]["ppp"]
    assert ppp["zero_seeds"] == 2
    assert ppp["sign_agreement"] is None
    assert ppp["wilcoxon"]["status"] == "all_zero"


def test_mixed_models_fit_and_export_cli_round_trip(tmp_path):
    tables = _tables()
    result = analyse_experiment(tables, metrics=("ppp",), bootstrap_replicates=100)
    models = result["results"][0]["possession_model"]
    assert models["additive"]["status"] == "ok", models["additive"]
    assert models["condition_by_time"]["status"] == "ok", models["condition_by_time"]
    assert models["additive"]["n_games"] == 12
    assert models["additive"]["n_possessions"] == 240
    assert any("condition" in name for name in models["additive"]["coefficients"])
    assert any(":" in name for name in models["condition_by_time"]["coefficients"])

    export_dir = tmp_path / "export"
    # Use the actual writer's complete field schema, so the CLI reads the same
    # CSV representation as a saved-log export.
    write_experiment_tables(tables, export_dir)
    output = tmp_path / "stats.json"
    cli = CliRunner().invoke(app, ["stats", str(export_dir), str(output),
                                   "--bootstrap-replicates", "100"])
    assert cli.exit_code == 0, cli.output
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["results"][0]["metrics"]["ppp"]["effect_mean"] == 3.5
    assert read_experiment_tables(export_dir).manifest["schema_version"] == 1
    with pytest.raises(FileExistsError):
        write_experiment_stats(result, output)


def test_invalid_input_is_rejected():
    tables = _tables(seeds=range(1))
    tables.games[1]["game_id"] = tables.games[0]["game_id"]
    with pytest.raises(StatsError, match="duplicate game_id"):
        analyse_experiment(tables, metrics=("ppp",), bootstrap_replicates=100)


def test_insufficient_possessions_reports_no_fit():
    tables = _tables(seeds=range(2))
    tables.possessions.clear()
    result = analyse_experiment(tables, metrics=("ppp",), bootstrap_replicates=100)
    assert result["results"][0]["possession_model"]["additive"]["status"] == (
        "insufficient_data"
    )


def test_failed_possession_fit_reports_error(monkeypatch):
    import statsmodels.formula.api as smf

    def fail_fit(*args, **kwargs):
        raise ValueError("singular synthetic design")

    monkeypatch.setattr(smf, "mixedlm", fail_fit)
    result = analyse_experiment(_tables(seeds=range(2)), metrics=("ppp",),
                                bootstrap_replicates=100)
    models = result["results"][0]["possession_model"]
    assert models["additive"]["status"] == "fit_failed"
    assert "singular synthetic design" in models["condition_by_time"]["error"]
