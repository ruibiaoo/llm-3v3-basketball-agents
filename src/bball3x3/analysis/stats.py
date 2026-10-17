"""Seed-paired ablation statistics and exploratory possession mixed models.

The seed is the bootstrap/test unit: its two opposite first-possession games
are averaged before inference. Possession models are descriptive Gaussian
random-intercept models; paired seed inference remains the primary analysis.
"""

from __future__ import annotations

import csv
import json
import math
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

from .exports import ExperimentTables

DEFAULT_METRICS = ("ppp", "maar", "ccar", "cssr", "maur", "grounded_maur", "iar")
_BINS = ("q1", "q2", "q3", "q4", "q5", "ot")


class StatsError(ValueError):
    """Export tables are missing required identity or pairing evidence."""


def read_experiment_tables(directory: str | Path) -> ExperimentTables:
    """Read the unmodified CSV/manifest bundle written by ``bball export``."""
    source = Path(directory)

    def rows(name: str) -> list[dict[str, Any]]:
        with (source / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    with (source / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise StatsError("Unsupported experiment export schema version")
    return ExperimentTables(rows("games"), rows("teams"), rows("possessions"), manifest)


def _true(value: Any) -> bool:
    return value is True or value == "True"


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _seed_key(value: Any) -> str:
    if value is None or str(value) == "":
        raise StatsError("Game is missing its seed")
    return str(value)


def _pair_effects(
    games: list[dict[str, Any]], teams: dict[str, dict[str, dict[str, Any]]],
    baseline: str, comparison: str, metric: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        grouped[_seed_key(game.get("seed"))].append(game)
    paired, excluded = [], []
    for seed, variants in sorted(grouped.items()):
        sides = [game.get("first_possession") for game in variants]
        if len(variants) != 2 or sorted(sides) != ["A", "B"]:
            excluded.append({"seed": seed, "reason": "missing_or_duplicate_start_side",
                             "game_ids": [game["game_id"] for game in variants]})
            continue
        differences = []
        reason = None
        for game in variants:
            game_id = game["game_id"]
            if not _true(game.get("game_complete")):
                reason = "incomplete_game"
                break
            if game.get("possession_validation") != "ok":
                reason = "invalid_possessions"
                break
            team_rows = teams.get(game_id, {})
            if set(team_rows) != {baseline, comparison}:
                reason = "missing_or_duplicate_team_condition"
                break
            values = []
            for condition in (baseline, comparison):
                row = team_rows[condition]
                if row.get(f"{metric}_status") != "ok":
                    reason = f"{condition}_{metric}_{row.get(f'{metric}_status', 'missing')}"
                    break
                value = _finite(row.get(f"{metric}_rate"))
                if value is None:
                    reason = f"{condition}_{metric}_invalid_rate"
                    break
                values.append(value)
            if reason:
                break
            differences.append(values[0] - values[1])
        if reason:
            excluded.append({"seed": seed, "reason": reason,
                             "game_ids": [game["game_id"] for game in variants]})
        else:
            paired.append({"seed": seed, "game_ids": [game["game_id"] for game in variants],
                           "game_differences": differences,
                           "difference": sum(differences) / 2})
    return paired, excluded


def _summary(paired: list[dict[str, Any]], excluded: list[dict[str, Any]],
             analysis_seed: int, bootstrap_replicates: int) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    values = [row["difference"] for row in paired]
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    zeros = len(values) - positive - negative
    result: dict[str, Any] = {
        "status": "ok" if len(values) >= 2 else "insufficient_pairs",
        "effect_mean": sum(values) / len(values) if values else None,
        "n_seeds": len(values), "n_games": 2 * len(values),
        "positive_seeds": positive, "negative_seeds": negative, "zero_seeds": zeros,
        "sign_agreement": max(positive, negative) / (positive + negative)
        if positive + negative else None,
        "paired_seeds": paired, "excluded_seeds": excluded,
        "bootstrap_95_ci": None, "wilcoxon": {"status": "insufficient_pairs", "p_value": None},
    }
    if len(values) < 2:
        return result
    rng = random.Random(analysis_seed)
    n = len(values)
    samples = sorted(sum(rng.choices(values, k=n)) / n for _ in range(bootstrap_replicates))
    # Linear interpolation of the empirical bootstrap distribution.
    def quantile(probability: float) -> float:
        position = (len(samples) - 1) * probability
        left = int(position)
        return samples[left] + (samples[min(left + 1, len(samples) - 1)] - samples[left]) * (
            position - left
        )

    result["bootstrap_95_ci"] = [quantile(0.025), quantile(0.975)]
    if positive + negative == 0:
        result["wilcoxon"] = {"status": "all_zero", "p_value": None}
    else:
        test = wilcoxon(values, zero_method="wilcox", alternative="two-sided")
        result["wilcoxon"] = {"status": "ok", "statistic": float(test.statistic),
                              "p_value": float(test.pvalue), "zero_method": "wilcox"}
    return result


def _fit_model(rows: list[dict[str, Any]], baseline: str, interaction: bool) -> dict[str, Any]:
    import pandas as pd
    import statsmodels.formula.api as smf

    games = {row["game_id"] for row in rows}
    bins = {row["time_bin"] for row in rows}
    conditions = {row["condition"] for row in rows}
    result: dict[str, Any] = {
        "status": "insufficient_data", "n_possessions": len(rows), "n_games": len(games),
        "time_bins": [name for name in _BINS if name in bins],
        "family": "Gaussian linear mixed model", "random_intercept": "game_id",
    }
    if (len(games) < 4 or len(rows) < 20 or len(bins) < 2 or len(conditions) != 2
            or any(sum(row["condition"] == condition for row in rows) < 8
                   for condition in conditions)
            or len({_finite(row["score_diff_before"]) for row in rows}) < 2):
        return result
    # Physical team is intentionally not a fixed effect: treatment is the
    # assigned condition, and both opposing teams remain in their game cluster.
    term = f"C(condition, Treatment(reference={baseline!r}))"
    formula = f"points ~ {term} * C(time_bin)" if interaction else (
        f"points ~ {term} + score_diff_before + C(time_bin)"
    )
    if interaction:
        formula += " + score_diff_before"
    result["formula"] = formula
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frame = pd.DataFrame(rows)
            fit = smf.mixedlm(formula, frame, groups=frame["game_id"]).fit(
                reml=False, method="lbfgs", disp=False
            )
        result["warnings"] = [str(item.message) for item in caught]
        if not fit.converged:
            result["status"] = "fit_failed"
            result["error"] = "Optimizer did not converge"
            return result
        interval = fit.conf_int()
        variance = float(fit.cov_re.iloc[0, 0])
        if not math.isfinite(variance) or any(
            not all(math.isfinite(value) for value in (
                float(estimate), float(fit.pvalues[name]),
                float(interval.loc[name, 0]), float(interval.loc[name, 1]),
            )) for name, estimate in fit.fe_params.items()
        ):
            result["status"] = "fit_failed"
            result["error"] = "Fit returned non-finite estimates or intervals"
            return result
        result["status"] = "ok"
        result["random_intercept_variance"] = variance
        result["coefficients"] = {
            name: {"estimate": float(value), "p_value": float(fit.pvalues[name]),
                   "ci_95": [float(interval.loc[name, 0]), float(interval.loc[name, 1])]}
            for name, value in fit.fe_params.items()
        }
    except (ValueError, ArithmeticError, IndexError, KeyError) as exc:
        result["status"] = "fit_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _model_rows(possessions: list[dict[str, Any]], game_ids: set[str],
                conditions: set[str]) -> tuple[list[dict[str, Any]], int]:
    eligible = []
    excluded = 0
    for row in possessions:
        if row.get("game_id") not in game_ids or row.get("condition") not in conditions:
            continue
        points, score = _finite(row.get("points")), _finite(row.get("score_diff_before"))
        if (not _true(row.get("model_eligible")) or row.get("time_bin") not in _BINS
                or points is None or score is None):
            excluded += 1
            continue
        eligible.append({"game_id": row["game_id"], "condition": row["condition"],
                         "points": points, "score_diff_before": score,
                         "time_bin": row["time_bin"]})
    return eligible, excluded


def analyse_experiment(
    tables: ExperimentTables, *, baseline: str = "A_full",
    metrics: tuple[str, ...] = DEFAULT_METRICS, analysis_seed: int = 4213,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    """Analyse each baseline-vs-ablation experiment from saved export tables."""
    if bootstrap_replicates < 100:
        raise StatsError("bootstrap_replicates must be at least 100")
    if not tables.games:
        raise StatsError("No games in experiment export")
    team_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in tables.teams:
        game_id, condition = row.get("game_id"), row.get("condition")
        if not game_id or not condition or condition in team_rows[game_id]:
            raise StatsError(f"Missing or duplicate condition in team rows for {game_id!r}")
        team_rows[game_id][condition] = row
    seen_games: set[str] = set()
    comparisons: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for game in tables.games:
        game_id = game.get("game_id")
        if not game_id or game_id in seen_games:
            raise StatsError(f"Missing or duplicate game_id: {game_id!r}")
        seen_games.add(game_id)
        conditions = {game.get("team_A_condition"), game.get("team_B_condition")}
        if baseline not in conditions or len(conditions) != 2:
            raise StatsError(f"Game {game_id!r} must have {baseline!r} and one ablation")
        other = next(condition for condition in conditions if condition != baseline)
        comparisons[(game.get("experiment_id") or "", other)].append(game)
    results = []
    for (experiment_id, comparison), games in sorted(comparisons.items()):
        paired_by_metric = {}
        for metric in metrics:
            paired, excluded = _pair_effects(games, team_rows, baseline, comparison, metric)
            paired_by_metric[metric] = _summary(paired, excluded, analysis_seed,
                                                bootstrap_replicates)
        ppp_game_ids = {game_id for pair in paired_by_metric["ppp"]["paired_seeds"]
                        for game_id in pair["game_ids"]} if "ppp" in metrics else set()
        rows, excluded_possessions = _model_rows(tables.possessions, ppp_game_ids,
                                                 {baseline, comparison})
        results.append({"experiment_id": experiment_id, "comparison": comparison,
                        "metrics": paired_by_metric, "possession_model": {
                            "additive": _fit_model(rows, baseline, False),
                            "condition_by_time": _fit_model(rows, baseline, True),
                            "excluded_possessions": excluded_possessions,
                        }})
    return {
        "schema_version": 1, "baseline": baseline,
        "analysis_seed": analysis_seed, "bootstrap_replicates": bootstrap_replicates,
        "pairing": "two A/B first-possession variants per seed; game difference averaged per seed",
        "missing_pair_policy": "exclude seed from that metric and report reason",
        "bootstrap_unit": "seed (both games kept together)",
        "wilcoxon_unit": "per-seed average difference; two-sided; zeros dropped",
        "model_note": "Exploratory Gaussian approximation to discrete points. Both opposing teams "
                      "share a game random intercept; cross-game seed dependence is not modeled. "
                      "Use seed-paired results for primary inference. "
                      "OT is a separate time category.",
        "source_excluded_games": tables.manifest.get("excluded_games", []),
        "results": results,
    }


def write_experiment_stats(result: dict[str, Any], path: str | Path) -> Path:
    """Write one JSON result, refusing to replace an existing analysis."""
    target = Path(path)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return target
