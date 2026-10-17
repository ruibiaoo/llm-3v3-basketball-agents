"""Reproducible adaptation curves from an exported experiment bundle."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exports import ExperimentTables

_METRICS = ("ppp", "maur", "cssr")
_BINS = ("q1", "q2", "q3", "q4", "q5", "ot")
_FIELDS = ("experiment_id", "comparison", "metric", "condition", "time_bin",
           "numerator", "denominator", "rate", "status", "n_games_with_exposure",
           "n_games_without_evidence", "n_games_not_applicable")


class PlotError(ValueError):
    """Export lacks reliable adaptation counts or pairing identity."""


@dataclass
class AdaptationPlotData:
    """Pooled curve points and all exclusions, before image rendering."""

    rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def _complete(value: Any) -> bool:
    return value is True or value == "True"


def _count(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise PlotError(f"{context}: count must be a nonnegative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PlotError(f"{context}: count must be a nonnegative integer") from exc
    if number < 0 or str(number) != str(value):
        raise PlotError(f"{context}: count must be a nonnegative integer")
    return number


def _adaptation(row: dict[str, Any], game_id: str) -> dict[str, Any]:
    value = row.get("adaptation_json")
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError) as exc:
        raise PlotError(f"Game {game_id!r} has invalid adaptation_json") from exc
    if not isinstance(data, dict):
        raise PlotError(f"Game {game_id!r} has invalid adaptation_json")
    return data


def _bin_rate(adaptation: dict[str, Any], metric: str, bin_name: str,
              game_id: str) -> tuple[str, int, int]:
    context = f"Game {game_id!r}, {metric}/{bin_name}"
    try:
        value = adaptation[metric][bin_name]
        status = value["status"]
        numerator = _count(value["numerator"], context)
        denominator = _count(value["denominator"], context)
    except (KeyError, TypeError) as exc:
        raise PlotError(f"{context}: missing adaptation count/status") from exc
    if status not in {"ok", "undefined", "not_applicable", "incomplete",
                      "insufficient_evidence"}:
        raise PlotError(f"{context}: unknown status {status!r}")
    if status == "ok" and (denominator == 0 or
                           (metric != "ppp" and numerator > denominator)):
        raise PlotError(f"{context}: invalid rate counts")
    return status, numerator, denominator


def build_adaptation_plot_data(
    tables: ExperimentTables, *, baseline: str = "A_full"
) -> AdaptationPlotData:
    """Pool count evidence for eligible paired games; never fill absent bins with zero.

    A seed is eligible only with exactly one A-start and one B-start game, both
    complete and possession-validated. Counts are pooled within condition/bin;
    this weights each exposed opportunity rather than each game equally.
    """
    if not tables.games:
        raise PlotError("No games in experiment export")
    team_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in tables.teams:
        game_id, condition = row.get("game_id"), row.get("condition")
        if not game_id or not condition or condition in team_rows[game_id]:
            raise PlotError(f"Missing or duplicate condition for game {game_id!r}")
        team_rows[game_id][condition] = row

    comparisons: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_ids: set[str] = set()
    for game in tables.games:
        game_id = game.get("game_id")
        if not game_id or game_id in seen_ids:
            raise PlotError(f"Missing or duplicate game_id: {game_id!r}")
        seen_ids.add(game_id)
        conditions = {game.get("team_A_condition"), game.get("team_B_condition")}
        if baseline not in conditions or len(conditions) != 2:
            raise PlotError(f"Game {game_id!r} must have baseline and one ablation")
        seed = game.get("seed")
        if seed is None or str(seed) == "":
            raise PlotError(f"Game {game_id!r} has no seed")
        comparison = next(name for name in conditions if name != baseline)
        comparisons[(game.get("experiment_id") or "", comparison)][str(seed)].append(game)

    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for (experiment_id, comparison), seeds in sorted(comparisons.items()):
        accepted_games: list[str] = []
        excluded_seeds: list[dict[str, Any]] = []
        for seed, variants in sorted(seeds.items()):
            game_ids = [game["game_id"] for game in variants]
            sides = [game.get("first_possession") for game in variants]
            reason = None
            if len(variants) != 2 or sorted(sides) != ["A", "B"]:
                reason = "missing_or_duplicate_start_side"
            elif any(not _complete(game.get("game_complete")) for game in variants):
                reason = "incomplete_game"
            elif any(game.get("possession_validation") != "ok" for game in variants):
                reason = "invalid_possessions"
            elif any(set(team_rows.get(game_id, {})) != {baseline, comparison}
                     for game_id in game_ids):
                reason = "missing_team_condition"
            if reason:
                excluded_seeds.append({"seed": seed, "game_ids": game_ids, "reason": reason})
            else:
                accepted_games.extend(game_ids)

        adaptations = {
            game_id: {condition: _adaptation(team_rows[game_id][condition], game_id)
                      for condition in (baseline, comparison)}
            for game_id in accepted_games
        }
        for metric in _METRICS:
            for condition in (baseline, comparison):
                for bin_name in _BINS:
                    numerator = denominator = exposed = no_evidence = not_applicable = 0
                    for game_id in accepted_games:
                        status, count, opportunities = _bin_rate(
                            adaptations[game_id][condition], metric, bin_name, game_id
                        )
                        if status == "ok":
                            numerator += count
                            denominator += opportunities
                            exposed += 1
                        elif status in ("incomplete", "insufficient_evidence"):
                            no_evidence += 1
                        elif status == "not_applicable":
                            not_applicable += 1
                    if exposed:
                        status = "partial_evidence" if no_evidence else "ok"
                        rate = numerator / denominator
                    elif no_evidence:
                        status, rate = "insufficient_evidence", None
                    elif not_applicable and not_applicable == len(accepted_games):
                        status, rate = "not_applicable", None
                    else:
                        status, rate = "no_exposure", None
                    rows.append({
                        "experiment_id": experiment_id, "comparison": comparison,
                        "metric": metric, "condition": condition, "time_bin": bin_name,
                        "numerator": numerator, "denominator": denominator,
                        "rate": rate, "status": status, "n_games_with_exposure": exposed,
                        "n_games_without_evidence": no_evidence,
                        "n_games_not_applicable": not_applicable,
                    })
        groups.append({"experiment_id": experiment_id, "comparison": comparison,
                       "n_paired_seeds": len(accepted_games) // 2,
                       "n_paired_games": len(accepted_games),
                       "excluded_seeds": excluded_seeds})
    return AdaptationPlotData(rows, {
        "schema_version": 1, "baseline": baseline, "metrics": list(_METRICS),
        "regulation_bins": list(_BINS[:5]), "overtime_bin": "ot",
        "bin_edges_elapsed_fraction": [0, 0.2, 0.4, 0.6, 0.8, 1],
        "bin_source": "game clock/phase before event, relative to configured regulation duration",
        "ppp_attribution": "possession start", "maur_cssr_attribution": "event start",
        "aggregation": "sum valid per-game numerators and denominators within condition/bin; "
                       "exclude missing evidence; no exposure is null, never zero",
        "pairing": "both A-start and B-start complete, possession-validated games per seed",
        "overtime_display": "separate marker, never connected to regulation curve",
        "source_excluded_games": tables.manifest.get("excluded_games", []),
        "groups": groups,
    })


def _render_plot(rows: list[dict[str, Any]], path: Path, baseline: str,
                 experiment_id: str, comparison: str, metric: str) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    palette = ((baseline, "#1f77b4"), (comparison, "#d95f02"))
    any_rate = False
    for condition, color in palette:
        points = {row["time_bin"]: row for row in rows
                  if row["metric"] == metric and row["condition"] == condition}
        regulation = [points[name]["rate"] for name in _BINS[:5]]
        curve = [float(value) if value is not None else math.nan for value in regulation]
        if any(math.isfinite(value) for value in curve):
            any_rate = True
            ax.plot(range(1, 6), curve, marker="o", linewidth=2, color=color,
                    label=condition)
        overtime = points["ot"]["rate"]
        if overtime is not None:
            any_rate = True
            ax.scatter([6], [overtime], marker="D", s=55, color=color,
                       label=condition if not any(math.isfinite(value) for value in curve)
                       else None)
    ax.axvline(5.5, linestyle=":", color="0.55", linewidth=1)
    ax.set_xticks(range(1, 7), labels=[*_BINS[:5], "OT"])
    ax.set_xlim(0.8, 6.3)
    ax.set_xlabel("Elapsed regulation-time quintile (OT separate)")
    ax.set_ylabel("Points per possession" if metric == "ppp" else "Share")
    if metric != "ppp":
        ax.set_ylim(0, 1.05)
    ax.set_title(f"{experiment_id}: {metric.upper()} — {baseline} vs {comparison}")
    ax.grid(axis="y", alpha=0.25)
    if any_rate:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No eligible exposure", transform=ax.transAxes,
                ha="center", va="center")
    figure.savefig(path, dpi=140, metadata={"Software": "bball3x3 WS6"})


def write_adaptation_plots(data: AdaptationPlotData, directory: str | Path) -> Path:
    """Write pooled CSV, provenance manifest and one PNG per comparison/metric.

    The output directory must not exist; unsafe experiment IDs never enter
    filenames. Rendering has no random component.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "adaptation.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(data.rows)
    manifest = {**data.manifest, "plots": []}
    baseline = manifest["baseline"]
    for index, group in enumerate(manifest["groups"], start=1):
        group_rows = [row for row in data.rows
                      if row["experiment_id"] == group["experiment_id"]
                      and row["comparison"] == group["comparison"]]
        for metric in _METRICS:
            filename = f"comparison-{index:03d}-{metric}.png"
            _render_plot(group_rows, destination / filename, baseline,
                         group["experiment_id"], group["comparison"], metric)
            manifest["plots"].append({"experiment_id": group["experiment_id"],
                                      "comparison": group["comparison"],
                                      "metric": metric, "file": filename})
    with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return destination
