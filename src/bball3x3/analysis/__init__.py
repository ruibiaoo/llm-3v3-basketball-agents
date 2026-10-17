"""Tools for reading and analysing game logs."""

from .exports import ExperimentTables, ExportError, build_experiment_tables, write_experiment_tables
from .log_reader import GameLogIndex, LogReadError, iter_jsonl_records, read_game_log
from .metrics import (
    AdaptationMetrics,
    GameMetrics,
    MemoryGrowthMetric,
    RateMetric,
    compute_game_metrics,
)
from .plots import AdaptationPlotData, PlotError, build_adaptation_plot_data, write_adaptation_plots
from .possessions import (
    PossessionError,
    PossessionSummary,
    PossessionTracker,
    reconstruct_possessions,
    validate_possessions,
)
from .qualitative import TranscriptError, render_possession_transcript
from .stats import StatsError, analyse_experiment, read_experiment_tables, write_experiment_stats

__all__ = [
    "GameLogIndex",
    "ExperimentTables",
    "ExportError",
    "GameMetrics",
    "AdaptationMetrics",
    "AdaptationPlotData",
    "LogReadError",
    "MemoryGrowthMetric",
    "RateMetric",
    "PossessionError",
    "PossessionSummary",
    "PossessionTracker",
    "PlotError",
    "TranscriptError",
    "StatsError",
    "analyse_experiment",
    "compute_game_metrics",
    "build_experiment_tables",
    "build_adaptation_plot_data",
    "iter_jsonl_records",
    "read_game_log",
    "read_experiment_tables",
    "reconstruct_possessions",
    "render_possession_transcript",
    "validate_possessions",
    "write_experiment_tables",
    "write_experiment_stats",
    "write_adaptation_plots",
]
