"""Offline analysis commands for saved game logs."""

from pathlib import Path

import typer

from .analysis import (
    analyse_experiment,
    build_adaptation_plot_data,
    build_experiment_tables,
    read_experiment_tables,
    read_game_log,
    render_possession_transcript,
    write_adaptation_plots,
    write_experiment_stats,
    write_experiment_tables,
)

app = typer.Typer(no_args_is_help=True, help="Inspect saved 3v3 basketball game logs.")


@app.callback()
def main() -> None:
    """Offline tools; GAME arguments are paths to game JSONL files."""


@app.command()
def transcript(
    game: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    possession: str = typer.Argument(..., help="Exact possession ID, e.g. p0001."),
) -> None:
    """Print observations, communication, decisions and outcomes for a possession."""
    try:
        rendered = render_possession_transcript(read_game_log(game), possession)
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(rendered, nl=False)


@app.command("export")
def export_experiment(
    output_dir: Path = typer.Argument(..., help="New directory for CSV tables and manifest."),
    games: list[Path] = typer.Argument(..., help="One or more saved game JSONL paths."),
    include_incomplete: bool = typer.Option(False, "--include-incomplete"),
    window: int = typer.Option(1, "--window", min=1, max=2),
) -> None:
    """Export game, team, and possession tables from saved logs."""
    try:
        tables = build_experiment_tables(games, include_incomplete=include_incomplete,
                                         window=window)
        write_experiment_tables(tables, output_dir)
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Exported {len(tables.games)} games, {len(tables.teams)} teams, "
        f"{len(tables.possessions)} possessions to {output_dir}"
    )


@app.command("stats")
def experiment_stats(
    export_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output_json: Path = typer.Argument(..., help="New JSON file for the analysis results."),
    baseline: str = typer.Option("A_full", "--baseline"),
    analysis_seed: int = typer.Option(4213, "--analysis-seed"),
    bootstrap_replicates: int = typer.Option(10_000, "--bootstrap-replicates", min=100),
) -> None:
    """Compute seed-paired effects and exploratory possession models from an export."""
    try:
        result = analyse_experiment(read_experiment_tables(export_dir), baseline=baseline,
                                    analysis_seed=analysis_seed,
                                    bootstrap_replicates=bootstrap_replicates)
        write_experiment_stats(result, output_json)
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Analysed {len(result['results'])} comparisons to {output_json}")


@app.command("plots")
def adaptation_plots(
    export_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output_dir: Path = typer.Argument(..., help="New directory for PNG curves and pooled CSV."),
    baseline: str = typer.Option("A_full", "--baseline"),
) -> None:
    """Render PPP, MAUR and CSSR curves from saved experiment tables."""
    try:
        data = build_adaptation_plot_data(read_experiment_tables(export_dir), baseline=baseline)
        write_adaptation_plots(data, output_dir)
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote {len(data.manifest['groups']) * 3} plots to {output_dir}")
