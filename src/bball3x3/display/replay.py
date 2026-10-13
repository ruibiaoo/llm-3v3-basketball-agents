"""Offline JSONL replay (post mode): state records to frames, GIF, charts.

Consumes only the WS6 display contract: `state` records carrying the full
`GameState` and `resolution` records carrying positions plus model terms.
The writer does not exist yet (WS6); this module is forward-compatible
with it and testable today against hand-built record streams.
"""

import argparse
import json
from pathlib import Path

from bball3x3.schemas.state import GameState


def iter_records(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_states(path: str | Path) -> list[GameState]:
    """Return the GameState of every `state` record in log order."""
    return [
        GameState(**record["state"])
        for record in iter_records(path)
        if record.get("record_type") == "state" and "state" in record
    ]


def shot_chart(path: str | Path) -> list[dict]:
    """Aggregate one row per `resolution` record with a shooter attached.

    Tolerant of writer evolution: rows carry whatever the record holds,
    and missing terms stay None rather than failing the replay.
    """
    rows = []
    for record in iter_records(path):
        if record.get("record_type") != "resolution":
            continue
        outcomes = record.get("outcomes") or {}
        terms = record.get("model_terms") or {}
        if "shooter" not in outcomes and "shooter" not in terms:
            continue
        rows.append(
            {
                "event_idx": record.get("event_idx"),
                "shooter": outcomes.get("shooter", terms.get("shooter")),
                "made": outcomes.get("made"),
                "points": outcomes.get("points"),
                "p_make": terms.get("p_make"),
                "pos": terms.get("shooter_pos", outcomes.get("shooter_pos")),
                "zone": terms.get("zone"),
                "contest": terms.get("contest"),
            }
        )
    return rows


def write_gif(
    states: list[GameState],
    out_path: str | Path,
    court: dict | None = None,
    fps: int = 2,
) -> Path:
    """Render states to an animated GIF. Needs matplotlib + Pillow."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise RuntimeError(
            "GIF replay needs matplotlib and Pillow (dev-only): "
            "uv add --optional display matplotlib pillow"
        ) from exc

    from bball3x3.display.square_render import render_frame

    frames = [render_frame(state, court=court) for state in states]
    fig, ax = plt.subplots()
    ax.axis("off")
    image = ax.imshow(frames[0])

    def animate(i: int):
        image.set_array(frames[i])
        return [image]

    animation = FuncAnimation(fig, animate, frames=len(frames))
    out = Path(out_path)
    animation.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def main(argv=None) -> Path:
    """CLI: uv run python -m bball3x3.display.replay <game.jsonl> --out replay.gif."""
    parser = argparse.ArgumentParser(description="Replay a game JSONL to GIF.")
    parser.add_argument("jsonl", help="game event log (.jsonl)")
    parser.add_argument("--out", required=True, help="output GIF path")
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args(argv)
    return write_gif(load_states(args.jsonl), args.out, fps=args.fps)


if __name__ == "__main__":
    main()
