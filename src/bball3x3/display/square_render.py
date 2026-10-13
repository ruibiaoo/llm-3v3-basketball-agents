"""Read-only square-court rendering (display package, WS1-now scope).

Mirrors the two-function shape of BasketWorld's ``core/rendering.py``
(``render_ascii`` / ``render_visual``) adapted to the square 15x11 metre
grid. No third-party import: ASCII needs the standard library only, and
frames import matplotlib lazily so the engine and test suite never
require it. Rendering reads state and never mutates it.
"""

import math

from bball3x3.schemas.state import GameState

DEFAULT_COURT = {
    "width_cells": 15,
    "depth_cells": 11,
    "basket": (7, 1),
    "arc_radius_m": 6.75,
}


class DisplayBackendMissing(RuntimeError):
    """Raised when frame rendering is called without matplotlib installed."""


def court_geometry(court: dict | None) -> dict:
    geometry = dict(DEFAULT_COURT)
    if court:
        geometry.update(court)
    geometry["basket"] = tuple(geometry["basket"])
    return geometry


def _holder(state: GameState) -> str | None:
    for pid, player in state.players.items():
        if player.has_ball:
            return pid
    return state.ball.holder


def render_ascii(state: GameState, court: dict | None = None) -> str:
    """Render one deterministic ASCII frame of the court."""
    geometry = court_geometry(court)
    width, depth = geometry["width_cells"], geometry["depth_cells"]
    basket = geometry["basket"]
    holder = _holder(state)

    header = (
        f"A {state.score.get('A', 0)} - {state.score.get('B', 0)} B | "
        f"game {state.game_clock_s:.1f}s | shot {state.shot_clock_s:.1f}s | "
        f"ball {holder} | event {state.event_idx} | {state.phase}"
    )
    footer = (
        f"fouls A{state.team_fouls.get('A', 0)} "
        f"B{state.team_fouls.get('B', 0)} | "
        f"clear={state.needs_clear} | offense={state.offense}"
    )
    rows = []
    for y in range(depth):
        cells = []
        for x in range(width):
            occupants = sorted(
                pid for pid, p in state.players.items() if p.pos == (x, y)
            )
            if (x, y) == basket and not occupants:
                cells.append(" B  ")
            elif not occupants:
                cells.append(" .  ")
            elif len(occupants) > 1:
                cells.append(" ## ")
            else:
                pid = occupants[0]
                cells.append(f"*{pid}*" if pid == holder else f" {pid} ")
        rows.append("|" + "|".join(cells) + "|")
    return "\n".join([header, *rows, footer])


def render_frame(
    state: GameState,
    court: dict | None = None,
    highlight: dict | None = None,
):
    """Render one matplotlib frame; returns an RGB numpy array.

    Deterministic: identical states produce byte-identical arrays.
    Requires matplotlib (dev-only); raises DisplayBackendMissing otherwise.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Arc, Circle, Rectangle
    except ImportError as exc:
        raise DisplayBackendMissing(
            "Frame rendering needs matplotlib (dev-only): "
            "uv add --optional display matplotlib pillow"
        ) from exc

    import numpy as np

    geometry = court_geometry(court)
    width, depth = geometry["width_cells"], geometry["depth_cells"]
    basket = geometry["basket"]
    arc_r = geometry["arc_radius_m"]
    holder = _holder(state)

    fig = Figure(figsize=(7.5, 5.5), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, depth - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(
        Rectangle(
            (-0.5, -0.5), width, depth, fill=False, edgecolor="black", linewidth=2
        )
    )
    ax.add_patch(
        Arc(
            basket,
            2 * arc_r,
            2 * arc_r,
            theta1=0,
            theta2=180,
            edgecolor="red",
            linewidth=2,
        )
    )
    ax.add_patch(Circle(basket, 0.35, fill=False, edgecolor="red", linewidth=3))

    # Deterministic declutter: players sharing a cell fan out around it so
    # none hide beneath another. Pure function of state; determinism holds.
    occupants: dict[tuple[int, int], list[str]] = {}
    for pid in sorted(state.players):
        occupants.setdefault(state.players[pid].pos, []).append(pid)
    offsets: dict[str, tuple[float, float]] = {}
    for pos, members in occupants.items():
        for i, pid in enumerate(members):
            if len(members) == 1:
                offsets[pid] = (0.0, 0.0)
            else:
                angle = 2 * math.pi * i / len(members)
                offsets[pid] = (0.30 * math.cos(angle), 0.30 * math.sin(angle))

    for pid in sorted(state.players):
        player = state.players[pid]
        color = "blue" if player.team == "A" else "red"
        dx, dy = offsets[pid]
        cx, cy = (player.pos[0] + dx, player.pos[1] + dy)
        ax.add_patch(
            Circle((cx, cy), 0.4, facecolor=color, edgecolor="white", alpha=0.9)
        )
        ax.text(
            cx,
            cy,
            pid,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
        if pid == holder:
            ax.add_patch(
                Circle((cx, cy), 0.55, fill=False, edgecolor="orange", linewidth=3)
            )

    if highlight and highlight.get("arrow"):
        (x1, y1), (x2, y2) = highlight["arrow"]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 2},
        )

    ax.set_title(
        f"A {state.score.get('A', 0)} - {state.score.get('B', 0)} B  "
        f"event {state.event_idx}  shot {state.shot_clock_s:.0f}s",
        fontsize=10,
    )
    canvas.draw()
    array = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    fig.clf()
    return array
