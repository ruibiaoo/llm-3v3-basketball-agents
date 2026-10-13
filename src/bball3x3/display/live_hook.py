"""Guarded live display hook (runner-owned call site, WS6 contract).

``live_hook(snapshot, ctx)`` takes a deep-copied snapshot plus a context
dict carrying ``event_idx``, ``exp_id``, ``game_id``, and the display
config. It draws no RNG, mutates nothing, and never raises: failures are
counted in the returned stats so the runner can total them without the
game ever failing on display.
"""

from pathlib import Path

from bball3x3.schemas.state import GameState

DEFAULTS = {
    "enabled": True,
    "mode": "live",
    "out_dir": "frames",
    "every_n_events": 1,
    "max_frames_per_game": 2000,
    "fail_isolated": True,
}


def _save_png(path: Path, frame) -> None:
    """Persist one RGB frame. Requires matplotlib (dev-only)."""
    from matplotlib.image import imsave

    imsave(path, frame)


def live_hook(snapshot: GameState, ctx: dict | None = None) -> dict:
    """Render one frame for a snapshot. Never raises; returns stats."""
    stats = {"frames_written": 0, "errors": 0, "skipped": 0}
    try:
        ctx = ctx or {}
        cfg = {**DEFAULTS, **(ctx.get("display") or {})}
        if not cfg.get("enabled", True) or cfg.get("mode", "live") != "live":
            stats["skipped"] = 1
            return stats
        event_idx = int(ctx.get("event_idx", snapshot.event_idx))
        throttle = max(1, int(cfg.get("every_n_events", 1)))
        if event_idx % throttle != 0:
            stats["skipped"] = 1
            return stats
        out_dir = Path(ctx.get("game_dir", ".")) / str(cfg.get("out_dir", "frames"))
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(list(out_dir.glob("frame_*.png"))) >= int(
            cfg.get("max_frames_per_game", 2000)
        ):
            stats["skipped"] = 1
            return stats
        from bball3x3.display.square_render import render_frame

        frame = render_frame(snapshot, court=ctx.get("court"))
        _save_png(out_dir / f"frame_{event_idx:05d}.png", frame)
        stats["frames_written"] = 1
    except Exception:
        stats["errors"] = 1
    return stats
