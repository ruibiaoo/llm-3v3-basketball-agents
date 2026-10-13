# display package (read-only visualisation; the engine never imports this).
#
# Lazy imports only: importing this package must never require matplotlib,
# Pillow, or third-party code, so the engine suite stays green without them.
__all__ = [
    "live_hook",
    "render_ascii",
    "render_frame",
    "load_states",
    "shot_chart",
    "write_gif",
]


def __getattr__(name: str):
    if name == "live_hook":
        from bball3x3.display.live_hook import live_hook

        return live_hook
    if name in ("render_ascii", "render_frame"):
        from bball3x3.display import square_render

        return getattr(square_render, name)
    if name in ("load_states", "shot_chart", "write_gif"):
        from bball3x3.display import replay

        return getattr(replay, name)
    raise AttributeError(f"module 'bball3x3.display' has no attribute {name!r}")
