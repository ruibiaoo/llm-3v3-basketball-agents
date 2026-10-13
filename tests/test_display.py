"""Tests for the read-only display package (plan section 5.6, WS6 contract).

Frame tests skip without matplotlib; everything else runs everywhere, so
the suite stays green with third_party/ absent and display deps missing.
"""

import ast
import json
from pathlib import Path

import pytest

from bball3x3.config import SimulationConfig, load_display_config
from bball3x3.display.live_hook import live_hook
from bball3x3.display.replay import load_states, shot_chart
from bball3x3.display.square_render import render_ascii
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.state import create_initial_state

GOLDEN_ASCII = """A 0 - 0 B | game 600.0s | shot 12.0s | ball A1 | event 0 | LIVE
| .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| .  | .  | .  | .  | .  | .  | .  | B  | .  | .  | .  | .  | .  | .  | .  |
| .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| .  | .  | .  | B2 | .  | .  | .  | .  | .  | .  | .  | B3 | .  | .  | .  |
| .  | .  | .  | .  | .  | .  | .  | B1 | .  | .  | .  | .  | .  | .  | .  |
| .  | .  | .  | A2 | .  | .  | .  | .  | .  | .  | .  | A3 | .  | .  | .  |
| .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| .  | .  | .  | .  | .  | .  | .  |*A1*| .  | .  | .  | .  | .  | .  | .  |
| .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
fouls A0 B0 | clear=False | offense=A"""


@pytest.fixture
def state():
    return create_initial_state(
        SimulationConfig(), GameRNG.from_seed(4), first_possession="A"
    )


def test_ascii_golden_frame(state):
    assert render_ascii(state) == GOLDEN_ASCII


def test_ascii_marks_holder_and_basket(state):
    frame = render_ascii(state)
    assert "*A1*" in frame
    assert "B  " in frame


def test_display_config_defaults():
    cfg = load_display_config("configs/display.yaml")
    assert cfg.enabled is False
    assert cfg.mode == "off"
    assert cfg.every_n_events == 1


def test_no_third_party_imports_in_src():
    root = Path("src/bball3x3")
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            if "third_party" in names or "basketworld" in names:
                offenders.append(str(path))
    assert offenders == []


def test_live_hook_disabled_is_noop(state, tmp_path):
    stats = live_hook(
        state,
        {"display": {"enabled": False}, "game_dir": str(tmp_path), "event_idx": 0},
    )
    assert stats == {"frames_written": 0, "errors": 0, "skipped": 1}
    assert list(tmp_path.iterdir()) == []


def test_live_hook_failure_isolated(state, tmp_path, monkeypatch):
    import bball3x3.display.live_hook as hook_module

    def boom(snapshot, court=None):
        raise RuntimeError("renderer down")

    monkeypatch.setattr(
        "bball3x3.display.square_render.render_frame", boom
    )
    before = state.model_dump()
    stats = live_hook(
        state,
        {"display": {"enabled": True}, "game_dir": str(tmp_path), "event_idx": 3},
    )
    assert stats["errors"] == 1
    assert stats["frames_written"] == 0
    assert state.model_dump() == before
    assert hook_module.DEFAULTS["fail_isolated"] is True


def test_live_hook_throttle_and_cap(state, tmp_path, monkeypatch):
    calls = []

    def fake_render(snapshot, court=None):
        calls.append(snapshot.event_idx)
        return [[0]]

    def fake_save(path, frame):
        Path(path).write_text("png", encoding="utf-8")

    monkeypatch.setattr(
        "bball3x3.display.square_render.render_frame", fake_render
    )
    monkeypatch.setattr("bball3x3.display.live_hook._save_png", fake_save)
    ctx = {
        "display": {"enabled": True, "every_n_events": 2, "max_frames_per_game": 1},
        "game_dir": str(tmp_path),
        "event_idx": 3,
    }
    assert live_hook(state, ctx)["skipped"] == 1
    assert calls == []
    ctx["event_idx"] = 4
    assert live_hook(state, ctx)["frames_written"] == 1
    assert live_hook(state, ctx)["skipped"] == 1  # cap reached


def test_replay_loads_states_and_shot_chart(state, tmp_path):
    log_path = tmp_path / "game.jsonl"
    records = [
        {"record_type": "state", "event_idx": 0, "state": state.model_dump()},
        {
            "record_type": "resolution",
            "event_idx": 0,
            "actions": {"A1": "SHOOT"},
            "outcomes": {"shooter": "A1", "made": True, "points": 1},
            "model_terms": {"p_make": 0.62, "zone": "RIM", "contest": 0.1},
            "rng_draws": {},
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    states = load_states(log_path)
    assert len(states) == 1
    assert states[0].ball.holder == "A1"
    chart = shot_chart(log_path)
    assert chart == [
        {
            "event_idx": 0,
            "shooter": "A1",
            "made": True,
            "points": 1,
            "p_make": 0.62,
            "pos": None,
            "zone": "RIM",
            "contest": 0.1,
        }
    ]


def test_render_frame_deterministic(state):
    matplotlib = pytest.importorskip("matplotlib")
    _ = matplotlib
    from bball3x3.display.square_render import render_frame

    first = render_frame(state)
    second = render_frame(state)
    assert first.shape == second.shape
    assert first.dtype == second.dtype
    assert bool((first == second).all())
