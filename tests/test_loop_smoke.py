"""WS1 smoke test: a heuristic game must finish with SIR = 0."""

from bball3x3.config import SimulationConfig
from bball3x3.engine.loop import play_heuristic_game


def test_heuristic_game_completes_clean():
    summary = play_heuristic_game(
        seed=7, config=SimulationConfig(), max_events=1500
    )
    assert summary["events"] > 10
    assert summary["events"] < 1500
    assert summary["violations"] == [], summary["violations"][:3]
    assert summary["sivr"] == 0.0
    total = summary["score"]["A"] + summary["score"]["B"]
    assert total > 0


def test_heuristic_games_deterministic():
    first = play_heuristic_game(seed=21, config=SimulationConfig(), max_events=400)
    second = play_heuristic_game(seed=21, config=SimulationConfig(), max_events=400)
    assert first["score"] == second["score"]
    assert first["events"] == second["events"]
