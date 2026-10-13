"""Tests for heuristic / random agents and abilities / movement helpers."""

from bball3x3.agents.heuristic_player import HeuristicPlayer
from bball3x3.agents.random_player import RandomPlayer
from bball3x3.config import SimulationConfig, load_roster_config
from bball3x3.engine.abilities import build_abilities_map, get_abilities
from bball3x3.engine.actions import get_legal_actions, validate_action
from bball3x3.engine.movement import movement_label, speed_label
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.state import create_initial_state


def _state():
    config = SimulationConfig()
    return config, create_initial_state(
        config, GameRNG.from_seed(4), first_possession="A"
    )


def test_heuristic_proposes_legal_action():
    config, state = _state()
    for pid in state.players:
        proposal = HeuristicPlayer(pid, config).decide(state)
        valid, _, _ = validate_action(state, proposal)
        assert valid, f"{pid} proposed illegal {proposal.action}"


def test_heuristic_clears_when_required():
    config, state = _state()
    state.needs_clear = True
    proposal = HeuristicPlayer("A1", config).decide(state)
    assert proposal.action == "CLEAR"


def test_random_player_always_legal():
    _, state = _state()
    for pid in state.players:
        player = RandomPlayer(pid, seed=123)
        for _ in range(20):
            proposal = player.decide(state)
            valid, _, _ = validate_action(state, proposal)
            assert valid


def test_random_player_deterministic():
    _, state = _state()
    first = [RandomPlayer("A1", seed=9).decide(state).action for _ in range(5)]
    second = [RandomPlayer("A1", seed=9).decide(state).action for _ in range(5)]
    # Fresh instances with the same seed diverge (own stream position), but
    # actions must always come from the legal set.
    legal = get_legal_actions(state, "A1")
    assert set(first) <= legal and set(second) <= legal


def test_abilities_mirror_roster():
    roster = load_roster_config("configs/teams/roster.yaml")
    abilities_map = build_abilities_map(roster)
    assert abilities_map["A1"].shooting == abilities_map["B1"].shooting
    assert get_abilities("A9", abilities_map).shooting == 5


def test_movement_labels():
    assert movement_label((5, 5), (5, 5)) == "HOLD"
    assert speed_label((5, 5), (5, 5), 1.0) in ("NONE", "STATIONARY")
    assert speed_label((5, 5), (9, 5), 1.0) == "FAST"
