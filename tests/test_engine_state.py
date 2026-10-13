
"""Tests for game initialization and basic state transitions."""

import pytest

from bball3x3.config import SimulationConfig
from bball3x3.engine.court import Court
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.state import (
    create_initial_state,
    move_player,
    set_ball_holder,
    switch_possession,
)


@pytest.fixture
def config():
    return SimulationConfig()


@pytest.fixture
def rng():
    return GameRNG.from_seed(42)


@pytest.fixture
def state(config, rng):
    return create_initial_state(
        config,
        rng,
        first_possession="A",
    )


def test_initial_player_count(state):
    assert len(state.players) == 6
    assert sum(p.team == "A" for p in state.players.values()) == 3
    assert sum(p.team == "B" for p in state.players.values()) == 3


def test_initial_score(state):
    assert state.score == {"A": 0, "B": 0}
    assert state.team_fouls == {"A": 0, "B": 0}


def test_initial_clocks(state):
    assert state.game_clock_s == 600.0
    assert state.shot_clock_s == 12.0


def test_initial_possession(state):
    assert state.offense == "A"
    assert state.ball.holder == "A1"
    assert state.ball.status == "HELD"
    assert state.players["A1"].has_ball


def test_only_one_ball_holder(state):
    holders = [
        p.id for p in state.players.values() if p.has_ball
    ]
    assert holders == ["A1"]


def test_initial_positions_in_bounds(state, config):
    court = Court(config.court)

    for player in state.players.values():
        assert court.is_in_bounds(player.pos)


def test_seeded_first_possession(config):
    rng1 = GameRNG.from_seed(42)
    rng2 = GameRNG.from_seed(42)

    state1 = create_initial_state(config, rng1)
    state2 = create_initial_state(config, rng2)

    assert state1.offense == state2.offense


def test_invalid_first_possession(config, rng):
    with pytest.raises(ValueError):
        create_initial_state(
            config, rng, first_possession="C"
        )


def test_set_ball_holder(state):
    set_ball_holder(state, "A2")

    assert state.ball.holder == "A2"
    assert state.players["A2"].has_ball
    assert not state.players["A1"].has_ball
    assert state.ball.pos == state.players["A2"].pos


def test_invalid_ball_holder(state):
    with pytest.raises(ValueError):
        set_ball_holder(state, "A4")


def test_switch_possession(state, config):
    switch_possession(state, "B", config)

    assert state.offense == "B"
    assert state.shot_clock_s == 12.0
    assert state.needs_clear is True
    assert state.ball.holder is None
    assert state.ball.status == "LOOSE"
    assert not any(p.has_ball for p in state.players.values())


def test_invalid_switch_possession(state, config):
    with pytest.raises(ValueError):
        switch_possession(state, "C", config)

    with pytest.raises(ValueError):
        switch_possession(state, "A", config)


def test_move_player(state, config):
    court = Court(config.court)

    old_pos = state.players["A2"].pos
    move_player(state, "A2", (4, 7), court)

    assert state.players["A2"].prev_pos == old_pos
    assert state.players["A2"].pos == (4, 7)


def test_move_ball_holder(state, config):
    court = Court(config.court)

    move_player(state, "A1", (7, 8), court)

    assert state.players["A1"].pos == (7, 8)
    assert state.ball.pos == (7, 8)


def test_invalid_move(state, config):
    court = Court(config.court)

    with pytest.raises(ValueError):
        move_player(state, "A1", (20, 20), court)

    with pytest.raises(ValueError):
        move_player(state, "A4", (5, 5), court)
