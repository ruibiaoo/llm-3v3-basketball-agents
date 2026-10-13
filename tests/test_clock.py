
"""Unit tests for simulation clock mechanics."""

import pytest

from bball3x3.config import SimulationConfig
from bball3x3.engine.clock import (
    advance_action_clock,
    advance_clocks,
    get_action_duration,
    is_regulation_expired,
    is_shot_clock_violation,
    reset_shot_clock,
)
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.state import create_initial_state


@pytest.fixture
def config():
    return SimulationConfig()


@pytest.fixture
def state(config):
    return create_initial_state(
        config,
        GameRNG.from_seed(42),
        first_possession="A",
    )


def test_action_durations(config):
    assert get_action_duration("PASS", config) == 1
    assert get_action_duration("SHOOT", config) == 1
    assert get_action_duration("DRIVE", config) == 2
    assert get_action_duration("CLEAR", config) == 2


def test_action_duration_case_insensitive(config):
    assert get_action_duration("pass", config) == 1


def test_configured_action_duration():
    config = SimulationConfig(
        time_costs_s={"PASS": 3}
    )
    assert get_action_duration("PASS", config) == 3


def test_unknown_action(config):
    with pytest.raises(ValueError):
        get_action_duration("UNKNOWN", config)


def test_invalid_action_duration():
    config = SimulationConfig(
        time_costs_s={"PASS": 0}
    )
    with pytest.raises(ValueError):
        get_action_duration("PASS", config)


def test_advance_clocks(state):
    result = advance_clocks(state, 2)

    assert result.elapsed_s == 2
    assert state.game_clock_s == 598
    assert state.shot_clock_s == 10
    assert not result.game_expired
    assert not result.shot_clock_expired


def test_advance_action_clock(state, config):
    result = advance_action_clock(
        state, "DRIVE", config
    )

    assert result.elapsed_s == 2
    assert state.game_clock_s == 598
    assert state.shot_clock_s == 10


def test_shot_clock_expiration(state):
    state.shot_clock_s = 1

    result = advance_clocks(state, 2)

    assert result.elapsed_s == 1
    assert state.shot_clock_s == 0
    assert state.game_clock_s == 599
    assert result.shot_clock_expired
    assert is_shot_clock_violation(state)


def test_game_clock_expiration(state):
    state.game_clock_s = 1

    result = advance_clocks(state, 2)

    assert result.elapsed_s == 1
    assert state.game_clock_s == 0
    assert state.shot_clock_s == 11
    assert result.game_expired
    assert is_regulation_expired(state)


def test_both_clocks_expire(state):
    state.game_clock_s = 1
    state.shot_clock_s = 1

    result = advance_clocks(state, 2)

    assert result.game_expired
    assert result.shot_clock_expired
    assert state.game_clock_s == 0
    assert state.shot_clock_s == 0


def test_clocks_never_negative(state):
    advance_clocks(state, 1000)

    assert state.game_clock_s >= 0
    assert state.shot_clock_s >= 0


def test_negative_elapsed_time(state):
    with pytest.raises(ValueError):
        advance_clocks(state, -1)


def test_dead_ball_clock_stopped(state):
    state.phase = "DEAD"

    result = advance_clocks(state, 5)

    assert result.elapsed_s == 0
    assert state.game_clock_s == 600
    assert state.shot_clock_s == 12


def test_free_throw_clock_stopped(state):
    state.phase = "FT"

    result = advance_clocks(state, 5)

    assert result.elapsed_s == 0
    assert state.game_clock_s == 600
    assert state.shot_clock_s == 12


def test_reset_shot_clock(state, config):
    state.shot_clock_s = 3

    reset_shot_clock(state, config)

    assert state.shot_clock_s == 12


def test_overtime_clock(state):
    state.phase = "OT"
    state.game_clock_s = 0

    result = advance_clocks(state, 2)

    assert result.elapsed_s == 2
    assert state.game_clock_s == 0
    assert state.shot_clock_s == 10
    assert not result.game_expired


def test_regulation_not_expired(state):
    assert not is_regulation_expired(state)


def test_no_initial_shot_clock_violation(state):
    assert not is_shot_clock_violation(state)
