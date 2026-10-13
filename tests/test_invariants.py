"""Tests for the 20 invariant checks (proposal Table 3)."""

import pytest

from bball3x3.config import SimulationConfig
from bball3x3.engine.court import Court
from bball3x3.engine.invariants import (
    CONSTRAINT_NAMES,
    InvariantViolationError,
    check,
    collect_violations,
)
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.state import create_initial_state


def _fresh():
    config = SimulationConfig()
    court = Court(config.court)
    state = create_initial_state(config, GameRNG.from_seed(1), first_possession="A")
    return config, court, state


def test_all_twenty_constraints_named():
    assert len(CONSTRAINT_NAMES) == 20


def test_initial_state_clean():
    config, court, state = _fresh()
    prev = state.model_copy(deep=True)
    assert collect_violations(prev, state, {}, court, config) == []


def test_duplicate_possession_caught():
    config, court, state = _fresh()
    prev = state.model_copy(deep=True)
    state.players["A2"].has_ball = True
    violations = collect_violations(prev, state, {}, court, config)
    assert any(v["constraint_id"] == 1 for v in violations)


def test_out_of_bounds_caught():
    config, court, state = _fresh()
    prev = state.model_copy(deep=True)
    state.players["A1"].pos = (99, 99)
    violations = collect_violations(prev, state, {}, court, config)
    assert any(v["constraint_id"] == 3 for v in violations)


def test_movement_limit_caught():
    config, court, state = _fresh()
    prev = state.model_copy(deep=True)
    state.players["A2"].pos = (14, 10)
    violations = collect_violations(
        prev, state, {"actions": {}, "elapsed_s": 1.0}, court, config
    )
    assert any(v["constraint_id"] == 9 for v in violations)


def test_score_without_event_caught():
    config, court, state = _fresh()
    prev = state.model_copy(deep=True)
    state.score["A"] += 2
    violations = collect_violations(prev, state, {"actions": {}}, court, config)
    assert any(v["constraint_id"] == 14 for v in violations)


def test_check_raises():
    config, court, state = _fresh()
    prev = state.model_copy(deep=True)
    state.players["A2"].has_ball = True
    with pytest.raises(InvariantViolationError):
        check(prev, state, {}, court, config)


def test_fatigue_bounds_caught():
    config, court, state = _fresh()
    prev = state.model_copy(deep=True)
    state.players["B1"].fatigue = 11.0
    violations = collect_violations(prev, state, {}, court, config)
    assert any(v["constraint_id"] == 18 for v in violations)
