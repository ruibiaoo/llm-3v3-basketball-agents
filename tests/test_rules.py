"""Tests for FIBA 3x3 rules: scoring, fouls, clearing, OT, violations."""

from bball3x3.config import SimulationConfig
from bball3x3.engine.court import Court
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.rules import (
    apply_made_shot,
    apply_missed_shot,
    apply_rebound_winner,
    apply_referee_decision,
    apply_shot_clock_violation,
    apply_turnover,
    foul_penalty,
    is_game_over,
    is_overtime_over,
    is_tied_at_expiry,
    start_overtime,
)
from bball3x3.engine.state import create_initial_state
from bball3x3.schemas.referee import RefereeDecision


def _state(seed=1, possession="A"):
    config = SimulationConfig()
    return config, create_initial_state(
        config, GameRNG.from_seed(seed), first_possession=possession
    )


def test_made_shot_scores_and_clears():
    config, state = _state()
    apply_made_shot(state, "A", 2, config)
    assert state.score["A"] == 2
    assert state.offense == "B"
    assert state.needs_clear is True


def test_miss_leaves_loose_then_rebound():
    config, state = _state()
    apply_missed_shot(state)
    assert state.ball.status == "LOOSE"
    apply_rebound_winner(state, "B2", config, offensive_rebound=False)
    assert state.ball.holder == "B2"
    assert state.needs_clear is True


def test_offensive_rebound_resets_clock():
    config, state = _state()
    state.shot_clock_s = 2.0
    apply_missed_shot(state)
    apply_rebound_winner(state, "A2", config, offensive_rebound=True)
    assert state.shot_clock_s == config.rules.shot_clock_s


def test_shot_clock_violation_turns_over():
    config, state = _state()
    apply_shot_clock_violation(state, config)
    assert state.offense == "B"
    assert state.needs_clear is True


def test_foul_bonus_ladder():
    config = SimulationConfig()
    assert foul_penalty(
        foul_count_after=3, was_shooting=False, shot_value=1,
        made_shot=False, config=config,
    )["free_throws"] == 0
    assert foul_penalty(
        foul_count_after=7, was_shooting=False, shot_value=1,
        made_shot=False, config=config,
    )["free_throws"] == 2
    bonus = foul_penalty(
        foul_count_after=10, was_shooting=False, shot_value=1,
        made_shot=False, config=config,
    )
    assert bonus["free_throws"] == 2 and bonus["retain_possession"] is True
    assert foul_penalty(
        foul_count_after=2, was_shooting=True, shot_value=2,
        made_shot=False, config=config,
    )["free_throws"] == 2
    assert foul_penalty(
        foul_count_after=2, was_shooting=False, shot_value=1,
        made_shot=True, config=config,
    )["reason"] == "AND_ONE"


def test_game_over_and_overtime():
    config, state = _state()
    assert not is_game_over(state, config)
    state.score["A"] = 21
    assert is_game_over(state, config)
    config2, state2 = _state()
    state2.game_clock_s = 0.0
    state2.score = {"A": 10, "B": 10}
    assert is_tied_at_expiry(state2)
    baseline = start_overtime(state2, config2)
    assert state2.phase == "OT"
    assert not is_overtime_over(state2, config2, baseline)
    state2.score["A"] += 2
    assert is_overtime_over(state2, config2, baseline)


def test_referee_none_is_noop():
    config, state = _state()
    before = state.model_copy(deep=True)
    result = apply_referee_decision(state, RefereeDecision(call="NONE"), config)
    assert result["applied"] is False
    assert state.score == before.score


def test_turnover_switches_possession():
    config, state = _state()
    apply_turnover(state, "B", config)
    assert state.offense == "B"
    _ = Court(config.court)
