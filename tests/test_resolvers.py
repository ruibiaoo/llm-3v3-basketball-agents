"""Tests for pass / drive / screen / rebound / steal / FT resolvers."""

from bball3x3.config import SimulationConfig
from bball3x3.engine.abilities import default_abilities
from bball3x3.engine.court import Court
from bball3x3.engine.resolvers import (
    resolve_drive,
    resolve_free_throw,
    resolve_pass,
    resolve_rebound,
    resolve_screen,
    resolve_shot,
    resolve_steal,
    update_fatigue,
)
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.state import create_initial_state


def _abilities(value=7):
    return {pid: default_abilities(value) for pid in ("A1", "A2", "A3", "B1", "B2", "B3")}


def test_resolve_shot_deterministic():
    config, court = SimulationConfig(), Court(SimulationConfig().court)
    state = create_initial_state(config, GameRNG.from_seed(3), first_possession="A")
    rng1, rng2 = GameRNG.from_seed(9), GameRNG.from_seed(9)
    o1, t1, _ = resolve_shot(
        state=state, shooter_id="A1", abilities_map=_abilities(),
        court=court, config=config, rng=rng1,
    )
    o2, t2, _ = resolve_shot(
        state=state, shooter_id="A1", abilities_map=_abilities(),
        court=court, config=config, rng=rng2,
    )
    assert o1 == o2
    assert t1["p_make"] == t2["p_make"]
    assert 0.0 <= t1["p_make"] <= 1.0


def test_resolve_pass_steal_vs_oob():
    config, court = SimulationConfig(), Court(SimulationConfig().court)
    state = create_initial_state(config, GameRNG.from_seed(3), first_possession="A")
    rng = GameRNG.from_seed(11)
    outcome, terms, _ = resolve_pass(
        state=state, passer_id="A1", target_id="A2", abilities_map=_abilities(),
        court=court, config=config, rng=rng,
    )
    assert 0.0 <= terms["p_complete"] <= 1.0
    if not outcome["completed"]:
        assert (outcome["steal_by"] is not None) != outcome["out_of_bounds"]


def test_resolve_drive_terms():
    config = SimulationConfig()
    rng = GameRNG.from_seed(5)
    outcome, terms, _ = resolve_drive(
        ball_handler_id="A1", primary_defender_id="B1", help_present=False,
        abilities_map=_abilities(), config=config, rng=rng,
    )
    assert 0.0 <= terms["p_success"] <= 1.0
    assert set(outcome) >= {"success", "stripped", "stopped"}


def test_resolve_screen_flags_advantage():
    rng = GameRNG.from_seed(5)
    outcome, _, _ = resolve_screen(screener_id="A2", rng=rng)
    assert outcome["screen_advantage"] == outcome["success"]


def test_resolve_rebound_picks_holder():
    config, court = SimulationConfig(), Court(SimulationConfig().court)
    state = create_initial_state(config, GameRNG.from_seed(3), first_possession="A")
    from bball3x3.engine.rules import apply_missed_shot

    apply_missed_shot(state)
    rng = GameRNG.from_seed(7)
    outcome, summary, _ = resolve_rebound(
        state=state, abilities_map=_abilities(), court=court, config=config,
        rng=rng, actions={pid: "CRASH_BOARDS" for pid in state.players},
        offense_team="A",
    )
    assert outcome["winner"] in state.players
    assert abs(sum(summary["weights"].values()) - 1.0) < 1e-6


def test_resolve_steal_and_ft():
    config = SimulationConfig()
    rng = GameRNG.from_seed(5)
    outcome, terms, _ = resolve_steal(
        defender_id="B1", ball_handler_id="A1",
        abilities_map=_abilities(), rng=rng,
    )
    assert 0.0 <= terms["p_success"] <= 1.0
    assert outcome["by"] in (None, "B1")
    outcome_ft, terms_ft, _ = resolve_free_throw(
        shooter_id="A1", abilities_map=_abilities(), config=config, rng=rng
    )
    assert 0.0 <= terms_ft["p_make"] <= 1.0
    assert outcome_ft["shooter"] == "A1"


def test_fatigue_accumulates_and_recovers():
    config = SimulationConfig()
    state = create_initial_state(config, GameRNG.from_seed(3), first_possession="A")
    before = state.players["A1"].fatigue
    update_fatigue(
        state=state, abilities_map=_abilities(), config=config, elapsed_s=2.0
    )
    assert state.players["A1"].fatigue > before
    update_fatigue(
        state=state, abilities_map=_abilities(), config=config,
        elapsed_s=0.0, dead_ball=True,
    )
    assert state.players["A1"].fatigue >= 0.0
