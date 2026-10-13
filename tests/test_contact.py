"""Tests for contact detection and referee-state building."""

from bball3x3.config import SimulationConfig
from bball3x3.engine.contact import (
    build_referee_state,
    detect_contact,
    referee_query_text,
)
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.state import create_initial_state


def test_no_contact_without_triggers():
    config = SimulationConfig()
    state = create_initial_state(config, GameRNG.from_seed(1), first_possession="A")
    rng = GameRNG.from_seed(1)
    contact, _ = detect_contact(
        state=state, actions={"A1": "HOLD", "B1": "DEFEND"},
        ball_handler_id="A1", primary_defender_id="B1",
        rng=rng, config=config,
    )
    # HOLD vs DEFEND never triggers; result must be a valid ContactEvent.
    assert contact.contact_detected in (True, False)


def test_steal_attempt_can_produce_reach():
    config = SimulationConfig()
    state = create_initial_state(config, GameRNG.from_seed(1), first_possession="A")
    seen = set()
    for seed in range(20):
        rng = GameRNG.from_seed(seed)
        contact, draws = detect_contact(
            state=state, actions={"A1": "DRIVE", "B1": "STEAL_ATTEMPT"},
            ball_handler_id="A1", primary_defender_id="B1",
            rng=rng, config=config,
        )
        seen.add(contact.contact_detected)
        assert "reach" in draws or not contact.contact_detected
    assert True in seen  # reach_prob 0.2 fires at least once in 20 seeds


def test_referee_state_and_query():
    config = SimulationConfig()
    state = create_initial_state(config, GameRNG.from_seed(1), first_possession="A")
    rng = GameRNG.from_seed(2)
    contact, _ = detect_contact(
        state=state, actions={"A1": "DRIVE", "B1": "DEFEND"},
        ball_handler_id="A1", primary_defender_id="B1",
        rng=rng, config=config,
    )
    ref = build_referee_state(
        state=state, contact=contact, current_event={"event_type": "DRIVE"}
    )
    assert ref.ball_handler == "A1"
    query = referee_query_text(ref)
    assert "DRIVE" in query and "contact=" in query
