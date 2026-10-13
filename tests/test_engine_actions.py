
"""Tests for action legality and invalid-action remapping."""

import pytest

from bball3x3.config import SimulationConfig
from bball3x3.engine.actions import (
    ALL_ACTIONS,
    get_legal_actions,
    remap_invalid_action,
    validate_action,
)
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.state import create_initial_state
from bball3x3.schemas.actions import ActionProposal


@pytest.fixture
def state():
    return create_initial_state(
        SimulationConfig(),
        GameRNG.from_seed(42),
        first_possession="A",
    )


def test_all_schema_actions_covered():
    assert len(ALL_ACTIONS) == 24


def test_ball_holder_actions(state):
    legal = get_legal_actions(state, "A1")

    assert "SHOOT" in legal
    assert "PASS" in legal
    assert "DRIVE" in legal
    assert "DEFEND" not in legal


def test_off_ball_actions(state):
    legal = get_legal_actions(state, "A2")

    assert "CUT" in legal
    assert "SCREEN" in legal
    assert "SHOOT" not in legal


def test_defensive_actions(state):
    legal = get_legal_actions(state, "B1")

    assert "DEFEND" in legal
    assert "CONTEST" in legal
    assert "SHOOT" not in legal


def test_unknown_player(state):
    with pytest.raises(ValueError):
        get_legal_actions(state, "A4")


def test_dead_ball_no_actions(state):
    state.phase = "DEAD"

    assert get_legal_actions(state, "A1") == set()


def test_valid_shoot(state):
    proposal = ActionProposal(
        player_id="A1",
        action="SHOOT",
    )

    valid, reason, _ = validate_action(state, proposal)

    assert valid
    assert reason is None


def test_valid_pass(state):
    proposal = ActionProposal(
        player_id="A1",
        action="PASS",
        target="A2",
    )

    valid, reason, _ = validate_action(state, proposal)

    assert valid
    assert reason is None


def test_invalid_off_ball_shoot(state):
    proposal = ActionProposal(
        player_id="A2",
        action="SHOOT",
    )

    valid, reason, _ = validate_action(state, proposal)

    assert not valid
    assert reason == "ILLEGAL_POSSESSION"


def test_invalid_defensive_shoot(state):
    proposal = ActionProposal(
        player_id="B1",
        action="SHOOT",
    )

    valid, reason, _ = validate_action(state, proposal)

    assert not valid
    assert reason == "ILLEGAL_POSSESSION"


def test_unknown_action(state):
    proposal = ActionProposal(
        player_id="A1",
        action="DUNK_FROM_MIDCOURT",
    )

    valid, reason, _ = validate_action(state, proposal)

    assert not valid
    assert reason == "UNKNOWN_ACTION"


def test_pass_requires_target(state):
    proposal = ActionProposal(
        player_id="A1",
        action="PASS",
    )

    valid, reason, _ = validate_action(state, proposal)

    assert not valid
    assert reason == "BAD_TARGET"


def test_pass_target_must_be_teammate(state):
    proposal = ActionProposal(
        player_id="A1",
        action="PASS",
        target="B1",
    )

    valid, reason, _ = validate_action(state, proposal)

    assert not valid
    assert reason == "BAD_TARGET"


def test_cannot_pass_to_self(state):
    proposal = ActionProposal(
        player_id="A1",
        action="PASS",
        target="A1",
    )

    valid, reason, _ = validate_action(state, proposal)

    assert not valid
    assert reason == "BAD_TARGET"


def test_clear_not_needed(state):
    proposal = ActionProposal(
        player_id="A1",
        action="CLEAR",
    )

    valid, reason, _ = validate_action(state, proposal)

    assert not valid
    assert reason == "INFEASIBLE"


def test_shoot_requires_clear(state):
    state.needs_clear = True

    proposal = ActionProposal(
        player_id="A1",
        action="SHOOT",
    )

    valid, reason, _ = validate_action(state, proposal)

    assert not valid
    assert reason == "ILLEGAL_POSSESSION"


def test_remap_invalid_action(state):
    proposal = ActionProposal(
        player_id="A2",
        action="SHOOT",
    )

    remapped, record = remap_invalid_action(
        state, proposal
    )

    assert remapped.action == "SPOT_UP"
    assert remapped.target is None
    assert record is not None
    assert record.original_action == "SHOOT"
    assert record.remapped_action == "SPOT_UP"
    assert record.reason == "ILLEGAL_POSSESSION"


def test_valid_action_not_remapped(state):
    proposal = ActionProposal(
        player_id="A1",
        action="SHOOT",
    )

    remapped, record = remap_invalid_action(
        state, proposal
    )

    assert remapped.action == "SHOOT"
    assert record is None


def test_lowercase_action_normalized(state):
    proposal = ActionProposal(
        player_id="A1",
        action="shoot",
    )

    remapped, record = remap_invalid_action(
        state, proposal
    )

    assert remapped.action == "SHOOT"
    assert record is None


def test_remap_preserves_memory_refs(state):
    proposal = ActionProposal(
        player_id="A2",
        action="SHOOT",
        memory_ref=["memory_1"],
    )

    remapped, record = remap_invalid_action(
        state, proposal
    )

    assert remapped.memory_ref == ["memory_1"]
    assert record is not None


def test_invalid_player_cannot_remap(state):
    proposal = ActionProposal(
        player_id="A4",
        action="SHOOT",
    )

    with pytest.raises(ValueError):
        remap_invalid_action(state, proposal)


def test_remap_dead_ball_fails(state):
    state.phase = "DEAD"

    proposal = ActionProposal(
        player_id="A1",
        action="SHOOT",
    )

    with pytest.raises(ValueError):
        remap_invalid_action(state, proposal)
