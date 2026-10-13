"""Tests for shot quality, contest and make probability."""

import pytest

from bball3x3.config import SimulationConfig
from bball3x3.engine.abilities import default_abilities
from bball3x3.engine.court import Court
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.shot_quality import (
    compute_contest,
    compute_shotquality,
    finalize_shot_terms,
    infer_shot_type,
    location_zone,
    shot_make_probability,
)
from bball3x3.engine.state import create_initial_state


@pytest.fixture
def config():
    return SimulationConfig()


@pytest.fixture
def court(config):
    return Court(config.court)


def test_location_zones(court):
    assert location_zone((7, 1), court) == "RIM"
    assert location_zone((7, 9), court) == "ARC2"


def test_infer_shot_type_priority():
    assert infer_shot_type(shot_clock_s=1.5) == "FORCED_LATE_CLOCK"
    assert infer_shot_type(after_drive_success=True) == "LAYUP_OFF_DRIVE"
    assert infer_shot_type(after_pass_reception=True) == "CATCH_AND_SHOOT"
    assert infer_shot_type(is_post_up=True) == "POST"
    assert infer_shot_type() == "OFF_DRIBBLE"


def test_shotquality_bounded(config):
    q = compute_shotquality(
        location=1.0, shot_type_score_val=1.0, screen_advantage=1.0, config=config
    )
    assert 0.0 <= q <= 1.0


def test_contest_close_vs_far(config):
    close = compute_contest(distance_m=0.5, defender_defense=7, config=config)
    far = compute_contest(distance_m=5.0, defender_defense=7, config=config)
    assert close > far
    assert far == 0.0


def test_contest_bonuses(config):
    base = compute_contest(distance_m=1.0, defender_defense=5, config=config)
    contested = compute_contest(
        distance_m=1.0, defender_defense=5, defender_contesting=True, config=config
    )
    helped = compute_contest(
        distance_m=1.0, defender_defense=5, help_present=True, config=config
    )
    assert contested > base
    assert helped > base


def test_make_probability_monotonic(config):
    open_look = shot_make_probability(
        shooting=7, shotquality=0.8, contest=0.1, fatigue=1.0, config=config
    )
    contested = shot_make_probability(
        shooting=7, shotquality=0.8, contest=0.9, fatigue=1.0, config=config
    )
    tired = shot_make_probability(
        shooting=7, shotquality=0.8, contest=0.1, fatigue=9.0, config=config
    )
    assert open_look > contested
    assert open_look > tired
    assert 0.0 <= contested <= 1.0


def test_finalize_terms_logs_all_fields(config, court):
    state = create_initial_state(config, GameRNG.from_seed(1), first_possession="A")
    shooter = default_abilities(7)
    from bball3x3.engine.shot_quality import describe_shot

    terms = describe_shot(
        state=state,
        shooter_id="A1",
        shooter=shooter,
        court=court,
        config=config,
    )
    final = finalize_shot_terms(terms, defender_abilities={}, config=config)
    for key in ("shooting", "shotquality", "contest", "fatigue", "p_make"):
        assert key in final
