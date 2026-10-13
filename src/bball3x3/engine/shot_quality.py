"""Shot quality, contest and make-probability (WS1 spec section 5).

Implements::

    P(make) = sigmoid(beta0 + beta1*shooting + beta2*shotquality
                      - beta3*contest - beta4*fatigue)

where ``shotquality = w_loc*location + w_type*shot_type + w_scr*screen``.
Beta3/beta4 are stored positive; the minus signs apply here.
Every intermediate term is returned for transparent logging.
"""

from math import exp

from bball3x3.config import SimulationConfig
from bball3x3.engine.court import Court
from bball3x3.schemas.state import Abilities, GameState

DEFAULT_SHOT_MODEL = {
    "beta0": -1.6,
    "beta1": 0.25,
    "beta2": 2.0,
    "beta3": 1.5,
    "beta4": 0.08,
    "shotquality_weights": {"location": 0.6, "shot_type": 0.25, "screen_advantage": 0.15},
    "location_score": {"RIM": 1.0, "SHORT": 0.7, "MID": 0.45, "ARC2": 0.3, "DEEP2": 0.15},
    "shot_type_score": {
        "LAYUP_OFF_DRIVE": 0.9,
        "CATCH_AND_SHOOT": 0.8,
        "POST": 0.6,
        "OFF_DRIBBLE": 0.5,
        "FORCED_LATE_CLOCK": 0.2,
    },
}

DEFAULT_CONTEST_MODEL = {
    "max_dist_cells": 3,
    "b_defense": 0.06,
    "contest_action_bonus": 0.2,
    "help_bonus": 0.15,
}

# Distance bands (metres) mapping to location_score keys. Placeholder
# thresholds, kept explicit for sensitivity analysis.
RIM_M = 2.0
SHORT_M = 4.25


def sigmoid(x: float) -> float:
    if x >= 30.0:
        return 1.0
    if x <= -30.0:
        return 0.0
    return 1.0 / (1.0 + exp(-x))


def shot_params(config: SimulationConfig) -> dict:
    return {**DEFAULT_SHOT_MODEL, **config.shot_model}


def contest_params(config: SimulationConfig) -> dict:
    return {**DEFAULT_CONTEST_MODEL, **config.contest_model}


def location_zone(pos: tuple[int, int], court: Court) -> str:
    """Map a court position to RIM / SHORT / MID / ARC2 / DEEP2."""
    dist_m = court.distance_to_basket(pos)
    beyond_arc = court.is_beyond_arc(pos)
    if not beyond_arc:
        if dist_m <= RIM_M:
            return "RIM"
        if dist_m <= SHORT_M:
            return "SHORT"
        return "MID"
    if dist_m <= court.arc_radius_m + 1.75:
        return "ARC2"
    return "DEEP2"


def location_score(zone: str, config: SimulationConfig) -> float:
    table = shot_params(config)["location_score"]
    if zone not in table:
        raise ValueError(f"Unknown location zone: {zone}")
    return float(table[zone])


def infer_shot_type(
    *,
    after_drive_success: bool = False,
    after_pass_reception: bool = False,
    is_post_up: bool = False,
    shot_clock_s: float = 12.0,
) -> str:
    """Infer shot_type from engine context (spec section 5)."""
    if shot_clock_s <= 2.0:
        return "FORCED_LATE_CLOCK"
    if after_drive_success:
        return "LAYUP_OFF_DRIVE"
    if after_pass_reception:
        return "CATCH_AND_SHOOT"
    if is_post_up:
        return "POST"
    return "OFF_DRIBBLE"


def shot_type_score(shot_type: str, config: SimulationConfig) -> float:
    table = shot_params(config)["shot_type_score"]
    if shot_type not in table:
        raise ValueError(f"Unknown shot type: {shot_type}")
    return float(table[shot_type])


def compute_shotquality(
    *,
    location: float,
    shot_type_score_val: float,
    screen_advantage: float,
    config: SimulationConfig,
) -> float:
    weights = shot_params(config)["shotquality_weights"]
    quality = (
        float(weights["location"]) * location
        + float(weights["shot_type"]) * shot_type_score_val
        + float(weights["screen_advantage"]) * screen_advantage
    )
    return max(0.0, min(1.0, quality))


def nearest_defender(
    state: GameState,
    shooter_id: str,
    court: Court,
) -> tuple[str | None, float]:
    """Return (defender_id, distance_m) of the closest live defender."""
    shooter = state.players[shooter_id]
    best_id: str | None = None
    best_dist = float("inf")
    for pid, player in state.players.items():
        if player.team == shooter.team:
            continue
        dist_m = court.distance(shooter.pos, player.pos)
        if dist_m < best_dist:
            best_dist = dist_m
            best_id = pid
    return best_id, best_dist


def compute_contest(
    *,
    distance_m: float,
    defender_defense: int = 5,
    defender_contesting: bool = False,
    help_present: bool = False,
    config: SimulationConfig,
) -> float:
    """Contest in [0, 1] from nearest-defender distance and bonuses."""
    params = contest_params(config)
    max_dist_cells = float(params["max_dist_cells"])
    # Court cells are 1 m, so metres and cells coincide at default config.
    base = max(0.0, 1.0 - distance_m / max_dist_cells)
    scaled = base * (1.0 + float(params["b_defense"]) * (defender_defense - 5))
    if defender_contesting:
        scaled += float(params["contest_action_bonus"])
    if help_present:
        scaled += float(params["help_bonus"])
    return max(0.0, min(1.0, scaled))


def shot_make_probability(
    *,
    shooting: int,
    shotquality: float,
    contest: float,
    fatigue: float,
    config: SimulationConfig,
) -> float:
    params = shot_params(config)
    logit = (
        float(params["beta0"])
        + float(params["beta1"]) * shooting
        + float(params["beta2"]) * shotquality
        - float(params["beta3"]) * contest
        - float(params["beta4"]) * fatigue
    )
    return sigmoid(logit)


def describe_shot(
    *,
    state: GameState,
    shooter_id: str,
    shooter: Abilities,
    court: Court,
    config: SimulationConfig,
    defender_actions: dict[str, str] | None = None,
    help_present: bool = False,
    after_drive_success: bool = False,
    after_pass_reception: bool = False,
    is_post_up: bool = False,
    screen_advantage: bool = False,
) -> dict:
    """Compute and return every shot-model term for logging."""
    defender_actions = defender_actions or {}
    shooter_state = state.players[shooter_id]
    zone = location_zone(shooter_state.pos, court)
    loc_score = location_score(zone, config)
    shot_type = infer_shot_type(
        after_drive_success=after_drive_success,
        after_pass_reception=after_pass_reception,
        is_post_up=is_post_up,
        shot_clock_s=state.shot_clock_s,
    )
    type_score = shot_type_score(shot_type, config)
    quality = compute_shotquality(
        location=loc_score,
        shot_type_score_val=type_score,
        screen_advantage=1.0 if screen_advantage else 0.0,
        config=config,
    )
    defender_id, defender_dist = nearest_defender(state, shooter_id, court)
    defender_defense = 5
    return {
        "zone": zone,
        "location_score": loc_score,
        "shot_type": shot_type,
        "shot_type_score": type_score,
        "screen_advantage": 1.0 if screen_advantage else 0.0,
        "shotquality": quality,
        "defender_id": defender_id,
        "defender_dist_m": defender_dist,
        "defender_defense": defender_defense,
        "defender_contesting": any(
            a.upper() == "CONTEST" for a in defender_actions.values()
        ),
        "help_present": help_present,
        "shooting": shooter.shooting,
        "fatigue": shooter_state.fatigue,
    }


def finalize_shot_terms(
    terms: dict,
    *,
    defender_abilities: dict[str, Abilities] | None = None,
    config: SimulationConfig,
) -> dict:
    """Fill contest and p_make once defender ratings are known."""
    defense = 5
    defender_id = terms.get("defender_id")
    if defender_id and defender_abilities and defender_id in defender_abilities:
        defense = defender_abilities[defender_id].defense
    contest = compute_contest(
        distance_m=terms["defender_dist_m"],
        defender_defense=defense,
        defender_contesting=terms["defender_contesting"],
        help_present=terms["help_present"],
        config=config,
    )
    p_make = shot_make_probability(
        shooting=int(terms["shooting"]),
        shotquality=float(terms["shotquality"]),
        contest=contest,
        fatigue=float(terms["fatigue"]),
        config=config,
    )
    return {**terms, "defender_defense": defense, "contest": contest, "p_make": p_make}
