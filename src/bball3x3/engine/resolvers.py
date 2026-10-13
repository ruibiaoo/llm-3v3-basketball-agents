"""Action resolvers (WS1 spec section 6).

Every resolver uses explicit coefficients from ``configs/simulation.yaml``,
seeded :class:`GameRNG` streams, and an abilities lookup (WS0 schemas stay
frozen). Each resolver returns ``(outcome, terms, rng_draws)`` so the WS6
event log can record every model term.
"""

from math import exp

from bball3x3.config import SimulationConfig
from bball3x3.engine.abilities import get_abilities
from bball3x3.engine.court import Court
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.shot_quality import (
    describe_shot,
    finalize_shot_terms,
)
from bball3x3.schemas.state import Abilities, GameState

DEFAULT_PASS_MODEL = {
    "base": 2.5,
    "b_passing": 0.2,
    "b_distance": -0.15,
    "b_lane_defender": -1.5,
    "b_def_intercept": 0.1,
    "lane_width_cells": 1.0,
}

DEFAULT_DRIVE_MODEL = {
    "base": 0.0,
    "b_ball_handling": 0.3,
    "b_defense": -0.3,
    "b_help": -0.8,
}

DEFAULT_REBOUND = {
    "off_base_prob": 0.28,
    "b_rebounding": 0.15,
    "b_boxout": -0.6,
    "b_proximity": 0.4,
}

DEFAULT_FATIGUE = {
    "per_s_active": 0.05,
    "per_s_sprint": 0.12,
    "stamina_scale": 0.08,
    "recover_dead_ball": 0.3,
    "max": 10.0,
}

SCREEN_SUCCESS_P = 0.65
STEAL_BASE = -0.5
STEAL_B_DEFENSE = 0.35
STEAL_B_HANDLING = -0.35


def _sigmoid(x: float) -> float:
    if x >= 30.0:
        return 1.0
    if x <= -30.0:
        return 0.0
    return 1.0 / (1.0 + exp(-x))


def pass_params(config: SimulationConfig) -> dict:
    return {**DEFAULT_PASS_MODEL, **config.pass_model}


def drive_params(config: SimulationConfig) -> dict:
    return {**DEFAULT_DRIVE_MODEL, **config.drive_model}


def rebound_params(config: SimulationConfig) -> dict:
    return {**DEFAULT_REBOUND, **config.rebound}


def fatigue_params(config: SimulationConfig) -> dict:
    return {**DEFAULT_FATIGUE, **config.fatigue}


def _point_to_segment_dist(
    p: tuple[int, int], a: tuple[int, int], b: tuple[int, int]
) -> float:
    """Distance from point p to segment a-b in cells."""
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def lane_defenders(
    state: GameState,
    passer_id: str,
    target_id: str,
    lane_width_cells: float = 1.0,
) -> list[str]:
    """Defenders within the passing lane (point-to-segment distance)."""
    passer = state.players[passer_id]
    target = state.players[target_id]
    defenders = []
    for pid, player in state.players.items():
        if player.team == passer.team:
            continue
        if _point_to_segment_dist(player.pos, passer.pos, target.pos) <= lane_width_cells:
            defenders.append(pid)
    return defenders


def resolve_shot(
    *,
    state: GameState,
    shooter_id: str,
    abilities_map: dict[str, Abilities] | None,
    court: Court,
    config: SimulationConfig,
    rng: GameRNG,
    defender_actions: dict[str, str] | None = None,
    help_present: bool = False,
    after_drive_success: bool = False,
    after_pass_reception: bool = False,
    is_post_up: bool = False,
    screen_advantage: bool = False,
) -> tuple[dict, dict, dict]:
    """Resolve SHOOT. Returns (outcome, terms, rng_draws)."""
    shooter = get_abilities(shooter_id, abilities_map)
    terms = describe_shot(
        state=state,
        shooter_id=shooter_id,
        shooter=shooter,
        court=court,
        config=config,
        defender_actions=defender_actions,
        help_present=help_present,
        after_drive_success=after_drive_success,
        after_pass_reception=after_pass_reception,
        is_post_up=is_post_up,
        screen_advantage=screen_advantage,
    )
    defender_abilities = {
        pid: get_abilities(pid, abilities_map) for pid in state.players
    }
    terms = finalize_shot_terms(
        terms, defender_abilities=defender_abilities, config=config
    )
    draw = rng.random("shooting")
    made = draw < terms["p_make"]
    points = court.shot_value(state.players[shooter_id].pos) if made else 0
    outcome = {"made": made, "points": points, "shooter": shooter_id}
    return outcome, terms, {"shooting": draw}


def resolve_pass(
    *,
    state: GameState,
    passer_id: str,
    target_id: str,
    abilities_map: dict[str, Abilities] | None,
    court: Court,
    config: SimulationConfig,
    rng: GameRNG,
) -> tuple[dict, dict, dict]:
    """Resolve PASS. Failure is a steal (defender in lane) or OOB turnover."""
    params = pass_params(config)
    passer = get_abilities(passer_id, abilities_map)
    distance_m = court.distance(
        state.players[passer_id].pos, state.players[target_id].pos
    )
    in_lane = lane_defenders(
        state, passer_id, target_id,
        lane_width_cells=float(params.get("lane_width_cells", 1.0)),
    )
    lane_count = len(in_lane)
    avg_lane_def = 5.0
    if in_lane:
        avg_lane_def = sum(
            get_abilities(pid, abilities_map).defense for pid in in_lane
        ) / len(in_lane)
    logit = (
        float(params["base"])
        + float(params["b_passing"]) * passer.passing
        + float(params["b_distance"]) * distance_m
        + float(params["b_lane_defender"]) * lane_count
        + float(params["b_def_intercept"]) * (avg_lane_def - 5.0)
    )
    p_complete = _sigmoid(logit)
    draw = rng.random("passing")
    completed = draw < p_complete
    terms = {
        "passing": passer.passing,
        "distance_m": distance_m,
        "lane_defenders": in_lane,
        "lane_count": lane_count,
        "avg_lane_defense": avg_lane_def,
        "p_complete": p_complete,
    }
    if completed:
        outcome = {"completed": True, "receiver": target_id}
    elif in_lane:
        outcome = {
            "completed": False,
            "receiver": None,
            "steal_by": in_lane[0],
            "out_of_bounds": False,
        }
    else:
        outcome = {
            "completed": False,
            "receiver": None,
            "steal_by": None,
            "out_of_bounds": True,
        }
    return outcome, terms, {"passing": draw}


def resolve_drive(
    *,
    ball_handler_id: str,
    primary_defender_id: str | None,
    help_present: bool,
    abilities_map: dict[str, Abilities] | None,
    config: SimulationConfig,
    rng: GameRNG,
) -> tuple[dict, dict, dict]:
    """Resolve DRIVE. Success advances toward the rim (LAYUP_OFF_DRIVE)."""
    params = drive_params(config)
    handler = get_abilities(ball_handler_id, abilities_map)
    defender_def = 5
    if primary_defender_id:
        defender_def = get_abilities(primary_defender_id, abilities_map).defense
    logit = (
        float(params["base"])
        + float(params["b_ball_handling"]) * handler.ball_handling
        + float(params["b_defense"]) * defender_def
        + float(params["b_help"]) * (1.0 if help_present else 0.0)
    )
    p_success = _sigmoid(logit)
    draw = rng.random("driving")
    success = draw < p_success
    terms = {
        "ball_handling": handler.ball_handling,
        "defense": defender_def,
        "help_present": help_present,
        "p_success": p_success,
    }
    if success:
        outcome = {"success": True, "stripped": False, "stopped": False}
    else:
        strip_draw = rng.random("driving")
        stripped = strip_draw < 0.25
        outcome = {"success": False, "stripped": stripped, "stopped": not stripped}
        draws = {"driving": draw, "driving_strip": strip_draw}
        return outcome, terms, draws
    return outcome, terms, {"driving": draw}


def resolve_screen(
    *,
    screener_id: str,
    rng: GameRNG,
) -> tuple[dict, dict, dict]:
    """Resolve SCREEN. Success flags screen_advantage for the next event."""
    draw = rng.random("driving")
    success = draw < SCREEN_SUCCESS_P
    outcome = {"success": success, "screen_advantage": success, "screener": screener_id}
    return outcome, {"p_success": SCREEN_SUCCESS_P}, {"screen": draw}


def resolve_rebound(
    *,
    state: GameState,
    abilities_map: dict[str, Abilities] | None,
    court: Court,
    config: SimulationConfig,
    rng: GameRNG,
    actions: dict[str, str],
    offense_team: str,
) -> tuple[dict, dict, dict]:
    """Weighted rebound lottery over proximity, rebounding and box-outs."""
    params = rebound_params(config)
    upper = {pid: act.upper() for pid, act in actions.items()}
    weights: dict[str, float] = {}
    terms: dict[str, dict] = {}
    for pid, player in state.players.items():
        abilities = get_abilities(pid, abilities_map)
        dist_m = court.distance(player.pos, state.ball.pos)
        proximity = max(0.0, 1.0 - dist_m / 6.0)
        w = (
            float(params["b_rebounding"]) * abilities.rebounding
            + float(params["b_proximity"]) * proximity * 5.0
        )
        act = upper.get(pid, "HOLD")
        if act == "BOX_OUT":
            w += 1.0
            w += float(params["b_boxout"]) * -1.0 if player.team != offense_team else 0.0
        if act == "CRASH_BOARDS":
            w += 0.8
        # Stamina edge: fresher legs rebound slightly better.
        w += (5.0 - player.fatigue) * 0.05
        weights[pid] = exp(w / 5.0)
        terms[pid] = {
            "rebounding": abilities.rebounding,
            "dist_m": dist_m,
            "proximity": proximity,
            "action": act,
            "weight": weights[pid],
        }
    total = sum(weights.values())
    draw = rng.random("rebounding")
    cumulative = 0.0
    winner: str = next(iter(weights))
    for pid, w in weights.items():
        cumulative += w / total
        if draw <= cumulative:
            winner = pid
            break
    outcome = {
        "winner": winner,
        "team": state.players[winner].team,
        "is_offensive": state.players[winner].team == offense_team,
    }
    summary = {
        "weights": {pid: round(w / total, 4) for pid, w in weights.items()},
        "per_player": terms,
        "off_base_prob": float(params["off_base_prob"]),
    }
    return outcome, summary, {"rebounding": draw}


def resolve_steal(
    *,
    defender_id: str,
    ball_handler_id: str,
    abilities_map: dict[str, Abilities] | None,
    rng: GameRNG,
) -> tuple[dict, dict, dict]:
    """Resolve STEAL_ATTEMPT (defense vs ball_handling)."""
    defender = get_abilities(defender_id, abilities_map)
    handler = get_abilities(ball_handler_id, abilities_map)
    logit = (
        STEAL_BASE
        + STEAL_B_DEFENSE * defender.defense
        + STEAL_B_HANDLING * handler.ball_handling
    )
    p_success = _sigmoid(logit)
    draw = rng.random("contact")
    success = draw < p_success
    outcome = {"success": success, "by": defender_id if success else None}
    terms = {
        "defense": defender.defense,
        "ball_handling": handler.ball_handling,
        "p_success": p_success,
    }
    return outcome, terms, {"steal": draw}


def resolve_free_throw(
    *,
    shooter_id: str,
    abilities_map: dict[str, Abilities] | None,
    config: SimulationConfig,
    rng: GameRNG,
    fatigue: float = 0.0,
) -> tuple[dict, dict, dict]:
    """Resolve one free throw from shooting with a fatigue penalty."""
    _ = config
    shooter = get_abilities(shooter_id, abilities_map)
    # Placeholder calibration: shooting 7 with fresh legs ~= 0.75.
    logit = -0.5 + 0.3 * shooter.shooting - 0.05 * fatigue
    # Fatigue is applied by the caller via the shooter's current value.
    p_make = _sigmoid(logit)
    draw = rng.random("free_throws")
    made = draw < p_make
    outcome = {"made": made, "shooter": shooter_id}
    return (
        outcome,
        {"shooting": shooter.shooting, "fatigue": fatigue, "p_make": p_make},
        {"free_throw": draw},
    )


def update_fatigue(
    *,
    state: GameState,
    abilities_map: dict[str, Abilities] | None,
    config: SimulationConfig,
    elapsed_s: float,
    sprint_players: set[str] | None = None,
    dead_ball: bool = False,
) -> None:
    """Apply the fatigue model in place (spec section 6)."""
    params = fatigue_params(config)
    sprint_players = sprint_players or set()
    for pid, player in state.players.items():
        abilities = get_abilities(pid, abilities_map)
        scale = 1.0 - float(params["stamina_scale"]) * (abilities.stamina - 5)
        scale = max(0.3, scale)
        increment = float(params["per_s_active"]) * elapsed_s * scale
        if pid in sprint_players:
            increment += float(params["per_s_sprint"]) * elapsed_s * scale
        if dead_ball:
            increment -= float(params["recover_dead_ball"])
        player.fatigue = max(0.0, min(float(params["max"]), player.fatigue + increment))
