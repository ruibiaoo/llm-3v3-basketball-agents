"""FIBA 3x3 rules (WS1 spec section 7).

Objective violations live here: shot clock, clearing, check-ball, team-foul
bonus, free throws, end conditions and overtime. Judgement-dependent
contact fouls stay with the WS5 referee; this module only applies the
referee's decided call via :func:`apply_referee_decision`.
"""

from bball3x3.config import SimulationConfig
from bball3x3.engine.court import Court
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.state import (
    create_initial_state,
    set_ball_holder,
    switch_possession,
)
from bball3x3.schemas.referee import RefereeDecision
from bball3x3.schemas.state import GameState

__all__ = [
    "create_initial_state",
    "coin_flip_first_possession",
    "is_game_over",
    "is_tied_at_expiry",
    "start_overtime",
    "is_overtime_over",
    "apply_made_shot",
    "apply_missed_shot",
    "apply_turnover",
    "apply_clear",
    "apply_checkball",
    "foul_penalty",
    "apply_referee_decision",
    "apply_shot_clock_violation",
]


def coin_flip_first_possession(rng: GameRNG) -> str:
    return rng.choice("first_possession", ["A", "B"])


def other_team(team: str) -> str:
    return "B" if team == "A" else "A"


def is_game_over(state: GameState, config: SimulationConfig) -> bool:
    """Regulation end: 21+ reached, or clock expired with a winner."""
    if state.phase == "OT":
        return False  # use is_overtime_over with the OT baseline
    if state.score["A"] >= config.rules.win_score:
        return True
    if state.score["B"] >= config.rules.win_score:
        return True
    if state.game_clock_s <= 0.0:
        return state.score["A"] != state.score["B"]
    return False


def is_tied_at_expiry(state: GameState) -> bool:
    return (
        state.phase == "LIVE"
        and state.game_clock_s <= 0.0
        and state.score["A"] == state.score["B"]
    )


def start_overtime(state: GameState, config: SimulationConfig) -> dict[str, int]:
    """Enter OT. Returns the score baseline that ends OT (+2 points)."""
    state.phase = "OT"
    state.game_clock_s = 0.0
    state.shot_clock_s = config.rules.shot_clock_s
    state.needs_clear = False
    return dict(state.score)


def is_overtime_over(
    state: GameState, config: SimulationConfig, ot_baseline: dict[str, int]
) -> bool:
    target = config.rules.overtime_points_to_win
    for team in ("A", "B"):
        if state.score.get(team, 0) - ot_baseline.get(team, 0) >= target:
            return True
    return False


def apply_made_shot(
    state: GameState,
    scoring_team: str,
    points: int,
    config: SimulationConfig,
) -> None:
    """Score, then give the defence the restart under the basket (must clear)."""
    state.score[scoring_team] = state.score.get(scoring_team, 0) + points
    state.event_idx += 1
    switch_possession(state, other_team(scoring_team), config, needs_clear=True)


def apply_missed_shot(state: GameState) -> None:
    """A miss leaves the ball loose for the rebound phase."""
    state.event_idx += 1
    for player in state.players.values():
        player.has_ball = False
    state.ball.status = "LOOSE"
    state.ball.holder = None


def apply_rebound_winner(
    state: GameState,
    winner_id: str,
    config: SimulationConfig,
    offensive_rebound: bool,
) -> None:
    """Assign the rebound. Offensive boards reset the shot clock (simplified)."""
    set_ball_holder(state, winner_id)
    state.event_idx += 1
    if offensive_rebound:
        state.shot_clock_s = config.rules.shot_clock_s
    else:
        # Defensive rebound: must clear behind the arc.
        state.needs_clear = True
        state.shot_clock_s = config.rules.shot_clock_s


def apply_turnover(
    state: GameState, new_offense: str, config: SimulationConfig
) -> None:
    switch_possession(state, new_offense, config, needs_clear=True)
    state.event_idx += 1


def apply_shot_clock_violation(
    state: GameState, config: SimulationConfig
) -> None:
    """Shot-clock expiry is an objective turnover (engine-owned)."""
    apply_turnover(state, other_team(state.offense), config)


def apply_clear(
    state: GameState,
    ball_handler_id: str,
    court: Court,
    config: SimulationConfig,
) -> float:
    """Automatic clear behind the arc. Returns seconds consumed."""
    _ = court
    set_ball_holder(state, ball_handler_id)
    state.needs_clear = False
    state.event_idx += 1
    costs = config.time_costs_s
    return float(costs.get("CLEAR", 2))


def apply_checkball(state: GameState, config: SimulationConfig) -> None:
    """Check-ball at the top after dead balls; offence keeps the ball."""
    state.phase = "LIVE"
    state.needs_clear = False
    state.shot_clock_s = config.rules.shot_clock_s
    holder = next(
        (pid for pid, p in state.players.items() if p.has_ball),
        f"{state.offense}1",
    )
    set_ball_holder(state, holder)
    state.event_idx += 1


def foul_penalty(
    *,
    foul_count_after: int,
    was_shooting: bool,
    shot_value: int,
    made_shot: bool,
    config: SimulationConfig,
) -> dict:
    """Map a foul to FTs and possible possession (spec section 7)."""
    bonus = config.rules.team_foul_bonus
    two_from = int(bonus.get("two_fts_from", 7))
    plus_poss_from = int(bonus.get("two_fts_plus_possession_from", 10))
    if made_shot:
        return {"free_throws": 1, "retain_possession": False, "reason": "AND_ONE"}
    if was_shooting:
        return {
            "free_throws": 1 if shot_value <= 1 else 2,
            "retain_possession": False,
            "reason": "SHOOTING_FOUL",
        }
    if foul_count_after >= plus_poss_from:
        return {
            "free_throws": 2,
            "retain_possession": True,
            "reason": "BONUS_10_PLUS",
        }
    if foul_count_after >= two_from:
        return {"free_throws": 2, "retain_possession": False, "reason": "BONUS_7_9"}
    return {"free_throws": 0, "retain_possession": False, "reason": "COMMON_FOUL"}


def apply_referee_decision(
    state: GameState,
    decision: RefereeDecision,
    config: SimulationConfig,
    *,
    was_shooting: bool = False,
    shot_value: int = 1,
    made_shot: bool = False,
) -> dict:
    """Apply a judged FOUL/VIOLATION. NONE returns without changes."""
    if decision.call == "NONE":
        return {"applied": False, "free_throws": 0, "retain_possession": False}
    if decision.call == "VIOLATION":
        apply_turnover(state, other_team(state.offense), config)
        return {"applied": True, "free_throws": 0, "retain_possession": False}
    offender_team = None
    if decision.offender and decision.offender in state.players:
        offender_team = state.players[decision.offender].team
    foul_team = offender_team or other_team(state.offense)
    state.team_fouls[foul_team] = state.team_fouls.get(foul_team, 0) + 1
    penalty = foul_penalty(
        foul_count_after=state.team_fouls[foul_team],
        was_shooting=was_shooting,
        shot_value=shot_value,
        made_shot=made_shot,
        config=config,
    )
    state.event_idx += 1
    return {"applied": True, **penalty, "foul_team": foul_team}
