
"""Game clock, shot clock, and action timing utilities."""

from dataclasses import dataclass

from bball3x3.config import SimulationConfig
from bball3x3.schemas.state import GameState

DEFAULT_TIME_COSTS = {
    "PASS": 1,
    "SHOOT": 1,
    "DRIVE": 2,
    "DRIBBLE_MOVE": 2,
    "POST_UP": 2,
    "HOLD": 1,
    "CLEAR": 2,
    # Off-ball, defensive and rebound actions default to one event (1 s).
    "CUT": 1,
    "CUT_BASELINE": 1,
    "SCREEN": 1,
    "ROLL": 1,
    "POP": 1,
    "SPOT_UP": 1,
    "MOVE": 1,
    "DEFEND": 1,
    "HELP_PAINT": 1,
    "SWITCH": 1,
    "DOUBLE_TEAM": 1,
    "CONTEST": 1,
    "STEAL_ATTEMPT": 1,
    "DENY": 1,
    "CRASH_BOARDS": 1,
    "BOX_OUT": 1,
    "GET_BACK": 1,
}


@dataclass(frozen=True)
class ClockResult:
    """Result of advancing the simulation clocks."""

    elapsed_s: float
    game_expired: bool
    shot_clock_expired: bool


def get_action_duration(
    action: str,
    config: SimulationConfig,
) -> float:
    """Return the configured duration of an action in seconds."""

    action = action.upper()

    time_costs = {
        **DEFAULT_TIME_COSTS,
        **config.time_costs_s,
    }

    if action not in time_costs:
        raise ValueError(f"Unknown action: {action}")

    duration = float(time_costs[action])

    if duration <= 0:
        raise ValueError(
            f"Action duration must be positive: {action}"
        )

    return duration


def advance_clocks(
    state: GameState,
    elapsed_s: float,
) -> ClockResult:
    """Advance the game and shot clocks without going below zero."""

    if elapsed_s < 0:
        raise ValueError("Elapsed time cannot be negative.")

    if state.phase not in ("LIVE", "OT"):
        return ClockResult(
            elapsed_s=0.0,
            game_expired=False,
            shot_clock_expired=False,
        )

    if state.phase == "OT":
        actual_elapsed = min(
            elapsed_s,
            state.shot_clock_s,
        )

        state.shot_clock_s = max(
            0.0,
            state.shot_clock_s - actual_elapsed,
        )

        return ClockResult(
            elapsed_s=actual_elapsed,
            game_expired=False,
            shot_clock_expired=state.shot_clock_s == 0.0,
        )

    actual_elapsed = min(
        elapsed_s,
        state.game_clock_s,
        state.shot_clock_s,
    )

    state.game_clock_s = max(
        0.0,
        state.game_clock_s - actual_elapsed,
    )

    state.shot_clock_s = max(
        0.0,
        state.shot_clock_s - actual_elapsed,
    )

    return ClockResult(
        elapsed_s=actual_elapsed,
        game_expired=state.game_clock_s == 0.0,
        shot_clock_expired=state.shot_clock_s == 0.0,
    )


def advance_action_clock(
    state: GameState,
    action: str,
    config: SimulationConfig,
) -> ClockResult:
    """Advance clocks according to the duration of an action."""

    duration = get_action_duration(action, config)

    return advance_clocks(state, duration)


def reset_shot_clock(
    state: GameState,
    config: SimulationConfig,
) -> None:
    """Reset the shot clock to its configured full duration."""

    state.shot_clock_s = config.rules.shot_clock_s


def is_shot_clock_violation(state: GameState) -> bool:
    """Check whether a live possession has exhausted its shot clock."""

    return (
        state.phase in ("LIVE", "OT")
        and state.shot_clock_s <= 0.0
    )


def is_regulation_expired(state: GameState) -> bool:
    """Check whether regulation time has expired."""

    return (
        state.phase != "OT"
        and state.game_clock_s <= 0.0
    )
