"""Invariant checker: the 20 constraints from proposal Table 3 (WS1 spec 10).

In development and tests any violation raises
:class:`InvariantViolationError`. In experiments the runner logs them
(SIR) and continues. State-only checks run without event context;
transition checks use the optional ``event`` dict::

    event = {
        "actions": {pid: action}, "elapsed_s": float,
        "scored": bool, "points": int, "scoring_team": str,
        "possession_changed": bool, "possession_reason": str | None,
        "shot_clock_reset": bool, "reset_reason": str | None,
        "rebound": bool, "rebound_after_miss": bool,
        "turnover": bool, "foul_called": bool,
    }
"""

from bball3x3.config import SimulationConfig
from bball3x3.engine.court import Court
from bball3x3.engine.movement import displacement_cells
from bball3x3.schemas.state import GameState

BALL_HANDLER_ONLY = {"SHOOT", "PASS", "DRIVE", "DRIBBLE_MOVE", "POST_UP", "CLEAR"}

CONSTRAINT_NAMES = {
    1: "unique_ball_possession",
    2: "ball_possessor_consistency",
    3: "valid_player_positions",
    4: "valid_ball_position",
    5: "valid_player_count",
    6: "valid_team_membership",
    7: "possession_dependent_actions",
    8: "action_physical_feasibility",
    9: "movement_limit",
    10: "game_clock_validity",
    11: "shot_clock_validity",
    12: "shot_clock_reset_consistency",
    13: "possession_transition_consistency",
    14: "score_consistency",
    15: "rebound_consistency",
    16: "turnover_consistency",
    17: "defensive_matchup_validity",
    18: "fatigue_bounds",
    19: "foul_consistency",
    20: "dead_ball_consistency",
}


class InvariantViolationError(AssertionError):
    def __init__(self, constraint_id: int, details: str):
        super().__init__(f"[{constraint_id}:{CONSTRAINT_NAMES[constraint_id]}] {details}")
        self.constraint_id = constraint_id
        self.constraint_name = CONSTRAINT_NAMES[constraint_id]
        self.details = details


def _violation(constraint_id: int, details: str) -> tuple[int, str, str]:
    return (constraint_id, CONSTRAINT_NAMES[constraint_id], details)


def check_state(
    state: GameState, court: Court, _config: SimulationConfig
) -> list[tuple[int, str, str]]:
    violations: list[tuple[int, str, str]] = []

    holders = [pid for pid, p in state.players.items() if p.has_ball]
    if len(holders) > 1:
        violations.append(_violation(1, f"multiple holders: {holders}"))

    if holders:
        holder = holders[0]
        if state.ball.status != "HELD" or state.ball.holder != holder:
            violations.append(
                _violation(2, f"ball {state.ball.status}/{state.ball.holder} vs {holder}")
            )
        elif state.ball.pos != state.players[holder].pos:
            violations.append(_violation(2, "ball pos differs from holder pos"))
    elif state.ball.status == "HELD":
        violations.append(_violation(2, "ball HELD with no holder flag"))

    for pid, player in state.players.items():
        if not court.is_in_bounds(player.pos):
            violations.append(_violation(3, f"{pid} out of bounds at {player.pos}"))
        if not court.is_in_bounds(player.prev_pos):
            violations.append(_violation(3, f"{pid} prev_pos out of bounds"))

    if not court.is_in_bounds(state.ball.pos):
        violations.append(_violation(4, f"ball out of bounds at {state.ball.pos}"))

    team_a = sum(1 for p in state.players.values() if p.team == "A")
    team_b = sum(1 for p in state.players.values() if p.team == "B")
    if team_a != 3 or team_b != 3:
        violations.append(_violation(5, f"counts A={team_a} B={team_b}, expected 3 each"))

    for pid, player in state.players.items():
        if pid[0] != player.team or player.team not in ("A", "B"):
            violations.append(_violation(6, f"{pid} has team {player.team}"))
        if pid not in state.matchups and player.team in ("A", "B"):
            pass  # matchups checked in rule 17; missing entries tolerated here

    for pid, matchup in state.matchups.items():
        if pid in state.players and matchup in state.players:
            if state.players[pid].team == state.players[matchup].team:
                violations.append(
                    _violation(17, f"{pid} guards teammate {matchup}")
                )
        elif matchup not in state.players:
            violations.append(_violation(17, f"matchup target {matchup} not active"))

    for pid, player in state.players.items():
        if not 0.0 <= player.fatigue <= 10.0:
            violations.append(_violation(18, f"{pid} fatigue {player.fatigue}"))

    if state.game_clock_s < 0.0 or state.game_clock_s > 600.0 + 1e-9:
        violations.append(_violation(10, f"game_clock {state.game_clock_s}"))
    if state.shot_clock_s < 0.0 or state.shot_clock_s > 12.0 + 1e-9:
        violations.append(_violation(11, f"shot_clock {state.shot_clock_s}"))

    if state.phase in ("DEAD", "FT"):
        if any(p.has_ball for p in state.players.values()):
            violations.append(
                _violation(20, f"has_ball set while phase={state.phase}")
            )

    return violations


def check_transition(
    prev: GameState,
    new: GameState,
    event: dict,
    config: SimulationConfig,
) -> list[tuple[int, str, str]]:
    violations: list[tuple[int, str, str]] = []
    actions: dict[str, str] = {
        pid: str(act).upper() for pid, act in event.get("actions", {}).items()
    }

    if prev.phase in ("LIVE", "OT"):
        holder = next(
            (pid for pid, p in prev.players.items() if p.has_ball), None
        )
        for pid, act in actions.items():
            if act in BALL_HANDLER_ONLY and pid != holder:
                violations.append(
                    _violation(7, f"{pid} attempts {act} while {holder} holds ball")
                )
            if pid in prev.players and act in ("PASS", "SCREEN", "SWITCH"):
                pass  # target-existence enforced by the action validator

    elapsed = float(event.get("elapsed_s", 2.0))
    max_step = 3.0 * max(elapsed, 0.5)
    for pid in prev.players:
        if pid in new.players:
            moved = displacement_cells(
                prev.players[pid].pos, new.players[pid].pos
            )
            if moved > max_step + 1e-9:
                violations.append(
                    _violation(9, f"{pid} moved {moved:.2f} cells in {elapsed}s")
                )

    if new.game_clock_s - prev.game_clock_s > 1e-9:
        violations.append(
            _violation(10, f"clock increased {prev.game_clock_s}->{new.game_clock_s}")
        )

    reset_reasons = {
        "POSSESSION_CHANGE",
        "OFFENSIVE_REBOUND",
        "CHECK_BALL",
        "OT_START",
        None,
    }
    if new.shot_clock_s > prev.shot_clock_s + 1e-9:
        reason = event.get("reset_reason")
        if not event.get("shot_clock_reset") or reason not in reset_reasons - {None}:
            violations.append(
                _violation(12, f"shot clock reset without reason ({reason})")
            )

    if new.offense != prev.offense:
        if not event.get("possession_changed"):
            violations.append(_violation(13, "possession changed without event flag"))
        elif not event.get("possession_reason"):
            violations.append(_violation(13, "possession change lacks a reason"))

    for team in ("A", "B"):
        gained = new.score.get(team, 0) - prev.score.get(team, 0)
        if gained < 0 or gained > 2:
            violations.append(_violation(14, f"{team} score jump {gained}"))
        elif gained > 0 and not event.get("scored"):
            violations.append(_violation(14, f"{team} +{gained} with scored=false"))

    if event.get("rebound") and not event.get("rebound_after_miss", True):
        violations.append(_violation(15, "rebound without a preceding missed shot"))

    if event.get("turnover") and new.offense == prev.offense:
        violations.append(_violation(16, "turnover kept possession"))

    for team in ("A", "B"):
        gained = new.team_fouls.get(team, 0) - prev.team_fouls.get(team, 0)
        if gained < 0 or gained > 2:
            violations.append(_violation(19, f"{team} foul jump {gained}"))
        elif gained > 0 and not event.get("foul_called"):
            violations.append(_violation(19, f"{team} foul added without a call"))

    if prev.phase in ("DEAD", "FT") and not event.get("resume_play", False):
        live_acts = [a for a in actions.values() if a not in ("HOLD",)]
        if live_acts and new.phase == prev.phase:
            violations.append(
                _violation(20, f"live actions {live_acts} while {prev.phase}")
            )

    _ = config
    return violations


def collect_violations(
    prev: GameState,
    new: GameState,
    event: dict | None,
    court: Court,
    config: SimulationConfig,
) -> list[dict]:
    """Return violation dicts (SIR logging shape) without raising."""
    event = event or {}
    out: list[dict] = []
    for cid, name, details in check_state(new, court, config):
        out.append(
            {
                "constraint_id": cid,
                "constraint_name": name,
                "details": details,
                "event_idx": new.event_idx,
            }
        )
    for cid, name, details in check_transition(prev, new, event, config):
        out.append(
            {
                "constraint_id": cid,
                "constraint_name": name,
                "details": details,
                "event_idx": new.event_idx,
            }
        )
    return out


def check(
    prev: GameState,
    new: GameState,
    event: dict | None,
    court: Court,
    config: SimulationConfig,
) -> None:
    """Raise on the first violation (dev/test mode)."""
    violations = collect_violations(prev, new, event, court, config)
    if violations:
        first = violations[0]
        raise InvariantViolationError(first["constraint_id"], first["details"])
