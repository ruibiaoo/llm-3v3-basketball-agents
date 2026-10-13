
"""Game initialization and basic state transitions."""

from bball3x3.config import SimulationConfig
from bball3x3.engine.court import Court
from bball3x3.engine.rng import GameRNG
from bball3x3.schemas.state import (
    BallState,
    GameState,
    PlayerState,
)

INITIAL_POSITIONS = {
    "A1": (7, 9),
    "A2": (3, 7),
    "A3": (11, 7),
    "B1": (7, 6),
    "B2": (3, 5),
    "B3": (11, 5),
}


def create_initial_state(
    config: SimulationConfig,
    rng: GameRNG,
    first_possession: str | None = None,
) -> GameState:
    """Create the initial state of a 3v3 basketball game."""

    court = Court(config.court)

    for position in INITIAL_POSITIONS.values():
        if not court.is_in_bounds(position):
            raise ValueError(
                f"Initial player position outside court: {position}"
            )

    if first_possession is None:
        first_possession = rng.choice(
            "first_possession", ["A", "B"]
        )

    if first_possession not in ("A", "B"):
        raise ValueError("First possession must be A or B.")

    players = {}

    for player_id, position in INITIAL_POSITIONS.items():
        team = player_id[0]

        players[player_id] = PlayerState(
            id=player_id,
            team=team,
            pos=position,
            prev_pos=position,
            movement="HOLD",
            speed="NONE",
            has_ball=False,
            fatigue=0.0,
        )

    ball_holder = f"{first_possession}1"
    players[ball_holder].has_ball = True

    ball = BallState(
        status="HELD",
        pos=players[ball_holder].pos,
        holder=ball_holder,
    )

    return GameState(
        event_idx=0,
        game_clock_s=config.rules.game_length_s,
        shot_clock_s=config.rules.shot_clock_s,
        score={"A": 0, "B": 0},
        team_fouls={"A": 0, "B": 0},
        offense=first_possession,
        needs_clear=False,
        phase="LIVE",
        players=players,
        ball=ball,
        matchups={
            "A1": "B1",
            "A2": "B2",
            "A3": "B3",
            "B1": "A1",
            "B2": "A2",
            "B3": "A3",
        },
    )


def set_ball_holder(state: GameState, player_id: str) -> None:
    """Give the ball to an existing player."""

    if player_id not in state.players:
        raise ValueError(f"Unknown player: {player_id}")

    for player in state.players.values():
        player.has_ball = False

    holder = state.players[player_id]
    holder.has_ball = True

    state.ball.status = "HELD"
    state.ball.holder = player_id
    state.ball.pos = holder.pos


def switch_possession(
    state: GameState,
    new_offense: str,
    config: SimulationConfig,
    needs_clear: bool = True,
) -> None:
    """Change the offensive team and reset the shot clock."""

    if new_offense not in ("A", "B"):
        raise ValueError("Offensive team must be A or B.")

    if new_offense == state.offense:
        raise ValueError("New offense must differ from current offense.")

    state.offense = new_offense
    state.shot_clock_s = config.rules.shot_clock_s
    state.needs_clear = needs_clear

    # Possession changes do not automatically assign a new ball holder.
    for player in state.players.values():
        player.has_ball = False

    state.ball.status = "LOOSE"
    state.ball.holder = None


def move_player(
    state: GameState,
    player_id: str,
    new_pos: tuple[int, int],
    court: Court,
) -> None:
    """Move a player to a valid court position."""

    if player_id not in state.players:
        raise ValueError(f"Unknown player: {player_id}")

    if not court.is_in_bounds(new_pos):
        raise ValueError(f"Position outside court: {new_pos}")

    player = state.players[player_id]

    player.prev_pos = player.pos
    player.pos = new_pos

    if player.has_ball:
        state.ball.pos = new_pos
