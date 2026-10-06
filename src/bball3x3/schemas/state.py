from typing import Literal

from pydantic import BaseModel, Field


# player ability ratings between 1 and 10
class Abilities(BaseModel):
    shooting: int = Field(ge=1, le=10)
    passing: int = Field(ge=1, le=10)
    defense: int = Field(ge=1, le=10)
    stamina: int = Field(ge=1, le=10)
    ball_handling: int = Field(ge=1, le=10)
    rebounding: int = Field(ge=1, le=10)

MovementLabel = str
Speed = Literal["NONE", "SLOW", "MEDIUM", "FAST", "STATIONARY"]
Phase = Literal["LIVE", "DEAD", "FT", "OT"]
BallStatus = Literal["HELD", "IN_PASS", "IN_SHOT", "LOOSE", "DEAD", "LIVE"]

class PlayerState(BaseModel):
    id: str
    team: Literal["A", "B"]
    pos: tuple[int, int]
    prev_pos: tuple[int, int]
    movement: MovementLabel = "HOLD"
    speed: str = "NONE"
    has_ball: bool = False
    fatigue: float = Field(default=0.0, ge=0.0, le=10.0)

class BallState(BaseModel):
    status: BallStatus = "HELD"
    pos: tuple[int, int]
    holder: str | None = None

class GameState(BaseModel):
    event_idx: int = 0
    game_clock_s: float = 600.0
    shot_clock_s: float = 12.0
    score: dict[str, int] = Field(default_factory=lambda: {"A": 0, "B": 0})
    team_fouls: dict[str, int] = Field(default_factory=lambda: {"A": 0, "B": 0})
    offense: Literal["A", "B"] = "A"
    needs_clear: bool = False
    phase: Phase = "LIVE"
    players: dict[str, PlayerState] = Field(default_factory=dict)
    ball: BallState
    matchups: dict[str, str] = Field(default_factory=dict)
