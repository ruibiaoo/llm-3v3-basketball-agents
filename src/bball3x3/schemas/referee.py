from typing import Any, Literal

from pydantic import BaseModel, Field

from .state import BallState, PlayerState

FoulType = Literal[
    "DEFENSIVE_FOUL",
    "OFFENSIVE_FOUL",
    "SHOOTING_FOUL",
    "LOOSE_BALL_FOUL",
    "UNSPORTSMANLIKE",
    "TECHNICAL",
]

JudgementViolation = Literal[
    "GOALTENDING",
    "BASKET_INTERFERENCE",
    "STALLING",
]

class RefereeDecision(BaseModel):
    call: Literal["NONE", "FOUL", "VIOLATION"] = "NONE"
    offender: str | None = None
    fouled_player: str | None = None
    foul_type: FoulType | None = None
    violation_type: JudgementViolation | None = None
    rule_ids_cited: list[str] = Field(default_factory=list)

class ContactEvent(BaseModel):
    contact_detected: bool = False
    players_involved: list[str] = Field(default_factory=list)
    contact_location: tuple[int, int] | None = None
    contact_type: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

class RefereeState(BaseModel):
    event_idx: int = 0
    game_clock_s: float = 600.0
    shot_clock_s: float = 12.0
    score: dict[str, int] = Field(default_factory=lambda: {"A": 0, "B": 0})
    team_fouls: dict[str, int] = Field(default_factory=lambda: {"A": 0, "B": 0})
    possession: Literal["A", "B"] = "A"
    ball_handler: str | None = None
    players: dict[str, PlayerState] = Field(default_factory=dict)
    matchups: dict[str, str] = Field(default_factory=dict)
    ball: BallState
    current_event: dict[str, Any] = Field(default_factory=dict)
    contact_event: ContactEvent = Field(default_factory=ContactEvent)
