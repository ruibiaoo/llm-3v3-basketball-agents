from typing import Literal

from pydantic import BaseModel

MemoryCategory = Literal[
    "defensive_tendency",
    "offensive_tendency",
    "playstyle",
    "preferred_zone",
    "weakness",
    "general",
]

class MemoryProposal(BaseModel):
    opponent: str
    category: str
    tendency: str

class CSProposal(BaseModel):
    opponent_strategy: str
    counter_strategy: str
    side: Literal["offense", "defense"]

class MemoryEntry(BaseModel):
    id: str
    opponent: str
    category: str
    tendency: str
    confidence: int = 1
    observed_by: str = "PREGAME"
    decision_event: int = 0

class CSEntry(BaseModel):
    key: str
    side: Literal["offense", "defense"]
    opponent_strategy: str
    counter_strategy: str
    created_by: str
    decision_event: int = 0
    times_referenced: int = 0
