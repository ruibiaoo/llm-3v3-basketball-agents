from typing import Any, Literal

from pydantic import BaseModel, Field

from .agent import AgentOutput, CoordinationConfig
from .memory import CSProposal, MemoryProposal
from .messages import Directive
from .referee import RefereeDecision
from .state import GameState


# event log record types for jsonl logging
class GameStartRecord(BaseModel):
    record_type: Literal["game_start"] = "game_start"
    game_id: str
    seed: int
    matchup: dict[str, Any]
    coordination: dict[str, CoordinationConfig]
    config_hash: str
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    model_slug: str
    roster: dict[str, Any]

class DecisionRecord(BaseModel):
    record_type: Literal["decision"] = "decision"
    event_idx: int
    player: str
    observation_hash: str = ""
    raw_output: str = ""
    parsed_output: AgentOutput | None = None
    parse_ok: bool = True
    invalid_reason: str | None = None
    remapped_action: str | None = None
    memory_ref: list[str] | None = None
    latency_ms: float = 0.0
    tokens: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0

class MessageRecord(BaseModel):
    record_type: Literal["message"] = "message"
    event_idx: int
    sender: str
    team: Literal["A", "B"]
    channel: Literal["teammate", "captain"]
    kind: Literal["action", "tactic", "info"]
    text: str
    directive: Directive | None = None
    cs: str | None = None
    delivered_at: int | None = None
    suppressed: bool = False

class CaptainCallRecord(BaseModel):
    record_type: Literal["captain_call"] = "captain_call"
    event_idx: int
    team: Literal["A", "B"]
    maintain: bool
    tactic: str | None = None
    cs: str | None = None
    directives: list[dict[str, Any]] = Field(default_factory=list)

class MemoryProposalRecord(BaseModel):
    record_type: Literal["memory_proposal"] = "memory_proposal"
    event_idx: int
    team: Literal["A", "B"]
    player: str
    proposal: MemoryProposal
    accepted: bool
    merged_into: str | None = None
    new_id: str | None = None
    similarity: float | None = None
    confidence_after: int | None = None

class CSProposalRecord(BaseModel):
    record_type: Literal["cs_proposal"] = "cs_proposal"
    event_idx: int
    team: Literal["A", "B"]
    player: str
    proposal: CSProposal
    accepted: bool
    key: str | None = None
    merged: bool = False

class ResolutionRecord(BaseModel):
    record_type: Literal["resolution"] = "resolution"
    event_idx: int
    actions: dict[str, str]
    outcomes: dict[str, Any] = Field(default_factory=dict)
    model_terms: dict[str, Any] = Field(default_factory=dict)
    rng_draws: dict[str, Any] = Field(default_factory=dict)

class RefereeRecord(BaseModel):
    record_type: Literal["referee"] = "referee"
    event_idx: int
    query: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    decision: RefereeDecision
    latency_ms: float = 0.0
    cost_usd: float = 0.0

class StateRecord(BaseModel):
    record_type: Literal["state"] = "state"
    event_idx: int
    state: GameState

class InvariantViolationRecord(BaseModel):
    record_type: Literal["invariant_violation"] = "invariant_violation"
    event_idx: int
    constraint_id: int
    constraint_name: str
    details: str

class PossessionEndRecord(BaseModel):
    record_type: Literal["possession_end"] = "possession_end"
    team: Literal["A", "B"]
    points: int
    end_reason: str
    start_event_idx: int
    end_event_idx: int

class GameEndRecord(BaseModel):
    record_type: Literal["game_end"] = "game_end"
    final_score: dict[str, int]
    ot_flag: bool
    totals: dict[str, Any] = Field(default_factory=dict)
    total_cost_usd: float = 0.0
