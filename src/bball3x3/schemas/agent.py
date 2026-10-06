from pydantic import BaseModel, Field

from .memory import CSProposal, MemoryProposal
from .messages import CaptainCall, Message


class CoordinationConfig(BaseModel):
    comms: bool = True
    captain: bool = True
    memory_write: bool = True

class AgentOutput(BaseModel):
    action: str
    target: str | None = None
    memory_ref: list[str] | None = None
    messages: list[Message] = Field(default_factory=list)
    captain_call: CaptainCall | None = None
    memory_update: MemoryProposal | None = None
    cs_update: CSProposal | None = None
    rationale: str | None = None
    raw_text: str = ""
    parse_ok: bool = True
    parse_errors: list[str] = Field(default_factory=list)
