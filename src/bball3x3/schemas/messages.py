from typing import Literal

from pydantic import BaseModel, Field


class Directive(BaseModel):
    to: str
    act: str
    target: str | None = None

class Message(BaseModel):
    sender: str
    is_captain_channel: bool = False
    kind: Literal["action", "tactic", "info"] = "info"
    text: str = ""
    directive: Directive | None = None
    cs: str | None = None

class CaptainCall(BaseModel):
    maintain: bool = False
    tactic: Message | None = None
    directives: list[Message] = Field(default_factory=list)
