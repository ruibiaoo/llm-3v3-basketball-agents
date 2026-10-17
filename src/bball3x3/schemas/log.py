from hashlib import sha256
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from .actions import ActionType, InvalidReason
from .agent import AgentOutput, CoordinationConfig
from .memory import CSProposal, MemoryCategory, MemoryProposal
from .messages import Directive
from .referee import RefereeDecision
from .state import GameState

# Shared field types.
Team = Literal["A", "B"]
EventIndex = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
ConstraintId = Annotated[int, Field(strict=True, ge=1, le=20)]
GameEndReason = Literal["CLOCK_EXPIRED", "WIN_SCORE", "OT_WIN"]


def require_both_teams(values: dict[Team, int]) -> dict[Team, int]:
    """Team totals must explicitly include both teams, even when zero."""
    if set(values) != {"A", "B"}:
        raise ValueError("Team totals must contain exactly A and B.")
    return values


TeamCounts = Annotated[
    dict[Team, NonNegativeInt],
    AfterValidator(require_both_teams),
]


class LogModel(BaseModel):
    """Reject misspelled fields and non-finite numeric values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


# ---------------------------------------------------------------------------
# Nested payloads used by the main records.
# ---------------------------------------------------------------------------


class PossessionRef(LogModel):
    possession_id: NonEmptyStr
    team: Team


class ScoringEvent(LogModel):
    team: Team
    points: Annotated[int, Field(strict=True, gt=0)]
    source: Literal["FIELD_GOAL", "FREE_THROW"]
    possession_id: NonEmptyStr


class ShotAttempt(LogModel):
    """One field-goal attempt; assist and screen links are engine outcomes."""

    team: Team
    shooter: NonEmptyStr
    made: bool
    assist_by: NonEmptyStr | None = None
    screen_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_assist(self) -> Self:
        if self.assist_by is not None and (not self.made or self.assist_by == self.shooter):
            raise ValueError("An assist requires a made shot by a different player.")
        return self


class ScreenEvent(LogModel):
    """A screen attempt, linked to a shot by its game-unique ID."""

    screen_id: NonEmptyStr
    team: Team
    screener: NonEmptyStr
    freed_player: NonEmptyStr

    @model_validator(mode="after")
    def validate_players(self) -> Self:
        if self.screener == self.freed_player:
            raise ValueError("A screen must involve two distinct teammates.")
        return self


class TurnoverEvent(LogModel):
    """An explicit loss of control, including team violations."""

    team: Team
    possession_id: NonEmptyStr
    reason: NonEmptyStr
    player: NonEmptyStr | None = None


class PossessionChange(LogModel):
    from_possession_id: NonEmptyStr
    to_possession_id: NonEmptyStr
    to_team: Team
    reason: NonEmptyStr

    # PERIOD_BOUNDARY represents starting an OT possession, which need not
    # involve the opposing team gaining control.
    kind: Literal["CONTROL_TRANSFER", "PERIOD_BOUNDARY"]

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        if self.from_possession_id == self.to_possession_id:
            raise ValueError("A possession change must create a new ID.")
        return self


class GameTotals(LogModel):
    decisions: NonNegativeInt
    decisions_by_team: TeamCounts
    transitions: NonNegativeInt
    possessions_by_team: TeamCounts

    @model_validator(mode="after")
    def validate_decision_total(self) -> Self:
        if self.decisions != sum(self.decisions_by_team.values()):
            raise ValueError("Decision total must equal the team totals.")
        return self


class InitialMemoryEntry(LogModel):
    """A team memory item available before the first decision."""

    id: NonEmptyStr
    opponent: NonEmptyStr
    category: MemoryCategory
    tendency: NonEmptyStr


# ---------------------------------------------------------------------------
# Game lifecycle.
# ---------------------------------------------------------------------------


class GameStartRecord(LogModel):
    record_type: Literal["game_start"] = "game_start"

    # Required explicitly so an unversioned legacy record is not silently
    # interpreted as a version-1 record.
    schema_version: Literal[1]

    game_id: NonEmptyStr
    experiment_id: NonEmptyStr
    replicate_id: NonEmptyStr
    seed: Annotated[int, Field(strict=True)]

    first_possession: Team
    team_conditions: dict[Team, NonEmptyStr]
    matchup: dict[str, Any]
    coordination: dict[Team, CoordinationConfig]
    roster: dict[str, Any]

    config_hash: NonEmptyStr
    config_snapshot: dict[str, Any]
    prompt_hashes: dict[str, str]
    model_slug: NonEmptyStr
    code_commit: str | None

    initial_state: GameState
    initial_possession_id: NonEmptyStr
    # None marks legacy/incomplete metadata; an explicit empty list means that
    # team's initial memory was checked and is empty.
    initial_memory: dict[Team, list[InitialMemoryEntry]] | None = None

    @model_validator(mode="after")
    def validate_initial_metadata(self) -> Self:
        if set(self.team_conditions) != {"A", "B"}:
            raise ValueError("team_conditions must contain A and B.")
        if set(self.coordination) != {"A", "B"}:
            raise ValueError("coordination must contain A and B.")
        if self.initial_state.event_idx != 0:
            raise ValueError("The initial state must have event_idx = 0.")
        if self.initial_state.score != {"A": 0, "B": 0}:
            raise ValueError("A complete game must start at 0-0.")
        if self.initial_state.offense != self.first_possession:
            raise ValueError("Initial offense must match first_possession.")
        if self.initial_memory is not None:
            if set(self.initial_memory) != {"A", "B"}:
                raise ValueError("initial_memory must contain exactly A and B.")
            for team, entries in self.initial_memory.items():
                ids = [entry.id for entry in entries]
                if len(ids) != len(set(ids)):
                    raise ValueError(f"Initial memory IDs for team {team} must be unique.")
        return self


class GameEndRecord(LogModel):
    record_type: Literal["game_end"] = "game_end"
    last_event_idx: EventIndex
    end_reason: GameEndReason
    final_score: TeamCounts
    ot_flag: bool
    totals: GameTotals
    total_cost_usd: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_transition_total(self) -> Self:
        if self.totals.transitions != self.last_event_idx:
            raise ValueError(
                "Transition total must equal the last contiguous event index."
            )
        return self


# ---------------------------------------------------------------------------
# Player decisions.
# ---------------------------------------------------------------------------


class DecisionRecord(LogModel):
    record_type: Literal["decision"] = "decision"
    event_idx: EventIndex
    player: NonEmptyStr
    team: Team

    raw_output: str
    parsed_output: AgentOutput | None
    parse_ok: bool

    # Required explicitly: null means checked and valid.
    invalid_reason: InvalidReason | None

    remapped_action: ActionType | None = None
    remapped_target: str | None = None

    # These are the actions handed to the resolver, not guaranteed outcomes.
    executed_action: ActionType
    executed_target: str | None

    # Exact player-facing observation before this decision. None means legacy
    # or unavailable evidence; an empty string is a recorded empty observation.
    observation_text: str | None = None
    observation_hash: str = ""
    memory_ref: list[str] | None = None
    latency_ms: float = Field(default=0.0, ge=0)
    tokens: dict[str, NonNegativeInt] = Field(default_factory=dict)
    cost_usd: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_proposal_and_execution(self) -> Self:
        if self.observation_text is not None:
            digest = sha256(self.observation_text.encode("utf-8")).hexdigest()
            if self.observation_hash and self.observation_hash != digest:
                raise ValueError("observation_hash must match SHA-256 of observation_text UTF-8.")
            self.observation_hash = digest
        if self.parse_ok:
            if self.parsed_output is None:
                raise ValueError("A successful parse requires parsed_output.")
            if not self.parsed_output.parse_ok:
                raise ValueError("The two parse_ok fields must agree.")
            if self.invalid_reason == "PARSE_FAIL":
                raise ValueError("A successful parse cannot be PARSE_FAIL.")
        else:
            if self.parsed_output is not None:
                raise ValueError("A final parse failure requires null output.")
            if self.invalid_reason != "PARSE_FAIL":
                raise ValueError("A failed parse must use PARSE_FAIL.")

        if self.invalid_reason is None:
            if self.remapped_action is not None or self.remapped_target is not None:
                raise ValueError("A valid proposal must not have a replacement.")

            if self.parsed_output is None:
                raise ValueError("A valid proposal requires parsed_output.")

            proposed = (
                self.parsed_output.action,
                self.parsed_output.target,
            )
            executed = (self.executed_action, self.executed_target)

            if proposed != executed:
                raise ValueError("A valid proposal must execute unchanged.")
        else:
            if self.remapped_action is None:
                raise ValueError("An invalid proposal requires a safe replacement.")

            replacement = (self.remapped_action, self.remapped_target)
            executed = (self.executed_action, self.executed_target)

            if replacement != executed:
                raise ValueError("Executed action must match the replacement.")

        return self


# ---------------------------------------------------------------------------
# Coordination records.
# ---------------------------------------------------------------------------


class MessageRecord(LogModel):
    record_type: Literal["message"] = "message"
    event_idx: EventIndex
    sender: NonEmptyStr
    team: Team
    channel: Literal["teammate", "captain"]
    kind: Literal["action", "tactic", "info"]
    text: str
    directive: Directive | None = None
    cs: str | None = None
    delivered_at: EventIndex | None = None
    suppressed: bool = False


class CaptainCallRecord(LogModel):
    record_type: Literal["captain_call"] = "captain_call"
    event_idx: EventIndex
    team: Team
    maintain: bool
    tactic: str | None = None
    cs: str | None = None
    directives: list[dict[str, Any]] = Field(default_factory=list)


class MemoryProposalRecord(LogModel):
    record_type: Literal["memory_proposal"] = "memory_proposal"
    event_idx: EventIndex
    team: Team
    player: NonEmptyStr
    proposal: MemoryProposal
    accepted: bool
    merged_into: str | None = None
    new_id: str | None = None
    similarity: float | None = None
    confidence_after: int | None = None


class MemoryWriteRecord(LogModel):
    """An accepted memory-store mutation and its decision-time visibility."""

    record_type: Literal["memory_write"] = "memory_write"
    event_idx: EventIndex
    team: Team
    memory_id: NonEmptyStr
    opponent: NonEmptyStr
    category: MemoryCategory
    tendency: NonEmptyStr
    visible_at: EventIndex
    evicted_ids: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_visibility(self) -> Self:
        if self.visible_at <= self.event_idx:
            raise ValueError("A memory write must become visible after its write event.")
        if self.memory_id in self.evicted_ids:
            raise ValueError("A memory write cannot evict the entry it just wrote.")
        if len(self.evicted_ids) != len(set(self.evicted_ids)):
            raise ValueError("evicted_ids must not contain duplicates.")
        return self


class CSProposalRecord(LogModel):
    record_type: Literal["cs_proposal"] = "cs_proposal"
    event_idx: EventIndex
    team: Team
    player: NonEmptyStr
    proposal: CSProposal
    accepted: bool
    key: str | None = None
    merged: bool = False


class CSWriteRecord(LogModel):
    """An accepted counter-strategy entry and when agents can first use it."""

    record_type: Literal["cs_write"] = "cs_write"
    event_idx: EventIndex
    team: Team
    key: NonEmptyStr
    side: Literal["offense", "defense"]
    opponent_strategy: NonEmptyStr
    counter_strategy: NonEmptyStr
    merged: bool = False
    visible_at: EventIndex

    @model_validator(mode="after")
    def validate_visibility(self) -> Self:
        if self.visible_at <= self.event_idx:
            raise ValueError("A CS write must become visible after its write event.")
        return self


# ---------------------------------------------------------------------------
# Engine and referee outcomes.
# ---------------------------------------------------------------------------


class ResolutionRecord(LogModel):
    record_type: Literal["resolution"] = "resolution"
    event_idx: EventIndex

    actions: dict[str, ActionType]
    action_targets: dict[str, str | None]
    outcomes: dict[str, Any] = Field(default_factory=dict)
    # When present, this must identify the opponents involved in each player's
    # action. None preserves compatibility with logs that predate this evidence.
    involved_opponents: dict[str, list[NonEmptyStr]] | None = None
    model_terms: dict[str, Any] = Field(default_factory=dict)
    rng_draws: dict[str, Any] = Field(default_factory=dict)

    possession_before: PossessionRef
    possession_after: PossessionRef | None
    scoring_events: list[ScoringEvent]
    possession_changes: list[PossessionChange]

    # None means the producer did not supply descriptive evidence. An empty
    # list means that event was checked and had no such outcomes.
    shots: list[ShotAttempt] | None = None
    screens: list[ScreenEvent] | None = None
    turnovers: list[TurnoverEvent] | None = None

    game_over: bool
    game_end_reason: GameEndReason | None

    @model_validator(mode="after")
    def validate_resolution_fields(self) -> Self:
        if self.actions.keys() != self.action_targets.keys():
            raise ValueError("Actions and targets must cover the same players.")
        if self.involved_opponents is not None:
            if self.involved_opponents.keys() != self.actions.keys():
                raise ValueError("involved_opponents must cover every acting player.")
            for player, opponents in self.involved_opponents.items():
                if len(opponents) != len(set(opponents)):
                    raise ValueError(f"Involved opponents for {player} must be unique.")
                if any(opponent[:1] == player[:1] for opponent in opponents):
                    raise ValueError(f"Involved opponents for {player} must be on the other team.")

        if self.game_over:
            if self.game_end_reason is None:
                raise ValueError("A finished game requires an end reason.")
            if self.possession_after is not None:
                raise ValueError("A finished game has no active next possession.")
        else:
            if self.game_end_reason is not None:
                raise ValueError("An ongoing game cannot have an end reason.")
            if self.possession_after is None:
                raise ValueError("An ongoing game requires an active possession.")

        return self


class RefereeRecord(LogModel):
    record_type: Literal["referee"] = "referee"
    event_idx: EventIndex
    query: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    decision: RefereeDecision
    latency_ms: float = Field(default=0.0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)


class StateRecord(LogModel):
    record_type: Literal["state"] = "state"
    event_idx: EventIndex

    # Use the normal typed state whenever possible.
    state: GameState | None

    # Preserve an invalid candidate snapshot if GameState rejects it.
    raw_state: dict[str, Any] | None = None
    state_validation_errors: list[str] = Field(default_factory=list)

    # Required explicitly so missing checks cannot look like a clean transition.
    invariant_checks_completed: bool
    constraints_checked: list[ConstraintId]

    @model_validator(mode="after")
    def validate_snapshot_and_checks(self) -> Self:
        if (self.state is None) == (self.raw_state is None):
            raise ValueError("Provide exactly one of state or raw_state.")

        if self.state is not None:
            if self.state.event_idx != self.event_idx:
                raise ValueError("State and record event indices must agree.")
            if self.state_validation_errors:
                raise ValueError("A typed state must not have validation errors.")
        else:
            if not self.state_validation_errors:
                raise ValueError("A raw state requires validation error details.")
            if self.raw_state.get("event_idx") != self.event_idx:
                raise ValueError("Raw state and record event indices must agree.")

        checked = set(self.constraints_checked)

        if len(checked) != len(self.constraints_checked):
            raise ValueError("constraints_checked must not contain duplicates.")

        if self.invariant_checks_completed and checked != set(range(1, 21)):
            raise ValueError("Completed checking requires all 20 constraints.")

        return self


class InvariantViolationRecord(LogModel):
    record_type: Literal["invariant_violation"] = "invariant_violation"
    event_idx: EventIndex
    constraint_id: ConstraintId
    constraint_name: NonEmptyStr
    details: NonEmptyStr


class PossessionEndRecord(LogModel):
    record_type: Literal["possession_end"] = "possession_end"
    possession_id: NonEmptyStr
    team: Team
    points: NonNegativeInt
    points_by_team: TeamCounts
    end_reason: NonEmptyStr
    start_event_idx: EventIndex
    end_event_idx: EventIndex

    @model_validator(mode="after")
    def validate_possession_totals(self) -> Self:
        if self.end_event_idx < self.start_event_idx:
            raise ValueError("A possession cannot end before it starts.")
        if self.points != self.points_by_team[self.team]:
            raise ValueError("points must equal the owning team's points.")
        return self


# The reader can use TypeAdapter(LogRecord) to select the correct model
# automatically from each JSON object's record_type.
LogRecord = Annotated[
    GameStartRecord
    | DecisionRecord
    | MessageRecord
    | CaptainCallRecord
    | MemoryProposalRecord
    | MemoryWriteRecord
    | CSProposalRecord
    | CSWriteRecord
    | ResolutionRecord
    | RefereeRecord
    | StateRecord
    | InvariantViolationRecord
    | PossessionEndRecord
    | GameEndRecord,
    Field(discriminator="record_type"),
]
