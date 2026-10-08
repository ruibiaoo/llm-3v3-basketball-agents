from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .schemas.agent import CoordinationConfig
from .schemas.state import Abilities


# llm configurations
class LLMProviderConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    app_name: str = "dsa4213-3x3-agents"
    routing: dict[str, Any] = Field(default_factory=dict)

class LLMPlayersConfig(BaseModel):
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 500
    seed: int | None = None
    output_format: Literal["xml", "json"] = "xml"
    include_rationale: bool = True
    timeout_s: float = 45.0
    max_retries: int = 3
    repair_attempts: int = 1

class LLMRefereeConfig(BaseModel):
    enabled: bool = False
    backend: Literal["llm", "rule_based"] = "rule_based"
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    max_tokens: int = 200
    output_format: Literal["xml", "json"] = "xml"
    top_k_rules: int = 3

class LLMEmbeddingsConfig(BaseModel):
    backend: str = "local"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"

class LLMRuntimeConfig(BaseModel):
    max_concurrent_requests: int = 12
    cache: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, float] = Field(default_factory=dict)

class LLMConfig(BaseModel):
    provider: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    players: LLMPlayersConfig = Field(default_factory=LLMPlayersConfig)
    referee: LLMRefereeConfig = Field(default_factory=LLMRefereeConfig)
    embeddings: LLMEmbeddingsConfig = Field(default_factory=LLMEmbeddingsConfig)
    runtime: LLMRuntimeConfig = Field(default_factory=LLMRuntimeConfig)

# court and rules configurations
class CourtConfig(BaseModel):
    width_m: float = 15.0
    depth_m: float = 11.0
    cell_m: float = 1.0
    basket: tuple[int, int] = (7, 1)
    arc_radius_m: float = 6.75

class RulesConfig(BaseModel):
    game_length_s: float = 600.0
    win_score: int = 21
    shot_clock_s: float = 12.0
    overtime_points_to_win: int = 2
    team_foul_bonus: dict[str, int] = Field(
        default_factory=lambda: {"two_fts_from": 7, "two_fts_plus_possession_from": 10}
    )

class SimulationConfig(BaseModel):
    court: CourtConfig = Field(default_factory=CourtConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    time_costs_s: dict[str, int] = Field(default_factory=dict)
    movement: dict[str, Any] = Field(default_factory=dict)
    shot_model: dict[str, Any] = Field(default_factory=dict)
    contest_model: dict[str, Any] = Field(default_factory=dict)
    pass_model: dict[str, Any] = Field(default_factory=dict)
    drive_model: dict[str, Any] = Field(default_factory=dict)
    rebound: dict[str, Any] = Field(default_factory=dict)
    contact: dict[str, Any] = Field(default_factory=dict)
    fatigue: dict[str, Any] = Field(default_factory=dict)

# roster configurations
class PlayerProfileConfig(BaseModel):
    slot: str
    name: str
    source_url: str = ""
    abilities: Abilities
    playstyle: str = ""
    synthetic_fields: list[str] = Field(default_factory=list)

class RosterConfig(BaseModel):
    source_note: str = ""
    captain: str = "P1"
    players: list[PlayerProfileConfig]

# initial memory configurations
class OpponentScoutingConfig(BaseModel):
    opponent_slot: str
    category: str
    tendency: str

class InitialMemoryConfig(BaseModel):
    game_plan: str = ""
    coach_instructions: str = ""
    opponent_scouting: list[OpponentScoutingConfig] = Field(default_factory=list)

# experiment configurations
class TeamMatchupConfig(BaseModel):
    label: str
    coordination: CoordinationConfig

class MatchupConfig(BaseModel):
    home: TeamMatchupConfig
    away: TeamMatchupConfig

class ExperimentConfig(BaseModel):
    experiment_id: str
    matchup: MatchupConfig
    seeds: list[int]
    swap_first_possession: bool = True
    max_concurrent_games: int = 4

# loader functions
def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_llm_config(path: str | Path = "configs/llm.yaml") -> LLMConfig:
    data = load_yaml(path)
    return LLMConfig(**data)

def load_simulation_config(path: str | Path = "configs/simulation.yaml") -> SimulationConfig:
    data = load_yaml(path)
    return SimulationConfig(**data)

def load_roster_config(path: str | Path = "configs/teams/roster.yaml") -> RosterConfig:
    data = load_yaml(path)
    return RosterConfig(**data)

def load_initial_memory_config(
    path: str | Path = "configs/teams/initial_memory.yaml",
) -> InitialMemoryConfig:
    data = load_yaml(path)
    return InitialMemoryConfig(**data)

def load_experiment_config(path: str | Path) -> ExperimentConfig:
    data = load_yaml(path)
    return ExperimentConfig(**data)
