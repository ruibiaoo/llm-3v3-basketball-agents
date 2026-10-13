"""Roster-to-state ability lookups (WS1).

WS0 schemas are frozen: :class:`PlayerState` carries no ability ratings.
Resolvers therefore take an explicit ``abilities_map`` built from
``configs/teams/roster.yaml``. Slots ``P1..P3`` are mirrored onto both
teams (``A1 -> P1``, ``B1 -> P1``, ...).
"""

from bball3x3.config import RosterConfig
from bball3x3.schemas.state import Abilities


def default_abilities(value: int = 5) -> Abilities:
    """Return a flat ability profile (useful for tests and fuzzing)."""
    return Abilities(
        shooting=value,
        passing=value,
        defense=value,
        stamina=value,
        ball_handling=value,
        rebounding=value,
    )


def build_abilities_map(roster: RosterConfig) -> dict[str, Abilities]:
    """Mirror roster slots P1..P3 onto game ids A1..A3 and B1..B3."""
    by_slot = {p.slot: p.abilities for p in roster.players}
    abilities_map: dict[str, Abilities] = {}
    for team in ("A", "B"):
        for slot, abilities in by_slot.items():
            try:
                number = int(slot[1:])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid roster slot: {slot}") from exc
            abilities_map[f"{team}{number}"] = abilities
    return abilities_map


def get_abilities(
    player_id: str,
    abilities_map: dict[str, Abilities] | None,
    fallback: int = 5,
) -> Abilities:
    """Return abilities for a player, falling back to flat defaults."""
    if abilities_map is not None and player_id in abilities_map:
        return abilities_map[player_id]
    return default_abilities(fallback)
