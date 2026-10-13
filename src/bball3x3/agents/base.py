"""Non-LLM baseline agents (WS1 spec section 12)."""

from bball3x3.schemas.actions import ActionProposal
from bball3x3.schemas.state import GameState


class BasePlayer:
    """Minimal player interface used by the WS1 validation loop."""

    def __init__(self, player_id: str):
        self.player_id = player_id

    @property
    def team(self) -> str:
        return self.player_id[0]

    def decide(self, state: GameState, **kwargs) -> ActionProposal:  # pragma: no cover
        raise NotImplementedError


def primary_defender(state: GameState, ball_handler_id: str) -> str | None:
    """Return the defender assigned to the ball handler, if any."""
    for defender, matchup in state.matchups.items():
        if matchup == ball_handler_id and defender in state.players:
            if state.players[defender].team != state.players[ball_handler_id].team:
                return defender
    return None


def ball_handler(state: GameState) -> str | None:
    for pid, player in state.players.items():
        if player.has_ball:
            return pid
    return None


def teammates(state: GameState, player_id: str) -> list[str]:
    team = state.players[player_id].team
    return [pid for pid, p in state.players.items() if p.team == team and pid != player_id]


def opponents(state: GameState, player_id: str) -> list[str]:
    team = state.players[player_id].team
    return [pid for pid, p in state.players.items() if p.team != team]
