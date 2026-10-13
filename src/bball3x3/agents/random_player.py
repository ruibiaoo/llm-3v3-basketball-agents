"""Uniform-random baseline for fuzzing (WS1)."""

import numpy as np

from bball3x3.agents.base import BasePlayer, opponents, teammates
from bball3x3.engine.actions import TARGET_PLAYER_ACTIONS, get_legal_actions
from bball3x3.schemas.actions import ActionProposal
from bball3x3.schemas.state import GameState


class RandomPlayer(BasePlayer):
    """Pick uniformly from legal actions. Own RNG so fuzzing is reproducible."""

    def __init__(self, player_id: str, seed: int = 0):
        super().__init__(player_id)
        self._rng = np.random.default_rng(seed + hash(player_id) % (2**31))

    def decide(self, state: GameState, **kwargs) -> ActionProposal:
        _ = kwargs
        try:
            legal = sorted(get_legal_actions(state, self.player_id))
        except ValueError:
            return ActionProposal(player_id=self.player_id, action="HOLD")
        if not legal:
            return ActionProposal(player_id=self.player_id, action="HOLD")
        action = str(self._rng.choice(legal))
        target = None
        if action in TARGET_PLAYER_ACTIONS:
            target = self._random_target(state, action)
            if target is None:
                return ActionProposal(player_id=self.player_id, action="HOLD")
        return ActionProposal(
            player_id=self.player_id, action=action, target=target
        )

    def _random_target(self, state: GameState, action: str) -> str | None:
        if action in ("PASS", "SCREEN"):
            options = teammates(state, self.player_id)
        elif action in ("DOUBLE_TEAM", "DENY"):
            options = opponents(state, self.player_id)
        elif action == "SWITCH":
            options = teammates(state, self.player_id)
        else:
            options = []
        if not options:
            return None
        return str(self._rng.choice(sorted(options)))
