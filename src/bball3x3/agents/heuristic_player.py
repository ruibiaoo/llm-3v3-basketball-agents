"""Heuristic baseline: shoots when open, passes to open teammates (WS1)."""

from bball3x3.agents.base import (
    BasePlayer,
    ball_handler,
    opponents,
    primary_defender,
    teammates,
)
from bball3x3.config import SimulationConfig
from bball3x3.engine.actions import get_legal_actions
from bball3x3.engine.court import Court
from bball3x3.engine.resolvers import lane_defenders
from bball3x3.engine.shot_quality import compute_contest
from bball3x3.schemas.actions import ActionProposal
from bball3x3.schemas.state import GameState

OPEN_DIST_M = 3.0
CONTEST_DIST_M = 2.0


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _nearest_opp_dist(state: GameState, player_id: str) -> tuple[str | None, float]:
    me = state.players[player_id]
    best, best_d = None, float("inf")
    for pid in opponents(state, player_id):
        d = _dist(me.pos, state.players[pid].pos)
        if d < best_d:
            best, best_d = pid, d
    return best, best_d


class HeuristicPlayer(BasePlayer):
    """Deterministic non-LLM policy for engine validation (SIR = 0)."""

    def __init__(self, player_id: str, config: SimulationConfig | None = None):
        super().__init__(player_id)
        self.config = config or SimulationConfig()
        self.court = Court(self.config.court)
        self.lane_width = float(
            (self.config.pass_model or {}).get("lane_width_cells", 1.0)
        )

    def decide(self, state: GameState, **kwargs) -> ActionProposal:
        _ = kwargs
        legal = get_legal_actions(state, self.player_id)
        if not legal:
            return ActionProposal(player_id=self.player_id, action="HOLD")
        me = state.players[self.player_id]

        # Rebound phase: chase or box out.
        if state.ball.status == "LOOSE":
            return self._rebound_decision(state, legal)

        if me.team != state.offense:
            return self._defense_decision(state, legal)

        if me.has_ball:
            return self._ball_handler_decision(state, legal)
        return self._off_ball_decision(state, legal)

    def _ball_handler_decision(
        self, state: GameState, legal: set[str]
    ) -> ActionProposal:
        if state.needs_clear and "CLEAR" in legal:
            return ActionProposal(player_id=self.player_id, action="CLEAR")
        _, nearest_d = _nearest_opp_dist(state, self.player_id)
        if state.shot_clock_s <= 2.0:
            if "SHOOT" in legal:
                return ActionProposal(player_id=self.player_id, action="SHOOT")
            return self._safe_pass(state, legal)
        contest_est = compute_contest(distance_m=nearest_d, config=self.config)
        if contest_est < 0.5 and "SHOOT" in legal:
            return ActionProposal(player_id=self.player_id, action="SHOOT")
        clear_mate = self._clearest_lane_mate(state)
        if clear_mate is not None and "PASS" in legal:
            return ActionProposal(
                player_id=self.player_id, action="PASS", target=clear_mate
            )
        if "DRIVE" in legal and nearest_d >= 1.0:
            return ActionProposal(player_id=self.player_id, action="DRIVE")
        # No clear lane and no drive: hold rather than donate possession.
        # The shot clock forces action at <= 2 s (handled above).
        return ActionProposal(player_id=self.player_id, action="HOLD")

    def _clearest_lane_mate(self, state: GameState) -> str | None:
        """Return the most open teammate whose passing lane is empty."""
        best, best_key = None, None
        for mate in teammates(state, self.player_id):
            lane = len(
                lane_defenders(
                    state, self.player_id, mate,
                    lane_width_cells=self.lane_width,
                )
            )
            _, openness = _nearest_opp_dist(state, mate)
            key = (lane, -openness)
            if best_key is None or key < best_key:
                best, best_key = mate, key
        if best is not None and len(
            lane_defenders(
                state, self.player_id, best, lane_width_cells=self.lane_width
            )
        ) == 0:
            return best
        return None

    def _safe_pass(self, state: GameState, legal: set[str]) -> ActionProposal:
        if "PASS" not in legal:
            return ActionProposal(player_id=self.player_id, action="HOLD")
        best, best_key = None, None
        for mate in teammates(state, self.player_id):
            lane = len(
                lane_defenders(
                    state, self.player_id, mate,
                    lane_width_cells=self.lane_width,
                )
            )
            _, openness = _nearest_opp_dist(state, mate)
            key = (lane, -openness)
            if best_key is None or key < best_key:
                best, best_key = mate, key
        mate = best or (
            teammates(state, self.player_id)[0]
            if teammates(state, self.player_id)
            else None
        )
        return ActionProposal(
            player_id=self.player_id, action="PASS", target=mate
        )

    def _most_open_teammate(self, state: GameState) -> str | None:
        best, best_d = None, -1.0
        for mate in teammates(state, self.player_id):
            _, d = _nearest_opp_dist(state, mate)
            if d > best_d:
                best, best_d = mate, d
        return best

    def _off_ball_decision(
        self, state: GameState, legal: set[str]
    ) -> ActionProposal:
        holder = ball_handler(state)
        if holder is None:
            if "SPOT_UP" in legal:
                return ActionProposal(player_id=self.player_id, action="SPOT_UP")
            return ActionProposal(player_id=self.player_id, action="HOLD")
        _, holder_d = _nearest_opp_dist(state, holder)
        defender_of_holder = primary_defender(state, holder)
        if (
            holder_d < CONTEST_DIST_M
            and defender_of_holder
            and "SCREEN" in legal
            and defender_of_holder in opponents(state, self.player_id)
        ):
            mate = next(
                (m for m in teammates(state, self.player_id) if m != self.player_id),
                None,
            )
            _ = mate
            # SCREEN targets a teammate (the screener helps them); pick holder.
            return ActionProposal(
                player_id=self.player_id, action="SCREEN", target=holder
            )
        if "CUT" in legal or "POP" in legal or "SPOT_UP" in legal:
            rim_dist = self.court.distance_to_basket(
                state.players[self.player_id].pos
            )
            if rim_dist < 3.0 and "POP" in legal:
                # Crowded paint: pop out and restore spacing.
                return ActionProposal(player_id=self.player_id, action="POP")
            if "CUT" in legal:
                return ActionProposal(player_id=self.player_id, action="CUT")
            if "SPOT_UP" in legal:
                return ActionProposal(player_id=self.player_id, action="SPOT_UP")
        return ActionProposal(player_id=self.player_id, action="HOLD")

    def _defense_decision(
        self, state: GameState, legal: set[str]
    ) -> ActionProposal:
        holder = ball_handler(state)
        my_matchup = state.matchups.get(self.player_id)
        if holder and my_matchup == holder:
            d = _dist(state.players[self.player_id].pos, state.players[holder].pos)
            if d <= CONTEST_DIST_M and "CONTEST" in legal:
                return ActionProposal(
                    player_id=self.player_id, action="CONTEST", target=None
                )
            if "DEFEND" in legal:
                return ActionProposal(
                    player_id=self.player_id, action="DEFEND", target=None
                )
        if my_matchup and my_matchup in state.players:
            if "DEFEND" in legal:
                return ActionProposal(
                    player_id=self.player_id, action="DEFEND", target=None
                )
        if "HELP_PAINT" in legal and holder is not None:
            return ActionProposal(player_id=self.player_id, action="HELP_PAINT")
        return ActionProposal(player_id=self.player_id, action="HOLD")

    def _rebound_decision(
        self, state: GameState, legal: set[str]
    ) -> ActionProposal:
        d_ball = _dist(state.players[self.player_id].pos, state.ball.pos)
        dists = sorted(
            _dist(p.pos, state.ball.pos) for p in state.players.values()
        )
        closest = dists[0] if dists else d_ball
        if d_ball <= closest + 0.5 and "CRASH_BOARDS" in legal:
            return ActionProposal(player_id=self.player_id, action="CRASH_BOARDS")
        if "BOX_OUT" in legal:
            return ActionProposal(player_id=self.player_id, action="BOX_OUT")
        return ActionProposal(player_id=self.player_id, action="HOLD")
