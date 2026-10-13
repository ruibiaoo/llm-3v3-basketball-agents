"""WS1 close-out proofs (owned by WS1): overtime, free throws, fuzz, calibration."""

from bball3x3.agents.random_player import RandomPlayer
from bball3x3.config import SimulationConfig
from bball3x3.engine.abilities import default_abilities
from bball3x3.engine.court import Court
from bball3x3.engine.invariants import check_state
from bball3x3.engine.loop import play_heuristic_game
from bball3x3.engine.resolvers import resolve_free_throw
from bball3x3.engine.rng import GameRNG
from bball3x3.engine.rules import apply_checkball, apply_missed_shot, apply_referee_decision
from bball3x3.engine.state import create_initial_state
from bball3x3.schemas.referee import RefereeDecision


def test_forced_overtime_full_game():
    """A tied clock-expiry game must enter OT and close at plus two."""
    config = SimulationConfig()
    start = create_initial_state(config, GameRNG.from_seed(3), first_possession="A")
    start.score = {"A": 12, "B": 12}
    start.game_clock_s = 0.0
    apply_missed_shot(start)  # LOOSE ball: the first event rebounds, never scores.

    summary = play_heuristic_game(seed=11, config=config, start_state=start)

    assert summary["overtime_reached"] is True
    assert summary["phase"] == "OT"
    assert summary["violations"] == []
    assert summary["events"] < 1500
    assert max(summary["score"].values()) >= 14


def test_free_throw_and_checkball_path():
    """Shooting fouls, the bonus ladder, and-one, and check-ball recovery."""
    config = SimulationConfig()
    court = Court(config.court)
    state = create_initial_state(config, GameRNG.from_seed(9), first_possession="A")
    abilities = {pid: default_abilities(7) for pid in state.players}
    rng = GameRNG.from_seed(9)

    def foul(fouls_before: int, **kwargs):
        state.team_fouls["B"] = fouls_before
        return apply_referee_decision(
            state,
            RefereeDecision(
                call="FOUL", offender="B1", fouled_player="A1",
                foul_type=kwargs.pop("foul_type", "DEFENSIVE_FOUL"),
            ),
            config,
            **kwargs,
        )

    shooting = foul(0, foul_type="SHOOTING_FOUL", was_shooting=True,
                    shot_value=1, made_shot=False)
    assert shooting["free_throws"] == 1
    assert shooting["foul_team"] == "B"
    made = 0
    for _ in range(shooting["free_throws"]):
        outcome, terms, _ = resolve_free_throw(
            shooter_id="A1", abilities_map=abilities, config=config, rng=rng,
            fatigue=state.players["A1"].fatigue,
        )
        assert 0.0 <= terms["p_make"] <= 1.0
        if outcome["made"]:
            state.score["A"] += 1
            made += 1
    assert made in (0, 1)

    seventh = foul(6, was_shooting=False, shot_value=1, made_shot=False)
    assert seventh["free_throws"] == 2
    assert seventh["retain_possession"] is False

    tenth = foul(9, was_shooting=False, shot_value=1, made_shot=False)
    assert tenth["free_throws"] == 2
    assert tenth["retain_possession"] is True

    and_one = foul(2, was_shooting=True, shot_value=1, made_shot=True)
    assert and_one["free_throws"] == 1
    assert and_one["retain_possession"] is False

    state.phase = "DEAD"
    apply_checkball(state, config)
    assert state.phase == "LIVE"
    assert state.needs_clear is False
    assert state.shot_clock_s == config.rules.shot_clock_s
    assert state.ball.holder is not None
    assert check_state(state, court, config) == []


def test_thousand_game_fuzz_sir_zero():
    """1,000 random-agent games: zero invariant violations, none truncated."""
    config = SimulationConfig()
    total_violations = 0
    total_events = 0
    truncated = 0
    for game_seed in range(1000):
        def factory(pid: str, _seed: int = game_seed) -> RandomPlayer:
            number = int(pid[1:])
            offset = number if pid[0] == "A" else 3 + number
            return RandomPlayer(pid, seed=_seed * 1000 + offset)

        summary = play_heuristic_game(
            seed=game_seed, config=config, agent_factory=factory
        )
        total_violations += len(summary["violations"])
        total_events += summary["events"]
        if summary["events"] >= 2000:
            truncated += 1
    print(f"\n1000-game fuzz: {total_events} events, "
          f"violations={total_violations}, truncated={truncated}")
    assert truncated == 0
    assert total_violations == 0


def test_calibration_readout():
    """Heuristic-vs-heuristic scoring shape over 40 seeds.

    Reference readout (seed 0-39, defense moves, lane-aware policy):
    FG1 0.316 (744/2356), FG2 0.472 on 36 attempts, PPP per
    team-possession ~0.18-0.19, ~52 team-possessions and ~491 events per
    game, 4 natural overtimes, zero violations, zero truncations.
    Tuning verdict: in-arc finishing runs low against the ~0.5 FIBA range
    and deep attempts are scarce; PPP target is ~0.9. Candidates are the
    contest curve under trailing defense, shot-type mix, and the open-shot
    threshold. Bounds below admit today and fail on regression.
    """
    games = [
        play_heuristic_game(seed=seed, config=SimulationConfig())
        for seed in range(40)
    ]
    att1 = sum(g["shots_1pt"]["att"] for g in games)
    made1 = sum(g["shots_1pt"]["made"] for g in games)
    att2 = sum(g["shots_2pt"]["att"] for g in games)
    made2 = sum(g["shots_2pt"]["made"] for g in games)
    fg1 = made1 / max(att1, 1)
    fg2 = made2 / max(att2, 1)
    ppp = {
        team: sum(g["score"][team] / max(sum(g["possessions"].values()), 1)
                  for g in games) / len(games)
        for team in ("A", "B")
    }
    poss = sum(sum(g["possessions"].values()) for g in games) / len(games)
    events = sum(g["events"] for g in games) / len(games)
    print(f"\ncalibration (n=40): FG1={fg1:.3f} ({made1}/{att1}), "
          f"FG2={fg2:.3f} ({made2}/{att2}), "
          f"PPP_A={ppp['A']:.3f} PPP_B={ppp['B']:.3f}, "
          f"poss/game={poss:.1f}, events/game={events:.1f}, "
          f"OT={sum(g['overtime_reached'] for g in games)}, "
          f"viol={sum(len(g['violations']) for g in games)}")

    assert att1 > 0 and att2 > 0
    assert 0.30 <= fg1 <= 0.80
    assert 0.20 <= fg2 <= 0.75
    assert 0.02 <= ppp["A"] <= 0.60
    assert 0.02 <= ppp["B"] <= 0.60
    assert 40 <= poss <= 250
    assert 50 <= events < 1500
    assert all(g["events"] < 2000 for g in games)
    assert sum(len(g["violations"]) for g in games) == 0
