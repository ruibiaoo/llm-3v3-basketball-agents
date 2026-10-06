
from bball3x3.config import (
    load_experiment_config,
    load_initial_memory_config,
    load_llm_config,
    load_roster_config,
    load_simulation_config,
)


def test_load_llm_config():
    cfg = load_llm_config("configs/llm.yaml")
    assert cfg.players.model == "openai/gpt-oss-120b"
    assert cfg.referee.backend == "rule_based"
    assert cfg.runtime.max_concurrent_requests == 12

def test_load_simulation_config():
    cfg = load_simulation_config("configs/simulation.yaml")
    assert cfg.court.width_m == 15.0
    assert cfg.court.depth_m == 11.0
    assert cfg.rules.win_score == 21
    assert cfg.rules.shot_clock_s == 12.0
    assert cfg.time_costs_s["PASS"] == 1
    assert cfg.shot_model["beta0"] == -1.6

def test_load_roster_config():
    cfg = load_roster_config("configs/teams/roster.yaml")
    assert cfg.captain == "P1"
    assert len(cfg.players) == 3
    p1 = cfg.players[0]
    assert p1.slot == "P1"
    assert p1.abilities.shooting == 7
    assert p1.abilities.rebounding == 6

def test_load_initial_memory_config():
    cfg = load_initial_memory_config("configs/teams/initial_memory.yaml")
    assert "Attack switches" in cfg.game_plan or "attack switches" in cfg.game_plan
    assert len(cfg.opponent_scouting) == 2
    assert cfg.opponent_scouting[0].opponent_slot == "P1"

def test_load_experiment_configs():
    for f in [
        "configs/experiments/pilot.yaml",
        "configs/experiments/ablation_comms.yaml",
        "configs/experiments/ablation_captain.yaml",
        "configs/experiments/ablation_memory.yaml",
    ]:
        cfg = load_experiment_config(f)
        assert cfg.experiment_id is not None
        assert len(cfg.seeds) > 0
        assert cfg.matchup.home.label == "A_full"
