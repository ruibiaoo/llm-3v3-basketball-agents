# Basketball Simulation Engine (WS1)

## Overview

Core simulation engine for the 3v3 basketball multi-agent project (WS1:
Simulation Engine and FIBA 3x3 Rules, `proposal/IMPLEMENTATION_PLAN.md`
section 6 / WS1).

The engine is the source of truth. LLMs only propose intent; every outcome
comes from these deterministic, seeded modules. The engine runs fully
without any LLM: heuristic and random agents play complete games.

Shared Pydantic schemas (`bball3x3.schemas`, WS0) are frozen and reused
everywhere. Player ability ratings live only in `configs/teams/roster.yaml`;
resolvers take an explicit `abilities_map` (see `abilities.py`) so no schema
change was needed.

## 1. Implemented Modules

### `court.py` — Court Geometry

15x11 grid at 1 m per cell. Arc test at 6.75 m Euclidean from the rim.
Named zones (`TOP`, `LEFT_WING`, `RIGHT_WING`, `LEFT_CORNER`,
`RIGHT_CORNER`, `LEFT_ELBOW`, `RIGHT_ELBOW`, `LEFT_BLOCK`, `RIGHT_BLOCK`,
`PAINT`) with 1-pt / 2-pt values. Tests: `tests/test_court.py`.

### `rng.py` — Reproducible Randomness

`GameRNG.from_seed(seed)` spawns independent NumPy streams via
`SeedSequence.spawn()` for `shooting`, `passing`, `driving`, `rebounding`,
`contact`, `free_throws`, `first_possession`. Matched seeds give common
random numbers across conditions. Tests: `tests/test_rng.py`.

### `state.py` — Game State Management

`create_initial_state(config, rng, first_possession=None)` builds six
players (A1-A3, B1-B3), initial positions, matchups, scores, fouls, clocks,
ball holder and `needs_clear`. Helpers `set_ball_holder`,
`switch_possession`, `move_player`. Tests: `tests/test_engine_state.py`.

### `clock.py` — Game and Shot Clocks

600 s regulation clock, 12 s shot clock, configurable per-action durations
for all 24 actions (ball-handler actions 1-2 s, everything else 1 s).
Clocks clamp at zero and stop in `DEAD`/`FT` phases; `OT` decrements the
shot clock only. Expiry never applies rules by itself — that is `rules.py`.
Tests: `tests/test_clock.py`.

### `actions.py` — Action Validation and Remapping (spec section 2-3)

All 24 actions in ball-holder / off-ball / defensive sets, plus a rebound
phase (`CRASH_BOARDS`, `BOX_OUT`, `GET_BACK`, `MOVE`, `HOLD`) whenever the
ball is `LOOSE`. `CLEAR` is legal only while `needs_clear`; `SHOOT` is
withheld until cleared. `SWITCH` targets a teammate to swap assignments
with. Invalid proposals remap to spec fallbacks — ball handler to `HOLD`
(safe `PASS` when the shot clock is <= 2), off-ball offense to `SPOT_UP`,
defender to `DEFEND`, rebound phase to `BOX_OUT` — each logged with
`invalid_reason` in `PARSE_FAIL | UNKNOWN_ACTION | ILLEGAL_POSSESSION |
INFEASIBLE | BAD_TARGET`. Tests: `tests/test_engine_actions.py`.

### `abilities.py` — Roster Lookup (schemas stay frozen)

`build_abilities_map(roster)` mirrors slots P1-P3 onto A1-A3 and B1-B3.
`get_abilities(pid, map)` falls back to flat defaults for tests/fuzzing.

### `shot_quality.py` — Shot Model (spec section 5)

`P(make) = sigmoid(beta0 + beta1*shooting + beta2*shotquality -
beta3*contest - beta4*fatigue)` with betas from `simulation.yaml`
(`-1.6, 0.25, 2.0, 1.5, 0.08`). `shotquality = w_loc*location +
w_type*shot_type + w_scr*screen` over `RIM | SHORT | MID | ARC2 | DEEP2`
and `LAYUP_OFF_DRIVE | CATCH_AND_SHOOT | POST | OFF_DRIBBLE |
FORCED_LATE_CLOCK` (forced when the shot clock is <= 2). Contest from
nearest-defender distance, `(1 + b_def*(defense-5))`, plus `CONTEST`-action
and help bonuses, clipped to [0, 1]. `describe_shot` + `finalize_shot_terms`
return every term for the event log. Tests: `tests/test_shot_quality.py`.

### `resolvers.py` — Action Outcomes (spec section 6)

Logistic models with explicit coefficients: pass (passing, distance, lane
defenders — failure is a steal if a defender is in the lane, else OOB),
drive (`ball_handling` vs `defense` + help; success sets up
`LAYUP_OFF_DRIVE`, failure is a stop/strip), screen (65 % success flags
`screen_advantage`), rebound (weighted lottery over proximity,
`rebounding`, `BOX_OUT`/`CRASH_BOARDS`, stamina), steal (`defense` vs
`ball_handling`, contact risk on `reach_prob`), free throw (shooting minus
fatigue). `update_fatigue` applies `per_s_active / per_s_sprint`,
stamina-scaled, with dead-ball recovery, capped at 10. Every resolver
returns `(outcome, terms, rng_draws)`. Tests: `tests/test_resolvers.py`.

### `contact.py` — Contact Detection (spec step 5)

Annotates only; the referee judges. Draws from `contact` coefficients:
drive-contest 0.25, screen 0.15, rebound 0.10, reach 0.20. Returns a
`ContactEvent` plus `build_referee_state` (Code 3) and the deterministic
templated query for WS5 RAG. Tests: `tests/test_contact.py`.

### `rules.py` — FIBA 3x3 Rules (spec section 7)

Objective rules only: 12 s shot-clock resets on possession change and
offensive rebound (simplified to full reset); automatic `CLEAR` behind the
arc after defensive rebounds, steals and made baskets; check-ball reset;
team-foul ladder (7th-9th: 2 FTs; 10th+: 2 FTs + possession; shooting foul
1 FT inside / 2 FTs beyond; and-one 1 FT); end at 600 s or first to 21,
tie at time to overtime (first to +2 in OT); shot-clock expiry as an
objective turnover. `apply_referee_decision` applies judged `FOUL` /
`VIOLATION` calls (`NONE` is a no-op). Tests: `tests/test_rules.py`.

### `invariants.py` — 20 Constraints (proposal Table 3, spec section 10)

All 20 checks as `check_state` (single snapshot) + `check_transition`
(prev, new, event). `collect_violations` returns SIR-logging dicts;
`check` raises `InvariantViolationError` (dev/test mode). Transition
context (`scored`, `possession_reason`, `reset_reason`, `rebound_after_miss`,
`turnover`, `foul_called`, …) disambiguates legal resets from violations.
Tests: `tests/test_invariants.py`.

### `movement.py` — Direction / Speed Labels (spec section 9)

Derives Code 1 labels from `prev_pos -> pos` over elapsed time:
`HOLD | CUTTING_LEFT/RIGHT/UP/DOWN | DRIVE_LEFT/RIGHT |
CUTTING_TOWARD_BASKET` and `STATIONARY | NONE | SLOW | MEDIUM | FAST`.
The WS3 observation builder consumes these. Covered in
`tests/test_agents.py`.

### `loop.py` — WS1 Validation Harness (not the WS7 runner)

`play_heuristic_game(seed, config, max_events)` runs the spec resolution
order — ball handler vs primary defender, screens, off-ball drift, contact
annotation, rules, fatigue, invariants — with heuristic agents on all six
players, OT handling and shot-clock turnovers. Used by the smoke test and
available to the future asyncio runner.

### Agents — `src/bball3x3/agents/` (spec section 12)

`HeuristicPlayer`: clears when required, shoots when open (>= 3 m),
passes to the most open teammate, drives otherwise; off-ball cuts or
screens for a contested holder; defenders contest their matchup inside
2 m, else defend/help; rebound phase crashes when closest, else boxes out.
`RandomPlayer`: uniform over legal actions with its own seeded RNG
(fuzzing). `BasePlayer` holds shared helpers (`ball_handler`,
`primary_defender`, `teammates`, `opponents`). Tests:
`tests/test_agents.py`, `tests/test_loop_smoke.py`.

## 2. Validation (agreed scope: unit tests + full pytest + smoke test)

From the repository root:

```powershell
uv run pytest -v
uv run ruff check .
```

New WS1 coverage: `test_shot_quality`, `test_resolvers`, `test_contact`,
`test_rules`, `test_invariants`, `test_agents`, `test_loop_smoke`,
`test_display`, `test_ws1_closeout`
(plus the spec-fallback update in `test_engine_actions`).

Current status: **145 passed, 1 skipped (matplotlib absent), ruff clean.**
Smoke reference (`play_heuristic_game(seed=7)`): **281 events, final 22-17,
SIR = 0.0, IAR = 0.0** (heuristic agents propose only legal actions, so
IAR 0 is expected; IAR > 0 will come from LLM/fuzz traffic). Close-out
proofs (`tests/test_ws1_closeout.py`): forced-overtime game enters OT and
closes at plus two; free-throw ladder (shooting, 7th, 10th, and-one) plus
check-ball recovery; 1,000 random-agent games over 411,043 events with
SIR = 0 and zero truncations; calibration readout FG1 0.316, FG2 0.472 on
scarce deep attempts, with PPP well below the ~0.9 FIBA target — in-arc
finishing under trailing defense stays open (see item 3 below).

## 3. Integration Notes

- `schemas/state.py`: game-state structures (frozen, WS0).
- `schemas/actions.py`: proposals and remap records.
- `config.py` + `configs/simulation.yaml`: every coefficient the resolvers
  use; each engine module ships defaults matching the YAML so
  `SimulationConfig()` works in tests.
- `court.py`: geometry for resolution distances and shot values.
- `rng.py`: seeded streams per resolver; matched seeds share draws.
- `state.py`: init and mutations; abilities arrive via `abilities_map`.
- `clock.py`: advances time only; `rules.py` owns expiry consequences.
- `actions.py`: validates before `resolvers.py`; each remap carries its
  reason for IAR.
- `contact.py`: annotates; WS5 judges every resolved state.
- `rules.py` + `invariants.py` + `loop.py`: close the event with
  score/possession/clocks/fatigue and the 20 checks.
- `display/` (read-only visualisation) consumes `GameState` snapshots and
  WS6 `state`/`resolution` records; the engine never imports it.

## 4. Remaining Work (outside this WS1 pass)

1. Full `bball run-game` CLI and asyncio game runner (WS7) reusing
   `loop.py` order with logging, referee calls and LLM agents.
2. Rule-based referee running every state (WS5-A) wired into the loop.
3. Resolver calibration follow-up: the readout (FG1 0.316, FG2 0.472 on
   scarce attempts, PPP ~0.18-0.19 vs ~0.9 target) points at in-arc
   finishing under trailing defense and deep-shot scarcity; tune the
   contest curve, shot-type mix, and the open-shot threshold, then add
   coefficient sensitivity ±20 %.
4. Done (WS1-owned): 1,000 random-agent games, 411,043 events, SIR = 0,
   zero truncations (`tests/test_ws1_closeout.py`).
5. Movement-limit thresholds against real `time_costs_s` per action and
   zone-target reachability once proposals carry zone targets.
