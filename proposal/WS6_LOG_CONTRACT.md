# WS6 Event Log Contract: First Delivery

**Status:** core schemas and JSONL writer/reader are implemented; this contract remains open for team review and producer integration.

**Scope:** define the records needed to compute points per possession (PPP), invalid action rate (IAR), and state inconsistency rate (SIR) from one saved game log.

**Sources:** [implementation plan, WS6](IMPLEMENTATION_PLAN.md#ws6--event-logging-metrics-and-analysis), [WS6 delivery plan](WS6_PLAN.md), and the existing [log schemas](../src/bball3x3/schemas/log.py). This document makes their counting and timing rules explicit. It does not change the simulation rules.

## 1. Deliverable and ownership

The runner writes one UTF-8 JSONL file per game. Each line contains one JSON object with a `record_type`. WS6 reads this file to calculate metrics without running the engine or calling an LLM.

| Component | Supplies | Responsibility |
|---|---|---|
| Runner | Game identity, event numbering, initial/final state, completion | Assemble records and write them in the defined order |
| WS3: player agents | Raw response, parsed proposal, final parse status | Preserve what each player proposed |
| WS1: engine | Legality result, selected action/target, outcomes, scoring, possession changes, invariant results | Supply authoritative facts; WS6 does not infer these from natural-language text |
| WS5: referee | Judgement for each event | Supply the existing `referee` record; engine applies the judgement before final outcomes are logged |
| WS6 | Record definitions, writer/reader, log validation, possession accounting and metrics | Check that records connect correctly and calculate results |

The runner owns the file. Concurrent agent tasks return data to the runner; they do not independently append to the same file. The group still needs to identify the runner integration owner.

## 2. Identity, timing and ordering

### 2.1 Identifiers

| Identifier | Convention |
|---|---|
| `game_id` | Unique within the experiment. Stored in `game_start`; all following lines inherit it until `game_end` |
| `experiment_id` | Identifies the experiment, including a separate identifier for synthetic fixtures or pilots |
| `seed` | Simulation seed; not a unique game identifier |
| `replicate_id` | Distinguishes games sharing a seed, including the two first-possession variants |
| `event_idx` | Integer starting at 1 for the first decision event; increases by exactly 1 for each completed transition |
| Decision identity | `(game_id, event_idx, player)`; exactly one final decision per active player per event |
| Transition identity | `(game_id, event_idx)`; no additional transition counter is needed for this first delivery |
| `possession_id` | Unique within a game, e.g. `p0001`; never reused, including in overtime |
| Team identity | Physical teams are `A` and `B`. `team_conditions` maps them to experimental labels such as `A_full` and `D_read_only` |

Do not confuse physical team `B` with the no-communication condition. The away team can represent any ablation condition.

### 2.2 Event timeline

The initial state is `S0`, with `GameState.event_idx = 0`, stored in `game_start.initial_state`. It is not a completed transition and is not a SIR denominator entry.

For event `e`:

1. Build six observations from state `S(e-1)` and information available at that point.
2. Collect the six final player proposals. Internal parse-repair attempts remain part of the same decision.
3. Check legality and select safe replacements for invalid proposals.
4. Write six finalized `decision` records with `event_idx = e`.
5. Resolve actions, obtain the referee judgement, and apply scoring, possession, clock and other rules.
6. Write the `referee` record and one final `resolution` record for `e`.
7. Run all 20 invariants. Write zero or more `invariant_violation` records for `e`.
8. Write one `state` record for `e`, containing `Se` and confirmation that invariant checking completed.
9. Write any `possession_end` records closed by this transition, in closure order.
10. If the game ended normally, write `game_end` as the final line. Otherwise continue with event `e+1`.

The `state` record marks a completed transition. Its outer `event_idx` and inner `state.event_idx` must agree. The state used to choose actions at event 6 is state 5; the state produced by those actions is state 6.

The implementation plan's pseudocode currently logs decisions before legality checking. Integration must buffer the decision until the legality result is available, so `invalid_reason = null` means checked and valid, not not-yet-checked. Records are not edited after writing.

Additional message, captain and memory records may occur between these records. They must not change this ordering of core records. For later coordination metrics, messages emitted at event `e` first become visible before the decisions at `e+1`; they cannot affect decisions at `e`.

## 3. Core record definitions

Unless marked optional, the fields below are required for this contract. Existing usage/diagnostic fields remain available. Defaults in today's Pydantic models do not establish that an omitted fact was measured.

### 3.1 `game_start`

**Purpose:** identify the run, starting state and experimental conditions. **Producer:** runner, using validated configuration.

| Fields | Type / meaning |
|---|---|
| `record_type` | Literal `game_start` |
| `schema_version` | Integer `1` for this proposed contract; unversioned existing records are legacy data |
| `game_id`, `experiment_id`, `replicate_id` | Non-empty strings |
| `seed` | Integer |
| `first_possession` | `A` or `B`; agrees with the initial control assignment |
| `team_conditions` | Map of both physical teams to experiment condition labels |
| `matchup`, `coordination`, `roster` | Existing metadata; roster includes the six instantiated player IDs and their physical teams |
| `config_hash`, `config_snapshot` | Hash and complete effective configuration used for this game, excluding credentials |
| `prompt_hashes`, `model_slug` | Existing provenance; use an explicit non-LLM identifier for heuristic/fixture agents |
| `code_commit` | Commit string, or explicit `null` if unavailable; record a dirty-working-tree indicator in the config/run metadata |
| `initial_state` | Full `GameState`, index 0, including both initial scores of 0 and all six players |
| `initial_possession_id` | Identifier of the possession owned by `first_possession` |
| `initial_memory` | `null` if unavailable; otherwise both team keys map to entries with `id`, `opponent`, `category`, and `tendency`. An explicit empty list means that team's starting memory was checked and empty |

Hash the exact saved effective config using a documented canonical JSON encoding (sorted keys, compact separators, UTF-8) and SHA-256. Never include API key values. Configuration hashes alone cannot reconstruct the run.

### 3.2 `decision`

**Purpose:** record one player's final proposal and the engine's legality result. **Producers:** WS3 and WS1, assembled by the runner after validation.

| Fields | Type / meaning |
|---|---|
| `record_type`, `event_idx`, `player` | Literal `decision`, event index, active player ID |
| `team` | Physical team `A` or `B`; must agree with roster membership |
| `raw_output` | Final response text; retain malformed text when parsing fails |
| `parsed_output` | Existing `AgentOutput`, or `null` for a final parse failure |
| `parse_ok` | Explicit boolean; agrees with the parsed output's status when present |
| `invalid_reason` | `null` for a valid proposal; otherwise one existing `InvalidReason` value |
| `remapped_action`, `remapped_target` | Replacement action and target when invalid; both `null` for an unchanged valid proposal. A replacement action may legitimately have a null target |
| `executed_action`, `executed_target` | Action and target handed to the resolver, including any safe replacement. This records selected intent, not guaranteed success |
| `observation_text` | Exact player-facing text supplied before the decision; optional for legacy logs. `null` means not recorded; an empty string is a recorded empty observation |
| `observation_hash` | With text present, SHA-256 of its exact UTF-8 bytes, automatically populated if empty; a supplied mismatch is rejected. Legacy hash-only records remain readable |
| `memory_ref`, `latency_ms`, `tokens`, `cost_usd` | Existing diagnostic fields; not denominators for these three metrics |

`InvalidReason` is one of `PARSE_FAIL`, `UNKNOWN_ACTION`, `ILLEGAL_POSSESSION`, `INFEASIBLE`, or `BAD_TARGET`.

- A final parse failure requires `parse_ok = false`, `parsed_output = null`, and `invalid_reason = "PARSE_FAIL"`.
- A valid parsed proposal requires `parse_ok = true`; its proposed action/target are preserved in `parsed_output`, even when illegal.
- If `invalid_reason` is non-null, a replacement action is required and must match `executed_action`; replacement and executed targets also agree.
- If valid, executed action/target equal the proposed action/target.
- If several legality checks fail, WS1 supplies one primary reason and may put additional details in diagnostic output. IAR counts the decision once regardless.
- A pass intercepted by a defender or a missed shot is an unsuccessful outcome, not an invalid proposal.
- A retry or repair is not a second decision. A request that produces no proposal because execution aborts is an incomplete run, not automatically an invalid basketball action.

### 3.3 `resolution`

**Purpose:** expose final outcomes after referee decisions and rules. **Producer:** WS1, assembled by the runner.

| Fields | Type / meaning |
|---|---|
| `record_type`, `event_idx` | Literal `resolution` and event index |
| `actions` | Existing map of all six player IDs to executed action names; agrees with decisions |
| `action_targets` | Map of those players to executed targets or `null` |
| `outcomes` | Existing per-action outcome details; useful for transcripts and diagnostics |
| `involved_opponents` | Optional map from every acting player ID to the opponent IDs involved in that executed action. Needed to validate grounded MAUR; `null` marks older/incomplete evidence |
| `model_terms`, `rng_draws` | Existing model inputs/probabilities and random draws actually used; empty when none |
| `possession_before`, `possession_after` | Objects `{possession_id, team}`. Before is required; after is `null` on normal game termination |
| `scoring_events` | Ordered list of `{team, points, source, possession_id}`; positive integer points, source e.g. `FIELD_GOAL` or `FREE_THROW`; empty list when no points |
| `possession_changes` | Ordered list of `{from_possession_id, to_possession_id, to_team, reason}` for genuine control transfers; empty if control remains with the same team |
| `game_over`, `game_end_reason` | Boolean and nullable reason; true requires a reason such as `CLOCK_EXPIRED`, `WIN_SCORE`, or `OT_WIN` |

Scoring events are the accounting source for score changes. Their `possession_id` is the accounting possession, including free throws attributed to the possession of the foul. Sum them by team to check the next state's score.

A normal game ending is not a control transfer: set `possession_after = null`, close the active possession, and do not fabricate a next possession. When a control transfer occurs, the ordered change list must connect `possession_before` to `possession_after`. If a transfer and termination occur together, only record a new possession if control actually transferred before termination.

WS1 supplies these facts. WS6 does not infer them from a change in ball holder, a score change, or the `offense` field alone.

### 3.4 `state`

**Purpose:** mark a completed transition and preserve its resulting state. **Producer:** runner after WS1's rules and invariants.

| Fields | Type / meaning |
|---|---|
| `record_type`, `event_idx` | Literal `state` and completed transition index |
| `state` | Full resulting `GameState`, index equal to the outer event index |
| `invariant_checks_completed` | Explicit boolean; must be true for a transition included in reported SIR |
| `constraints_checked` | Unique constraint IDs checked; must equal the set 1 through 20 from proposal Table 3 |

Absence of violation records means zero failures only when all checks completed. A checker exception or skipped check makes SIR unavailable for final analysis; it must not be treated as a pass.

**Invalid snapshot escape path:** experiment mode must be able to record an inconsistent state even if Pydantic rejects it, for example fatigue outside its normal bounds. Extend this record to allow `state = null` with `raw_state` containing the exact JSON-compatible candidate snapshot and `state_validation_errors` containing validation errors. In the normal path, `raw_state = null` and errors are empty. Exactly one of `state` and `raw_state` is present. All invariant results still refer to this same candidate snapshot; do not clamp or repair it before measuring SIR. Unknown/missing invariant results remain incomplete evidence.

### 3.5 `invariant_violation`

**Purpose:** explain one violated constraint. **Producer:** WS1 invariant checker.

Required fields are the existing `record_type = "invariant_violation"`, `event_idx`, `constraint_id` (integer 1-20), `constraint_name`, and non-empty `details`.

Emit one record per violated constraint per transition. Combine multiple findings for the same constraint in `details`. Different constraints can reference the same transition; this does not create multiple inconsistent transitions.

### 3.6 `possession_end`

**Purpose:** finalize a possession for PPP and possession-level analysis. **Producer:** runner/WS6 possession tracker consuming explicit WS1 outcomes.

| Fields | Type / meaning |
|---|---|
| `record_type` | Literal `possession_end` |
| `possession_id`, `team` | Unique possession identifier and the team that owned it |
| `start_event_idx`, `end_event_idx` | Inclusive event bounds; initial possession starts at 1 |
| `points` | Non-negative integer points credited to the possession's owning team |
| `points_by_team` | Map `{A: integer, B: integer}` of all points attributed to this possession; `points` equals the owner's entry |
| `end_reason` | Explicit reason from the engine, e.g. `MADE_BASKET`, `DEFENSIVE_REBOUND`, `TURNOVER`, `CLOCK_EXPIRED`, `WIN_SCORE`, `OT_WIN`, or `REGULATION_END` |

Points are accumulated from `resolution.scoring_events`, not guessed from shot counts. `points_by_team` accommodates exceptional scoring credited to the other team, such as an engine-supported technical free throw, without losing score reconciliation or inventing a possession.

Event bounds describe when control was held, not exclusive decision membership. If control changes during event 2, the outgoing possession ends at 2 and the incoming possession starts at 2, even though the incoming team's first offensive decision may occur at 3. Their bounds can overlap; use possession IDs to assign scoring and the control timeline to interpret events.

### 3.7 `game_end`

**Purpose:** mark normal completion and expose totals for reconciliation. **Producer:** runner.

Required fields: existing `record_type = "game_end"`, `final_score`, `ot_flag`, `totals`, `total_cost_usd`, plus `last_event_idx` and `end_reason`.

Required entries in `totals`:

- `decisions`: total decision records; also `decisions_by_team` for A and B.
- `transitions`: total completed transitions.
- `possessions_by_team`: total closed possessions owned by each team.

These are cross-checks, not substitutes for deriving counts from the underlying records. The final score agrees with the last state and the sum of scoring events. A normal `game_end` means the game finished, not that it had zero invariant violations.

An aborted run has no normal `game_end`. Closing the file or leaving a logger context manager does not create one automatically.

### 3.8 `memory_write`

**Purpose:** record a committed shared-memory mutation separately from the agent proposal. Fields are `event_idx`, `team`, `memory_id`, `opponent`, `category`, `tendency`, `visible_at`, and `evicted_ids`. `visible_at` must be later than the write event; evicted IDs stop being valid references at that event. Initial entries are recorded in `game_start.initial_memory`.

### 3.9 `cs_write`

**Purpose:** record a committed counter-strategy entry separately from a proposal. Fields are `event_idx`, `team`, `key`, `side`, `opponent_strategy`, `counter_strategy`, `merged`, and `visible_at`. `visible_at` must be later than the write event, and a tactic cannot cite the key before that event.

## 4. Possession counting rules

1. **Start:** open the initial possession from `game_start`; open later possessions only when the engine records a genuine control transfer.
2. **Same team:** a pass, loose ball without opponent control, or offensive rebound keeps the same possession ID. A shot-clock reset alone does not start a possession.
3. **Transfer:** close the outgoing possession on opponent control after a made basket, defensive rebound, steal or turnover. Use the engine's explicit control transfer, not merely a change in `ball.holder`.
4. **Free throws:** attribute them to the possession where the foul occurred. For an and-one, do not close the scoring possession until the associated free throws and control outcome are resolved. Retaining possession after free throws does not add another possession.
5. **Game end:** close the final active possession at normal termination even if the opponent never gains control. A zero-point final possession still counts. Do not open a phantom possession after the winning score or final whistle.
6. **Overtime:** proposed convention: close the regulation possession at tied clock expiry with `REGULATION_END`, then open an OT possession when the engine establishes OT control. Do not reset possession IDs. This administrative boundary is explicit even if the same team receives OT control; agree this convention with WS1 before integration.
7. **Interrupted game:** leave its active possession incomplete. Do not call it a zero-point completed possession or include the run in final experiment summaries.

## 5. Metric definitions and denominators

### 5.1 PPP: points per possession

For physical team `T` in a complete game:

```text
PPP(T) = total points credited to T / number of closed possessions owned by T
```

The numerator is the sum of `points_by_team[T]` across all possession-end records. It must equal `game_end.final_score[T]` because complete games start at 0-0. In ordinary possessions only the owning team scores, so this is also the sum of its `points` fields. The denominator counts unique possession IDs whose `team = T`.

Report per team per game, carrying the condition label separately. Do not average possession-specific ratios or use number of shots as the denominator.

### 5.2 IAR: invalid action rate

```text
IAR(T) = decisions by T with invalid_reason != null / all final decisions by T
IAR(game) = all invalid final decisions / all final decisions
```

Count each `(event_idx, player)` once. Include final parse failures and invalid proposals later replaced by safe actions. Internal retries do not enlarge the denominator. A legal action that fails in resolution remains valid.

Report each primary reason's count and its count divided by the same decision denominator. The primary reason counts sum to the invalid-decision numerator.

### 5.3 SIR: state inconsistency rate

```text
bad_events = distinct event_idx values in invariant_violation records
SIR(game) = number of bad_events / number of completed, fully checked transitions
```

For a final result, all completed transitions must have all 20 checks; otherwise report insufficient evidence instead of a rate over only the checked subset. Do not count the initial state, decision records, individual violation lines, or possession endings as additional transitions.

For constraint `c`, report the number of distinct transitions violating `c`, divided by the same transition denominator. Per-constraint counts can overlap and need not sum to the overall numerator. SIR is a game-level measure; do not assign a violation to one team without a separately defined attribution rule.

### 5.4 Result shape and unavailable data

Return each metric with `numerator`, `denominator`, `value`, and `status`. Keep fractions unrounded internally; round only for display.

| Situation | Result |
|---|---|
| Valid inputs and positive denominator | `status = "ok"`, numeric value, including a measured zero |
| Complete evidence but denominator is zero | `status = "undefined"`, `value = null` |
| Truncated/aborted game | `status = "incomplete"`, excluded from final experiment aggregates |
| Missing required evidence or incompatible contract | Reject with a clear validation error; do not fabricate metric values |

## 6. Worked synthetic example

This is a hand-authored accounting example, not experimental evidence or an engine calibration game. Configure a five-second test game starting at 0-0, with A in control of `p0001`. All players A1-A3 and B1-B3 submit one decision at every event.

### 6.1 Complete decision count and transition outline

Entries below are final proposed actions. Omitted targets are null except where shown. At event 1, A2's invalid pass is replaced with `SPOT_UP` at `LEFT_CORNER`; all other actions are accepted.

| Event | A1 | A2 | A3 | B1 | B2 | B3 | Final outcome |
|---|---|---|---|---|---|---|---|
| 1 | SHOOT | PASS to A3 (invalid: A2 has no ball) | HOLD | HOLD | HOLD | HOLD | A1 misses; A gets the offensive rebound; p0001 continues; 4 seconds remain |
| 2 | SHOOT | HOLD | HOLD | HOLD | HOLD | HOLD | A makes a 2-point shot; B gains control of p0002; 3 seconds remain |
| 3 | HOLD | HOLD | HOLD | CLEAR | HOLD | HOLD | B clears; p0002 continues; 1 second remains |
| 4 | HOLD | HOLD | HOLD | SHOOT | HOLD | HOLD | B makes a 1-point shot at clock expiry; game ends 2-1; p0002 closes |

For the SIR example, deliberately inject two structurally representable state errors into event 2: the ball names B1 as holder while B1 has `has_ball = false` (constraint 2), and B2 is assigned to defend teammate B3 (constraint 17). The test restores valid fields at event 3. These faults are deliberate test inputs, not consequences of a normal made basket. All 20 checks run on all four transitions.

### 6.2 Selected sample records

These are **field projections** illustrating the proposed contract. They intentionally omit full state snapshots, metadata, other players' decisions and diagnostics. They are not a complete importable JSONL game and must not be passed to a strict reader as one. A future fixture will expand the outline into the full record stream.

```json
[
  {
    "record_type": "decision",
    "event_idx": 1,
    "player": "A2",
    "team": "A",
    "raw_output": "<A2 action=\"PASS\" target=\"A3\"/>",
    "parsed_output": {"action": "PASS", "target": "A3", "parse_ok": true},
    "parse_ok": true,
    "invalid_reason": "ILLEGAL_POSSESSION",
    "remapped_action": "SPOT_UP",
    "remapped_target": "LEFT_CORNER",
    "executed_action": "SPOT_UP",
    "executed_target": "LEFT_CORNER"
  },
  {
    "record_type": "resolution",
    "event_idx": 2,
    "possession_before": {"possession_id": "p0001", "team": "A"},
    "possession_after": {"possession_id": "p0002", "team": "B"},
    "scoring_events": [
      {"team": "A", "points": 2, "source": "FIELD_GOAL", "possession_id": "p0001"}
    ],
    "possession_changes": [
      {"from_possession_id": "p0001", "to_possession_id": "p0002", "to_team": "B", "reason": "MADE_BASKET"}
    ],
    "game_over": false,
    "game_end_reason": null
  },
  {
    "record_type": "invariant_violation",
    "event_idx": 2,
    "constraint_id": 2,
    "constraint_name": "ball_possessor_consistency",
    "details": "Injected test fault: ball.holder is B1 but B1.has_ball is false."
  },
  {
    "record_type": "invariant_violation",
    "event_idx": 2,
    "constraint_id": 17,
    "constraint_name": "defensive_matchup_validity",
    "details": "Injected test fault: B2 is assigned to defend teammate B3."
  },
  {
    "record_type": "possession_end",
    "possession_id": "p0001",
    "team": "A",
    "points": 2,
    "points_by_team": {"A": 2, "B": 0},
    "start_event_idx": 1,
    "end_event_idx": 2,
    "end_reason": "MADE_BASKET"
  },
  {
    "record_type": "possession_end",
    "possession_id": "p0002",
    "team": "B",
    "points": 1,
    "points_by_team": {"A": 0, "B": 1},
    "start_event_idx": 2,
    "end_event_idx": 4,
    "end_reason": "CLOCK_EXPIRED"
  },
  {
    "record_type": "game_end",
    "last_event_idx": 4,
    "end_reason": "CLOCK_EXPIRED",
    "final_score": {"A": 2, "B": 1},
    "ot_flag": false,
    "totals": {
      "decisions": 24,
      "decisions_by_team": {"A": 12, "B": 12},
      "transitions": 4,
      "possessions_by_team": {"A": 1, "B": 1}
    },
    "total_cost_usd": 0.0
  }
]
```

### 6.3 Hand-calculated expected results

| Metric | Calculation | Expected value |
|---|---|---|
| A PPP | 2 points / 1 possession | 2.0 |
| B PPP | 1 point / 1 possession | 1.0 |
| A IAR | 1 invalid proposal / 12 proposals | 0.083333... (8.33%) |
| B IAR | 0 invalid proposals / 12 proposals | 0.0 |
| Game IAR | 1 invalid proposal / 24 proposals | 0.041666... (4.17%) |
| Game SIR | 1 inconsistent transition / 4 transitions | 0.25 (25%) |
| Constraint 2 rate | 1 affected transition / 4 transitions | 0.25 |
| Constraint 17 rate | 1 affected transition / 4 transitions | 0.25 |

The offensive rebound did not add a possession. The replacement did not erase A2's invalid proposal. Two violations at event 2 produced one inconsistent transition. B's last possession counted even though there was no subsequent A possession.

## 7. Required schema changes

These changes are proposed for a later implementation step; no Python changes are made by this document.

| Existing model / area | Change | Reason |
|---|---|---|
| `GameStartRecord` | Add contract version, experiment/replicate IDs, first possession, team-condition map, config snapshot, code commit, initial state and possession ID | Identify games and establish the starting accounting state |
| `DecisionRecord` | Add team, replacement target and executed action/target; constrain `invalid_reason` to `InvalidReason`; validate parse/remap consistency | Compute IAR without losing the original proposal or executed target |
| `ResolutionRecord` | Add typed scoring events, possession references/changes, action targets and game-end fields | Reconstruct possession/score changes from explicit facts |
| `StateRecord` | Add invariant-check coverage and diagnostic raw-state path | Distinguish successful checks from missing checks; record inconsistent states without hiding them |
| `InvariantViolationRecord` | Bound IDs to 1-20; validate uniqueness per event/constraint at stream level | Reliable constraint breakdown and joins |
| `PossessionEndRecord` | Add possession ID and points-by-team; validate point totals and event bounds | Count possessions once and reconcile every scored point |
| `GameEndRecord` | Add last event and termination reason; specify required totals structure | Establish completion and verify computed counts |
| Log module and schema exports | Add typed nested payloads and a discriminated union of all supported records | Dispatch records by `record_type` and validate known payloads |
| Reader / stream validator | Enforce cross-record ordering, identities and reconciliation | Per-record Pydantic validation alone cannot check a whole game |

Preserve existing names where possible. New required evidence must not be silently defaulted to a successful result when loading old logs. An unversioned log can be inspected as legacy data, but is not automatically a contract-v1 game.

## 8. Validation and acceptance checklist

Before producing final metrics from a game, validate:

- Exactly one `game_start` first and one normal `game_end` last; no trailing records.
- Supported version and six distinct roster players, three on each physical team.
- Contiguous completed event IDs and exactly six distinct final decisions per event.
- One resolution, one referee record and one post-transition state per event; all joins resolve.
- Decision legality/remap fields agree and resolution actions/targets match them.
- All 20 invariant checks completed for every transition; violation records refer to checked constraints and existing transitions.
- Initial state excluded from transition counts; duplicate records rejected rather than silently double-counted.
- Possession IDs open/close consistently, never close twice, and no active possession remains at normal game end.
- Scoring events belong to real accounting possessions; their sums agree with possession totals, each state's score changes and final score.
- Recomputed decision/transition/possession counts agree with `game_end.totals`.

An invariant failure is measured simulation behaviour, not automatically a malformed log. Keep it for SIR and flag affected metrics if accounting is also inconsistent. For example, a score contradiction can make PPP unavailable while the fully recorded invariant checks still support a diagnostic SIR. Do not silently drop bad transitions or games to improve validity rates. Experiment exclusion policy must be explicit before the main runs.

For the first implementation, test the worked example plus separate cases for final parse failure, zero denominators, offensive rebound, free throws/and-one, retained possession, final zero-point possession, overtime boundary, duplicate records, missing invariant checks and interrupted files.

**This first document is complete when a teammate can identify which fields their component supplies and reproduce Section 6's three metrics by hand.** Before producer integration, confirm the event-index convention, final decision logging after legality, possession/OT boundaries, and invariant snapshot escape path with WS1 and the runner owner.

## 9. Deferred extensions

The repository now includes typed initial-memory, memory-write, CS-write, per-action opponent-involvement, and observation-text fields. `analysis/qualitative.py` and `bball transcript <game.jsonl> <possession_id>` render saved observations and the ordered event history. Producers still need to populate the new fields in real runs.

Transcript boundaries are reconstructed from resolutions. Transfer events can belong to two adjacent possession spans: the renderer includes the full event as context, identifies `possession_before` for its decisions, and preserves scoring possession IDs. Open possessions can be inspected in unfinished logs. Earlier messages with delivery in the span and the latest prior captain replacement plus subsequent maintain records are labeled context, not new emissions. Missing delivery metadata is explicitly marked as an inferred t+1 convention. Missing observations are never reconstructed from hashes or game state.

Stable message/call IDs and captain-call identity across `maintain=true` still need a follow-up coordination contract and producer integration. Context shown in a transcript is not proof of exact per-player inbox visibility; the logged observation supplies that evidence. This first delivery defines PPP/IAR/SIR as its core and does not claim that those metrics complete all of WS6.
