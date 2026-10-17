# WS6: step-by-step delivery plan

Status (2026-10-16): logging, indexing, possession validation, primary and
descriptive metrics, adaptation-bin calculations, observation logging,
transcripts, experiment exports, seed-paired statistics/possession models,
and adaptation plots are **implemented and tested**. `bball export` produces
game, team, and possession CSV tables plus a JSON manifest from saved logs;
`bball stats` produces analysis JSON, and `bball plots` produces PPP/MAUR/CSSR
curves, pooled CSV and provenance manifest. `tests/test_exports.py` covers pairing identifiers,
rate evidence, possession covariates, CSV/CLI output, incomplete games, invalid
possession accounting, exceptional scoring, and overtime bins. The full suite
passes (109 tests), and Ruff passes. Real-run producer integration remains.
The original review and sequencing notes below are historical.

Sources: `IMPLEMENTATION_PLAN.md`, especially WS0, WS4, WS6 and milestones M1/M5/M6; original proposal pages 8-13 and 17-20, including Figure 3 on page 11.

## 1. Outcome and current starting point

WS6 must turn each game's event log into reproducible metrics, possession transcripts, and experiment analysis. Every reported result must be recoverable from the saved log without calling an LLM or accessing live engine/memory objects.

The current checkout contains:

- Pydantic state, action, message, memory, referee and log schemas.
- YAML configs and loaders, a lockfile, and schema/config tests.
- Implemented WS6 logging and analysis packages; the runner and other producer packages remain skeletons.
- No generated game logs or implemented simulation to integrate with yet.
- No checked-in GitHub Actions workflow. WS0 scaffolding exists, but its entire milestone should not be assumed complete.

Start using small synthetic logs with hand-calculated expected results. Real-game integration can follow when the runner is available. No API key, model selection, rulebook index, or final player roster is needed for this work.

## 2. Ownership and integration boundaries

| Owner | Responsibility needed by WS6 |
|---|---|
| WS6 | Record contracts, JSONL writer/reader, log validation, possession analysis, metrics, statistics, transcripts, exports and analysis tests |
| WS1 / engine | Legality outcomes, executed actions and targets, scoring and possession changes, resolver terms/RNG draws, invariant results |
| WS2 / LLM infrastructure | Usage, latency, cost, model/provider metadata and retry accounting |
| WS3 / player agents | Actual observations, raw/parsed outputs, parse results, rationales and prompt hashes |
| WS4 / coordination and memory | Message delivery/suppression, active captain calls, memory/CS visibility and mutations |
| WS5 / referee | Every-event judgement, retrieval evidence, fallback/parse status and usage |
| Runner integration owner | Invoke the logger in the agreed event order, assign game/event IDs, capture run metadata and signal completion |
| WS7 | Run experiments and use WS6 outputs for reporting |

Share the contract and a sample log with the relevant owners before they wire up producers. WS6 should consume their explicit outcomes, not recreate basketball rules or infer legality from prose. The implementation plan does not clearly assign runner ownership; identify that owner during integration.

## 3. Step 1: settle the log and metric contract

First deliverable: `proposal/WS6_LOG_CONTRACT.md` with a field table, event timeline, metric denominator rules, and a worked example. Then make the corresponding focused schema changes in `src/bball3x3/schemas/log.py` and exports in `schemas/__init__.py`.

The following are proposed conventions for that contract, not existing implemented behaviour.

### Gaps to address

| Current gap | Proposed contract requirement | Why it matters |
|---|---|---|
| No log schema version | Version in game metadata; reject unsupported versions clearly | Prevent incompatible logs being silently combined |
| Seed alone does not identify swapped games | Experiment ID, unique game ID, replicate/start-side identifier, first possession, explicit physical-team-to-condition mapping | Correct pairing; physical team `B` can represent condition B, C or D |
| Config hash without saved config; no commit field | Sanitized config snapshot, commit identifier, actual backend/model/provider metadata and prompt hashes | Explain and reproduce the run |
| No initial state or instantiated memory | Save initial state, initial per-team memory entries with assigned IDs, and empty initial CS stores | Reconstruct first possession and Team D's valid memory references |
| Observation hash without content | Store actual observation text in the JSONL record | Fulfil the transcript requirement from the log alone |
| Remapped action has no target | Preserve proposed and executed action/target separately | Distinguish agent intent, invalid proposals and actual outcomes |
| No stable IDs linking captain calls/messages | Unique message/call IDs and explicit delivery/visibility information | Avoid double-counting the same captain directive or tactic |
| No complete memory/CS mutation history | Log accepted writes/merges/evictions, resulting entries, rejection reasons and when changes become visible | Reconstruct the store at decision time, including after eviction |
| Only proposal records currently exist | Add explicit `memory_write` and `cs_write` records, linked to proposals | Separate attempted updates from committed changes |
| Grounded MAUR has no structured involvement evidence | Per-action involved opponent IDs, supplied by the engine | A reference to an opponent elsewhere on court is not automatically grounded |
| Resolution outcomes are unconstrained dictionaries | Specify required typed scoring, control-transfer and per-action outcome fields with WS1 | Handle free throws, rebounds, possessions and later descriptive metrics |
| State event numbering is not defined | Explicit transition ID and distinction between initial state and post-transition state | Correct joins and SIR denominator |
| No possession identifier | Stable possession ID plus start/end events, points, end reason and closure status | Reliable transcripts and score reconciliation |
| Referee fallback can resemble genuine `NONE` | Explicit parse/fallback status and raw response where applicable | Measure referee calls without hiding failures |

Add only fields required by concrete consumers. Keep constructors compatible where sensible, but do not default missing evidence to a successful or valid result. A reader should distinguish legacy/incomplete data from a measured zero.

### Proposed metric conventions to settle with the producers

1. **Message timing:** a message emitted at event 5 is visible before decisions at event 6; the primary alignment check uses the recipient's decision at event 6. This matches proposal Figure 3 and WS4. WS6's phrase "received at t" followed by "decisions at t+1" is ambiguous and should be clarified in the contract. For the k=2 robustness check, consider decisions at events 6 and 7, with explicit rules for unavailable future decisions.
2. **Alignment unit:** primary MAAR/CCAR count eligible recipient decisions once, not once per directive. Proposed rule: a decision aligns if its proposed action and target match at least one eligible directive; also report conflicting directives separately. Ignore the target only when the directive has none. Keep executed-action alignment as an explicitly separate diagnostic if needed. Settle this choice before calculating results.
3. **Captain persistence:** retain call identity across `maintain=true`. CCAR uses directives actually visible in each observation. CSSR counts a newly emitted tactic once, not once per repeated display. Define supersession and k=2 handling explicitly.
4. **Reference validity:** validate against the player's team memory/CS store visible before its decision. A write produced during that event cannot retroactively validate another simultaneous output. Include initial entries and evictions; an ID existing at game end is insufficient.
5. **MAUR:** at least one valid reference counts once per decision. Invalid references do not count. Grounded MAUR additionally requires structured evidence that the referenced opponent participated in that action. Do not treat self-reported references as proof of causal influence.
6. **IAR:** count each final player proposal once, including final parse failures. Count invalidity before remapping; do not count internal repair attempts as additional player decisions. The current loop sketch logs decisions before legality checking, so producers must either finalize that record after legality or emit a linked validation outcome.
7. **SIR:** count distinct completed transitions with at least one invariant violation divided by all completed transitions. Two broken constraints in one transition contribute one to the numerator. Initial state is excluded. Invariant evidence must still be recordable if the resulting state fails normal Pydantic bounds.
8. **Undefined rates:** return numerator, denominator, value and status. A zero denominator is undefined, not zero. Feature-disabled metrics are not applicable: no teammate MAAR for condition B; no CCAR or evaluated CSSR for condition C. D's CSSR is zero when it emits eligible tactics and none can have a valid CS reference; with no tactics it remains undefined.
9. **Possessions:** offensive rebounds retain the current possession. Attribute free throws to the possession of the foul; do not close early on an and-one or invent a new possession for a retained-possession foul. Explicitly close the last possession at normal game end, including winning shots and clock expiry; never invent a subsequent possession after termination. Define regulation-to-OT handling with WS1. An interrupted run is incomplete and excluded from experiment results by default.
10. **Scoring and time:** validate possession-attributed points against final scores, including any exceptional free-throw attribution. Save clocks and pre-possession score difference for adaptation/statistics. Define regulation bins and separate OT handling before plotting.

**Step 1 is done when:** every primary metric and transcript field has a named record source, timing is unambiguous, and a tiny example can be calculated by hand. Schema tests cover the newly agreed requirements. Changes to shared schemas should be communicated to the other workstream owners.

## 4. Implementation stages

Implement and explain one stage at a time. Each stage should be reviewable independently.

| Stage | Files / deliverable | Completion check |
|---|---|---|
| 1. Contract | Contract document, focused log schema/export updates | Each metric's inputs and event timing are specified; schema tests pass |
| 2. JSONL infrastructure | `logging/event_log.py`, `tests/test_event_log.py` | All supported records round-trip, Unicode/newlines survive, invalid records give line-numbered errors, completed/incomplete games are distinguishable |
| 3. Minimal analysis | `analysis/possessions.py`, initial `analysis/metrics.py`, focused tests and synthetic fixtures | Hand-calculated PPP, IAR and SIR match exactly; offensive rebound, foul/FT, terminal possession and duplicate violation cases pass |
| 4. Coordination/adaptation metrics | Extend `analysis/metrics.py`; message/memory fixtures | MAAR, CCAR, CSSR, MAUR and grounded MAUR obey timing, validity and ablation rules |
| 5. Inspection and exports | `analysis/qualitative.py`, metric export functions, CLI integration | A saved log yields a readable possession transcript and machine-readable team/game/possession tables without live simulation objects |
| 6. Experiment analysis | `analysis/stats.py`, adaptation plots/notebooks and descriptive metrics | Paired effects, confidence intervals, stability and plots reproduce from exported logs with a fixed analysis seed |
| 7. Real-run integration | Runner hooks and integration tests with its owner | A heuristic game produces a complete validated log; score/possession checks pass; offline metrics and transcript agree with recorded events |

Integrate stages 1-3 with the runner as soon as it is ready; do not delay the M1 engine milestone until statistics are complete. M1 needs valid logs and core validity metrics. M5 needs all primary metrics; M6 needs statistics and qualitative outputs.

### Stage 2 design

- Begin with a single writer owned by each game runner and one UTF-8 JSONL file per game. Agent tasks return results to the runner rather than writing concurrently into the same file.
- Proposed interface: `EventLogger(path)` as a context manager, `write(record)`, `flush()`, and `read_events(path)` returning typed records selected by `record_type`.
- Add a discriminated record union. Validate records and required stream structure, including exactly one start/end for complete logs, expected event links, and duplicate decisions.
- Refuse accidental overwrite. A normal context-manager exit must not invent a successful `game_end`; only the runner knows the game finished.
- Start with strict reading. If interrupted-log inspection is later supported, make it explicit, flag truncation, and never silently use partial games in final analysis.
- Use Pydantic and the Python standard library initially. Add analysis dependencies only when their stages need them.

### Stage 3 first worked fixture

Create a clearly synthetic short log with known expected counts. For example, Team A scores 3 points over 2 closed possessions, so PPP is 1.5. In a separate validity fixture, one invalid proposal among six yields IAR 1/6; two invariant records attached to one of three transitions yield SIR 1/3.

Include dedicated possession sequences for offensive rebounds, turnovers, and-one free throws, retained possession after free throws, a game-winning basket, clock expiry and OT. Derive boundaries from explicit engine outcomes and cross-check any logged `possession_end` records; do not infer every possession change from score changes or ball-holder changes alone.

### Stage 4 adversarial fixtures

- A teammate directive emitted at t influences t+1, never t; distinguish k=1 and k=2.
- Right action/wrong target, directive without a target, multiple/conflicting directives, and end-of-game unavailable response windows.
- Suppressed messages never create eligible alignment decisions. Condition B still allows captain directives.
- Maintained captain calls and duplicated representations do not inflate tactic counts.
- Initial memory, invalid references, opponent-team IDs, writes from the current event, merges, evictions and disabled writes.
- Condition C and zero-denominator cases yield explicit not-applicable/undefined results.
- A newly created CS entry cannot validate an earlier tactic; condition D cannot acquire new entries.

### Stages 5-6 analysis outputs

Stage 5 export status: **implemented and tested**. `analysis/exports.py` reads saved JSONL logs and produces game/team rows with experiment, physical-team conditions, seed, first-possession replicate, metric counts/rates/status, completion status, and configuration identity. Possession rows include points, pre-possession score difference from the owning team's perspective, clock, regulation/OT bin, and model eligibility. `bball export <new-output-dir> <game.jsonl>...` writes three CSV files and a JSON manifest. Incomplete games are excluded by default and listed in the manifest; completed games with invalid possession accounting retain game/team rows but do not receive unreliable possession rows. Existing output directories are not overwritten. The CLI entry point is registered in `pyproject.toml`; runner CLI consolidation still depends on its owner.

Stage 6 statistics status: **implemented and tested**. `analysis/stats.py` and
`bball stats <export-dir> <new-result.json>` calculate per-seed A-minus-X
effects, deterministic seed-cluster bootstrap intervals, signed-rank p-values,
sign stability and explicit missing-pair reasons. Additive and condition-by-time
Gaussian possession models use a game random intercept and pre-possession score
difference; insufficient or failed fits are explicit. The model is exploratory
because it does not model dependence between the two games of a seed; paired
seed analysis remains primary. Adaptation plots are implemented and tested in
`analysis/plots.py` and `tests/test_plots.py`: valid game counts are pooled by
condition/bin, zero-exposure bins remain blank, and OT is a separate marker.
Real-game validation is not yet implemented.

For each ablation, compute A-minus-X differences within each game, then aggregate the two first-possession variants within each seed. Bootstrap seed clusters, preserving both variants together; run the planned Wilcoxon test on per-seed differences. Report effect size, seed count, interval, ties/zero differences and sign stability. Define handling for missing pairs rather than treating them as complete. These are the implementation plan's requested analyses; settle the statistical specification before main experiments are frozen.

Then implement the planned possession model and condition-by-time interaction. Define the response/model family, pre-possession covariates, treatment of the two opposing teams in a game and dependence across games sharing a seed before fitting. Handle insufficient data or failed fits explicitly.

For adaptation, propose five fixed regulation-clock bins based on configured game duration, with OT reported separately. Early-ended games have missing later exposure, not artificial zero performance. Record binning and possession-boundary conventions in output metadata.

Add the plan's descriptive metrics once their event inputs exist: assists, turnovers, screen-to-shot conversion, same-zone coordination conflicts, memory growth and referee call rate. Define their denominators and attribution windows before coding. Produce tables, curves, and success/failure transcripts for WS7's manual review.

## 5. Verification and working sequence

For implementation, set up the repo environment with `uv sync --extra dev`, then establish the existing baseline with `uv run pytest` and `uv run ruff check .`. Use focused tests for the active stage before running the suite. Record pre-existing failures separately from regressions.

The initial review did not run that baseline. The current checkout has since been verified with 109 passing tests and a clean Ruff check (2026-10-16).

The original immediate-next-session instruction was **Step 1 only** (now completed):

1. Walk through one decision event using the schemas already in the repo.
2. Write the concrete log contract and settle the proposed conventions above.
3. Apply the smallest schema additions needed by that contract, with meaningful validation tests.
4. Review the worked example and schema changes before proceeding to the JSONL writer.

The first end-to-end WS6 milestone is: **synthetic records -> JSONL -> typed reader -> possession table + PPP/IAR/SIR**, with expected values verified by hand and tests. This establishes a usable interface for the engine team while keeping each learning step small.
