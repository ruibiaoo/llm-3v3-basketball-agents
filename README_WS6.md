# WS6: event logging, metrics and analysis

WS6 turns a saved 3v3 basketball game log into validated possessions, metrics,
possession transcripts, experiment tables, statistics and adaptation plots. It
works offline: analysis reads JSONL files and does not call an LLM or require a
live game object.

The offline deliverables are implemented and tested with synthetic game logs.
**Real-game integration is still pending:** the runner does not yet produce a
complete game log. The [implementation plan](proposal/IMPLEMENTATION_PLAN.md#ws6--event-logging-metrics-and-analysis)
defines the deliverables; the [WS6 log contract](proposal/WS6_LOG_CONTRACT.md)
defines the records and event timing.

## setup

Use Python 3.11 or newer. From the repository root, install the project and
development dependencies:

```bash
uv sync --extra dev
```

For a standard `venv`/`pip` installation, follow the [root README](README.md#setup).
The commands below use `uv run`; with an activated environment, omit `uv run`
or use `python -m bball3x3` in place of `bball`.

## deliverables

| Deliverable | Implementation | What it provides |
|---|---|---|
| Log schemas and JSONL I/O | `schemas/log.py`, `logging/event_log.py` | Typed records, one file per game, validation on write and read |
| Reader and possession handling | `analysis/log_reader.py`, `analysis/possessions.py` | File-order indexes, possession reconstruction, score and closure checks |
| Game metrics | `analysis/metrics.py` | Primary and descriptive counts, rates and evidence statuses |
| Possession transcripts | `analysis/qualitative.py` | Readable event context for manual review |
| Experiment exports | `analysis/exports.py` | Game, team and possession CSV tables plus a manifest |
| Statistics and adaptation plots | `analysis/stats.py`, `analysis/plots.py` | Seed-paired effects, exploratory possession models and PNG curves |
| Real-game producer/runner integration | Pending | A full simulation writing and validating its own log |

## game logs and possessions

Each JSONL file starts with `game_start`, contains event records in file order,
and ends with `game_end` only when the game actually finishes. Records cover
decisions, messages, captain calls, memory and coordination-store proposals and
writes, resolutions, referee decisions, state snapshots, invariant violations
and possession endings. `EventLogger` writes typed records without overwriting
an existing file. `read_game_log()` validates known records and groups them by
record type and event. An unfinished log can be read for inspection; it is not
silently treated as a completed game.

`validate_possessions()` checks saved possession endings against resolution
outcomes, state scores and game totals. An offensive rebound keeps the current
possession; free throws remain with the possession in which the foul occurred.
For a complete game, every possession must close and the final points must
reconcile with `game_end`.

To inspect an existing log from Python:

```python
from bball3x3.analysis import compute_game_metrics, read_game_log, validate_possessions

game = read_game_log("logs/game.jsonl")
possessions = validate_possessions(game)
metrics = compute_game_metrics(game)
print(game.is_complete, len(possessions.closed))
print(metrics.ppp["A"].to_dict())
```

The paths in this README are examples; the repository does not currently
include a generated real-game log.

## metrics

The primary metrics are points per possession (PPP), teammate-message action
alignment (MAAR), captain-call action alignment (CCAR), the share of tactic
calls with a valid coordination-strategy key (CSSR), memory-augmented usage
(MAUR and grounded MAUR), invalid-action rate (IAR), state inconsistency rate
(SIR) and referee call rate. MAAR and CCAR use
a one-event response window by default; `--window 2` selects the two-event
robustness check during export.

Extra descriptive outputs are assist rate, turnover rate, screen-to-shot
conversion, same-zone coordination conflicts, memory growth and PPP/MAUR/CSSR
by five regulation-time quintiles. Overtime has its own bin. See the
[descriptive metric contract](proposal/WS6_DESCRIPTIVE_METRICS.md) for exact
numerators, denominators and required resolution evidence.

Every rate reports `numerator`, `denominator`, `rate` and `status`. A missing
denominator is `undefined`, missing source evidence is
`insufficient_evidence`, a disabled feature may be `not_applicable`, and an
unfinished game is `incomplete`. These states are not converted into zero
rates. `compute_game_metrics(game).to_dict()` returns a JSON-friendly summary.

## possession transcripts

Print one possession from a saved log:

```bash
uv run bball transcript logs/game.jsonl p0001
```

The transcript includes recorded observations, messages, captain calls,
proposed and executed actions, rationales, memory references and mutations,
referee calls, outcomes and state snapshots. It can also inspect an open
possession in a valid unfinished log. Use the exact possession ID stored in
the file.

## experiment exports

Export one or more saved logs to a **new** directory:

```bash
uv run bball export outputs/ablation-comms logs/game-a.jsonl logs/game-b.jsonl
```

The output contains `games.csv`, `teams.csv`, `possessions.csv` and
`manifest.json`. Rows retain game/seed identity, physical-team conditions,
metric counts and statuses. Possession rows include points, pre-possession
score difference, start clock and time bin. The export never overwrites an
existing directory.

Incomplete games are excluded by default and listed in the manifest. Use
`--include-incomplete` to inspect them; open possessions have no outcome and
are not model eligible. A complete game with invalid possession accounting
retains game/team rows and an error, but no unreliable possession rows.

## experiment statistics

Analyse an export and write a new JSON result file:

```bash
uv run bball stats outputs/ablation-comms outputs/ablation-comms-stats.json
```

For each comparison, the analysis pairs the two opposite first-possession
games of each seed, computes A-minus-ablation effects, a seed-cluster bootstrap
95% interval, a two-sided Wilcoxon test and sign stability. Missing or invalid
pairs are excluded with reasons. The default bootstrap seed is 4213; use
`--analysis-seed` to change it. Possession-level Gaussian models include
condition, pre-possession score difference and time bin, with an exploratory
condition-by-time interaction. Insufficient or failed fits are reported rather
than used as results. The paired seed effects are the primary inference.

## adaptation plots

Render curves from the same export into a new directory:

```bash
uv run bball plots outputs/ablation-comms outputs/ablation-comms-plots
```

This writes `adaptation.csv`, `manifest.json` and PPP, MAUR and CSSR PNGs per
comparison. Plotting requires both first-possession variants of an eligible
seed. Rates pool valid counts; bins without exposure stay blank, and overtime
is displayed separately from regulation.

## verification and integration status

Run the WS6 tests and linter from the repository root:

```bash
uv run pytest tests/test_event_log.py tests/test_analysis.py tests/test_possessions.py tests/test_descriptive_metrics.py tests/test_qualitative.py tests/test_exports.py tests/test_stats.py tests/test_plots.py
uv run ruff check .
```

On Windows, one current export test expects the text `File exists`, whereas
Windows may report `Cannot create a file when that file already exists`. The
export correctly refuses to overwrite the directory; the assertion is tied
to the operating system's wording.

To complete WS6 end to end, the engine, agents, coordination/memory system and
referee must supply the contracted records, and the game runner must write
them in event order through `EventLogger`. Then run a full heuristic game and
verify its saved log with `read_game_log()`, `validate_possessions()`, metrics,
export and transcript output. Until that succeeds, WS6 is **offline
implementation complete; real-game integration pending**.
