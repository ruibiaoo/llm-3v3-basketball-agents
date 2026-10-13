# Adopting BasketWorld: Display and Parity, Minimal Changes

Source: [`github.com/EvanZ/basketworld`](https://github.com/EvanZ/basketworld),
MIT License, Copyright 2025 Evan Zamir.
Parent decision: `proposal/IMPLEMENTATION_PLAN.md`, subsection
"WS1-adjacent: BasketWorld (EvanZ, MIT) — evaluation and bounded use",
whose verdict stands: **our engine remains the source of truth**.
This document is the how-to for the two approved uses that need it:
**frontend display** of our simulations and **parity/fuzz reference**.
Nothing here changes the graded deliverable, which stays backend-only.

## 0. Ground rules

1. **Never run their sim as our engine.** Hex grid, 24-second clock,
   first-basket-or-step-cap episodes, 8 atomic actions. None of it is
   FIBA 3x3, and porting it costs more than finishing our engine.
2. **Take display and ideas, not the runtime.** What transfers:
   the ASCII and matplotlib renderer *structure* (`render_ascii` /
   `render_visual` in `basketworld/envs/core/rendering.py`), the
   frame-to-GIF pipeline from their `__main__` demo blocks, the
   collision rule (same-cell targeting leaves both in place), and the
   point-to-segment lane reasoning as a cross-check on our pass resolver.
3. **Leave the JAX path alone.** `basketworld_jax/` (Flax, Optax, Orbax),
   `train/`, MLflow logging, and the FastAPI-plus-Vue app backend are
   excluded. They buy nothing for an LLM decision-event project.
4. **Zero churn in the graded environment.** No new required dependencies,
   no CI changes. Everything below lives behind lazy imports or in
   gitignored directories.

## 1. One-time setup (no `pyproject.toml` changes)

```powershell
# Read-only reference clone; gitignored, never imported by src or tests.
git clone https://github.com/EvanZ/basketworld.git third_party/basketworld
```

Add to `.gitignore`:

```text
third_party/
```

Attribution file (required by the MIT condition on substantial reuse):

```text
third_party/BASKETWORLD_ATTRIBUTION.md
```

containing the repository URL, the MIT License text from their
`LICENSE.md`, and a list of every excerpt copied into our tree.
Any file carrying their code keeps their copyright notice at the top.

Display-only rendering needs `matplotlib` and `Pillow`, used exactly as
their renderers do (`fig.savefig` to a buffer, `PIL.Image`, `numpy` array).
Install them as dev tools only when the display work starts:

```powershell
uv add --optional display matplotlib pillow
```

or the equivalent dev-group entry. The engine and test suite must import
cleanly without them.

## 2. Frontend display (recommended path: offline renderer)

Their Vue app (`app/frontend`) is driven by their FastAPI backend
(`app/backend`), which loads their checkpoints by MLflow `run_id` and
replays through *their* runtime. Pointing that stack at our engine means
implementing a shim of their backend API plus a square-to-hex view-model
mapping, then auditing every hex assumption in their grid components.
That is Phase 2 at the earliest. The minimal path renders our own games
offline, following their renderer structure:

### 2.1 New module (ours, no third-party import)

```text
src/bball3x3/display/
├── __init__.py        # lazy imports only; nothing engine-side depends on this
├── square_render.py   # render_ascii(state) + render_frame(state) -> PNG bytes
└── replay.py          # JSONL state records -> frames -> GIF/MP4, shot charts
```

`square_render.py` mirrors their two-function shape:

- `render_ascii(state: GameState)` — prints shot clock, ball holder, and a
  15×11 ASCII court. Mark the basket at `court.basket`, offense as
  `A1..A3`, defense as `B1..B3`, ball holder ringed. Our coordinates are
  used directly; **no hex conversion exists anywhere in this path**.
- `render_frame(state, court, last_resolution=None)` — matplotlib figure
  in their style: court rectangle, painted arc at 6.75 m, zone labels,
  player discs in team colours, ball-holder ring, optional pass arrow
  from the last `resolution` record and live make-probability text from
  its logged `p_make`. Returns an RGB array via the same
  buffer-to-`PIL`-to-`numpy` pipeline their `render_visual` uses.

### 2.2 Replay from our logs (no sim involved)

`replay.py` reads one game JSONL (WS6 records), takes each `state` record
in order, renders a frame, and writes the sequence with the same
`PillowWriter` animation pattern as their demo `__main__` blocks:

```powershell
uv run python -m bball3x3.display.replay runs/<exp_id>/<game_id>.jsonl --out replay.gif
```

Shot charts aggregate `resolution` records (shooter position, zone,
`p_make`, made flag) — the analogue of their statistical summaries, fed
by our logged model terms instead of theirs.

### 2.3 Their Vue app (deferred stretch, no exit criterion)

Out of graded scope. Only if an interactive app is later required:

1. Serve our games from a shim implementing the endpoints their frontend
   already calls (game list, game state, self-play stream). The shim
   replays our JSONL; it never imports their sim.
2. Map our view model onto theirs: square `(x, y)` positions, holder id,
   game and shot clocks, score, fouls, phase. Their grid components
   assume hex cells, so each component needs a square-cell audit before
   it can render a single frame.
3. Keep the shim in `src/bball3x3/display/shim.py`, dev-only, outside the
   graded path.

### 2.4 Live runner hook (in scope for pilot and experiment games)

The one display call the runner owns. Contract, mirrored in WS6:

- **Signature.** `display.live_hook(snapshot, ctx) -> None`, where
  `snapshot` is a deep copy of the `GameState` and `ctx` carries
  `event_idx`, `exp_id`, `game_id`, and the display config. No return
  value; nothing the hook returns is read.
- **Call site.** `runner/game.py`, immediately after the log write,
  guarded by `display.enabled` and `mode == live`, throttled by
  `every_n_events`, capped by `max_frames_per_game`. Exceptions are
  caught, counted in the game totals, and never fail the game.
- **Import hygiene.** `matplotlib` is imported lazily inside the hook, so
  the engine and runner import graphs never gain the dependency and the
  suite stays green without it.
- **Budget gate.** The pilot measures mean render milliseconds per event
  before any main run may enable `live`. Over budget, main runs stay on
  `post` (render from finished JSONL) or `off`. Render cost is
  wall-clock only; display adds no tokens and no API cost.
- **Tests.** Frame determinism plus a raising-hook isolation test ship
  with `display/`; the parity script and `third_party/` stay out of CI.

## 3. Parity and fuzz reference (simple env only, WS7 Phase 1 owner)

The pedagogical env `basketworld/envs/basketworld_3x3_env.py` (12×8 grid,
8 actions, needs `gymnasium` plus `numpy`) is sufficient. Never the v2 or
JAX envs. A dev-only script plays random games on both engines and
compares possession dynamics, turnover rates, and game lengths:

```powershell
uv run --with gymnasium python notebooks/basketworld_parity.py
```

Rules: investigate divergences, never auto-sync coefficients; the script
is excluded from CI and imports BasketWorld lazily so the suite never
sees it. Owned by WS7 Phase 1; figures are methods-only disclosure,
excluded from all results tables and findings.

## 4. If their sim must be modified anyway (discouraged)

Should the team overrule the verdict, the surgical order — each step
verified against our invariant suite before the next — is:

1. `basketworld/envs/core/geometry.py`, `movement.py`: replace axial/hex
   distance with Euclidean metres on the 15×11 square grid.
2. Env `__init__`: 24 s clock to 12 s; episode end to 600 s or 21,
   tie to overtime at plus two.
3. New `rules.py` equivalent: clearing, check-ball, 7th/10th bonus
   ladder, shooting-foul free throws, 1-pt/2-pt scoring.
4. Action space 8 to 24 with role legality and remap records.
5. `core/shooting.py`: interpolation to the named-term logistic, logging
   every term.
6. Seeded per-resolver RNG streams replacing bare `random` calls.

Expect this to exceed the remaining WS1 work by multiples. Prefer
sections 2 and 3.

## 5. License compliance checklist

- [ ] `third_party/BASKETWORLD_ATTRIBUTION.md` records URL, MIT text,
      and every reused excerpt.
- [ ] Copied files retain the `Copyright (c) 2025 Evan Zamir` notice.
- [ ] No BasketWorld import in `src/` outside `display/` and dev scripts.
- [ ] Full suite passes with `third_party/` absent.
- [ ] Report limitations section names the reused display patterns.
