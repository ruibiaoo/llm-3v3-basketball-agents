# WS6 descriptive metric contract

This document defines the extra descriptive metrics in
`IMPLEMENTATION_PLAN.md` (WS6). All are computed from saved game records by
`compute_game_metrics()`. Rates retain their numerator, denominator and status.

## Resolution evidence supplied by the engine

Each `resolution` has three optional typed lists. The producer must set every
list on every event, using `[]` when the event had none. `null` means the
producer did not provide that evidence, so the related rate is unavailable.

| Field | Required facts |
|---|---|
| `shots` | One `ShotAttempt` per field-goal attempt: team, shooter, made, optional `assist_by`, optional `screen_id`. Made-shot counts must agree with `FIELD_GOAL` scoring-event counts for each team and event. |
| `screens` | One `ScreenEvent` per executed screen: game-unique `screen_id`, team, screener, freed player. The screener must have executed `SCREEN` in that event. |
| `turnovers` | One `TurnoverEvent` per lost possession: team, possession ID, reason, optional player. The possession must close in that event. |

The assist player must be a teammate other than the shooter. A `screen_id` on a
shot links it to a screen by the same team that freed that shooter. Unknown or
contradictory screen IDs make the conversion rate insufficiently evidenced.

## Counting rules

| Metric | Numerator | Denominator |
|---|---|---|
| Assist rate, per team | Made field goals with `assist_by` | All made field goals |
| Turnover rate, per team | Distinct owned possessions with an explicit turnover event | All closed possessions owned by that team |
| Screen-to-shot conversion, per team | Screens linked to at least one field-goal attempt by the freed player in the same event or next event | All executed screens |
| Coordination conflict rate, per team | Events where at least two teammates' executed actions target the same named court zone | Events with at least two distinct teammates executing zone-targeted actions |

The ten named zones are `TOP`, `LEFT_WING`, `RIGHT_WING`, `LEFT_CORNER`,
`RIGHT_CORNER`, `LEFT_ELBOW`, `RIGHT_ELBOW`, `LEFT_BLOCK`, `RIGHT_BLOCK`, and
`PAINT`. A conflict is counted once per team-event even if several players
collide. Disabled or suppressed messages do not affect this metric; it uses
executed decisions.

Memory growth counts unique active memory IDs at event 0 and after each
event's committed `memory_write` records. A write to an existing ID is a merge
and does not grow the store; `evicted_ids` remove entries before the new entry
is counted. The output includes initial, final, net and peak counts plus the
event series. Initial memory must be recorded in `game_start.initial_memory`
or supplied as team-scoped known IDs for an older log. An accepted proposal
without a matching committed write is insufficient evidence for growth.

## Adaptation bins

The game-start regulation clock defines five fixed bins of equal elapsed
regulation time: `q1` covers 0–20%, `q2` 20–40%, through `q5` 80–100%.
Overtime is `ot` and is never folded into `q5`. Bin assignment uses the state
at the **start** of an event. A possession created by an explicit
`PERIOD_BOUNDARY` is assigned to `ot` even when its start event began in
regulation.

PPP assigns a possession and all its credited points to the bin where that
possession started. MAUR assigns each decision to its event-start bin. CSSR
assigns each tactic call to its event-start bin. The existing whole-game
validity rules for PPP, MAUR and CSSR also apply to their bins. A bin with no
denominator is `undefined`; an early-ended game does not acquire artificial
zero rates in later bins. Missing clocks, states or required source evidence
yield `insufficient_evidence`. Unfinished games yield `incomplete`.

These series are numeric summaries only. Statistical tests, charts, transcript
tools and game-runner wiring are separate WS6 deliverables.
