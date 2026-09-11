# ChronoPace — Historical Race Replay

Replay a real F1 race **as if ChronoPace were operating during it**, using only
information available up to the current lap. No hindsight. The real 2024 telemetry
flows through the **existing** ChronoPace decision pipeline, lap by lap.

## Architecture

```
FastF1 historical session            app/data/fastf1_service.load_replay()
  (2024 Italian GP, LEC vs PIA)
        │  cache: .fastf1_cache/  (downloaded once, then offline)
        ▼
list[NormalizedLap]                   one CAUSAL lap each:
  · lap N fields = lap N samples + rolling stats of laps < N only
  · total_laps = scheduled race distance (53, known pre-race)
  · energy_is_modeled = True on every lap
        ▼
for lap N in start..end:              app/replay/historical.run_historical_replay()
  censored = [nl for nl in full if nl.lap <= N]      ◄── HINDSIGHT BARRIER
  provider = ReplayProvider(censored)
  snapshot = run_decision(provider, lap=N, config=DecisionConfig(seed))   ◄── existing pipeline
        ▼
compact per-lap summary  (no ground truth, no future data)
```

`run_decision` is `app.decision.engine.run_decision` — the same function the
synthetic scenarios and the demo layer call. There is **no** replay-specific
decision engine, Monte Carlo, rival estimator, or confidence gate.

## Where hindsight is prevented — three independent guards

1. **`load_replay` builds each lap causally.** The rival sector-delta baseline and
   the throttle/brake energy baselines are *rolling means of earlier laps only*
   (`_rolling_mean`), and the gap is the cumulative lap-time difference *through*
   lap N. `total_laps` comes from the race registry (`scheduled_laps = 53`), not
   the actually-completed count.
2. **`run_historical_replay` slices to `laps <= N`** and hands the engine a fresh
   `ReplayProvider` that *physically contains only* laps 1..N.
3. **`run_pipeline` censors again** — `history = [nl for nl in all_laps if nl.lap <= target]`.

Tested by `tests/test_historical_replay.py`:
`test_lap_n_provider_contains_only_laps_up_to_n` (the provider handed to the engine
for lap N has `lap_list == [1..N]`) and
`test_future_telemetry_cannot_change_a_lap_n_decision` (corrupt every lap > 8 with
wild values → the Lap 8 decision is byte-identical).

## Dynamic strategic-rival selection — telemetry-tick cadence

A real race is not a permanent two-car duel, and the fight can change between
corners. `POST /api/v1/replay/historical` (`dynamic_rival: true`, the default)
re-selects the opponent that matters at **every FastF1 telemetry tick** (the
provider's real cadence — ~4 Hz / ~330 samples per lap on 2024 data, **not** 1 ms),
from the whole field, using only data up to that tick.

```
session (all drivers' car_data, already parsed)
  -> app.replay.field_state.FieldTimeline.build(session, our_driver)   [cached per session]
       (T, D) matrices: race-progress / speed / running-position / lap-status,
       every channel sampled onto our tick grid with a strict previous-value hold
  -> vectorised relevance = strategic_rival.score_matrix(...)   (locked to _score_candidate)
  -> tick pick + hysteresis (switch_margin 0.10, min_dwell_ticks 12)  -> TickSelection[]
  -> per-lap dominant rival by time-share   ==  the value on NormalizedLap.strategic_rival
```

* **Lap-cadence decisions, tick-cadence rival.** The heavy pipeline (energy
  inference, Monte Carlo, gates, decision) still runs once per lap, on the
  **dominant** tick-level rival for that lap. The full sub-lap stream is exposed
  separately: `GET /api/v1/replay/historical/{race}/{lap}/timeline`.
* The lap value is a **summary** of the ticks, not an independently computed
  rival: `strategic_rival.changes_this_lap`, `.tick_share`, `.tick_level`.
* **Hysteresis.** A challenger must beat the incumbent's *live* score by
  `switch_margin` **and** the incumbent must have been held `min_dwell_ticks`
  (~3 s) — so two cars at a near-identical gap don't trade the title on
  reconstruction noise. The tick-level closing-rate signal is hard-clamped below
  the switch margin (public data can't resolve a 0.2 s closing rate).
* **Direction** (ahead/behind) is the official per-lap running position held
  forward — public GPS does not resolve a 0.2 s side-by-side. An overtake flips
  the role at the lap the position table updates (≤ ~1 lap latency); a genuine
  sub-second side-by-side reads as `POSITION_BATTLE`.
* **Identity-aware energy.** One particle filter **per tracked driver**. Each sees
  only that driver's own observations; a returning rival *resumes* its filter, and
  no filter is ever fed another driver's data.
* **No hindsight.** `FieldTimeline` samples are previous-value-hold only; track
  length and reference lap time come from laps ≤ 6 (circuit constants). Corrupting
  any telemetry after tick *t* leaves every selection at *t* byte-identical
  (`tests/test_tick_rival_replay.py::test_future_telemetry_cannot_change_tick_selection`).

### Lap-level (fallback)

When per-tick telemetry is unavailable the lap-level selector
(`strategic_rival.build_strategic_rivals`, `strategic_rival: {tick_level: false}`)
is used instead — same scoring model over per-lap position / gap / trend / pace.

### (the lap-level path in detail)

```
full field (all drivers' position / gap / trend / pace / pit status, <= lap N)
      -> app.replay.strategic_rival.build_strategic_rivals   (deterministic score)
      -> the SELECTED rival's observable kinematics
      -> the existing Rival Energy Estimator (particle filter)
      -> opportunity / Monte Carlo / gates / decision   (unchanged)
```

* The score is `w_proximity·proximity + w_adjacency·adjacency +
  w_gap_trend·trend + w_pace·pace + w_directional·directional`, damped for a
  pitting car or a car past the strategic gap ceiling. All weights/thresholds are
  in `RivalSelectorConfig` (MODEL_ASSUMPTION — a deterministic model, not learned).
* Role (`strategic_rival.classify_role`, one function for both the lap-level and
  tick-level paths):
  * `ATTACK_TARGET` — directly ahead **and** within the overtake-proximity range (racing)
  * `DEFENDING_THREAT` — directly behind **and** within that range (racing)
  * `POSITION_BATTLE` — adjacent in the running order **and** within
    `position_battle_max_gap_s` (2.0 s) — a genuine fight for track position
  * `STRATEGICALLY_RELEVANT` — relevant, but neither in immediate range nor a
    close position fight (adjacent-but-distant, or 2+ places away)
  * `NONE` — nobody clears the relevance floor (the decision still runs)
* **Identity-aware energy**: one particle filter **per tracked driver**. Each sees
  only that driver's own observations; when the strategic rival switches away and
  later returns, its filter **resumes** from where it left off — it is not
  restarted, and it never sees another driver's data. Per-driver deterministic
  seeding.
* `rival:` in the snapshot gains `driver` / `role` / `strategic_position` /
  `strategic_gap_s` / `strategic_rival_ahead` / `relevance_score` (all additive;
  `None` on the synthetic path and the fixed two-car replay). The trace emits
  `STRATEGIC_RIVAL_SELECTED` and `STRATEGIC_RIVAL_CHANGED`.
* `GET .../{lap}/timeline?debug=true` returns, per tick, **every candidate's
  relevance score + gap + ahead/behind + closing rate + position**, the selected
  rival, whether a switch happened, and *why* (`margin_exceeded` /
  `hysteresis_held` / `dwell_not_met` / `held` / `initial` / `incumbent_gone`).
  Every number comes from the engine.
* `dynamic_rival: false` keeps the original fixed two-car analysis against `rival`.
* The `rival` request parameter is now the **focus / fallback** rival (used on
  laps where no opponent clears the relevance floor).

2024 Monza (LEC): the selector tracks **NOR, PIA, SAI, VER, OCO** across the race
— NOR/PIA in the opening stint, midfield cars during Leclerc's out-lap recovery,
SAI while Piastri pits late, PIA again as Piastri closes to the flag. Twelve
switches, all causal.

## Pit stops and who is ahead

Two things the replay derives from **data available at lap N** (never the future):

* **Ahead vs behind.** The relative gap is the unsigned cumulative lap-time
  difference, but it is placed on the correct side using the drivers' real running
  `Position` that lap (falling back to the cumulative-time sign). When our driver
  leads, the gap is `gap_to_car_behind_s` and `gap_to_car_ahead_s` is `None` — so
  the engine frames those laps as *defending*, and `PUSH_MODE` can actually be
  selected. Before this, every lap was framed as "chase the rival ahead", even the
  15 laps Leclerc led at Monza.
* **Pit / out / invalid laps.** FastF1's `PitInTime` / `PitOutTime` / `IsAccurate`
  mark a lap as `pit`, `out_lap` or `invalid_for_energy_inference` (a backstop also
  trips on a lap time > 5 s off the rolling baseline). Such a lap keeps its raw
  telemetry but is **not clean racing evidence**: its four rival observables are
  dropped (the particle filter predicts, it does not update — a +20 s pit-lap
  sector delta never collapses the posterior to "battery empty"), the relative gap
  is withheld (no fake overtake window), our modelled SoC is carried across
  unchanged, and the Data Quality Gate marks the lap `DEGRADED`.

## Telemetry cadence (honest)

FastF1 historical car telemetry is **~4 Hz** (source-dependent — measured per
replay from the reconstructed tick grid, ~330 samples/lap on 2024 data). It is
**not a 128 Hz raw stream** and not a live feed. The replay response carries a
`telemetry` block: `{ source, kind: "HISTORICAL TELEMETRY REPLAY",
cadence_hz_measured, cadence_note, ticks_total, is_live: false, is_synthetic:
false }`. Synthetic mode is labelled `data_mode: "SYNTHETIC"` and
`energy_is_modeled: false` (the simulator's SoC is its own ground truth).

## Modelled energy — explicit accounting

F1 publishes **no ERS state of charge**, so our SoC is modelled. It is an explicit
per-lap accounting relationship (`ReplayEnergyModel`), causal, deterministic:

```
deployed(lap)  = 2.6 MJ * throttle_fraction
recovered(lap) = 2.3 MJ * brake_fraction  +  1.9 MJ * (1 - throttle_fraction)
net_swing      = (recovered - deployed) - (recovered_nominal - deployed_nominal)
SoC_next       = clip( SoC + net_swing + 0.05*(4.5 - SoC),  0.3 MJ,  9.0 MJ )
```

Every field is exposed and labelled MODELED: `soc_mj`, `soc_pct`,
`soc_capacity_mj`, `deployed_this_lap_mj`, `recovered_this_lap_mj`,
`net_swing_mj`, `modeled_mgu_k_peak_kw`, `mgu_k_power_ceiling_kw`. The compliance
block's `MGU-K power ceiling` check now verifies `modeled_mgu_k_peak_kw <=
max_ers_k_power_kw` (a real `pass`/`breach`, not a hardcoded string). The 2024
cars did **not** run under the 2026 energy budget.

## What is REAL vs MODELED vs INFERRED

| REAL (2024 FastF1 / official F1 timing) | MODELED (`MODEL_ASSUMPTION`) | INFERRED (probabilistic) |
|---|---|---|
| speed, throttle, brake, gear, RPM, DRS, distance; lap & sector timing; running position; relative gap | our own SoC / deployment / recovery / net swing / MGU-K peak power; the ChronoPace 2026 decision model, Monte Carlo outcomes, opportunity probabilities, regulatory-legality assumptions; which opponent is "strategically relevant" | the **strategic rival's hidden energy** distribution (mean / std / bucket) from observable performance — no car's real ERS SoC is public or used |

Each replay lap carries a `provenance` block spelling this out, plus the note:
**REAL 2024 OBSERVATION + CHRONOPACE 2026 MODEL = REPLAYED 2026 DECISION CONTEXT.**

**The 2024 cars did not run under ChronoPace's 2026 energy rules.** Regulatory
constants (`VERIFIED_FIA` vs `MODEL_ASSUMPTION`) are unchanged — see
`docs/regulation.md`.

Never claim *"we know Piastri's actual battery state."* ChronoPace **infers a
probabilistic rival energy state from observable historical performance** —
labelled `RIVAL ENERGY INFERENCE` everywhere.

## Run it

```bash
pip install -r requirements-fastf1.txt          # fastf1==3.4.4
uvicorn app.main:app --port 8000
curl -s localhost:8000/api/v1/replay/races | jq
curl -s -X POST localhost:8000/api/v1/replay/historical \
  -H 'content-type: application/json' \
  -d '{"race":"2024_italian_gp","start_lap":1,"end_lap":53,"seed":42}' | jq '.laps[9]'
curl -s "localhost:8000/api/v1/replay/historical/2024_italian_gp/11?full_snapshot=true" | jq
```

The **first** call for a session downloads ~30 MB to `.fastf1_cache/` (needs
network, ~15–60 s). Every call after that is offline and deterministic. If
`fastf1` is missing or the download fails → HTTP `503` with a clear message.

## Race library — Season → Grand Prix → Session

Two layers:

* **Curated** (`app/replay/races.py::HISTORICAL_RACES`) — the featured races, each
  with a default driver/rival and a note. `GET /api/v1/replay/races` (no args).
* **Discovery** (`app/replay/discovery.py`) — any telemetry-supported race in the
  **2019–2025** window, straight from `fastf1.get_event_schedule`:

  ```
  GET /api/v1/replay/seasons
  GET /api/v1/replay/races?season=2023
  GET /api/v1/replay/sessions?season=2023&race=Italian Grand Prix
  POST /api/v1/replay/historical  { "season": 2023, "event": "Italian Grand Prix",
                                    "session": "R", "driver": "VER", "rival": "SAI" }
  ```

  `event` accepts a Grand Prix name (loose match) or a round number. Unknown /
  pre-telemetry season → `503`; unknown event/session → `404`; a non-registry
  race with no `driver` → `422`. Schedules are cached per season in-process; a
  cached race needs no network.

To add a **featured** race, add a `HISTORICAL_RACES` entry (`year`, `event`
string FastF1 accepts, `scheduled_laps`, default driver/rival). Nothing else
changes.
