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

## What is REAL vs MODELED

| REAL (2024 FastF1 / official F1 timing) | MODELED (`MODEL_ASSUMPTION`) |
|---|---|
| lap & sector timing, speed, throttle, brake, gear, RPM, DRS, distance | **rival hidden energy** — *inferred* probabilistically from observable performance (terminal speed, clipping point, corner-exit accel, sector delta). The 2024 cars' real ERS SoC is not public and is **never used.** |
| actual observable driver performance, lap by lap | **our own SoC / energy budget** — a 2026-model mean-reverting trajectory (`ReplayEnergyModel.next_soc`). `energy_is_modeled = true` on every lap. |
| relative gap (cumulative lap-time difference through lap N) | the **ChronoPace 2026 decision model**, Monte Carlo outcomes, opportunity probabilities, and regulatory-legality assumptions. |

**The 2024 cars did not run under ChronoPace's 2026 energy rules.** Real 2024
telemetry flows through the 2026 decision model. Regulatory constants
(`VERIFIED_FIA` vs `MODEL_ASSUMPTION`) are unchanged — see `docs/regulation.md`.

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

## Adding another race

Add an entry to `app/replay/races.py::HISTORICAL_RACES` (`year`, `event` string
FastF1 accepts, `scheduled_laps`, default driver/rival). Nothing else changes.
