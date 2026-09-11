# ChronoPace — Interactive Demo & Rival-Estimator Validation

This layer lets you **change race inputs** and watch the deterministic Python
engine respond, and it lets you **prove the Rival Energy Estimator works** by
scoring it against a hidden ground truth it never receives.

It adds **no** strategy logic. The pipeline is unchanged:

```
inputs -> TelemetrySimulator -> NormalizedLap -> feature extraction
       -> Rival Energy Estimator -> Opportunity Horizon -> Regulatory Gate
       -> Monte Carlo -> Confidence Gate -> Decision Engine -> DecisionSnapshot
```

## Run it

```bash
uvicorn app.main:app --reload --port 8000
# Swagger: http://localhost:8000/docs  (the "demo" tag)
```

CLI:

```bash
python scripts/run_decision.py B --lap 25          # existing
```

## The five demo presets

`GET /api/v1/demo/presets` → then `POST /api/v1/demo/preset/{key}`.

| Key | Inputs | What the engine does (computed, not hardcoded) |
|---|---|---|
| `HIGH_ENERGY_STRONG_OPPORTUNITY` | scenario B, lap 25 | `USE_OVERTAKE_BONUS_MODE / ATTACK_NOW` |
| `LIMITED_ENERGY_SAME_OPPORTUNITY` | B, lap 25, `initial_soc_mj = 1.6` | `CONSERVE_MODE / CONSERVE` — same window, energy not worth spending |
| `BETTER_FUTURE_OPPORTUNITY` | scenario C, lap 20 | `CONSERVE_MODE / WAIT_5_LAPS` — the horizon defers |
| `LOW_CONFIDENCE_BAD_DATA` | B, lap 24, `telemetry_glitch` | `BALANCED_MODE / HOLD`, `override_reason = DATA_QUALITY` |
| `RIVAL_ENERGY_HIGH` / `RIVAL_ENERGY_LOW` | B, lap 28, gap 0.6, hidden `rival_initial_soc_mj` = 8.5 / 1.0 | decision robust here; the **estimator posterior** moves ~6.5→~3.2 MJ from the *same* visible gap |

Presets 5a/5b share `pair_group = "RIVAL_ENERGY_CHANGE"` — show them side by side.

## Manual control (free-form)

`POST /api/v1/demo/decision` (or `POST /api/v1/decision` — same `overrides`, minus
the validation block):

```json
{
  "scenario": "B", "seed": 42, "lap": 25,
  "overrides": {
    "initial_soc_mj": 7.2,
    "rival_initial_soc_mj": 6.8,
    "gap_to_car_ahead_s": 0.58,
    "noise_scale": 1.0
  }
}
```

`rival_initial_soc_mj` sets only the **simulator's hidden state**. The estimator
sees only the four kinematic observables generated from it.

## Proving the Rival Energy Estimator to a judge

1. **Same conditions, different hidden state.** Run `RIVAL_ENERGY_HIGH` then
   `RIVAL_ENERGY_LOW`. Identical visible gap (0.6 s) and a similar speed profile.
   `rival_validation.estimated_reserve_mj` moves from ~6.5 MJ (bucket HIGH) to
   ~3.2 MJ (bucket LOW/MEDIUM). The estimator was never told either number.

2. **Watch it update.** `POST /api/v1/demo/rival-trace` with `up_to_lap: 20`.
   Each `steps[i]` is the **actual particle-filter state** after observation *i*:
   `estimated_reserve_mj`, `estimated_std_mj`, `distribution`,
   `effective_sample_size`, `uncertainty_trend`. The mean walks toward the hidden
   state; the std stays > 0.35 MJ (a deliberate floor) — it never collapses.

3. **Show the histogram.** `final_particle_summary` is a compact view of the
   1000-particle posterior — 12 weighted bins + 5 percentiles. Not thousands of
   particles.

4. **Show it's genuinely hidden.** `rival_validation.ground_truth_reserve_mj`
   appears **only** under `/api/v1/demo/*`. `POST /api/v1/decision` has no such
   field, and `RivalObservation` (the estimator's input) has no SoC field at all.

**Honest framing:** the estimator is **not calibrated or validated against real
F1 telemetry.** Errors are ~0.3–0.7 MJ mid-range and larger at the extremes
(particles are bounded at 0–9 MJ). The demonstrable claims are: *it responds to
changing evidence in the right direction, and it keeps honest uncertainty.*

## FastF1 historical replay

Every endpoint above also accepts `"source": "fastf1"` with a `fastf1` block
(`year, event, session, our_driver, rival_driver`). It routes through the
existing `app/data/fastf1_service.load_replay()` → `ReplayProvider` → **the same
pipeline**. Needs `pip install -r requirements-fastf1.txt` and a cached session.
`rival_validation.ground_truth_reserve_mj` is `null` (F1 publishes no rival ERS
SoC; replayed SoC is modelled — `energy_is_modeled: true`).

---

## Recommended frontend layout — CHRONOPACE CONTROL ROOM

```
┌──────────────────────────── INPUTS ────────────────────────────┐
│  Our Energy        [====|====]  7.2 MJ                          │
│  Rival Hidden SoC  [======|==]  6.8 MJ   ⚠ DEMO ONLY (hidden)   │
│  Gap to rival      [=|=======]  0.58 s                          │
│  Lap               25 / 50                                      │
│  Telemetry noise   [|========]  1.0×      [ ] glitch feed       │
│  Preset ▾  HIGH ENERGY / STRONG OPPORTUNITY                     │
│                     [  RUN DECISION  ]                          │
└────────────────────────────────────────────────────────────────┘

┌──────────────── RIVAL ENERGY ESTIMATOR ────────────────────────┐
│  Estimated Reserve   6.55 MJ  ± 0.82 MJ    (28 observations)    │
│  LOW  ▏8%   MEDIUM ▍21%   HIGH ███████ 71%                      │
│  ┌ posterior histogram (12 bins) ─────────────────┐            │
│  │        ▁▂▄██▆▃▁                                 │            │
│  └──────────────────────────────────────────────────┘         │
│  ── DEMO ONLY ─────────────────────────────────────            │
│  Ground Truth        6.80 MJ      Estimation Error   -0.25 MJ   │
│  within 1σ ✓                                                    │
│                                                                │
│  [ ▶ REPLAY ESTIMATOR ]   lap 1 ──────●──────────── lap 28      │
│    (animates rival-trace: mean, ±σ band, bucket bars)          │
└────────────────────────────────────────────────────────────────┘

┌───────────── CHRONOPACE RECOMMENDATION ────────────────────────┐
│   USE OVERTAKE BONUS  ·  ATTACK NOW        confidence 82%       │
│   ✓ compliance   ✓ 5/5 confidence gates   feasible: 5 actions  │
│   reasons: strong current window · rival low energy · ...      │
│   [ decision trace ▾ ]  (11 steps, from telemetry to FINAL)    │
└────────────────────────────────────────────────────────────────┘
```

All numbers come from the backend: `snapshot.*` for the recommendation,
`rival_validation.*` for the estimator panel, `rival-trace.steps[]` for the
animation. The frontend computes nothing.
