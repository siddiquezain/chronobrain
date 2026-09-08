# ChronoPace API Contract

Base URL: `http://localhost:8000`

## Canonical decision endpoint (v1) — the one source of truth

```
POST /api/v1/decision
Body (all optional):
{
  "source": "synthetic",          // "synthetic" | "fastf1"
  "scenario": "B",                 // synthetic scenario A–E
  "seed": 42,
  "total_laps": 50,
  "lap": 30,                       // default: last lap
  "with_narrative": false,         // also fill `narrative` via LLM/fallback
  "overrides": {                   // manual race-input control (synthetic only)
    "initial_soc_mj": 7.2,         //   our SoC
    "rival_initial_soc_mj": 6.8,   //   HIDDEN rival SoC — sets the sim's ground truth;
                                    //   the estimator still infers it. Never echoed back here.
    "gap_to_car_ahead_s": 0.58,
    "gap_to_car_behind_s": 2.4,
    "rival_terminal_speed_kmh": 312.0,
    "lap_energy_deployed_mj": 1.2,
    "noise_scale": 1.0,            //   >1 = noisier rival telemetry
    "rival_obs_dropout": 0.0       //   fraction of laps with no rival observation
  },
  "fastf1": {                      // required only when source == "fastf1"
    "year": 2024, "event": "Monza", "session": "R",
    "our_driver": "VER", "rival_driver": "LEC", "laps": [30,31,32]
  }
}
→ DecisionSnapshot
```

```
GET /api/v1/scenario/{A..E}?seed=42&lap=30&narrative=false   → DecisionSnapshot
GET /api/v1/health   → { status, service, version, fastf1_available }
```

`DecisionSnapshot` shape:

```jsonc
{
  "meta":       { "lap", "total_laps", "data_mode", "source_detail", "seed",
                  "pipeline_version", "config_fingerprint", "generated_at" },
  "decision":   { "mode", "action", "confidence", "stage2_mode",
                  "confidence_overridden", "override_reason" },
  "data_quality": { "status",            // GOOD | DEGRADED | INVALID
                    "quality_score", "freshness_laps", "dropped_samples",
                    "out_of_order", "missing_fields": [ ], "checks": [ ] },
  "window":     { "n_laps", "speed_trend_kmh_per_lap", "gap_ahead_trend_s_per_lap",
                  "soc_trend_mj_per_lap", "rival_terminal_speed_trend",
                  "rival_sector_delta_trend", "closing",
                  "opportunity_trend" },   // IMPROVING | STABLE | DECAYING
  "energy":     { "soc_mj", "soc_pct", "lap_start_soc_mj", "deployed_this_lap_mj",
                  "deployment_headroom_mj", "projected_reserve_mj",
                  "projected_end_of_race_mj", "can_afford_aggressive", "energy_is_modeled" },
  "rival":      { "mean_reserve_mj", "reserve_std_mj", "n_observations", "confidence",
                  "estimate_uncertain", "bucket",                     // LOW|MEDIUM|HIGH
                  "distribution": { "low", "medium", "high" },        // sums to 1
                  "freshness_laps", "p_defend",                       // P(rival actively defends)
                  // dynamic strategic-rival selection — additive, null on the
                  // synthetic path and the fixed two-car replay:
                  "driver", "role",                                   // ATTACK_TARGET|DEFENDING_THREAT|POSITION_BATTLE|STRATEGICALLY_RELEVANT|NONE
                  "strategic_position", "strategic_gap_s",
                  "strategic_rival_ahead", "relevance_score" },
  "opportunity":{ "recommended_strategy", "prefers_wait", "foregone_strategy",
                  "foregone_value_gap_s", "future_energy_value_active", "opportunity_uncertain",
                  "opportunity_trend", "current_window_overtake_prob",
                  "projected_window_overtake_prob", "projected_window_lap",
                  "ranked_strategies": [ { "name","delay_laps","mean_horizon_delta_s",
                                           "std_horizon_delta_s","ci_lower_s",
                                           "end_soc_mj","energy_spent_mj",
                                           "current_opportunity_value","future_opportunity_value",
                                           "energy_opportunity_cost","strategic_value" } ],
                  "uncertainty_note" },
  "monte_carlo":{ "n_iterations", "seed", "recommended_mode",
                  "ranked_modes": [ { "mode","mean_laptime_delta_s","std_laptime_delta_s",
                                      "overtake_probability","sharpe_ratio","energy_cost_mj" } ] },
  "compliance": { "legal", "legal_modes", "illegal_modes",
                  "checks": [ { "rule","provenance","status","detail" } ] },  // provenance: VERIFIED_FIA|MODEL_ASSUMPTION|DEMO_CONSTANT
  "confidence": { "overall","statistical_reliability_passed","practical_significance_passed",
                  "dcli_passed","rival_confidence_passed","data_quality_passed",
                  "ci_lower_bound_s","t_statistic","dcli_score" },
  "constraints": { "regulatory",          // PASS | FAIL
                   "data_quality",         // GOOD | DEGRADED | INVALID
                   "energy" },             // OK | RESERVE_LOW
  "candidate_actions":     [ "ATTACK_NOW","WAIT_2_LAPS","WAIT_5_LAPS","CONSERVE","HOLD" ],
  "feasible_actions":      [ "..." ],      // the subset that survived legality + energy
  "rejected_alternatives": [ { "action","reason" } ],
  "reason_codes": [ "..." ],          // deterministic; from app/decision/reason_codes.VOCAB
  "reasons":      [ "..." ],          // human strings, same order
  "trace":        [ { "stage","detail" } ],   // deterministic decision trace
  "narrative":    "..."               // null unless with_narrative; explanation only, never authoritative
}
```

Determinism: identical body ⇒ identical response except `meta.generated_at`. The
*decision* is reproducible on any machine; the full JSON is byte-identical only
when the same ML model artifact is present (it feeds `opportunity.current_window_overtake_prob`
and `meta.config_fingerprint`). Train it with `python scripts/train_models.py`
(seeded — everyone gets the same model).

## Interactive demo + Rival Estimator validation — `/api/v1/demo/*`

Same deterministic pipeline. These endpoints add **manual input control**, a
**`rival_validation`** block (estimator vs. the simulator's hidden ground truth —
DEMO ONLY), and a **sequential estimator replay** for animation. Ground truth is
**never** in `POST /api/v1/decision`.

```
GET  /api/v1/demo/presets
  -> { "presets": [ { key, label, description, scenario, lap, overrides, expectation, pair_group } ] }

POST /api/v1/demo/preset/{key}?seed=42        // run a named preset
POST /api/v1/demo/decision                    // free-form; body = DecisionRequest + { telemetry_glitch }
  -> {
       "snapshot": DecisionSnapshot,           // identical shape to POST /api/v1/decision
       "rival_validation": {
         "enabled": true,
         "label": "ESTIMATED RIVAL ENERGY vs GROUND TRUTH — DEMO ONLY",
         "estimated_reserve_mj": 6.49, "estimated_std_mj": 0.90,
         "bucket": "HIGH", "distribution": { "low", "medium", "high" },
         "confidence": 0.72, "n_observations": 28, "estimate_uncertain": false,
         "latest_observation": { terminal_speed_kmh, clipping_point_fraction,
                                 corner_exit_accel_g, sector_delta_s },
         "particle_summary": { n_particles, effective_sample_size,
                               percentiles: {p05,p25,p50,p75,p95},
                               histogram: { bin_edges_mj:[13], weights:[12] } },
         "ground_truth_reserve_mj": 7.77,       // <- HIDDEN. simulator only. not telemetry.
         "error_mj": -1.28, "abs_error_mj": 1.28, "within_1_sigma": false,
         "ground_truth_note": "..."
       },
       "inputs_resolved": { ...effective sim preset..., "_preset": "...", "_expectation": "..." }
     }

POST /api/v1/demo/rival-trace                 // body { source, scenario, seed, up_to_lap, overrides, fastf1 }
  -> {
       "source": "synthetic",
       "ground_truth_available": true,          // false for fastf1 (F1 publishes no rival SoC)
       "note": "GROUND TRUTH — DEMO ONLY. ...",
       "steps": [ { lap, had_observation, observation, estimated_reserve_mj, estimated_std_mj,
                    n_observations, bucket, distribution, effective_sample_size,
                    uncertainty_trend: "more_certain|less_certain|flat",
                    ground_truth_reserve_mj, error_mj } ],   // one step per lap — the real filter state
       "final_particle_summary": { ...same as particle_summary above... }
     }
```

`source: "fastf1"` works on every demo endpoint too — it routes through the same
`load_replay()` -> `ReplayProvider` -> pipeline. `rival_validation.enabled` stays
`true` but `ground_truth_reserve_mj` is `null`.

## Historical race replay — `/api/v1/replay/*`

Real historical F1 telemetry, censored lap by lap, fed to the **same** `run_decision`
pipeline. No hindsight, no ground truth (F1 publishes no rival ERS SoC — ChronoPace
*infers* it). Needs `pip install -r requirements-fastf1.txt`; first call per session
downloads to `.fastf1_cache/`, later calls are offline.

```
GET  /api/v1/replay/races
  -> { "fastf1_available": true,
       "races": [ { key, name, circuit, year, session, scheduled_laps,
                    default_driver, default_rival, note } ] }

POST /api/v1/replay/historical
{ "race": "2024_italian_gp", "driver": "LEC", "rival": "PIA",   // rival = focus/fallback
  "start_lap": 1, "end_lap": 53, "seed": 42, "dynamic_rival": true }
  -> {
       "race": { key, name, circuit, year, session, scheduled_laps,
                 laps_with_telemetry, driver, rival, note },
       "driver": "LEC", "rival": "PIA", "total_laps": 53,
       "laps_replayed": [1, 53],
       "source": "fastf1_historical_replay", "model": "chronopace_2026",
       "strategic_rival": {                 // dynamic causal selection over the field
         "dynamic": true, "tick_level": true, "focus_rival": "PIA", "selector": "...",
         "drivers_tracked": ["NOR","PIA","SAI"],
         "changes": [ { lap, from, to, role, gap_s, relevance_score } ],  // lap-boundary
         "lap_boundary_changes": [ ... ], "intra_lap_changes_total": 60,
         "avg_switches_per_lap": 1.09,
         "timeline_endpoint": "GET /api/v1/replay/historical/2024_italian_gp/{lap}/timeline" },
       "provenance": { "REAL": [...], "MODELED (MODEL_ASSUMPTION)": [...], "note": "..." },
       "laps": [ {
         "lap": 11,
         "decision": "PUSH",            // ATTACK | WAIT | CONSERVE | HOLD | PUSH
         "action": "PUSH", "mode": "PUSH_MODE",
         "confidence": 0.60, "confidence_overridden": false, "override_reason": null,
         "reason_codes": [ ... ],
         "strategic_rival": { "driver": "NOR", "role": "DEFENDING_THREAT",
                              "position": 3, "gap_s": 0.59, "ahead": false,
                              "relevance_score": 0.83,
                              "tick_level": true,        // derived from telemetry-tick selection
                              "changes_this_lap": 2,     // sub-lap rival switches this lap
                              "tick_share": { "NOR": 0.61, "PIA": 0.39 } },
         "rival_energy_inference": {          // INFERRED, not measured — no ground truth
           "label": "RIVAL ENERGY INFERENCE (probabilistic, from observable performance)",
           "driver": "NOR",                  // whose observables this estimate is built from
           "estimated_reserve_mj", "std_mj", "bucket", "distribution",
           "confidence", "n_observations", "p_defend" },
         "opportunity": { recommended_strategy, prefers_wait, opportunity_trend,
                          current_window_overtake_prob, foregone_strategy, foregone_value_gap_s },
         "monte_carlo": { n_iterations, seed, recommended_mode, ranked_modes:[...] },
         "compliance": { legal, legal_modes },
         "energy": { soc_mj, can_afford_aggressive, energy_is_modeled:true },
         "data_quality": { status, quality_score },
         "gap_to_rival_s", "rival_role", "position", "drs", "our_speed_kmh",
         "lap_status", "rival_lap_status",
         "trace": [ { stage, detail } ]        // + STRATEGIC_RIVAL_SELECTED / _CHANGED
       } ]
     }
// dynamic_rival:false  -> the original fixed two-car analysis against `rival`.

GET  /api/v1/replay/historical/{race}/{lap}?driver=LEC&rival=PIA&seed=42&full_snapshot=false

GET  /api/v1/replay/historical/{race}/{lap}/timeline?driver=LEC&max_ticks=400
  -> { race, lap, our_driver, focus_rival,
       tick_cadence_hz, total_ticks_this_lap, returned_ticks, downsample_step,
       summary: { dominant_driver, dominant_role, dominant_ahead, dominant_gap_s,
                  first_driver, last_driver, n_changes, share_by_driver },
       change_events: [ { t, lap_fraction, from, to, role } ],
       ticks: [ { t, session_time_s, driver, role, ahead, gap_s, relevance,
                  switched, raw_leader } ],           // downsampled to max_ticks
       provenance: { REAL, MODELED } }
     // the telemetry-tick strategic-rival stream for one lap. Causal: each tick's
     // pick depends only on data <= that tick. 503 when the replay used the
     // lap-level fallback (no per-tick telemetry).
  -> { race, lap, censored_to_lap: {lap}, summary: <one lap object>,
       snapshot?: <full DecisionSnapshot when full_snapshot=true> }
```

`503` if `fastf1` is missing or the session cannot load (never fabricates data).
`404` for an unknown race key. Identical request ⇒ identical decisions (seeded;
FastF1's own cache makes the parse reproducible).

## Determinism

Identical body ⇒ identical response except `meta.generated_at` (for both the
production and demo endpoints). The *decision* is reproducible on any machine; the
full JSON is byte-identical only when the same ML model artifact is present (it
feeds `opportunity.current_window_overtake_prob` and `meta.config_fingerprint`).
Train it with `python scripts/train_models.py` (seeded — everyone gets the same model).

`POST /api/v1/decision` is the **single authoritative ChronoPace decision source.**
The frontend consumes this endpoint (or the WebSocket `decision_snapshot`, which
is the same object) and nothing else for decisions.

## Health

```
GET /health
→ { "status": "ok", "service": "ChronoPace", "version": "1.0.0" }
```

## DEPRECATED — legacy Stack B endpoints (do not use for decisions)

`GET /api/race/state`, `POST /api/race/update`, `GET /api/energy/state`,
`GET /api/overtake/current`, `GET /api/strategy/recommendation` are served by the
legacy Stack B engines. Their numbers **do not match** the v1 `DecisionSnapshot`.
They are flagged `deprecated: true` in the OpenAPI schema and every response
carries `Deprecation: true` + `Link: </api/v1/decision>; rel="successor-version"`.
They only return data while a simulation loop is running (else `404`).

(Legacy shapes unchanged — `RaceState`, `EnergyState`, `OvertakeAnalysis`,
`StrategyRecommendation` — see the OpenAPI schema. Not documented further here
because the frontend must not depend on them.)

### Simulation (active)

```
POST /api/simulation/start
Body: { "scenario": "B", "tick_interval_s": 2.0, "seed": 42 }
→ { "status": "started", "scenario": "B", "seed": 42 }

POST /api/simulation/stop
→ { "status": "stopped" | "not_running" }

GET /api/simulation/status
→ { "running": bool, "scenario": str, "tick": int, "elapsed_s": float }
```

## WebSocket

```
ws://localhost:8000/ws

Client → Server:  "ping"
Server → Client:  "pong"

Server broadcast (every simulation tick) — ONLY the authoritative snapshot:
{
  "type": "decision_snapshot",
  "tick": <int>,
  "snapshot": DecisionSnapshot   // identical shape to POST /api/v1/decision
}
```

No legacy race_state / energy / strategy payload is sent on the wire.

## Deployment Modes

| Mode | Description |
|------|-------------|
| `CONSERVE_MODE` | Minimum deployment — preserve energy |
| `BALANCED_MODE` | Standard race pace |
| `ARM_OVERTAKE_MODE` | Position for next-lap bonus deployment |
| `USE_OVERTAKE_BONUS_MODE` | Deploy banked overtake bonus |
| `PUSH_MODE` | Maximum legal deployment |
