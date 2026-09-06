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
                  "freshness_laps", "p_defend" },                     // P(rival actively defends)
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

Determinism: identical body ⇒ identical response except `meta.generated_at`.

The WebSocket broadcast (`/ws`) additionally carries `decision_snapshot` per tick.

## REST Endpoints (existing — unchanged)

### Health

```
GET /health
→ { "status": "ok", "service": "ChronoPace", "version": "1.0.0" }
```

### Race State

```
GET /api/race/state
→ RaceState { timestamp, lap, total_laps, position, gap_to_car_ahead_s,
              gap_to_car_behind_s, speed_kmh, soc_mj, soc_pct,
              tyre_compound, tyre_age_laps, sector, drs_available }

POST /api/race/update
Body: RaceStateUpdate (all fields optional)
→ RaceState (merged state)
```

### Energy

```
GET /api/energy/state
→ EnergyState { soc_mj, soc_pct, remaining_mj, deployed_this_lap_mj,
                harvested_this_lap_mj, deployment_headroom_mj,
                projected_reserve_mj, projected_end_of_race_mj,
                can_afford_aggressive }
```

### Overtake

```
GET /api/overtake/current
→ OvertakeAnalysis { score, probability, gap_s, closing_speed_mps,
                     slipstream_factor, braking_zone_score, corner_exit_score,
                     energy_advantage_score, contributing_factors,
                     closing_speed_score, gap_score }
```

### Strategy

```
GET /api/strategy/recommendation
→ StrategyRecommendation { timestamp, lap, recommended_mode, confidence,
                            overtake{}, energy{}, risk{}, regulatory{},
                            decision{}, explanation, utility, reason_codes }
```

### Simulation

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

Server broadcast (every tick):
{
  "race_state": { ... },
  "energy":     { ... },
  "overtake":   { ... },
  "strategy":   { ... }
}
```

## Deployment Modes

| Mode | Description |
|------|-------------|
| `CONSERVE_MODE` | Minimum deployment — preserve energy |
| `BALANCED_MODE` | Standard race pace |
| `ARM_OVERTAKE_MODE` | Position for next-lap bonus deployment |
| `USE_OVERTAKE_BONUS_MODE` | Deploy banked overtake bonus |
| `PUSH_MODE` | Maximum legal deployment |
