# ChronoPace API Contract

Base URL: `http://localhost:8000`

## REST Endpoints

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
