# ChronoPace — Architecture

## Overview

ChronoPace is a decision-support engine that answers: **"Is this moment worth spending finite electrical energy?"**

It processes real-time Formula E telemetry through a deterministic 4-stage intelligence pipeline and broadcasts strategy recommendations over WebSocket.

## Directory Structure

```
ChronoPace-Backend/
├── rule_gate.py            # Stage 1 — FIA 2026 Regulatory Gate
├── rival_estimator.py      # Particle filter rival SoC estimation
├── planner.py              # Stage 2 — Monte Carlo lap-time planner
├── confidence_gate.py      # Stage 3 — Statistical + DCLI significance filter
├── opportunity_engine.py   # Multi-lap horizon comparison (ATTACK_NOW / WAIT / HOLD)
├── narrator.py             # Stage 4 — Structured narrative + optional Claude haiku
├── telemetry_simulator.py  # Scenario-based simulation telemetry
├── app/                    # FastAPI service layer
│   ├── main.py             # App entry point
│   ├── api/                # HTTP routes + WebSocket
│   ├── core/               # Config, logging, state
│   ├── engines/            # Stateless computation engines
│   ├── ml/                 # RandomForest overtake success classifier
│   ├── models/             # Pydantic data models
│   ├── pipeline/           # Ingestion → preprocessing → engines
│   ├── simulation/         # Scenario config + tick simulator
│   └── utils/              # Math helpers and validation
```

## Intelligence Pipeline

```
Telemetry
    │
    ▼
Stage 1: RegulatoryGate (rule_gate.py)
    • FIA 2026 Art.5.4.10 — deployment cap (9 MJ/lap)
    • FIA 2026 Art.5.4.9  — SoC swing cap (4 MJ)
    • Proximity check     — gap ≤ 1.0s for overtake modes
    • Bonus banking       — overtake_qualified_last_lap flag
    → GateResult (legal_modes, violations, base_cap_mj)
    │
    ▼
Stage 2: MonteCarloPlanner (planner.py)
    • 10,000 iterations, FIXED — seeded via SeedSequence.spawn(5)
    • 5 canonical mode streams — byte-identical across runs
    • Rival SoC modulation via RivalStateEstimator particle filter
    • Sharpe ratio selection, capped ±999
    → PlannerResult (mode_results, recommended_mode, run_id)
    │
    ▼
Stage 3: ConfidenceGate (confidence_gate.py)
    • Statistical reliability — Welch t-test (p < 0.05)
    • Practical significance — |Δlaptime| > 0.05s, |ΔP(overtake)| > 3pp
    • DCLI — Driver Cognitive Load Index ≤ 60.0
    • Rival confidence — rival SoC std ≤ 1.5 MJ
    • Override: falls back to BALANCED_MODE naming all failing gates
    → ConfidenceGateResult
    │
    ▼
Stage 4: Narrator (narrator.py)
    • build_reason_codes() — fixed 13-code vocabulary
    • derive_confidence()  — deterministic from CI margin
    • narrate()            — Claude haiku if ANTHROPIC_API_KEY, else structured fallback
    → StrategyCall (decision, confidence, reason_codes, narration)
```

## Sign Convention (Critical)

The Monte Carlo planner uses: **diff = mean(runner_up) - mean(top)**

Negative delta = runner_up is faster (smaller lap time) than top. This is intentional — lower lap time is better.

## Key Constants (FIA 2026)

| Constant | Value | Regulation |
|---------|-------|------------|
| `max_deployment_per_lap_mj` | 9.0 MJ | Art. 5.4.10 |
| `max_delta_soc_mj` | 4.0 MJ | Art. 5.4.9 |
| `overtake_bonus_mj` | 0.5 MJ | Bonus pool |
| `overtake_detection_gap_threshold_s` | 1.0 s | Proximity |

## One-Way Import Law

`rule_gate.py` must **never** import from `rival_estimator.py`. The regulatory gate is a pure function of telemetry; rival uncertainty must not influence legality checks.
