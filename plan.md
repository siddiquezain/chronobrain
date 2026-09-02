# TrackShift 2026 — Motorsport Intelligence Engine

## Claude Code Master Implementation Prompt

You are the lead backend/ML engineer for **TrackShift 2026**, an AI-assisted motorsport intelligence system focused on:

> **Energy & Overtake Intelligence**

Build the complete backend from scratch in a new repository.

The backend must be production-quality in structure, highly explainable, modular, testable, and suitable for a hackathon demonstration. Do NOT build a toy CRUD API or a collection of disconnected scripts.

The system must simulate/ingest race-state data, maintain the current race state, evaluate energy availability, detect and score overtake opportunities, enforce regulatory constraints, and produce an explainable strategy recommendation that the separate Next.js frontend can consume.

---

# 1. CORE ARCHITECTURE

Use this architecture:

```text
                    ┌─────────────────────────┐
                    │     Race Data Input     │
                    │                         │
                    │ CSV / simulation /      │
                    │ future live telemetry   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Data Pipeline       │
                    │                         │
                    │ validation              │
                    │ normalization           │
                    │ feature engineering     │
                    └────────────┬────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │      Motorsport Intelligence        │
              │                                     │
              │ Energy Engine                        │
              │ Overtake Engine                      │
              │ Risk Engine                          │
              │ Regulatory Gate                      │
              │ Strategy Optimizer                   │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Decision / State API  │
                    │                         │
                    │ REST + WebSocket        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     TrackShift UI       │
                    │       Next.js            │
                    └─────────────────────────┘
```

The backend must remain completely independent of the frontend repository.

---

# 2. TECHNOLOGY STACK

Use:

* Python 3.11+
* FastAPI
* Pydantic v2
* NumPy
* Pandas
* scikit-learn
* joblib
* Uvicorn
* pytest
* httpx for API tests
* WebSockets through FastAPI
* python-dotenv
* logging module

Prefer lightweight dependencies.

Do NOT introduce unnecessary infrastructure such as Kafka, Redis, Celery, Kubernetes, Docker orchestration, databases, etc. unless genuinely required.

For the hackathon, an in-memory race state plus file-based datasets is sufficient.

---

# 3. REPOSITORY STRUCTURE

Create this structure:

```text
trackshift-engine/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── race.py
│   │   │   ├── energy.py
│   │   │   ├── overtake.py
│   │   │   ├── strategy.py
│   │   │   └── simulation.py
│   │   └── websocket.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── state.py
│   │
│   ├── models/
│   │   ├── race.py
│   │   ├── telemetry.py
│   │   ├── energy.py
│   │   ├── overtake.py
│   │   ├── strategy.py
│   │   └── regulatory.py
│   │
│   ├── engines/
│   │   ├── energy_engine.py
│   │   ├── overtake_engine.py
│   │   ├── risk_engine.py
│   │   ├── regulatory_gate.py
│   │   └── strategy_engine.py
│   │
│   ├── ml/
│   │   ├── features.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── models/
│   │
│   ├── pipeline/
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── race_pipeline.py
│   │
│   ├── simulation/
│   │   ├── simulator.py
│   │   ├── scenarios.py
│   │   └── telemetry_generator.py
│   │
│   └── utils/
│       ├── math.py
│       └── validation.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── tests/
│   ├── test_energy.py
│   ├── test_overtake.py
│   ├── test_regulatory.py
│   ├── test_strategy.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── scripts/
│   ├── train_models.py
│   └── run_simulation.py
│
├── models/
│
├── docs/
│   ├── architecture.md
│   ├── api-contract.md
│   └── methodology.md
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── pyproject.toml
```

Keep modules small and logically separated.

---

# 4. SYSTEM OBJECTIVE

At every race update, the system should answer:

### Energy

* How much usable energy remains?
* What is current SoC?
* How much energy is being deployed?
* How much is being harvested?
* What is the projected energy reserve?
* Can the driver afford an aggressive deployment?

### Overtake

* Is an overtake opportunity developing?
* What is the gap?
* What is the closing speed?
* How strong is the slipstream?
* How favorable is the braking zone?
* How favorable is the corner exit?
* What is the estimated probability of a successful overtake?
* What is the risk?

### Regulation

* Is the proposed action legal?
* Does it violate configured deployment constraints?
* Is the action inside the permitted strategy/action space?

### Strategy

Choose the best legal action based on:

```text
Immediate Gain
        +
Overtake Probability
        +
Energy Availability
        +
Future Energy Cost
        -
Failure Risk
        -
Defensive/Strategic Risk
```

The final recommendation must be explainable.

---

# 5. TELEMETRY / RACE STATE MODEL

Create a Pydantic model representing race state.

At minimum include:

```text
timestamp
lap
total_laps

position
gap_to_car_ahead
gap_to_car_behind

speed
closing_speed

throttle
brake
braking_point

sector
corner_id

straight_distance
distance_to_next_corner

slipstream_factor

soc
energy_deployment
energy_harvest
energy_remaining
energy_budget

tyre_age
tyre_compound

drs_available
overtake_opportunity

track_position
```

Use sensible units and document them.

Do not silently mix km/h and m/s.

Internally prefer SI units where appropriate.

---

# 6. ENERGY ENGINE

Implement a deterministic and explainable energy model.

Track:

```text
SoC
Energy Remaining
Energy Deployed
Energy Harvested
Energy Budget
Projected End-of-Lap Energy
Projected End-of-Race Energy
```

Implement:

```text
update_energy_state()
calculate_energy_budget()
calculate_projected_reserve()
calculate_deployment_headroom()
```

The engine should distinguish:

### Harvesting

Energy gained through regenerative braking / configured harvesting model.

### Deployment

Energy consumed during acceleration/deployment phases.

### Reserve

Energy that should remain available for future strategic situations.

Calculate something conceptually like:

```text
projected_reserve =
    current_energy
    + expected_future_harvest
    - expected_future_deployment
```

Avoid pretending this is an FIA-accurate physical model unless actual validated parameters are available.

Clearly label the system as a **decision-support simulation/model**.

---

# 7. ENERGY DEPLOYMENT MODES

Implement five strategy modes:

```text
HARVEST
BALANCED
ATTACK
DEFEND
OVERTAKE_BONUS
```

Each mode should have configurable characteristics:

```text
deployment intensity
expected energy cost
risk
priority
```

Example conceptual behavior:

```text
HARVEST
→ conserve energy / recover

BALANCED
→ normal race pace

ATTACK
→ higher deployment for pace

DEFEND
→ prioritize maintaining position

OVERTAKE_BONUS
→ aggressive short-duration deployment when
   a high-quality overtake window exists
```

Do NOT hardcode arbitrary magic numbers throughout the code.

Put configuration in one location.

---

# 8. OVERTAKE WINDOW ENGINE

Build an overtake opportunity detector.

Calculate features including:

```text
gap
closing_speed
relative_speed
straight_distance
distance_to_braking_zone
slipstream_factor
corner_exit_quality
track_position
energy_advantage
```

Create a normalized:

```text
overtake_score ∈ [0, 1]
```

and:

```text
overtake_probability ∈ [0, 1]
```

The score should be explainable.

For example:

```text
Overtake Score =
    0.25 × closing_speed_score
  + 0.20 × gap_score
  + 0.20 × braking_zone_score
  + 0.15 × slipstream_score
  + 0.10 × corner_exit_score
  + 0.10 × energy_advantage_score
```

These weights must be configurable.

Do not claim they represent real F1 team weights.

---

# 9. RISK ENGINE

Create an independent risk engine.

Estimate:

```text
overtake_risk
energy_risk
strategic_risk
overall_risk
```

Risk should increase when:

* gap is large
* closing speed is weak
* braking opportunity is poor
* energy reserve is low
* proposed deployment is excessive
* track position is unfavorable

Keep this deterministic and explainable.

---

# 10. REGULATORY GATE

Create:

```text
regulatory_gate.py
```

The regulatory gate must be evaluated BEFORE optimization.

Architecture:

```text
Candidate Actions
       ↓
Regulatory Gate
       ↓
Legal Actions
       ↓
Strategy Optimizer
       ↓
Best Legal Action
```

Never allow the optimizer to recommend an action that the regulatory gate has marked illegal.

Represent results as:

```json
{
  "legal": true,
  "violations": [],
  "constraints_checked": []
}
```

Make regulatory constraints configurable.

IMPORTANT:

Do not invent claims that a particular simulated constraint exactly matches the FIA sporting/technical regulations.

Clearly distinguish:

```text
FIA-inspired constraint
```

from:

```text
official FIA rule implementation
```

unless a verified rule has actually been implemented.

---

# 11. STRATEGY ENGINE

Create candidate actions:

```text
HARVEST
BALANCED
ATTACK
DEFEND
OVERTAKE_BONUS
```

For every action calculate:

```text
expected_position_gain
overtake_probability
energy_cost
future_energy_impact
risk
strategic_value
```

Then calculate an overall utility:

```text
utility =
    immediate_gain
    + overtake_probability * opportunity_value
    + strategic_value
    - energy_cost_penalty
    - risk_penalty
```

Then:

```text
candidate actions
        ↓
regulatory filtering
        ↓
utility calculation
        ↓
best legal action
```

Return:

```json
{
  "recommended_mode": "OVERTAKE_BONUS",
  "confidence": 0.91,
  "utility": 0.84,
  "reason_codes": [
    "HIGH_CLOSING_SPEED",
    "STRONG_SLIPSTREAM",
    "SUFFICIENT_ENERGY_RESERVE",
    "FAVORABLE_BRAKING_ZONE"
  ],
  "explanation": "A high-quality overtake window is developing and sufficient energy is available for a short aggressive deployment."
}
```

The explanation should be generated from structured facts, not random LLM text.

---

# 12. ML COMPONENT

Implement a REAL but lightweight ML pipeline.

Do not create fake ML merely to put "AI" on the project.

The model should estimate something meaningful, such as:

```text
overtake_success_probability
```

Use a synthetic but physically/intuitively reasonable dataset initially.

Features:

```text
gap
closing_speed
relative_speed
slipstream_factor
braking_zone_quality
corner_exit_quality
energy_advantage
track_position
```

Target:

```text
overtake_success
```

Train a:

```text
RandomForestClassifier
```

or another interpretable lightweight classifier.

Output:

```text
predicted probability
```

and:

```text
feature importance
```

Save the trained model with joblib.

Provide:

```bash
python scripts/train_models.py
```

which generates the model.

Also provide prediction functionality.

---

# 13. IMPORTANT ML REQUIREMENT

Do NOT use an ML model when deterministic logic is more appropriate.

Use:

```text
ML → probability / prediction
Rules → legality
Physics-inspired calculations → race state
Optimizer → decision
```

This separation is extremely important.

The architecture should be explainable to a jury.

---

# 14. SIMULATION ENGINE

Because true live F1 telemetry may not be available during the hackathon, build a realistic simulation layer.

Create scenarios such as:

### Scenario A — Normal Race

Balanced energy and no strong overtake opportunity.

Expected:

```text
BALANCED
```

### Scenario B — Strong Overtake Window

Small gap, high closing speed, strong slipstream, sufficient energy.

Expected:

```text
OVERTAKE_BONUS
```

### Scenario C — Low Energy

Strong opportunity but insufficient reserve.

Expected:

```text
BALANCED
```

or:

```text
HARVEST
```

depending on conditions.

### Scenario D — Defensive Situation

Car behind approaching quickly.

Expected:

```text
DEFEND
```

### Scenario E — Illegal Candidate

Create a candidate action violating a configured regulatory constraint.

Expected:

```text
REJECTED BY REGULATORY GATE
```

These scenarios must be deterministic when given a fixed seed.

---

# 15. LIVE SIMULATION API

Create an endpoint that starts a simulation.

Example:

```text
POST /api/simulation/start
```

and:

```text
POST /api/simulation/stop
```

and:

```text
GET /api/simulation/status
```

The simulation should periodically generate race-state updates.

Broadcast updates through WebSocket.

Example:

```text
WS /ws/race
```

Each update should contain:

```json
{
  "race_state": {},
  "energy": {},
  "overtake": {},
  "regulatory": {},
  "strategy": {}
}
```

The frontend must be able to consume this without knowing the internals.

---

# 16. REST API

Implement:

```text
GET  /health

GET  /api/race/state

GET  /api/energy/state

GET  /api/overtake/current

GET  /api/strategy/recommendation

POST /api/race/update

POST /api/simulation/start

POST /api/simulation/stop

GET  /api/simulation/status

WS   /ws/race
```

Add:

```text
GET /docs
```

through FastAPI's automatic Swagger documentation.

Use consistent response schemas.

---

# 17. API CONTRACT

Create:

```text
docs/api-contract.md
```

Document every endpoint.

For every response include:

```text
field
type
unit
meaning
example
```

This document is extremely important because the frontend is being developed in a separate repository.

Also make sure the API responses are stable and predictable.

---

# 18. ERROR HANDLING

Implement proper:

* validation
* HTTP status codes
* structured error responses
* logging
* graceful handling of missing telemetry
* invalid race state handling

Do not silently return nonsense values.

---

# 19. TESTING

Create meaningful pytest tests.

At minimum test:

### Energy

* energy decreases after deployment
* energy increases after harvesting
* reserve calculation
* insufficient energy detection

### Overtake

* high closing speed improves score
* large gap reduces score
* strong slipstream improves score
* poor braking zone reduces score

### Regulatory

* legal action passes
* illegal action is rejected

### Strategy

* high-quality overtake + sufficient energy → OVERTAKE_BONUS
* low energy → conservative strategy
* defensive situation → DEFEND

### API

Test:

```text
/health
/api/race/state
/api/energy/state
/api/overtake/current
/api/strategy/recommendation
```

Aim for strong coverage of core decision logic.

---

# 20. DATA PIPELINE

Implement a pipeline that can eventually accept:

```text
CSV
JSON
simulated telemetry
future authorized telemetry
```

Do not make FastF1 a hard dependency of the core engine.

Create an adapter-style interface so future telemetry providers can be plugged in.

For example:

```text
TelemetrySource
    ├── CSVTelemetrySource
    ├── SimulationTelemetrySource
    └── FutureLiveTelemetrySource
```

The core engine should not care where the telemetry came from.

---

# 21. FASTF1 / LIVE DATA

Do NOT claim FastF1 provides zero-latency official live telemetry.

If a FastF1 adapter is included, keep it optional and isolated.

The production architecture must support:

```text
Authorized telemetry
        ↓
Telemetry adapter
        ↓
TrackShift pipeline
```

Do not make unsupported claims about access to proprietary F1 telemetry.

---

# 22. LOGGING

Implement structured useful logs.

Examples:

```text
Race state updated
Energy state calculated
Overtake window detected
Regulatory action rejected
Strategy recommendation generated
Simulation tick processed
```

Avoid excessive logging.

---

# 23. CONFIGURATION

Centralize configuration.

For example:

```text
ENERGY_CAPACITY
MIN_ENERGY_RESERVE
DEPLOYMENT_COSTS
OVERTAKE_WEIGHTS
RISK_WEIGHTS
REGULATORY_LIMITS
SIMULATION_INTERVAL
```

Use environment variables where appropriate.

Never scatter magic numbers throughout the application.

---

# 24. README

Write an excellent README.

It should explain:

1. What TrackShift is
2. Problem being solved
3. Architecture
4. Intelligence pipeline
5. ML component
6. Energy engine
7. Overtake engine
8. Regulatory gate
9. Strategy optimizer
10. API
11. Simulation
12. Installation
13. Running locally
14. Training the model
15. Running the simulation
16. Connecting the frontend

Include architecture diagrams using Mermaid where useful.

Example:

```text
Telemetry
   ↓
Feature Engineering
   ↓
Energy + Overtake Analysis
   ↓
ML Probability
   ↓
Regulatory Gate
   ↓
Strategy Optimizer
   ↓
Explainable Recommendation
```

---

# 25. DEVELOPMENT PHILOSOPHY

IMPORTANT:

This is a serious engineering project.

Avoid:

* meaningless abstractions
* excessive classes
* unnecessary microservices
* fake AI
* random hardcoded outputs
* placeholder functions
* TODOs pretending to be complete
* unexplained magic numbers
* overcomplicated databases
* unnecessary authentication
* unnecessary cloud infrastructure

Prefer:

* simple
* modular
* explainable
* deterministic
* testable
* extensible

Every important calculation must be understandable by another engineer.

---

# 26. JURY DEMONSTRATION REQUIREMENT

The backend must support this exact demo narrative:

```text
Race is progressing
        ↓
Car ahead is 0.7s away
        ↓
Closing speed increases
        ↓
Slipstream detected
        ↓
Braking zone is favorable
        ↓
Overtake probability increases
        ↓
Energy reserve checked
        ↓
Sufficient energy available
        ↓
Regulatory gate approves action
        ↓
Strategy engine evaluates options
        ↓
OVERTAKE_BONUS selected
        ↓
Frontend receives recommendation
```

Then demonstrate a contrasting case:

```text
Strong overtake opportunity
        ↓
BUT low energy reserve
        ↓
OVERTAKE_BONUS becomes strategically expensive
        ↓
Regulatory gate still passes
        ↓
Optimizer chooses BALANCED
```

This demonstrates that the system is not simply:

> "See opportunity → attack."

It balances:

**opportunity + energy + risk + legality + future race state.**

---

# 27. OUTPUT FORMAT FOR RECOMMENDATION

Every strategy recommendation should expose enough information for the frontend to create a premium dashboard.

Return data such as:

```json
{
  "timestamp": "...",
  "lap": 42,

  "recommended_mode": "OVERTAKE_BONUS",

  "confidence": 0.91,

  "overtake": {
    "score": 0.88,
    "probability": 0.84,
    "gap": 0.72,
    "closing_speed": 8.4,
    "slipstream": 0.81,
    "braking_zone": 0.91
  },

  "energy": {
    "soc": 63,
    "remaining": 8.2,
    "deployment_headroom": 2.4,
    "projected_reserve": 5.8
  },

  "risk": {
    "overtake": 0.18,
    "energy": 0.22,
    "overall": 0.21
  },

  "regulatory": {
    "legal": true,
    "violations": []
  },

  "decision": {
    "utility": 0.86,
    "reason_codes": [
      "HIGH_CLOSING_SPEED",
      "STRONG_SLIPSTREAM",
      "FAVORABLE_BRAKING_ZONE",
      "SUFFICIENT_ENERGY"
    ]
  }
}
```

The frontend should have everything it needs to render the dashboard without recalculating backend logic.

---

# 28. IMPLEMENTATION ORDER

Do NOT try to build everything randomly.

Implement in this order:

### Phase 1

Repository + environment + project structure.

### Phase 2

Pydantic data models.

### Phase 3

Race state management.

### Phase 4

Energy engine.

### Phase 5

Overtake engine.

### Phase 6

Risk engine.

### Phase 7

Regulatory gate.

### Phase 8

Strategy optimizer.

### Phase 9

Synthetic dataset + Random Forest model.

### Phase 10

Simulation engine.

### Phase 11

REST API.

### Phase 12

WebSocket.

### Phase 13

Testing.

### Phase 14

Documentation.

### Phase 15

End-to-end demonstration.

After each phase, run tests and fix issues before moving on.

---

# 29. QUALITY BAR

Before considering the project complete, verify:

```text
[ ] Backend starts successfully
[ ] /health works
[ ] Swagger docs work
[ ] Race state can be updated
[ ] Energy calculations work
[ ] Overtake scoring works
[ ] ML prediction works
[ ] Regulatory gate works
[ ] Strategy optimization works
[ ] Simulation works
[ ] WebSocket broadcasts updates
[ ] API responses are stable
[ ] Tests pass
[ ] README is complete
[ ] API contract is documented
[ ] No fake placeholder logic remains
[ ] No unexplained magic numbers remain
```

---

# 30. FINAL INSTRUCTION

Start by inspecting the empty repository.

Then implement the system incrementally according to the phases above.

Do not merely create files with placeholder code.

Actually implement the core functionality.

After each major phase:

1. run tests
2. inspect failures
3. fix them
4. verify imports
5. verify API behavior
6. continue

At the end, run the complete test suite and perform an end-to-end simulation.

Finally provide a concise implementation report containing:

```text
Implemented
Architecture
ML approach
API endpoints
How to run
How to train
How to run simulation
Test status
Known limitations
Next recommended improvements
```

The final backend must be something we can genuinely demonstrate as the **TrackShift Motorsport Intelligence Engine**, not merely a prototype folder structure.
