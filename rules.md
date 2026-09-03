We now need to build the ChronoPace backend around a strict UI/API contract.

ChronoPace is a real-time energy-and-overtake decision engine.

IMPORTANT ARCHITECTURE RULE:

Python deterministically computes and verifies every numerical,
regulatory, simulation, and decision result.

The LLM must NEVER make the numerical decision.

The LLM, if used, only narrates an already verified JSON decision.

==================================================
BACKEND RESPONSIBILITIES
==================================================

The backend owns:

1. FastF1 telemetry/data ingestion
2. Telemetry replay
3. Feature extraction
4. Rival Energy Estimator
5. Opportunity Horizon
6. Monte Carlo Planner
7. Regulatory Gate
8. Confidence Gate
9. Deployment Mode Engine
10. Final ChronoPace Decision

The frontend owns:

- presentation, layout, charts, animations, 3D scene, user interaction
- lightweight UI values: formatting numbers, converting units for display,
  computing bar widths / percentages from values the backend already sent,
  colour thresholds, local toggle state

The frontend must NOT independently calculate race intelligence.

Race intelligence = anything that represents a strategic or ML decision:

Backend calculates:
- SoC estimation
- energy deployment / harvesting
- rival energy estimation
- overtake probability
- tyre degradation model
- predicted lap time
- Monte Carlo simulations
- strategy optimisation
- recommended action
- confidence scores
- regulatory / constraint checks

Frontend calculates:
- socPct = (currentMj / maxMj) * 100  ← formatting a value the backend sent
- bar widths, colour thresholds        ← pure presentation
- which panel tab is active            ← local UI state

==================================================
DEPLOYMENT MODES
==================================================

These are FIXED and must not be simplified:

CONSERVE_MODE
BALANCED_MODE
ARM_OVERTAKE_MODE
USE_OVERTAKE_BONUS_MODE
PUSH_MODE

==================================================
DECISION ACTIONS
==================================================

The planning layer may evaluate:

ATTACK_NOW
WAIT_2_LAPS
WAIT_5_LAPS
HOLD

==================================================
INITIAL API
==================================================

Create:

GET /api/v1/health

POST /api/v1/decision

POST /api/v1/decision should accept:

{
  "session_id": "monza_replay",
  "lap": 34,
  "driver": "CAR_23",
  "rival": "RIVAL_1"
}

Return a structured response containing:

meta
decision
energy
rival
opportunity
monte_carlo
compliance

==================================================
DECISION
==================================================

Example structure:

{
  "mode": "USE_OVERTAKE_BONUS_MODE",
  "action": "ATTACK_NOW",
  "confidence": 0.82,
  "reason": "..."
}

==================================================
ENERGY
==================================================

Return:

deployable_mj
harvest_rate_mj_per_lap
energy_state

==================================================
RIVAL
==================================================

Return:

energy_distribution:
  low
  medium
  high

estimated_reserve_mj
reserve_std_mj
confidence

clipping:
  detected
  location_percent
  terminal_speed_kmh

IMPORTANT:

The Rival Energy Estimator estimates a hidden state.
It does NOT claim to directly measure the rival's battery.

==================================================
OPPORTUNITY
==================================================

Return:

current opportunity
recommended future window

Example:

{
  "current": {
    "location": "T1",
    "success_probability": 0.64
  },
  "recommended_window": {
    "lap": 36,
    "location": "T1",
    "success_probability": 0.78
  }
}

Do not hardcode these values in the final engine.

==================================================
MONTE CARLO
==================================================

Return:

number of simulations
strategies
expected value
success probability
best strategy

The planner should compare spending energy now against preserving
energy for future opportunities.

==================================================
COMPLIANCE
==================================================

Return:

legal
checks[]

Each check should contain:

rule
status

The regulatory layer must be authoritative over strategy.

A strategy that is illegal must never become the final recommendation.

==================================================
IMPORTANT IMPLEMENTATION ORDER
==================================================

Do NOT attempt to build everything at once.

Phase 1:
- project structure
- FastAPI
- /health
- /decision schema
- Pydantic models
- API contract

Phase 2:
- FastF1 service
- historical session loading
- local caching
- replay mechanism

Phase 3:
- telemetry feature extraction

Phase 4:
- Rival Energy Estimator

Phase 5:
- Opportunity Horizon

Phase 6:
- Monte Carlo Planner

Phase 7:
- Regulatory Gate

Phase 8:
- Confidence Gate

Phase 9:
- final Decision Engine

Phase 10:
- frontend integration

==================================================
DATA HONESTY
==================================================

For the prototype, historical FastF1 telemetry can be replayed as if
it were arriving live.

Do NOT claim we have access to private real-time F1 team telemetry.

The engine should clearly expose:

data_mode = "REPLAY"

Production architecture can later replace the replay provider with
an authorized live telemetry source.

==================================================
DO NOT
==================================================

- build a separate "Telemetry Agent"
- put strategy calculations in React
- hardcode final intelligence values
- let an LLM decide strategy
- invent telemetry
- invent API fields
- couple Python code to frontend components
- use WebSockets yet

Use REST first.

Make the API contract clean enough that the frontend team can begin
integration without needing to understand the internal Python
implementation.