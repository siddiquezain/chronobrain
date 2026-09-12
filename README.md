# ChronoPace Backend

Motorsport decision-intelligence engine for **2026 Formula 1** energy-deployment
strategy, built for the **TrackShift 2026** challenge.

> "Is this moment worth spending finite electrical energy, or is a better
> opportunity coming?"

ChronoPace is not a battery predictor and not an overtake predictor. It combines
telemetry → our energy state → hidden rival-energy inference → current/future
overtake opportunity → regulatory legality → Monte Carlo → confidence → a final
deployment decision → a human-readable explanation.

**The invariant:** Python computes every number deterministically; the LLM only
turns the finished decision into a sentence. It never decides the strategy.

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (optional) train the overtake-success model — a heuristic fallback is used if absent
python scripts/train_models.py

# run the API
uvicorn app.main:app --reload --port 8000

# one authoritative decision from the terminal
python scripts/run_decision.py B --lap 30
```

## The canonical endpoint

```bash
curl -s -X POST localhost:8000/api/v1/decision \
  -H 'content-type: application/json' \
  -d '{"scenario":"B","seed":42,"lap":30,"with_narrative":true}' | jq
```

Returns one `DecisionSnapshot`: `decision`, `data_quality`, `window`, `energy`,
`rival`, `opportunity` (with Future Energy Value), `monte_carlo`, `compliance`,
`confidence`, `constraints`, `candidate_actions` / `feasible_actions` /
`rejected_alternatives`, `reason_codes`, `trace`, and an optional `narrative`.
Identical request ⇒ identical response (bar the timestamp). Full shape in
[docs/api-contract.md](docs/api-contract.md).

The pipeline: **Data Quality Gate → event-time window → energy + rival estimate
(+ P_defend) → regulatory gate → candidate → feasible set → Monte Carlo →
Opportunity Horizon with Future Energy Value → 5-gate Confidence Gate → decision
engine → verified snapshot → optional narrator.** Python computes and verifies
every number; the LLM only phrases the finished result.

## Telemetry sources

```
FastF1 historical replay  ─┐
                           ├─→  Telemetry Normalizer  →  ChronoPace decision engine
Synthetic simulator       ─┘
```

- **Synthetic** (default): deterministic scenario generator, no dependencies.
- **FastF1 replay**: `pip install -r requirements-fastf1.txt`, then
  `POST /api/v1/decision {"source":"fastf1","fastf1":{...}}`. FastF1 exposes car
  motion + FIA timing only — **F1 publishes no ERS state of charge**, so SoC on a
  replayed lap is *modelled* (`energy_is_modeled: true`). This is historical replay
  of public data, not a live feed and not team telemetry.

## Run tests

```bash
python -m pytest -q
```

## Scenarios (deterministic, seed 42, lap 30)

| Scenario | Situation | Decision |
|---|---|---|
| A | Normal race, gap too large to attack | `BALANCED_MODE` / HOLD |
| B | Strong window, high SoC, bonus banked | `USE_OVERTAKE_BONUS_MODE` / ATTACK_NOW |
| C | Same window as B but **low SoC** | `CONSERVE_MODE` / CONSERVE — the attack isn't worth the energy |
| D | Car close behind (defending) | `PUSH_MODE` / PUSH |
| E | Lap deployment over the cap | **rejected — `compliance.legal = false`**, `BALANCED_MODE` / HOLD |

`B` vs `C` is the point: the same opportunity produces a different decision
because of our energy state.

## API

**Authoritative (use only this for decisions):**
- `POST /api/v1/decision` — one `DecisionSnapshot`
- `GET  /api/v1/scenario/{A..E}` — convenience GET for the demo scenarios
- `GET  /api/v1/health` — includes `fastf1_available`
- `WS /ws` — per simulation tick, broadcasts `{ type, tick, snapshot }` (the same `DecisionSnapshot`, nothing else)

**Support:** `GET /health`, `POST /api/simulation/start|stop`, `GET /api/simulation/status`

**Deprecated (Stack B, NOT authoritative):** `/api/race/*`, `/api/energy/*`,
`/api/overtake/*`, `/api/strategy/*` — flagged `deprecated` in OpenAPI, every
response carries `Deprecation: true`. The frontend must not read decisions from these.

Swagger UI at `http://localhost:8000/docs`.

## Interactive demo

`/api/v1/demo/*` — change race inputs (our SoC, hidden rival SoC, gap, noise) and
watch the engine respond; score the Rival Energy Estimator against a hidden ground
truth it never receives; replay the estimator lap-by-lap. Ground truth is never in
`POST /api/v1/decision`. See **[docs/demo.md](docs/demo.md)**.

```bash
curl -s localhost:8000/api/v1/demo/presets | jq
curl -s -X POST localhost:8000/api/v1/demo/preset/RIVAL_ENERGY_HIGH | jq .rival_validation
curl -s -X POST localhost:8000/api/v1/demo/rival-trace \
  -d '{"scenario":"B","seed":42,"up_to_lap":20,"overrides":{"rival_initial_soc_mj":7.5}}' \
  -H 'content-type: application/json' | jq '.steps[-1]'
```

## Historical race replay

`/api/v1/replay/*` — replay a real F1 race (first: **2024 Italian GP**, our
driver **LEC**) through the same pipeline, lap by lap, using only data available
up to the current lap. The **strategic rival is re-selected from the whole field
every lap** (`app.replay.strategic_rival`) — NOR/PIA/SAI/VER/OCO across 2024
Monza — so the Rival Energy Estimator follows whoever actually matters. Needs
`pip install -r requirements-fastf1.txt`. See **[docs/historical-replay.md](docs/historical-replay.md)**.

```bash
curl -s -X POST localhost:8000/api/v1/replay/historical \
  -d '{"race":"2024_italian_gp","start_lap":1,"end_lap":53,"seed":42}' \
  -H 'content-type: application/json' | jq '.strategic_rival.changes, .laps[10].strategic_rival'
```

## Docs

- [docs/architecture.md](docs/architecture.md) — the pipeline and the LLM boundary
- [docs/api-contract.md](docs/api-contract.md) — every endpoint and the snapshot shape
- [docs/demo.md](docs/demo.md) — the interactive demo + how to demonstrate the rival estimator
- [docs/historical-replay.md](docs/historical-replay.md) — real-race replay, and where hindsight is prevented
- [docs/methodology.md](docs/methodology.md) — the algorithms
- [docs/regulation.md](docs/regulation.md) — every regulatory constant + provenance

## 2026 F1 Architecture

```
TELEMETRY (FastF1 historical replay / Synthetic)
         |
TELEMETRY NORMALIZER
         |
  +------+-------+
  |      |       |
OUR    RIVAL   RACE
ENERGY INTEL   CONTEXT
       (Particle Active Aero
        Filter + Overtake Mode
        RF Model)
  +------+-------+
         |
OPPORTUNITY ENGINE (now / +1 / +2 / +3/+5)
         |
REGULATORY GATE (2026 FIA feasibility)
         |
MONTE CARLO PLANNER (10,000 seeded rollouts/mode)
         |
CONFIDENCE / SIGNIFICANCE GATE (5 gates)
         |
DECISION ENGINE → DecisionSnapshot
         |
LLM NARRATOR (explanation only, never alters decisions)
```

## 2026 F1 Terminology

- **DRS abolished**: the 2026 Active Aero system replaces it.
- **Overtake Mode**: activates when gap to car ahead ≤ 1.0 s — grants +0.5 MJ extra deployable energy.
- **5 Deployment Modes**: CONSERVE_MODE, BALANCED_MODE, ARM_OVERTAKE_MODE, USE_OVERTAKE_BONUS_MODE, PUSH_MODE.
- **Rival energy**: always INFERRED via particle filter — F1 teams do not publish battery SoC.

## Important Assumptions

- **Energy is MODELED**: ChronoPace reconstructs SoC from throttle/deployment traces. Not measured.
- **Rival energy is INFERRED**: particle-filter estimate from observable kinematics. Not measured.
- **ML training data is SYNTHETIC**: structural domain knowledge, not real F1 outcome data.
- **FastF1 provider uses historical data** (pre-2026 races replayed as 2026 proxies).

## Environment Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key  # optional, for LLM narration
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Copy `.env.example` to `.env`:
- `ANTHROPIC_API_KEY` — optional; enables Claude narration (explanation only)

## API Contract

`POST /api/v1/decision` returns a `DecisionSnapshot` with:

| Block | Description |
|-------|-------------|
| `decision` | Final mode, action, confidence |
| `energy` | Our SoC, deployment headroom, projections |
| `rival` | Inferred rival energy distribution (INFERRED) |
| `opportunity` | Multi-lap strategy comparison |
| `monte_carlo` | 10,000 seeded rollouts per mode |
| `compliance` | FIA 2026 regulatory check |
| `confidence` | 5-gate significance check |
| `counterfactual` | What if we don't act? |
| `context_attribution` | Tyre compound + active aero attribution |
| `narrative` | Optional LLM explanation (never authoritative) |

## Demo Commands

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Scenario B — strong overtake window (gap=0.6s, high SoC, qualified last lap)
curl -s -X POST http://localhost:8000/api/v1/decision \
  -H "Content-Type: application/json" \
  -d '{"source":"synthetic","scenario":"B","seed":42}' | python -m json.tool

# All 5 scenarios
for SC in A B C D E; do
  echo "=== Scenario $SC ==="
  curl -s -X POST http://localhost:8000/api/v1/decision \
    -H "Content-Type: application/json" \
    -d "{\"source\":\"synthetic\",\"scenario\":\"$SC\",\"seed\":42}" | \
    python -c "import json,sys; d=json.load(sys.stdin); print(f'Mode: {d[\"decision\"][\"mode\"]}, Action: {d[\"decision\"][\"action\"]}')"
done

# With LLM narrative (requires ANTHROPIC_API_KEY)
curl -s -X POST http://localhost:8000/api/v1/decision \
  -H "Content-Type: application/json" \
  -d '{"source":"synthetic","scenario":"B","seed":42,"with_narrative":true}' | \
  python -c "import json,sys; d=json.load(sys.stdin); print(d.get('narrative','no narrative'))"
```
