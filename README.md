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

## Docs

- [docs/architecture.md](docs/architecture.md) — the pipeline and the LLM boundary
- [docs/api-contract.md](docs/api-contract.md) — every endpoint and the snapshot shape
- [docs/demo.md](docs/demo.md) — the interactive demo + how to demonstrate the rival estimator
- [docs/methodology.md](docs/methodology.md) — the algorithms
- [docs/regulation.md](docs/regulation.md) — every regulatory constant + provenance

## Environment

Copy `.env.example` to `.env`:
- `ANTHROPIC_API_KEY` — optional; enables Claude narration (explanation only)
