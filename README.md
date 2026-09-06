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

Returns one `DecisionSnapshot`: `decision`, `energy`, `rival`, `opportunity`,
`monte_carlo`, `compliance`, `confidence`, `reason_codes`, `trace`, and an optional
`narrative`. Identical request ⇒ identical response (bar the timestamp). Full shape
in [docs/api-contract.md](docs/api-contract.md).

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

Canonical:
- `POST /api/v1/decision` — one authoritative `DecisionSnapshot`
- `GET  /api/v1/scenario/{A..E}` — convenience GET for the demo scenarios
- `GET  /api/v1/health` — includes `fastf1_available`

Existing (unchanged):
- `GET /health`, `GET /api/race/state`, `GET /api/energy/state`,
  `GET /api/overtake/current`, `GET /api/strategy/recommendation`
- `POST /api/simulation/start|stop`, `GET /api/simulation/status`
- `WS /ws` — per-tick broadcast, now also carrying `decision_snapshot`

Swagger UI at `http://localhost:8000/docs`.

## Docs

- [docs/architecture.md](docs/architecture.md) — the pipeline and the LLM boundary
- [docs/api-contract.md](docs/api-contract.md) — every endpoint and the snapshot shape
- [docs/methodology.md](docs/methodology.md) — the algorithms
- [docs/regulation.md](docs/regulation.md) — every regulatory constant + provenance

## Environment

Copy `.env.example` to `.env`:
- `ANTHROPIC_API_KEY` — optional; enables Claude narration (explanation only)
