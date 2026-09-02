# ChronoPace Backend

Motorsport intelligence engine for Formula E energy deployment strategy.

> "Is this moment worth spending finite electrical energy?"

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Train the ML model
python scripts/train_models.py

# Run the server
uvicorn app.main:app --reload --port 8000

# Run a scenario simulation in the terminal
python scripts/run_simulation.py B
```

## Run Tests

```bash
python -m pytest -q
```

## Scenarios

| Scenario | Description | Expected Mode |
|---------|-------------|---------------|
| A | Normal race, balanced | BALANCED_MODE |
| B | Strong overtake window | ARM_OVERTAKE_MODE or USE_OVERTAKE_BONUS_MODE |
| C | Low energy reserve | CONSERVE_MODE or BALANCED_MODE |
| D | Defensive — car behind close | PUSH_MODE or BALANCED_MODE |
| E | Illegal candidate (over cap) | REJECTED by regulatory gate |

## API

- `GET /health` — health check
- `GET /api/race/state` — current race state
- `GET /api/energy/state` — energy state
- `GET /api/overtake/current` — overtake analysis
- `GET /api/strategy/recommendation` — strategy recommendation
- `POST /api/simulation/start` — start demo simulation
- `POST /api/simulation/stop` — stop simulation
- `GET /api/simulation/status` — simulation status
- `WS /ws` — real-time broadcast

Full API docs at `http://localhost:8000/docs` (Swagger UI).

## Architecture

See [docs/architecture.md](docs/architecture.md), [docs/api-contract.md](docs/api-contract.md), [docs/methodology.md](docs/methodology.md).

## Environment Variables

Copy `.env.example` to `.env` and set:
- `ANTHROPIC_API_KEY` — optional, enables Claude haiku narrative generation
- See `.env.example` for all options
# chronobrain
