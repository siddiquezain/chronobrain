# ChronoPace — Architecture

## Overview

ChronoPace is a decision-support engine that answers: **"Is this moment worth
spending finite electrical energy?"** — not just *where* to deploy, but *whether
the spend is worth it at all* versus a better future opportunity.

It processes **2026 Formula 1** telemetry (historical FastF1 replay **or** the
synthetic simulator) through a deterministic pipeline and returns one authoritative
`DecisionSnapshot`.

## The invariant

```
              PYTHON  =  COMPUTATION / TRUTH
              LLM     =  EXPLANATION ONLY
```

Every number — energy, probabilities, rival state, opportunity ranking, Monte
Carlo outcomes, legality, confidence, the final mode — is produced by
deterministic Python **before** the narrator runs. The LLM is handed a finished
`DecisionSnapshot` and may only write `snapshot.narrative`. It cannot change a
mode, a probability, or a compliance result. If the LLM is unavailable, a
structured fallback fills `narrative` and nothing else changes.

## End-to-end flow

```
        TELEMETRY SOURCES
   ┌───────────────┴───────────────┐
FastF1 historical replay      Synthetic simulator
(app/data/fastf1_service.py)  (telemetry_simulator.py)
   │  session load, DataFrame,     │
   │  DRS/throttle coding,         │
   │  driver/session selection     │
   └───────────────┬───────────────┘
                   ▼
        TELEMETRY NORMALIZER        app/data/normalizer.py
        TelemetrySample[] → NormalizedLap
                   ▼
        ┌──────────────────────────────────────────┐
        │  DECISION PIPELINE  (app/decision/engine) │
        │                                          │
        │  Data Quality Gate   (app/data/quality)   │  GOOD/DEGRADED/INVALID -> Stage 3
        │  event-time window   (app/decision/window)│  short-horizon trends
        │  feature extraction (energy, ML P(o/t))   │
        │  rival particle filter  (rival_estimator) │  + P_defend scalar
        │  Stage 1  RegulatoryGate  (rule_gate)     │  legality only
        │  candidate actions -> feasible set        │  before the planner runs
        │  Stage 2  MonteCarloPlanner (planner)     │  ranks legal modes
        │           OpportunityEngine + FEV         │  ATTACK_NOW/WAIT_N/HOLD, energy carried fwd
        │  Stage 3  ConfidenceGate (5 gates: stat/  │  may override → BALANCED
        │           practical/DCLI/rival/data-qual) │
        │  DECISION ENGINE (fusion)                 │  pick from the feasible set
        └──────────────────────┬───────────────────┘
                   ▼
        DecisionSnapshot  (verified JSON)  ── + DecisionTrace + rejected alternatives
                   ▼
        Stage 4  LLM Narrator   app/narrative/narrator.py   (optional, additive)
                   ▼
        POST /api/v1/decision  →  React frontend
```

## Modules

Stack A — the intelligence core (repo root, unchanged contracts):

| File | Role |
|---|---|
| `rule_gate.py` | Stage 1 — regulatory legality only. Never imports `rival_estimator`. |
| `rival_estimator.py` | 1000-particle filter → `RivalSocEstimate(mean, std)`. |
| `planner.py` | Stage 2 — 10,000-iteration Monte Carlo, `SeedSequence.spawn(5)` in canonical mode order. |
| `opportunity_engine.py` | Multi-lap `ATTACK_NOW / WAIT_N / HOLD` comparison; uncertainty widens with horizon distance. |
| `confidence_gate.py` | Stage 3 — statistical + practical significance + DCLI + rival-confidence; override → `BALANCED_MODE`. |
| `narrator.py` | Stack A Stage 4 helpers (`build_reason_codes`, `derive_confidence`). |
| `telemetry_simulator.py` | Deterministic scenario generator (A–E). |

New in `app/`:

| Path | Role |
|---|---|
| `app/data/` | Telemetry provider abstraction: `TelemetrySample`, `NormalizedLap`, `SyntheticProvider`, `ReplayProvider`, `fastf1_service` (the only module that imports `fastf1`), **`quality.py`** (Data Quality Gate). |
| `app/decision/` | `run_decision()` orchestrator, `DecisionConfig`, `DecisionContext`, **`window.py`** (event-time trends), **`actions.py`** (candidate/feasible set), unified `reason_codes`, `DecisionSnapshot`, `DecisionTrace`, the fusion step, **`outcome_log.py`** (opt-in prediction→outcome log). |
| `app/narrative/` | Stage 4 — `narrate(snapshot)`; the only place a language model runs. |
| `app/regulation/` | Provenance catalogue for every `GateConfig` constant (see `docs/regulation.md`). |
| `app/engines/` | Legacy Stack B engines, all **deprecated**. Only serve the deprecated GET endpoints during a simulation; the v1 pipeline does not import them. |

## Five deployment modes (fixed, all reachable)

`CONSERVE_MODE`, `BALANCED_MODE`, `ARM_OVERTAKE_MODE`, `USE_OVERTAKE_BONUS_MODE`,
`PUSH_MODE`. `ARM` = attack this lap within proximity, which also qualifies next
lap's Overtake-Mode bonus; `USE_OVERTAKE_BONUS` = spend a bonus banked last lap.
They are **not** merged. The decision engine resolves "attack now" to `USE_BONUS`
when a bonus is banked, to `ARM` when in proximity with none banked, and only to
`PUSH` when there is a car to defend from behind. All five appear as final
decisions across the test suite (`test_arm_mode.py::test_all_five_modes_are_reachable`).

The decision engine also emits a user-facing `action`:
`ATTACK_NOW | WAIT_2_LAPS | WAIT_5_LAPS | HOLD | PUSH | CONSERVE`.

## Determinism

`same NormalizedLap list + same DecisionConfig + same seed ⇒ byte-identical
DecisionSnapshot` (excluding `meta.generated_at`). All randomness is seeded
(`np.random.default_rng` / `SeedSequence`) from `DecisionConfig.seed`. The single-
lap API replays laps 1..N each call rather than caching mutable state, so results
never depend on call history. Enforced by `tests/test_decision_determinism.py`.

## Sign convention (critical)

Monte Carlo / confidence gate use **`diff = mean(runner_up) − mean(top)`**
(runner-up first). Negative delta = faster. Reversing the subtraction silently
inverts the gate; `test_integration.py` checks it explicitly.

## Regulatory constants

See **[docs/regulation.md](regulation.md)** — every constant is tagged
`VERIFIED_FIA` / `MODEL_ASSUMPTION` / `DEMO_CONSTANT` and cross-checked against
`GateConfig` by `test_regulation_provenance.py`.

## One-way import law

`rule_gate.py` must **never** import from `rival_estimator.py`. Legality is a pure
function of the car's own verified telemetry; rival uncertainty must not touch it.
