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
        │  feature extraction (energy, ML P(o/t))   │
        │  rival particle filter  (rival_estimator) │
        │  Stage 1  RegulatoryGate  (rule_gate)     │  legality only
        │  Stage 2  MonteCarloPlanner (planner)     │  ranks legal modes
        │           OpportunityEngine               │  ATTACK_NOW / WAIT_N / HOLD
        │  Stage 3  ConfidenceGate                  │  may override → BALANCED
        │  DECISION ENGINE (fusion)                 │  final mode + action + codes
        └──────────────────────┬───────────────────┘
                   ▼
        DecisionSnapshot  (verified JSON)  ── + DecisionTrace
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
| `app/data/` | Telemetry provider abstraction: `TelemetrySample`, `NormalizedLap`, `SyntheticProvider`, `ReplayProvider`, `fastf1_service` (the only module that imports `fastf1`). |
| `app/decision/` | `run_decision()` orchestrator, `DecisionConfig`, `DecisionContext`, unified `reason_codes`, `DecisionSnapshot`, `DecisionTrace`, the fusion step. |
| `app/narrative/` | Stage 4 — `narrate(snapshot)`; the only place a language model runs. |
| `app/regulation/` | Provenance catalogue for every `GateConfig` constant (see `docs/regulation.md`). |
| `app/engines/` | Legacy Stack B engines (energy/overtake/ML feature extractors + legacy `StrategyEngine`). `EnergyEngine`/`OvertakeEngine`/ML feed the new pipeline; `StrategyEngine`/`RiskEngine`/`AppRegulatoryGate` remain only behind the pre-existing GET endpoints. |

## Five deployment modes (fixed)

`CONSERVE_MODE`, `BALANCED_MODE`, `ARM_OVERTAKE_MODE`, `USE_OVERTAKE_BONUS_MODE`,
`PUSH_MODE`. `ARM` = qualify/prepare the Overtake-Mode bonus for next lap;
`USE_OVERTAKE_BONUS` = spend a bonus banked last lap. They are **not** merged.

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
