# ChronoPace — Methodology

## Core Question

> "Is this moment worth spending finite electrical energy?"

Every algorithm in ChronoPace exists to answer this question with bounded uncertainty and regulatory compliance.

## Stage 1 — Regulatory Gate

A pure rule-based filter. No ML, no uncertainty — the rules either pass or fail.

**FIA 2026 constraints modelled:**
- Art. 5.4.10: Maximum deployment per lap = 9 MJ (+ 0.5 MJ overtake bonus when banked)
- Art. 5.4.9: Maximum SoC swing per lap = 4 MJ
- Proximity rule: overtake modes require gap ≤ 1.0 s in current lap *or* banked qualification
- Speed taper: deployment capped as a function of speed (simulated via base_cap_mj per mode)

**Why pure rules?** Regulations are deterministic constraints. Any ML influence on legality decisions would introduce the risk of confidently wrong answers with real sporting consequences.

## Stage 2 — Monte Carlo Planner

Simulates 10,000 race trajectories per legal mode using NumPy's seeded RNG.

**Key design choices:**
- `n_iterations=10_000` is fixed and never varied — reproducibility is a first-class requirement
- RNG streams are spawned via `SeedSequence.spawn(5)` in canonical mode order — BALANCED_MODE always gets stream 1 regardless of which other modes are legal, preserving byte-identical results for debugging
- Rival SoC modulation: when a `RivalSocEstimate` is available, mode utilities are adjusted based on estimated rival energy state
- Sharpe ratio selection capped at ±999 prevents numerical explosion when std ≈ 0

**Mode dynamics** (`DEFAULT_MODE_DYNAMICS`) are engineered illustrative values — not measured from real Formula 1 PU data. They encode the structural relationships (PUSH is faster but costs more energy; CONSERVE is slower but saves energy) without claiming empirical precision.

## Stage 3 — Confidence Gate

Five sequential gates, all must pass for the planner's recommendation to stand:

1. **Statistical reliability** — Welch t-test between top and runner-up mode samples (p < 0.05)
2. **Practical significance** — |Δlaptime| > 0.05 s and |ΔP(overtake)| > 3 percentage points
3. **DCLI** — Driver Cognitive Load Index ≤ 60.0 (guards against issuing recommendations that would overwhelm the driver during a complex situation)
4. **Rival confidence** — rival SoC standard deviation ≤ 1.5 MJ (don't commit to an energy strategy when rival state is too uncertain)
5. **Data quality / opportunity clarity** — the Data Quality Gate's `quality_score` ≥ 0.6 **and** the opportunity horizon can actually separate the strategies. Degraded/stale telemetry or an ambiguous opportunity makes the gate abstain.

Override: if any gate fails, `recommended_mode` → BALANCED_MODE. `override_reason` names **all** failing gates.

**Sign convention**: `diff = mean(runner_up) − mean(top)`. Negative = runner_up has smaller (faster) lap time.

## Data Quality Gate

A pure, deterministic check that runs **before** feature extraction. It does not
fix or invent telemetry and it does not choose a strategy — it produces a verdict
(`GOOD | DEGRADED | INVALID`, `quality_score` 0–1, freshness, dropped samples,
missing fields) that flows into Stage 3. Bad/stale telemetry → lower
`quality_score` → the confidence gate abstains → BALANCED_MODE.

## Event-Time Window

A short sliding window (default 5 laps) over the normalized lap history. Sorts and
de-duplicates by lap number (tolerating out-of-order input), then linear-fits each
channel to produce trends: speed, gap-to-car-ahead (negative = closing), SoC,
rival terminal speed, rival sector delta. Yields an `opportunity_trend`
(`IMPROVING | STABLE | DECAYING`) that feeds the opportunity horizon.

## Candidate Actions vs. Feasible Set

Candidate strategic actions (`ATTACK_NOW`, `WAIT_2_LAPS`, `WAIT_5_LAPS`,
`CONSERVE`, `HOLD`) are planning alternatives, **not** the five deployment modes.
The regulatory gate plus an energy check reduce them to the *feasible* set before
the Monte Carlo horizon evaluates anything — the planner never optimises an action
already known to be illegal or unaffordable. A `WAIT_N` is feasible only if the
car can harvest enough energy *during* the wait to fund the attack at the window.

## Opportunity Horizon — Future Energy Value

The horizon carries modelled SoC forward lap-by-lap across each candidate
strategy. A strategy that would starve the reserve before its window cannot
actually attack (that lap falls back to BALANCED + a missed-attack penalty). Each
strategy gets a `strategic_value`:

```
strategic_value = pace  +  current_opportunity_value  +  future_opportunity_value
                        −  energy_opportunity_cost
```

`energy_opportunity_cost` rises as the horizon-end reserve nears the floor, so a
strategy that spends its last megajoule pays for it. Strategies are ranked by
`strategic_value`. This is how ChronoPace answers *"is spending energy now better
than preserving it for a better future opportunity?"* — not with a Dynamic
Programming solver, just a valuation term inside the existing Monte Carlo horizon.

A **prime window** (banked bonus + model-confident + affordable) is only deferred
for a *substantially* better future window (3× the normal decisive margin).

## Stage 4 — Narrator

Generates human-readable strategy calls from structured facts.

- `build_reason_codes()` selects from a fixed 13-code vocabulary — never generates free text
- `derive_confidence()` is deterministic: CI lower bound → confidence score [0, 1]
- `narrate()` calls Claude claude-haiku-4-5-20251001 if `ANTHROPIC_API_KEY` is set, otherwise falls back to a structured template. The LLM never changes the decision — only the phrasing.

## Particle Filter — Rival SoC Estimator

A 1,000-particle sequential importance resampling filter with 4 observable signals:

| Signal | Meaning |
|--------|---------|
| `terminal_speed_kmh` | Higher terminal speed → more energy deployed |
| `clipping_point_fraction` | Motor current clipping → near deployment limit |
| `corner_exit_accel_g` | Strong exit acceleration → high energy deployment |
| `sector_delta_s` | Sector time relative to baseline → integrated energy spend |

Predict step: Gaussian drift (σ = 0.3 MJ) simulates lap-to-lap SoC change.
Update step: 4-signal Gaussian likelihood, systematic resampling.

## ML — Overtake Success Classifier

RandomForestClassifier (100 trees, seed=42) trained on synthetic data with 8 features. Used to improve overtake probability estimates in the overtake engine.

**Why synthetic data?** Real Formula 1 overtake outcome data is not publicly available at the resolution required. The synthetic dataset encodes structural domain knowledge (small gap + high closing speed + DRS → higher success probability) while keeping the classifier interface ready for real data when available.
