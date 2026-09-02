# context.md — ChronoPace

> **CHRONOPACE — Decision Intelligence for the Energy Battle**
> *Deciding when energy is worth spending.*

The question this product answers is **not** "where should energy be
deployed?" — it's **"is this the moment our finite energy is worth
spending?"** That reframe is the whole product. Every section below exists
to support one of: legality, rival-state uncertainty, opportunity cost, or
confidence/abstention. If a proposed feature doesn't touch one of those
four things, it's probably not core — see §18 for this as an explicit
filter.

Read this before touching any code in this repo. It exists so a teammate
(or an agent) picking up this project mid-build doesn't have to
reverse-engineer intent from diffs or from what's on screen. This is the
**whole project**, not just the dashboard — read it end to end before
assuming you know what's built and what's still just designed, because
those two things currently do not match, and that mismatch matters.

**Current reality in one line**: the *decision engine* (the actual
"intelligence" in this hackathon's theme) is fully designed but has
**zero lines of Python written**. The *dashboard* you can click through
locally is fully built and working, but every number on it is hand-typed
mock data — nothing on screen is computed by anything. If code and this
file ever disagree once the backend exists, trust the code and flag the
mismatch rather than trusting whichever is more convenient.

## 1. What this project is

**ChronoPace** is a real-time, deterministic decision-support engine for
2026 Formula 1 energy/overtake strategy, built for the **TrackShift 2026
Innovation Challenge** hackathon (official theme: AI Motorsport
Intelligence, problem one of three — name it correctly in any judge-facing
material).

`TrackShift` = the hackathon name. `ChronoPace` = the product name.

> **Core positioning**: ChronoPace is a deterministic race-strategy engine
> that decides when finite electrical energy is worth spending by
> combining regulatory legality, probabilistic rival-state estimation,
> Monte Carlo planning, and future opportunity cost.
>
> **Core differentiator**: We don't just optimize the attack. We decide
> whether the attack is worth spending the energy on.

**Competitive positioning**: a different TrackShift team ("Deploy AI") is
reportedly building toward *where/when* energy should be deployed, with
driver/car/track-specific learned modeling. Don't attack that project or
make unsupported comparative claims in any pitch material — the accurate,
defensible distinction is one of scope, not quality: Deploy AI answers
*where/when*; ChronoPace answers *whether the spend is worth it at all*,
via legality-first optimization, rival uncertainty, opportunity cost, and
deliberate abstention. Win on that distinction, not by claiming to be
"more AI."

A companion document, **`ChronoPace-Solution-Design.pdf`** (repo root),
covers the same architecture in narrative form — problem statement, the
solution, and the reasoning behind each major decision — written to hand
to teammates who want the pitch-level version instead of this
reference-level one. This file is the one to trust for exact field names,
formulas, and conventions when actually writing code.

## 2. The one architectural law

**Python computes every number, deterministically and testably. An LLM
(Claude) is only ever allowed to turn an already-correct, already-verified
JSON payload into a plain-language sentence. It never computes, never
decides, and never has authority to change a number it was given.**

This is the safety argument the whole project rests on: an LLM generates
tokens probabilistically and can state a wrong number with full
confidence, which is unacceptable when the number is an energy deployment
figure or a compliance decision. Any change that lets a language model
influence a numeric or compliance outcome is a violation of the core
design, not a refactor — flag it explicitly rather than making it quietly.
No LLM call exists anywhere in Stages 1-3. The LLM appears exactly once,
in Stage 4, only after Stages 1-3 have produced final, verified numbers.
The *decision* (Stages 1-3) is fast; the *narration* (Stage 4, an API
call) is not — keep those two claims separate, never call the narrated
output "real-time." (The dashboard's current mock data doesn't violate
this law — it isn't computing anything at all yet, it's a static
placeholder standing in for where Stage 1-3's real output will go.)

## 3. Architecture — the four-stage pipeline

Memorable version, one question per stage — useful shorthand in a pitch,
not a substitute for the exact contracts below:

```text
OBSERVE                                  (Telemetry Simulator, §17 step 2)
   ↓
CAN WE?               → Regulatory Gate           (Stage 1)
   ↓
SHOULD WE?             → Monte Carlo Planner        (Stage 2)
   ↓
NOW OR LATER?           → Opportunity Horizon         (§9, layered on top)
   ↓
DO WE KNOW ENOUGH?       → Confidence Gate              (Stage 3)
   ↓
ACTION
   ↓
EXPLAIN                   → LLM Narrator                  (Stage 4, §10)
```

| Stage | File | Role | Status |
|---|---|---|---|
| 1 — Regulatory Gate | `rule_gate.py` | Legality only, not viability | Specified, **not built** |
| 2 — Monte Carlo Planner | `planner.py` | Ranks legal modes by simulated outcome | Specified, **not built** |
| 3 — Confidence Gate | `confidence_gate.py` | Two-part significance + DCLI + rival-uncertainty check | Specified, **not built** |
| 4 — LLM Narrator | `narrator.py` | Turns the verified decision into a sentence | Not started — blocked on Stage 3 |

A fifth component, the **Rival Energy State Estimator** (`rival_estimator.py`,
§6), sits alongside this pipeline: it feeds Stage 2 and Stage 3 as a
probabilistic input, but is not itself one of the four numbered stages and
never touches Stage 1. A sixth, the **Opportunity Horizon**
(`opportunity_engine.py`, §9), sits *on top of* Stages 2-3 rather than
inside the numbered sequence — it chains the single-lap pipeline across
several future laps to compare whole strategies, not just this lap's
modes. **None of these six Python files exist on disk yet** — see §12
for what does.

**Data flow has two valid shapes, depending on how far the build has
gotten — both are real, not one superseding the other mid-build:**

- **Pre-Horizon (built first, §17 steps 1-7)**: `TelemetryInput` →
  `RegulatoryGate.evaluate()` → `GateResult` (`legal_modes`, `base_cap_mj`,
  `violations`) → (`legal_modes` + `PlanningContext`, optionally carrying a
  `RivalSocEstimate`) → `MonteCarloPlanner.plan()` → `PlannerResult`
  (ranked `ModeProjection`s, `recommended_mode`) → (`PlannerResult` +
  `DriverLoadInput`, optionally the same rival estimate) →
  `ConfidenceGate.evaluate()` → `ConfidenceGateResult` (final
  recommendation, possibly overridden to `BALANCED_MODE`). This is the
  complete single-lap pipeline and a legitimate demo on its own.
- **Post-Horizon (§17 step 8 onward)**: `opportunity_engine.py` chains
  `MonteCarloPlanner.plan()` across a bounded future horizon per candidate
  strategy (§9), and `ConfidenceGate.evaluate()`'s inputs become the
  **top-ranked strategy vs. runner-up strategy** instead of top mode vs.
  runner-up mode — same gate, extended inputs, not a second gate.

Either way, the final `ConfidenceGateResult` → Stage 4 (§10, not yet
built) → `StrategyCall` (§10's narrator-input contract).

Run tests from the repo root once the backend exists: `python -m pytest -q`.

## 4. Deployment modes — five, not four

| Mode | What it represents | Stage 1 legality condition |
|---|---|---|
| `CONSERVE_MODE` | Minimal deployment, banks energy for later | Base cap only |
| `BALANCED_MODE` | Default baseline deployment; the fallback target when Stage 3 overrides | Base cap only |
| `ARM_OVERTAKE_MODE` | Attack this lap while within proximity of the car ahead; also what qualifies the bonus for next lap | Base cap, **and** must be within `overtake_detection_gap_threshold_s` at the detection point |
| `USE_OVERTAKE_BONUS_MODE` | Spend a banked bonus from *last* lap's qualification | Base cap + bonus, **and** `overtake_qualified_last_lap` must be true — illegal (not merely capped lower) otherwise |
| `PUSH_MODE` | Sustained max-legal deployment (e.g. defending, a qualifying-style lap) | Base cap only |

**Why arming and spending are separate modes, not one mode with a hidden
flag**: the 2026 Overtake Mode bonus (+0.5 MJ) is banked on the lap you
qualify (within the gap threshold at the detection point) and can only be
*spent* on the following lap — a genuine sequential decision, not a
same-lap bonus. Splitting it into two modes keeps `ModeDynamics` (§7) a
clean per-mode lookup table instead of forcing conditional logic into it,
lets Stage 2 rank "should I arm" and "should I spend" as the independent
strategic choices they actually are, and keeps each mode's Stage 1
legality a single condition instead of one mode with two different rules
depending on hidden state. A single `HOLD_BALANCED` or abstention mode was
considered and rejected: abstention is a Stage-3 statement about
*confidence*, not an on-track action a car executes — it already resolves
to recommending `BALANCED_MODE` with `overridden=True` on the existing
`ConfidenceGateResult`, not a sixth deployment mode.

**Qualification is a telemetry fact, not a mode-choice fact.**
`GateResult.qualifies_for_overtake_bonus_next_lap` is computed purely from
`gap_to_car_ahead_s` — a car qualifies for next lap's bonus by being close
enough at the detection point, *regardless of which mode ChronoPace
actually recommended that lap*. Do not wire "qualification" to whether
`ARM_OVERTAKE_MODE` was the chosen strategy; that would be wrong and is an
easy mistake to make when implementing this.

**Presentation-layer relabeling — proposed, not resolved.** Pitch material
has floated user-facing action words (ATTACK / ARM / WAIT / HOLD /
CONSERVE) as friendlier stand-ins for the five mode names above. These are
UI labels only — the five backend enum values above are what code uses,
and do not change. The exact 1:1 mapping is **not settled**: ATTACK and
HOLD map cleanly (`USE_OVERTAKE_BONUS_MODE`, `BALANCED_MODE`), but WAIT
reads as an Opportunity Horizon *strategy* name (`WAIT_N`, §9) rather than
a single-lap mode, and `PUSH_MODE` ("sustained max-legal deployment, e.g.
defending") doesn't fit ATTACK/ARM/WAIT/HOLD/CONSERVE at all — it may need
its own label (e.g. DEFEND, echoing its own description) or to keep its
backend name in the UI. Resolve this mapping explicitly before
implementing the relabel (§17 step 11); don't let a component silently
invent one.

## 5. Stage 1 — Regulatory Gate (`rule_gate.py`)

**FIA 2026 Power Unit Technical Regulations — corrected constants**
(earlier drafts of this project inverted the deployment and recovery
figures; these are the corrected values, cite the article number
alongside the constant in code so a reviewer can check it without reading
logic):

| Constant | Value | Source |
|---|---|---|
| `max_ers_k_power_kw` | 350 kW | Art. 5.4.7 — max instantaneous MGU-K power |
| `max_deployment_per_lap_mj` | 9.0 MJ | Art. 5.4.10 — max energy, MGU-K HV DC bus, per lap (**deployment** side) |
| `max_delta_soc_mj` | 4.0 MJ | Art. 5.4.9 — max SoC swing per lap |
| `recoverable_energy_baseline_mj` | 8.5 MJ | Baseline **recovery/harvest** figure, event-variable (as low as 5 MJ at energy-starved circuits, up to 9 MJ at energy-rich ones, 7 MJ in qualifying) — informational/config-only, does not gate deployment legality. No confirmed article number for this specific figure; comment as "citation pending" rather than inventing one. |
| `overtake_detection_gap_threshold_s` | 1.0 s | F1 Sporting Regulations — proximity eligibility at the detection point |
| `overtake_bonus_mj` | 0.5 MJ | Banked on the qualifying lap, spendable only the following lap |
| `taper_normal_start_kmh` / `taper_normal_end_kmh` | 290 → 355 km/h | Normal deployment taper band |
| `taper_overtake_full_power_end_kmh` | 337 km/h | Full 350 kW sustained to this speed under the Overtake Mode bonus, then tapers |

**The 8.5 MJ figure is not a deployment cap.** A car can gross-deploy up
to 9 MJ in a lap while only net-changing SoC by 4 MJ, because it is also
harvesting during the same lap — these are governed by different articles
(5.4.10 vs 5.4.9) and must not be conflated. The +0.5 MJ bonus applies to
`max_deployment_per_lap_mj` only, **never** to `max_delta_soc_mj`.

**`TelemetryInput`**: `lap_number`, `current_soc_mj`, `lap_start_soc_mj`
(optional — drives the SoC-swing check), `lap_energy_deployed_mj`,
`gap_to_car_ahead_s` (optional), `overtake_qualified_last_lap` (bool,
default `False` — true only if this car was within the gap threshold on
the *immediately preceding* lap; use-it-or-lose-it, the calling
orchestrator must not carry `True` forward more than one lap).

**`GateResult`**: `legal_modes`, `violations` (dict, always all 5
mode-keys present, `[]` for legal modes), `base_cap_mj` (**dict, not a
scalar** — the bonus makes the cap mode-dependent),
`qualifies_for_overtake_bonus_next_lap` (bool, threaded by the caller into
next lap's `TelemetryInput`).

**Binding design decisions (do not relitigate without new information):**
- **The gate enforces legality, not viability.** A mode with 0.1 MJ of
  remaining legal budget is still legal — whether it's worth using is
  Stage 2's job. Don't add a "not worth it" pruning rule here.
- **The SoC-swing check fails open** (skips silently, logs a note) when
  `lap_start_soc_mj` is `None`, rather than failing closed. A deliberate
  demo-usability choice — flagged as a candidate for fail-closed in a
  production/safety-certified build. Don't change the default without
  flagging it as a safety-posture change.
- **`USE_OVERTAKE_BONUS_MODE` does not re-check the gap** — only the
  banked flag. Spending a banked bonus after the gap has since closed is
  legal; whether it's still tactically wise is Stage 2's job (same
  legality-not-viability principle, applied to the new mechanic).

**Naming**: the interactive demo concept once called the "Judge Sabotage
Slider" is now the **Compliance Probe** — push telemetry to the legal
boundary and watch which modes get rejected, with the specific violated
rule cited via `violations[mode]`. The dashboard's `ComplianceProbe`
component (§12) already uses this name — keep it consistent everywhere,
including backend code comments/docstrings.

## 6. Rival Energy State Estimator (`rival_estimator.py`)

No F1 team publishes ERS state of charge for any car — not even your own
team gets a rival's. This module infers a **posterior distribution** over
a rival's SoC from kinematic proxies (a particle filter), not a point
value.

**Reconciling with this project's own binding rule**: an earlier,
rejected version of this idea (acoustic FFT on broadcast audio) came with
a binding rule — its output "must be a labeled, confidence-scored
estimate feeding a context/narration layer only — never a value the
regulatory gate or planner treats as ground truth." This module feeds
Stage 2's Monte Carlo and Stage 3's confidence gate directly, which
appears to conflict with that rule on its face. The reconciliation,
deliberate and stated here so it's a recorded decision, not an implicit
one:
- The estimate propagates as a **distribution** (`mean_soc_mj` +
  `std_soc_mj`), sampled per-iteration inside the Monte Carlo — never
  collapsed to a single point value asserted as fact anywhere downstream.
  This is the same treatment the planner already gives its own
  `ModeDynamics` priors.
- It **never reaches Stage 1.** `TelemetryInput` and `RegulatoryGate` take
  no rival-estimate input at all — legality stays based only on the car's
  own verified telemetry. Enforced at the module level: `rival_estimator.py`
  may import shared constants *from* `rule_gate.py`; `rule_gate.py` must
  **never** import from or reference `rival_estimator.py`. That one-way
  asymmetry is the concrete, checkable form of "never reaches Stage 1."
- Stage 3 is made structurally **more cautious**, not more trusting, when
  the estimate's uncertainty is high (§8) — the system hedges rather than
  trusts an uncertain estimate. That's what keeps this consistent with
  "never ground truth" in spirit, not just in the letter of "it's
  technically a distribution."

**Observables — expanded from the original 2-signal scope.** The first
draft trimmed this to `terminal_speed_kmh` and `clipping_point_fraction`
to stay lightweight. The team deliberately re-scoped this to a fuller,
still-tractable set, because the demo story ("we infer this from
observable behavior") is stronger with real breadth behind it:

| Field | What it captures | Why it's legitimate |
|---|---|---|
| `terminal_speed_kmh` | Peak straight-line speed | Direct proxy for power available at that moment |
| `clipping_point_fraction` | Where on the straight the speed trace flattens (0-1) | Proxy for how early the power-limited taper kicks in |
| `corner_exit_accel_g` | Longitudinal acceleration on corner exit | Higher deployment enables harder exit acceleration, up to a traction ceiling — this is the "corner-exit acceleration gradient" signal from the project's original rival-estimator pitch, reinstated here |
| `sector_delta_s` | Rival's sector time vs. their own rolling baseline for that sector | A standard, real, FIA-timing-derived quantity every team already receives; a genuine pace/deployment proxy independent of the straight-line signals above |

All four are things every team already gets from FIA timing/GPS for every
car on track — nothing here requires access any team lacks. "ERS
deployment pattern" and "previous lap behavior" (both named in early
brainstorming) are **not** separate fields: the *pattern* is what these
four signals look like *together*, and "previous lap behavior" is already
captured by the filter's own recursive structure (`n_observations`,
persistent particle weights across calls) — the posterior after lap 14 is
already informed by laps 1-13, that's what a particle filter *is*. Adding
a fifth field to represent "history" would be redundant with the
mechanism itself.

**`RivalEstimatorConfig`** (frozen dataclass): `n_particles=1000`,
`soc_min_mj=0.0`, `soc_max_mj=9.0` (mirrors `GateConfig.max_deployment_per_lap_mj`
— import it, don't retype it, so the two never drift apart),
`process_noise_std_mj=0.3`, `observation_noise_std_speed_kmh=5.0`,
`observation_noise_std_clip_fraction=0.08`,
`observation_noise_std_accel_g=0.15` (new), `observation_noise_std_sector_delta_s=0.12`
(new), `expected_soc_drift_per_lap_mj=-0.5`,
`ess_resample_threshold_fraction=0.5`, `default_seed=42`. The predict-step
drift is still a single configurable constant — this deliberately does
not jointly infer the rival's own mode choice, which would be a much
larger undertaking than this module's scope calls for even after the
expansion.

**`RivalStateEstimator`**: `predict()` advances particles by the drift
constant plus process noise, clipped to `[soc_min, soc_max]`. `update(observation)`
reweights particles by a Gaussian likelihood combining **all four**
observables, each comparing a particle's *expected* kinematics at that
SoC against the *actual* observation:
- `expected_speed`/`expected_clip` — unchanged, reuse Stage 1's taper
  curve as an evidence proxy (a repurposing of those constants for a
  different purpose than their regulatory meaning, not a regulatory
  claim itself — document as such).
- `expected_accel_g = baseline_accel_g + (soc / soc_max) * accel_gain_g`
  (new illustrative constants, e.g. `baseline_accel_g ≈ 1.0`,
  `accel_gain_g ≈ 0.3`) — more SoC, more deployable exit acceleration.
- `expected_sector_delta_s = -(soc / soc_max) * sector_gain_s` (new
  illustrative constant, e.g. `sector_gain_s ≈ 0.4`) — more SoC, a more
  negative (faster) sector delta.

The four per-observable log-likelihoods are summed (independence
assumption, standard for a first-pass particle filter), max-subtracted
before `exp` for numerical stability, then normalized and resampled via
systematic resampling when effective sample size drops too low.
`estimate()` returns the weighted mean/std as a `RivalSocEstimate`
(`mean_soc_mj`, `std_soc_mj`, `n_observations`). Fully vectorized NumPy,
seeded RNG, same conventions as the rest of the project (§13). All four
new constants (`baseline_accel_g`, `accel_gain_g`, `sector_gain_s`, the
two new observation-noise stds) are engineered/illustrative, same caveat
as `ModeDynamics` — not measured from real car data.

**Validation**: besides standard unit tests, this module has a synthetic
recovery test — generate observations from a *known*, hidden SoC using
the estimator's own observation model plus injected noise, run the
filter, and confirm the recovered mean converges within tolerance and
`std_soc_mj` shrinks as evidence accumulates. This is the standard
validation approach for a latent-variable filter, and it's the answer to
a judge asking "how do you know this estimator works" — a unit-test suite
alone doesn't answer that question, this test does.

## 7. Stage 2 — Monte Carlo Planner (`planner.py`)

**There is no separate "Energy State Model" module** — this is worth
saying explicitly because the concept (track current SoC, deployable
energy, reserve, recent trend, and the energy cost of each candidate
action; carry that state forward as actions are simulated) is real and
load-bearing, it's just not its own file. `TelemetryInput` (§5) is the
snapshot; the *forward-simulated* state it implies lives here, inside each
Monte Carlo iteration, and — for multi-lap comparisons — inside
`opportunity_engine.py`'s lap-to-lap carry described in §9. If a teammate
goes looking for `energy_state.py`, it doesn't exist and shouldn't; the
state-tracking is distributed across these two places by design, not
missing.

`ModeDynamics` (frozen dataclass, one row per mode — now 5 rows) holds
racecraft priors: `mean_laptime_delta_s` (negative = faster/better,
`BALANCED_MODE` = 0.0 baseline), `std_laptime_delta_s`,
`overtake_success_prob`. **This last field means different things by
mode** — for `ARM_OVERTAKE_MODE` it's the probability of *qualifying*
(closing to the gap threshold); for `USE_OVERTAKE_BONUS_MODE` and
`PUSH_MODE` it's the probability of *completing* an overtake. Same field
name, different semantics — document this explicitly, it's an easy thing
to miss.

**These priors are engineered, not measured.** The simulation math is
correct regardless of these constants; their realism is not validated.
This is the single most important calibration gap in the project — never
present the planner's numeric output as validated without repeating this
caveat, in docstrings and in any user-facing summary.

`PlannerConfig`: `n_iterations = 10_000` (fixed and reported in output,
never varied — see §8 for why a variable rollout count would let the
confidence gate manufacture significance), `default_seed`, shared taper
constants (290/355/337 km/h — these describe power-unit hardware
behavior common to all modes, not a per-mode strategy choice, so they
live in config rather than being duplicated across 5 `ModeDynamics` rows),
`overtake_gap_max_bonus`, `failed_overtake_penalty_s`,
`defense_penalty_weight = 0.3` (deliberately modest, not higher, so a
single noisy rival estimate can't swing outcomes nearly as much as if it
were asserted as fact).

`PlanningContext`: `gap_to_car_ahead_s` (optional), `rival_soc_estimate:
Optional[RivalSocEstimate] = None`. Deliberately does *not* carry
`overtake_qualified_last_lap` — that fact belongs exclusively to Stage
1's `legal_modes` output; duplicating it here would create two sources of
truth that could drift apart.

**Seeded determinism**: `numpy.random.SeedSequence(seed).spawn(5)` gives
one independent child RNG stream per mode, in a fixed canonical order —
**not** `legal_modes`'s order. Without this, a mode's samples would
silently depend on which *other* modes happened to be legal that call,
breaking the seeded-determinism guarantee this project's eventual replay
("Time Machine") feature depends on. This must be `.spawn(5)`, not
`.spawn(4)` — a leftover 4 would silently corrupt the newest mode's
stream.

**Rival-estimate modulation** (applies only to `ARM_OVERTAKE_MODE` and
`USE_OVERTAKE_BONUS_MODE`, only when a rival estimate is supplied, using
that mode's own seeded stream): sample a rival SoC per Monte Carlo
iteration from `Normal(mean_soc_mj, std_soc_mj)`, clip to
`[0, max_deployment_per_lap_mj]`, convert to a 0-1 "defense factor," and
scale the effective overtake-success probability down by
`defense_penalty_weight * defense_factor`. When no rival estimate is
supplied, behavior is byte-identical to the no-rival-estimator case. This
mapping is engineered/illustrative, same caveat as `ModeDynamics` — not a
claim about real rival-car defensive physics.

`get_raw_samples(result, mode)` is keyed by `PlannerResult.run_id` (a
monotonic counter), not just by mode — caching only "the most recent
`plan()` call's" samples would silently corrupt an earlier `PlannerResult`
handed to Stage 3 after a second `plan()` call. Raises `ValueError` on a
stale/unknown `run_id`. Sharpe ratio (`-mean/std`) is capped at ±999 when
variance is near zero, never returned as `inf`/`NaN` — a numerical-
stability fix so downstream statistics always receive finite inputs.

## 8. Stage 3 — Confidence Gate (`confidence_gate.py`)

**Why a plain significance test doesn't work here**: standard error scales
as `s/√n`. With `n_iterations` fixed at 10,000, `s/√n` is tiny, so almost
any nonzero true difference between two modes clears a naive `t ≥ 2.0`
threshold — the gate would measure "is n big enough" (always yes, since n
is fixed and large), not "does the difference actually matter." Under
that design, abstention — this project's signature behavior — becomes
nearly impossible to trigger.

**The fix — a two-part gate.** Both statistical reliability AND practical
significance must pass, together with DCLI and the rival-confidence
check, before a non-`BALANCED_MODE` recommendation reaches Stage 4:

1. **Statistical reliability**: compute a one-sided 95% confidence lower
   bound on the difference between the top two ranked modes' means
   (`diff = mean(runner_up) − mean(top)`, positive = top mode confidently
   better — same sign convention throughout, see the warning below), using
   the existing Welch-Satterthwaite degrees of freedom:
   `ci_lower_bound_s = diff − t_crit(confidence_level, df) × se`. Passes
   when `ci_lower_bound_s > min_actionable_laptime_delta_s` — **not**
   merely `> 0`. A one-sided test is the methodologically correct choice
   here (this is a superiority test — does the top mode beat the runner-up
   by a margin — not a two-sided "is there any difference" test).
2. **Practical significance**: `min_actionable_laptime_delta_s = 0.05`
   (seconds, laptime-equivalent) is the default floor for every
   comparison. When either top-two mode is `ARM_OVERTAKE_MODE` or
   `USE_OVERTAKE_BONUS_MODE`, an additional check runs on the
   overtake-probability axis with `min_actionable_overtake_prob_pp = 0.03`
   (3 percentage points) — both required when applicable, since a
   laptime-only view can understate what matters for overtake-flavored
   comparisons.
3. **DCLI (Driver Cognitive Load Index)**: a second, fully independent
   gate — not a substitute for the significance test. Combines three
   normalized [0,1] signals (time since last mode change, recent laptime
   variability, proximity pressure) into a 0-100 score; passes below 60.
   Fully invented — no source document exists for this formula — flagged
   illustrative, same pattern as `ModeDynamics`.
4. **Rival confidence**: `rival_confidence_passed = (rival_estimate is
   None) or (rival_estimate.std_soc_mj <= 1.5)`. This is what makes "the
   gate falls back to `BALANCED_MODE` when rival uncertainty is high" a
   direct, testable rule — not just an emergent side effect of the wider
   Monte Carlo variance a noisy rival estimate also produces in Stage 2
   (§7). Both effects are real and intentionally coexist: one is genuine
   outcome-uncertainty propagation, the other a direct sanity check on
   the estimator's own confidence.

`n_iterations` is fixed and echoed into `ConfidenceGateResult` precisely
because a *variable* rollout count would let the gate manufacture
statistical significance by running longer — fixing and reporting it is
what makes the test honest.

**Sign-convention warning — the single highest-risk correctness point in
this entire project**: under "negative delta = faster/better," the
difference must be computed as `mean(runner_up) − mean(top)`, runner-up
first. Subtracting in the other order silently inverts the whole gate's
pass/fail logic with no error — a confidently-better top mode would
produce a negative statistic and wrongly fail. Test this explicitly and
prominently; do not let an implementation "simplify" the subtraction
order.

`ConfidenceGateResult`: `recommended_mode`, `stage2_recommended_mode`
(the original top pick, preserved even when overridden — Stage 4 needs
this to narrate *what changed*), `runner_up_mode`, `t_statistic`,
`degrees_of_freedom`, `ci_lower_bound_s` (the actual bound, not just a
boolean — this is exactly the kind of already-verified number Stage 4's
narrator should use, e.g. "confident of at least a 0.08s gain"),
`statistical_reliability_passed`, `practical_significance_passed`,
`dcli_score`, `dcli_passed`, `rival_confidence_passed`, `n_iterations`,
`overridden`, `override_reason` (when multiple gates fail, name all of
them, not just the first).

## 9. Opportunity Horizon (`opportunity_engine.py`) — multi-lap strategy comparison

**The reframe this exists for**: ChronoPace's original question was "is
this mode legal and worth it *this lap*?" The sharper question — and the
one the product is now built around — is "is *this* the best moment to
spend limited energy, or will a better opportunity appear later?" That
question cannot be answered by a single-lap comparison; it requires
comparing a small number of complete **future strategies**, not just five
options for the next lap.

**This is a new module, not a rewrite of Stage 2.** `planner.py` keeps
doing exactly what §7 describes — one legal-mode comparison, one lap.
`opportunity_engine.py` sits on top of it and **chains repeated calls** to
that same single-lap machinery across a bounded future horizon, once per
candidate strategy, then hands the aggregated results to an extended
Stage 3 comparison. Nothing about the single-lap physics changes; what's
new is comparing *sequences* of laps instead of one lap.

**Candidate strategies — a fixed, named set, not an open search.** Bounded
on purpose: an unconstrained multi-lap optimization is a much harder
problem than a hackathon needs, and a strategist reading the output needs
named options, not a search result. A configurable `delay_laps` list
(illustrative default `[0, 2, 5]`) plus a permanent `HOLD` baseline:
- **`ATTACK_NOW`** (`delay_laps = 0`) — attempt the overtake sequence
  starting this lap (`ARM_OVERTAKE_MODE` → `USE_OVERTAKE_BONUS_MODE` the
  following lap), `BALANCED_MODE` for the remainder of the horizon.
- **`WAIT_N`** (one per entry in `delay_laps` greater than 0) —
  `BALANCED_MODE`/`CONSERVE_MODE` for `N` laps, then the same attempt
  sequence, then `BALANCED_MODE` for the remainder.
- **`HOLD`** — `BALANCED_MODE` for the entire horizon. The safe baseline
  every other strategy is measured against, same role `BALANCED_MODE`
  already plays in the single-lap gate.

**Simulation mechanism.** For each strategy, for each of `n_iterations`
Monte Carlo draws: walk the horizon lap by lap, drawing that lap's
`ModeDynamics` sample exactly as `planner.py` already does (including
rival-estimate modulation where applicable), carrying the simulated
energy state forward lap-to-lap using the corrected FIA constants (§5),
and accumulating the laptime-delta contribution. The result is one sample
array per strategy, the same shape `get_raw_samples()` already returns
per mode — the extension is that each "sample" is now a summed
multi-lap outcome instead of a single lap's.

**Uncertainty must widen with distance, honestly.** A rival estimate
computed from *this* lap's kinematics is not equally trustworthy 5 laps
out — nothing re-observes the rival in between. Effective standard
deviation at horizon offset `d` laps is scaled by a new, explicitly
illustrative `PlannerConfig` constant: `effective_std = base_std * (1 +
horizon_uncertainty_growth * d)` (default `horizon_uncertainty_growth ≈
0.15`, i.e. roughly 15% wider per lap of lookahead), applied to both the
mode's own `std_laptime_delta_s` and the rival estimate's `std_soc_mj`
when either is used inside a horizon-offset lap. This is what keeps a
"wait 5 laps" recommendation from claiming false confidence — it is
mechanically required to look less certain than "wait 2 laps," which
should in turn look less certain than "attack now."

**Stage 3 extension, not a new gate design.** The exact same two-part
significance construction from §8 (statistical reliability CI, practical
significance, DCLI, rival confidence) applies to the **top-ranked
strategy vs. the runner-up strategy's aggregated horizon outcome**,
instead of top mode vs. runner-up mode. Two new fields express the
"opportunity cost" framing directly, rather than leaving the viewer to
compare two numbers themselves: `foregone_strategy` (the runner-up
strategy's name) and `foregone_value_gap` (how much aggregated value the
chosen strategy beats it by) — this is what lets the eventual narration
say "attacking now beats waiting 2 laps by 0.3 expected positions,"
rather than just naming a winner.

**Build order note**: this depends on Stage 2's single-lap machinery
existing and tested, and its Stage 3 extension depends on Stage 3 being
built first. It is an addition to the build sequence in §17, not a
reordering of it — build the single-lap pipeline (Stages 1-3 + rival
estimator) completely first, exactly as already planned, then add this
layer on top.

## 10. Stage 4 — LLM Narrator (`narrator.py`) — input contract

Not built, and per §2/§17 build order, not to be started before Stage 3
(and ideally Opportunity Horizon) are green. Specified here now anyway,
because Stage 3's output fields (§8) should be designed with this consumer
in mind, not bolted on later.

**Proposed input JSON** — the boundary between deterministic intelligence
and narration, illustrative values only:

```json
{
  "decision": "USE_OVERTAKE_BONUS_MODE",
  "confidence": 0.82,
  "expected_position_gain": 1.2,
  "energy_cost": 2.1,
  "rival_energy_estimate": { "mean": 2.1, "uncertainty": 0.6 },
  "reason_codes": [
    "STRONG_OVERTAKE_OPPORTUNITY",
    "LOW_ESTIMATED_RIVAL_RESERVE",
    "SUFFICIENT_OWN_ENERGY",
    "CURRENT_OPPORTUNITY_OUTVALUES_PROJECTED_WAIT"
  ]
}
```

**This is a summary view, not a new computation.** Every field here must
be assembled by Python from fields already specified elsewhere in this
document — `ConfidenceGateResult` (§8), extended by Opportunity Horizon
(§9) once that exists. Two fields need a defined, deterministic derivation
before Stage 4 is built (open — flagged, not yet decided):
- **`confidence`** (a single [0,1] scalar) must reduce from
  `ci_lower_bound_s`'s margin above the practical-significance floor
  and/or `dcli_score` — candidates to evaluate at build time, not a
  second, independently-eyeballed number that could disagree with
  `statistical_reliability_passed`/`practical_significance_passed`.
- **`reason_codes`** is a **fixed vocabulary** Python selects from based on
  which gates passed/failed and which thresholds were crossed (e.g. a
  `RIVAL_UNCERTAINTY_HIGH` code when `rival_confidence_passed` is
  `False`, a `CURRENT_OPPORTUNITY_OUTVALUES_PROJECTED_WAIT` code when
  Opportunity Horizon's top strategy is `ATTACK_NOW`) — never
  LLM-generated. The LLM narrates the codes it's given; it does not invent
  new ones or omit ones that don't fit a nice sentence.

**Target narration style** (LLM output, given the JSON above):

```text
Decision: USE_OVERTAKE_BONUS_MODE
Confidence: 82%
Reasons:
- Strong overtake opportunity
- Low estimated rival reserve
- Sufficient own energy
- Current opportunity exceeds projected future value
```

**Restating §2 specifically for this stage** — the LLM must NOT: change
`0.82` to any other value, change the decision/action, alter any energy
number, override a legality result, invent a reason code not present in
the input, or introduce any race-state fact not given to it. If the LLM
is removed entirely, the JSON above is still a complete, useful,
machine-readable decision — that's the test for whether Stage 4 stayed in
its lane.

## 11. Core Demo Scenarios

Build the product around a small number of fixed, reproducible scenarios
rather than an open-ended live demo — the point to land with judges is
concrete and specific: **the same overtake opportunity can produce a
different optimal decision depending on our energy state and how far out
the horizon looks.**

- **Scenario A — attack.** High own SoC, low rival estimate, high current
  opportunity → high confidence → `USE_OVERTAKE_BONUS_MODE` (ATTACK).
- **Scenario B — hold.** Same opportunity and same low rival estimate as
  A, but **low own SoC** → `BALANCED_MODE` (HOLD/WAIT). Note for whoever
  implements this: it isn't yet decided *which mechanism* produces this
  outcome — Stage 2 may simply rank `BALANCED_MODE` first once low SoC
  changes the energy-cost tradeoff, or Stage 2 could still rank an
  aggressive mode first with Stage 3 then abstaining on confidence
  grounds. Both are legitimate demonstrations of the system; which one
  actually happens depends on the real `ModeDynamics`/gate numbers once
  built. Don't hard-code the dashboard to assume a specific one of these
  before it's known.
- **Scenario C — horizon comparison.** Requires Opportunity Horizon (§9)
  built. Same opportunity, compare `ATTACK_NOW` vs. `WAIT_2` vs. `WAIT_5`
  vs. `HOLD` side by side, uncertainty visibly widening with distance.

The existing `DecisionBanner` "Demo: toggle scenario" control (§12)
currently flips between a generic pass-case and a generic override-case —
it should eventually be rebuilt around these three named scenarios
specifically, once real fixtures exist to drive them. Tracked as UI work
in §17 step 11, not done yet. Once the backend exists, these must be real
computed outputs from specific seeded `TelemetryInput` fixtures — three
more hand-typed JS objects would repeat the exact problem this document's
opening section warns about (a mock that looks like intelligence but
isn't).

## 12. Dashboard UI (`frontend/`) — built, but running on fake data

Unlike Stages 1-4, this part genuinely exists on disk and runs. It is a
**visual prototype of what ChronoPace looks like once the backend is
real** — every panel, layout, and interaction was designed to match the
data shapes above exactly, but right now every number behind it is
hand-typed, not computed.

**Stack**: React 19 + Vite. `@react-three/fiber` + `@react-three/drei` +
`three` for the 3D car viewport. Plain CSS Modules (no Tailwind/UI kit) —
one `.module.css` file per component. No routing, no state management
library; local `useState` only.

**Layout** (`App.jsx`): a centered "CHRONOPACE" wordmark above a
two-column dashboard —
- **Left column**: `DecisionBanner` (full width) → a row of
  `ComplianceProbe` + `MonteCarloPlanner` side by side → `RivalEstimator`.
- **Right column**: `TelemetryHeader` → `Viewport3D`.

**Components, one-to-one with the backend concepts above**:
| Component | Mirrors | What it shows |
|---|---|---|
| `DecisionBanner` | `ConfidenceGateResult` (§8) | The "EXECUTE: {mode}" / "OVERRIDE → {mode}" call, the 4 gate-pass pills, the CI-bound/t-stat/DCLI/rival-σ stats line. Has a working "Demo: toggle scenario" button that flips between a canned pass-case and a canned override-case — the only interactive element on the page right now. |
| `ComplianceProbe` | `GateResult` (§5) | The FIA constant checks (MGU-K power, lap deployment, delta-SoC swing, overtake-bonus banking) as PASS/BREACH rows citing article numbers, plus a worked breach example. |
| `MonteCarloPlanner` + `DistributionSparkline` | `PlannerResult` / `ModeProjection` (§7) | The 5-mode ranked table — laptime delta ± std, Sharpe ratio, per-mode sparkline, iteration count. |
| `RivalEstimator` + `PosteriorPlot` | `RivalSocEstimate` (§6) | The posterior density plot (mean/σ), terminal speed, clipping point, attack tendency, and the "modeled, not measured" disclaimer. |
| `TelemetryHeader` | `TelemetryInput` (§5) | Session/lap/car number, speed, ERS SoC gauge. |
| `Viewport3D` | — (no backend equivalent) | Renders the team's own `rb22.glb` model (`frontend/public/models/`) on an auto-rotating turntable, lit by a controlled overhead spotlight plus fill lighting, standing on a dark stage with the cyan ring markers. Presentation only — carries no data. |
| `GlassPanel` / `Icons` | — | Shared card shell (dark glass, blurred backdrop, glowing cyan border) and the hand-drawn SVG icon set every panel uses. No emoji anywhere in the UI, by design. |

**Visual identity**: Rajdhani (display type) + JetBrains Mono (numeric/
technical readouts) via Google Fonts; a near-black background with a
radial vignette over a subtle repeating "carbon fiber" weave; electric
blue as the primary/energy accent, green/amber/red for pass/warn/fail
status, dark glassmorphic cards throughout. This went through several
redesign passes before landing here — this is the version the team
settled on, not a first draft.

**The one thing to actually understand about the data**: everything
renders from `frontend/src/data/mockTelemetry.js`, a single static file.
Its exports are named and shaped to match the real Pydantic models above
field-for-field on purpose (`complianceChecks`, `modeProjections`,
`rivalEstimate`, `confidenceGatePass`/`confidenceGateOverride`, etc.) —
so that once the backend exists, pointing the dashboard at it is a
**data-source swap, not a component rewrite**. Do not let a UI change
quietly invent a field the backend doesn't have, or rename one the
backend does — that would turn this from "ready to wire up" into "needs
reconciling."

**Not built yet**: the bridge between this UI and a real backend. The
plan is a small **FastAPI** service wrapping Stages 1-3 (FastAPI uses
Pydantic natively, so the exact models in §5-§8 become the API's request/
response schemas with no translation layer), called from the frontend
with `fetch`. Also undecided: where real telemetry to feed that API would
come from — leading candidate is replaying real 2026 FastF1 timing data
rather than synthetic data, for credibility, but nothing is built or
committed here.

**Running it locally**: `cd frontend && npm install && npm run dev`
(or, inside this Claude Code project, the `chronopace-frontend` config in
`.claude/launch.json` — port 5173 by default, falls back automatically if
that's taken).

## 13. Conventions — follow exactly (backend/Python)

- **Pydantic v2** for all cross-stage I/O models (`BaseModel`, `Field`
  with constraints like `ge=0`). Plain `@dataclass(frozen=True)` for
  internal config objects not serialized across a boundary (`GateConfig`,
  `PlannerConfig`, `ModeDynamics`, `RivalEstimatorConfig`).
- **pytest**, one test file per module (`test_<module>.py`), organized by
  rule/behavior with a `# --- section --- #` banner. `test_integration.py`
  is one deliberate, documented exception — it exists specifically to
  catch cross-stage bugs (sign convention, `run_id` plumbing, RNG-stream
  independence, the rival-estimator's Stage 2/3 dual effects) that
  per-module unit tests can't see.
- Every public class and non-trivial function has a docstring explaining
  **why**, not just what — record the design decision, not just the
  signature.
- Constants mapping to a real-world regulation are named and commented
  with the article they encode, so a reviewer can check them against real
  FIA text without reading logic. Constants that are numerical-stability
  fixes (the ±999 Sharpe cap) or pure engineering choices (DCLI weights,
  `defense_penalty_weight`) are commented as such — never conflate the
  two categories.
- NumPy vectorized Monte Carlo and particle-filter operations — batched
  array ops, never a per-iteration Python loop.
- Seeded RNG (`np.random.default_rng` / `SeedSequence`) everywhere
  randomness is used, with a default seed and an optional override.
  Never unseeded — this is what makes "deterministic pipeline" true
  despite Monte Carlo simulation and particle filtering both being
  involved, and what an eventual replay/"Time Machine" feature depends on.

**Validation is layered, not just "green tests"**: deterministic unit
tests → synthetic Core Demo Scenarios (§11) → Monte Carlo sanity checks →
(later) historical-telemetry backtesting → calibration. Below the
per-module test banners already specified in each stage's own section,
every module's suite should collectively prove these eight things — cross-
reference, don't re-derive a new test scheme:
1. Regulatory Gate rejects illegal actions correctly (§5 tests).
2. Energy accounting stays consistent across a simulated sequence of
   actions (§7's energy-state note, §9's lap-to-lap carry).
3. Rival uncertainty changes downstream outcomes (§7's rival-modulation
   tests, §8's rival-confidence tests).
4. The recommendation changes when our own energy state changes (§7).
5. Opportunity Horizon accounts for opportunity cost, not just raw
   per-lap value (§9).
6. Confidence decreases appropriately as uncertainty increases (§8, §9's
   `effective_std` growth).
7. The system abstains when evidence is insufficient — this is the one
   with no natural home in a per-module suite; make sure
   `test_integration.py` has a case that actually triggers
   `overridden=True` (§8).
8. The LLM narrator preserves every deterministic output unchanged (§10)
   — a literal test asserting the narrated numbers match the input JSON.

## 14. Explicit "do not" list (binding)

- No graph database, vector database, or "Obsidian/Graphify" memory
  system anywhere in the runtime.
- No multi-agent debate loops. No giving the LLM authority to pick a
  strategy.
- No LLM call anywhere in Stages 1-3.
- `rule_gate.py` must never import from or reference `rival_estimator.py`
  — the concrete, checkable form of "rival data never reaches Stage 1."
- No acoustic FFT telemetry processing — separating one power unit's
  signature from nineteen others through broadcast audio compression is
  an unsolved source-separation problem, not a narrowing one. If SoC
  inference is needed, use the particle filter (§6), which works from
  real, public kinematic data instead.
- No game-theory thermal spoofing — this would mean deliberately
  manipulating a rival's strategy system, which directly contradicts this
  project's own central claim ("check what's legal before optimizing
  anything").
- No full active-aero actuation gate — if it comes up at all, it's a
  one-line drag modifier, not a named feature.
- No FastF1 / real-telemetry integration in the hackathon build —
  **resolved, not pending anymore**: build a deterministic
  `telemetry_simulator.py` first (generates the seeded, reproducible
  `TelemetryInput`/`RivalObservation` sequences the Core Demo Scenarios
  (§11) and test fixtures need), and keep every consumer
  telemetry-source-agnostic so a real feed can replace it later without a
  redesign. `RivalObservation`, `TelemetryInput`, and `PlanningContext` are
  plain Pydantic models that don't care whether their values came from the
  simulator or real data — that's what makes the swap possible. FastF1 (or
  any real feed) is a **later calibration layer**, not a hackathon-build
  dependency; don't block Tier 1 progress on it.
- Never claim, in any pitch material or in the UI: a specific dataset
  size, a measured model accuracy, real-time F1 data access, an
  endorsement, a Haas deployment, or access to proprietary Haas telemetry.
  None of that exists. The honest, defensible framing (§16) is:
  "the prototype architecture is deterministic and testable; calibration
  against real race data is the next validation layer" — say that, not
  something stronger.
- No moving the legality check later in the pipeline. It has come up as
  "check compliance last, right before executing" in narrative framing —
  that's fine for how a *demo* walks through the reasoning out loud, but
  the actual computation must keep checking legality **first** (Stage 1,
  before Monte Carlo runs at all), on the existing principle that an
  illegal mode doesn't exist as an option rather than getting vetoed at
  the end. Narrative order and computation order are allowed to differ;
  don't let the former quietly become the latter.
- No renaming or reshaping a field in `mockTelemetry.js` without checking
  it against the matching Pydantic model above — the whole point of the
  mock data is that it won't need touching when the real API lands.
- No presenting an Opportunity Horizon strategy (§9) N laps out with the
  same confidence framing as a next-lap recommendation. The uncertainty
  growth in §9 is a hard requirement, not a nice-to-have — an
  unqualified "wait 5 laps" claim is exactly the overconfident-sounding
  behavior the whole project's abstention design exists to avoid.

## 15. Repo layout

```
context.md                        This file
ChronoPace-Solution-Design.pdf     Companion narrative doc (problem/solution/why) for sharing with teammates

--- backend — specified below, NOT YET ON DISK ---
telemetry_simulator.py            Deterministic, seeded scenario generator — produces the TelemetryInput /
                                     RivalObservation sequences behind the Core Demo Scenarios (§11) and test
                                     fixtures; the only planned telemetry source for the hackathon build (§14)
test_telemetry_simulator.py         Telemetry simulator tests
rule_gate.py                      Stage 1 — RegulatoryGate, DeploymentMode enum, Pydantic I/O models
test_rule_gate.py                   Stage 1 tests
rival_estimator.py                 Rival Energy State Estimator — particle filter, one-way import from rule_gate.py only
test_rival_estimator.py             Rival estimator tests, including synthetic recovery
planner.py                          Stage 2 — MonteCarloPlanner, ModeDynamics priors
test_planner.py                      Stage 2 tests
confidence_gate.py                  Stage 3 — ConfidenceGate, two-part significance + DCLI + rival check
test_confidence_gate.py              Stage 3 tests
opportunity_engine.py               Opportunity Horizon (§9) — chains planner.py across a multi-lap
                                       horizon per candidate strategy, built on top of Stages 1-3, not a
                                       reordering of them
test_opportunity_engine.py           Opportunity Horizon tests
narrator.py                          Stage 4 (§10) — not built until Stage 3 is green
test_integration.py                 End-to-end pipeline tests (the one deliberate exception to one-file-per-module)
conftest.py                          Shared pytest fixtures
requirements.txt                     numpy, pydantic, pytest, scipy (all pinned)

--- frontend — actually built ---
frontend/
  src/
    App.jsx, App.module.css          Top-level layout (§12)
    components/
      DecisionBanner.jsx/.module.css
      ComplianceProbe.jsx/.module.css
      MonteCarloPlanner.jsx/.module.css, DistributionSparkline.jsx
      RivalEstimator.jsx/.module.css, PosteriorPlot.jsx
      TelemetryHeader.jsx/.module.css
      Viewport3D.jsx/.module.css
      GlassPanel.jsx/.module.css, Icons.jsx
    data/mockTelemetry.js            The single source of every number on screen right now (§12)
    index.css, main.jsx
  public/models/rb22.glb             3D car asset
  vite.config.js, package.json
```

Backend files live at the repository root, no `src/` layout, once they
exist — §17 has the build order.

## 16. What actually exists right now — summary

**Data/validation framing, for any pitch material**: *"The prototype
architecture is deterministic and testable; calibration against real race
data is the next validation layer."* Say that, not anything implying a
current dataset, measured accuracy, or real-time F1 access — none of
those exist yet (§14).

| Component | Status |
|---|---|
| Architecture, data contracts, regulatory constants | Fully specified in this document |
| Telemetry Simulator (`telemetry_simulator.py`) | Designed, not implemented — first thing to build (§17) |
| Stage 1 — Regulatory Gate | Designed, not implemented |
| Rival Energy State Estimator (4 observables) | Designed, not implemented |
| Stage 2 — Monte Carlo Planner | Designed, not implemented |
| Stage 3 — Confidence Gate | Designed, not implemented |
| Opportunity Horizon (`opportunity_engine.py`) | Designed, not implemented — builds on Stages 2-3 |
| Stage 4 — LLM Narrator, input contract (§10) | Not started (by design — waits on Stage 3) |
| Dashboard UI | **Built and running**, against static mock data — does **not** yet reflect the 4-observable rival estimator, the Opportunity Horizon layer, or the three Core Demo Scenarios (§11); that UI work hasn't started |
| Backend ↔ frontend bridge | Not built — planned as a small FastAPI service |
| Real telemetry source | **Resolved**: `telemetry_simulator.py` (synthetic, deterministic) for the hackathon build; FastF1/real data is a later calibration layer, not a dependency (§14) |

## 17. Immediate next task

**Judgment call, logged rather than silently applied**: a teammate
proposal framed the build as TIER 1 (gate → estimator → planner → a raw
recommendation, *without* the confidence gate) shipping before TIER 2
(confidence gate + Opportunity Horizon, framed as "differentiation").
This document deliberately keeps the build order below instead, with
Stage 3 immediately after Stage 2 and before Opportunity Horizon. Reason:
the statistical-honesty fix in §8 (replacing a gameable `t≥2.0` with a
real significance-plus-margin test) is not a nice-to-have layered on top
of a working recommender — it's the fix for the exact overconfidence bug
this project exists to correct. A build that has a "working" pipeline
capable of confidently recommending a mode before the honesty gate exists
risks exactly that ungated behavior making it into a demo or a screenshot
by accident. Keep Stage 3 where it already was in the sequence.

Build in this order, keeping `python -m pytest -q` fully green after every
step, not just at the end:

1. `requirements.txt`, `conftest.py` skeleton.
2. `telemetry_simulator.py` + tests — deterministic, seeded generator for
   `TelemetryInput`/`RivalObservation` sequences. Built first because
   everything below needs example telemetry, and because it's what the
   Core Demo Scenarios (§11) and every test fixture actually run against
   — not something to improvise ad hoc per test file.
3. Stage 1 (`rule_gate.py` + tests) — the 5-mode enum and corrected
   constants everything else depends on.
4. `rival_estimator.py` + tests — no dependency on Stage 2/3, only a
   one-way constants import from Stage 1, so it can be built and fully
   validated in isolation right after Stage 1 exists.
5. Stage 2 (`planner.py` + tests) — needs the rival estimator's output
   type, not its Stage 3 integration.
6. Stage 3 (`confidence_gate.py` + tests).
7. `test_integration.py` — needs all of the above wired together. This is
   the single-lap pipeline, complete, and it is a legitimate demo on its
   own — don't treat step 8 as a blocker for showing progress. This is
   also the point at which Core Demo Scenarios A and B (§11) become real,
   computed outputs for the first time.
8. `opportunity_engine.py` + tests (§9) — the multi-lap strategy layer.
   Depends on everything above being built and green; extends Stage 3's
   comparison rather than replacing it. This is what makes Core Demo
   Scenario C (§11) possible.
9. Stage 4 (`narrator.py` + §10's input contract) — not started, do not
   begin until Stage 3 (and ideally the Opportunity Horizon layer, since
   it changes what there is to narrate) are built and green. Resolve the
   `confidence`/`reason_codes` derivation (§10) as part of this step, not
   before — it needs the real gate output shape in hand.
10. Once the backend above is green: build the FastAPI bridge exposing it,
    and point `frontend/src/data/mockTelemetry.js`'s consumers at real
    `fetch` calls instead — the mock file then becomes a fallback/demo
    mode rather than the only mode. (Real telemetry source is resolved,
    §14/§16 — no decision left to make here, just implementation.)
11. UI work still needed, independent of backend progress and safe to do
    against richer mock data in the meantime: the 4-observable Rival
    Estimator display, a "Why NOW?" consolidated reasoning panel, the
    presentation-layer relabeling of the 5 modes (§4 — resolve the exact
    mapping first, it isn't settled), an interactive Attack-vs-Wait
    comparison, an Opportunity Horizon visualization, and rebuilding the
    demo-scenario toggle around the three named Core Demo Scenarios (§11).
    None of this is built yet either. **Explicitly lower priority than the
    above**: further 3D viewport/visual polish. The car and telemetry
    visuals are supporting context for the decision, not the product —
    put new UI effort into making the reasoning legible before making the
    car prettier.

## 18. Engineering principles — the fast filter

When a new feature idea comes up, in any teammate's pitch or your own,
check it against this list before speccing it in. In priority order:

1. Correctness over visual complexity.
2. Deterministic calculation over LLM reasoning.
3. Explicit uncertainty (`mean ± std`) over fake precision (a bare point
   value with no error bar).
4. Legal action filtering before optimization, always — never the reverse.
5. Reproducibility (seeded RNG, everywhere).
6. Testability (if you can't write a test for it, it isn't specified yet).
7. Explainability (Stage 4 should be able to narrate *why*, not just
   *what*).
8. Honest prototype-status labeling — "designed" and "built" are different
   words in this document on purpose; don't blur them in a pitch either.
9. Modular architecture (one file, one job — §15's layout is not
   incidental).
10. Demo reliability over feature count — three scenarios that always
    work (§11) beat ten that might not.

The one-line version, worth repeating whenever scope creep shows up:
**does the proposed feature improve energy modeling, legality, rival
uncertainty, opportunity cost, confidence/abstention, or explainability?
If not, it's probably not core.**
