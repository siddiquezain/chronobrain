# ChronoPace — Regulatory Constants & Provenance

ChronoPace models the **2026 Formula 1** power-unit and Overtake-Mode rules. It was
built in 2026 from public F1 / FIA communications, **not** from the confidential
technical-regulation text. Every constant below is therefore tagged with where it
comes from and whether it still needs checking against the final published
regulations.

| Provenance | Meaning |
|---|---|
| `VERIFIED_FIA` | Published 2026 regulation or official F1 communication. |
| `MODEL_ASSUMPTION` | A ChronoPace modelling choice. May be grounded in public figures, but is not a hard rule *as applied here*. |
| `DEMO_CONSTANT` | Chosen only to make a demo scenario behave (none are load-bearing in the engine). |

The values live in **one place — `rule_gate.GateConfig`.** The v1 decision
pipeline reads them from there. The (deprecated) Stack B `Settings` object now
*defaults* its regulatory fields from `GateConfig()` rather than re-declaring the
numbers, so the two stacks can no longer silently diverge (an env var still
overrides for local experiments). The catalogue that classifies each constant is
`app/regulation/constants.py`, and `test_regulation_provenance.py` fails the build
if the catalogue drifts from `GateConfig` or an assumption is promoted to
`VERIFIED_FIA` without a source.

Inline code comments no longer cite specific FIA article numbers for the
`MODEL_ASSUMPTION` constants (they previously said "Art. 5.4.10 / 5.4.9"); the
comments now match the provenance catalogue.

## Constants

| `GateConfig` field | Value | Provenance | Needs verification | Notes |
|---|---|---|---|---|
| `max_ers_k_power_kw` | 350 kW | `VERIFIED_FIA` | no | 2026 MGU-K max electrical power (up from 120 kW). |
| `overtake_bonus_mj` | 0.5 MJ | `VERIFIED_FIA` | no | Extra deployable energy under Override / Overtake Mode when within 1.0 s of the car ahead. Officially stated as 0.5 MJ. |
| `overtake_detection_gap_threshold_s` | 1.0 s | `VERIFIED_FIA` | no | Proximity to become eligible for Overtake Mode (replaces the DRS 1.0 s detection rule). |
| `taper_overtake_full_power_end_kmh` | 337 km/h | `VERIFIED_FIA` | yes (taper shape) | Officially quoted speed to which full 350 kW is sustained under Overtake Mode; the taper curve below it is modelled. |
| `max_delta_soc_mj` | 4.0 MJ | `MODEL_ASSUMPTION` | yes | The **4 MJ figure is verified** (usable energy stored in the battery at any instant is capped at 4 MJ for 2026). Applying it as a per-lap **SoC-swing** limit in the gate is a ChronoPace interpretation, not a literal clause. |
| `max_deployment_per_lap_mj` | 9.0 MJ | `MODEL_ASSUMPTION` | yes | Public figures put deployable/recoverable energy at ~8–9 MJ/lap depending on circuit. 9.0 is the top-of-range value used as a single fixed cap; not a confirmed universal constant. |
| `recoverable_energy_baseline_mj` | 8.5 MJ | `MODEL_ASSUMPTION` | yes | Nominal per-lap harvest (~8.5 MJ, circuit-variable ~5–9 MJ). **Informational only — does not gate legality.** |
| `taper_normal_start_kmh` | 290 km/h | `MODEL_ASSUMPTION` | yes | Start of the normal-deployment power taper band. Engineered estimate of PU hardware behaviour. |
| `taper_normal_end_kmh` | 355 km/h | `MODEL_ASSUMPTION` | yes | End of the normal-deployment power taper band. Engineered estimate. |

## Engine policy constants (not regulation at all)

These live in `app/decision/config.py` (`DecisionConfig`) and are ChronoPace
decision-*policy*, flagged `MODEL_ASSUMPTION` in code comments:

- `horizon_decisive_margin_s = 0.15` — how much better a future window must look before the engine defers an attack (×3 for a prime window).
- `low_reserve_mj = 2.0`, `reserve_floor_mj = 1.0` — energy thresholds; a strategy may not deploy below the floor in the horizon.
- `rival_low_soc_mj = 3.0`, `rival_high_soc_mj = 6.0` — rival-state bucket edges.
- `window_laps = 5` — event-time sliding-window length.
- `overtake_reward_s = 0.3`, `future_window_bias_s = 0.12` — Future Energy Value: laptime-equivalent value of taking a window, and the bonus for waiting when the trend is IMPROVING.
- `data_quality_floor = 0.6` — below this `quality_score` the confidence gate abstains.
- `_DEFEND_GAP_S = 1.0` (engine.py) — rearward gap under which PUSH-to-defend is justified.
- `_HARVEST_PER_LAP = 0.35` (opportunity_engine.py), `harvest_per_lap_mj = 0.3` (actions.py) — modelled per-lap energy recovery for a non-attacking car.
- `_p_defend(...)` (engine.py) — deterministic scalar P(rival defends) from rival SoC bucket + gap trend + estimate uncertainty. A small statistical input to the planner, **not** a behavioural model.
- `ReplayEnergyModel` (normalizer.py) — turns a FastF1 lap's throttle/brake trace into a modelled SoC path, because **F1 publishes no ERS state of charge**. Every replayed lap carries `energy_is_modeled = True`.

## What still needs a human check

1. `max_deployment_per_lap_mj` — confirm the real 2026 per-lap electrical-deployment limit (and whether it is a single number or circuit-indexed).
2. `max_delta_soc_mj` — confirm whether the 4 MJ store cap is enforced as a swing limit or purely as a battery-capacity ceiling.
3. `taper_normal_start/end_kmh` and the Overtake-Mode taper shape — no published curve; currently engineered.

## Sources

- <https://www.formula1.com/en/latest/article/2026-regulations-explained-all-you-need-to-know-about-f1s-new-power-units.14jfv7a36905uDJDdNyfQd>
- <https://www.raceteq.com/articles/2026/05/f1s-2026-energy-system-explained>
- <https://www.espn.com/racing/f1/story/_/id/48090668/2026-f1-rules-whats-new-cars-how-changes-affect-racing>
- <https://f1chronicle.com/f1-overtake-mode-2026-explained/>
