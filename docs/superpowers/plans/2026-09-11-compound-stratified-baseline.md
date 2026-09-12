# Compound-Stratified Rival Observation Baseline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stratify `RivalObservationBaseline` by tyre compound so that Z-scores are computed within compound-matched laps, removing the ~15 km/h compound-driven speed variation that currently swamps the energy signal.

**Architecture:** Add `rival_compound: Optional[str]` to `NormalizedLap` and extract it from FastF1's `Compound` column. Thread the compound string through `condense_lap()` → engine → `RivalStateEstimator.update()` → `RivalObservationBaseline`. Inside the baseline, maintain a per-compound `_CompoundAccumulator` alongside the existing pooled stats; `z_score()` uses the compound-specific accumulator when it has ≥ `min_obs` observations, and falls back to the pooled baseline otherwise. `RivalObservation` (the leakage-boundary object) is unchanged — compound travels as a separate metadata argument.

**Tech Stack:** Python dataclasses, numpy, FastF1 `session.laps["Compound"]`, existing particle-filter infrastructure.

---

## File Map

| File | Change |
|------|--------|
| `app/data/samples.py` | Add `rival_compound: Optional[str] = None` to `NormalizedLap` |
| `app/data/fastf1_service.py` | Add `_compound_by_lap()` helper; pass compound to `condense_lap()` in both `load_replay()` and `_load_replay_dynamic()` |
| `app/data/normalizer.py` | Add `rival_compound: Optional[str] = None` param to `condense_lap()`; pass it to `NormalizedLap` |
| `rival_estimator.py` | Add `_CompoundAccumulator` dataclass; extend `RivalObservationBaseline`; extend `RivalStateEstimator.update()` |
| `app/decision/engine.py` | Pass `nl.rival_compound or "UNKNOWN"` to `est.update()` |
| `scripts/collect_and_train_rival_model.py` | Pass `nl.rival_compound or "UNKNOWN"` to `baseline.update()` and `baseline.z_score()` |
| `tests/test_compound_baseline.py` | New: unit tests for compound-stratified baseline |
| `tests/test_fastf1_normalizer.py` | Extend: verify `rival_compound` threads through `condense_lap()` |

---

### Task 1: Add `rival_compound` to `NormalizedLap` and thread through `condense_lap()`

**Files:**
- Modify: `app/data/samples.py` (around line 148 — after `rival_sector_delta_s`)
- Modify: `app/data/normalizer.py` (lines 147–253: `condense_lap` signature + `NormalizedLap` construction)
- Test: `tests/test_fastf1_normalizer.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_fastf1_normalizer.py`, add:

```python
def test_condense_lap_rival_compound_threaded():
    """rival_compound kwarg on condense_lap ends up on NormalizedLap."""
    from app.data.normalizer import condense_lap
    from app.data.samples import TelemetrySample

    sample = TelemetrySample(lap=1, speed_kmh=280.0, throttle=0.8, brake=0.1)
    nl = condense_lap(
        [sample], [],
        lap=1, total_laps=50, data_mode="REPLAY",
        prev_soc_mj=None,
        gap_to_car_ahead_s=None,
        rival_compound="SOFT",
    )
    assert nl.rival_compound == "SOFT"


def test_condense_lap_rival_compound_defaults_none():
    from app.data.normalizer import condense_lap
    from app.data.samples import TelemetrySample

    sample = TelemetrySample(lap=1, speed_kmh=280.0, throttle=0.8, brake=0.1)
    nl = condense_lap(
        [sample], [],
        lap=1, total_laps=50, data_mode="REPLAY",
        prev_soc_mj=None,
        gap_to_car_ahead_s=None,
    )
    assert nl.rival_compound is None
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_fastf1_normalizer.py::test_condense_lap_rival_compound_threaded tests/test_fastf1_normalizer.py::test_condense_lap_rival_compound_defaults_none -v
```

Expected: FAIL — `condense_lap` has no `rival_compound` parameter / `NormalizedLap` has no `rival_compound` field.

- [ ] **Step 3: Add `rival_compound` field to `NormalizedLap`**

In `app/data/samples.py`, after line 148 (`rival_sector_delta_s: Optional[float] = None`), add:

```python
    rival_compound: Optional[str] = Field(
        None,
        description="Rival's tyre compound on this lap (SOFT/MEDIUM/HARD/INTERMEDIATE/WET). "
        "None when unavailable (synthetic, pre-FastF1 extraction laps).",
    )
```

The full `NormalizedLap` rival-observable section now looks like:

```python
    # --- rival kinematic observables (feed rival_estimator particle filter) ---
    strategic_rival: Optional[StrategicRivalInfo] = Field(None, ...)
    rival_lap_status: LapStatus = Field("racing", ...)
    rival_terminal_speed_kmh: Optional[float] = Field(None, ge=0.0)
    rival_clipping_point_fraction: Optional[float] = Field(None, ge=0.0, le=1.0)
    rival_corner_exit_accel_g: Optional[float] = Field(None, ge=0.0)
    rival_sector_delta_s: Optional[float] = None
    rival_compound: Optional[str] = Field(
        None,
        description="Rival's tyre compound on this lap (SOFT/MEDIUM/HARD/INTERMEDIATE/WET). "
        "None when unavailable (synthetic, pre-FastF1 extraction laps).",
    )
```

- [ ] **Step 4: Add `rival_compound` param to `condense_lap()` and pass it through**

In `app/data/normalizer.py`, change the `condense_lap` signature (line 147):

```python
def condense_lap(
    our_samples: Sequence[TelemetrySample],
    rival_samples: Sequence[TelemetrySample],
    *,
    lap: int,
    total_laps: int,
    data_mode: DataMode,
    prev_soc_mj: Optional[float],
    gap_to_car_ahead_s: Optional[float],
    gap_to_car_behind_s: Optional[float] = None,
    energy_model: Optional[ReplayEnergyModel] = None,
    rival_sector_baseline_s: Optional[float] = None,
    throttle_baseline: Optional[float] = None,
    brake_baseline: Optional[float] = None,
    lap_status: LapStatus = "racing",
    rival_lap_status: LapStatus = "racing",
    strategic_rival: Optional[StrategicRivalInfo] = None,
    source_detail: str = "",
    rival_compound: Optional[str] = None,   # NEW
) -> NormalizedLap:
```

In the `return NormalizedLap(...)` block at the end of `condense_lap()` (around line 228), add `rival_compound=rival_compound` to the constructor:

```python
    return NormalizedLap(
        lap=lap,
        total_laps=total_laps,
        data_mode=data_mode,
        our_speed_kmh=round(our_speed, 3),
        our_soc_mj=end_soc,
        our_soc_capacity_mj=model.capacity_mj,
        our_lap_start_soc_mj=round(lap_start_soc, 4),
        our_lap_energy_deployed_mj=deployed,
        our_lap_energy_recovered_mj=recovered,
        our_lap_net_swing_mj=net_swing,
        our_mean_throttle=round(_clip01(mean_throttle), 4),
        our_mean_brake=round(_clip01(mean_brake), 4),
        gap_to_car_ahead_s=gap_to_car_ahead_s,
        gap_to_car_behind_s=gap_to_car_behind_s,
        position=position,
        sector=sector,
        drs_available=drs_open,
        lap_status=lap_status,
        rival_lap_status=rival_lap_status,
        strategic_rival=strategic_rival,
        rival_compound=rival_compound,      # NEW
        energy_is_modeled=True,
        raw_sample_count=len(our_samples),
        source_detail=source_detail,
        **rival_fields,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_fastf1_normalizer.py::test_condense_lap_rival_compound_threaded tests/test_fastf1_normalizer.py::test_condense_lap_rival_compound_defaults_none -v
```

Expected: PASS.

- [ ] **Step 6: Run full suite to check no regressions**

```bash
pytest --tb=short -q
```

Expected: same pass count as before (all green).

- [ ] **Step 7: Commit**

```bash
git add app/data/samples.py app/data/normalizer.py tests/test_fastf1_normalizer.py
git commit -m "feat: add rival_compound field to NormalizedLap and condense_lap"
```

---

### Task 2: Extract rival compound from FastF1 and populate `load_replay()` paths

**Files:**
- Modify: `app/data/fastf1_service.py`

Background: FastF1's `session.laps` DataFrame has a `Compound` column with values like `SOFT`, `MEDIUM`, `HARD`, `INTERMEDIATE`, `WET`. We add a `_compound_by_lap(session, driver)` helper — same shape as `_lap_status_by_lap()` — and pass the result to both `condense_lap()` call sites.

- [ ] **Step 1: Add `_compound_by_lap()` helper to `fastf1_service.py`**

After the `_lap_status_by_lap` function (after line 598), add:

```python
def _compound_by_lap(session, driver: str) -> Dict[int, str]:
    """Per-lap tyre compound string, e.g. 'SOFT', 'MEDIUM', 'HARD'. Empty dict when unavailable."""
    laps = session.laps.pick_drivers(driver) if hasattr(session.laps, "pick_drivers") else session.laps.pick_driver(driver)
    out: Dict[int, str] = {}
    for _, lap in laps.iterlaps():
        ln = int(lap["LapNumber"])
        c = lap.get("Compound")
        if c is not None and not _is_nan(c):
            out[ln] = str(c).upper()
    return out
```

- [ ] **Step 2: Use it in `load_replay()` (fixed-rival path)**

In `load_replay()`, after the existing line:
```python
rival_status = _lap_status_by_lap(ses, rival_driver)
```

Add:
```python
rival_compound_map = _compound_by_lap(ses, rival_driver)
```

Then in the `condense_lap(...)` call inside the `for ln in lap_numbers:` loop (around line 273), add the new kwarg:

```python
        nl = condense_lap(
            our_samples[ln],
            rival_samples.get(ln, []),
            lap=ln,
            total_laps=total_laps,
            data_mode="REPLAY",
            prev_soc_mj=prev_soc,
            gap_to_car_ahead_s=(None if we_lead else gap_usable),
            gap_to_car_behind_s=(gap_usable if we_lead else None),
            energy_model=model,
            rival_sector_baseline_s=rival_lap_times.get(ln),
            throttle_baseline=thr_base.get(ln, all_thr.get(ln)),
            brake_baseline=brk_base.get(ln, all_brk.get(ln)),
            lap_status=o_status,
            rival_lap_status=r_status,
            source_detail=src,
            rival_compound=rival_compound_map.get(ln),   # NEW
        )
```

- [ ] **Step 3: Use it in `_load_replay_dynamic()` (dynamic-rival path)**

In `_load_replay_dynamic()`, after:
```python
status_by_driver = {d: _lap_status_by_lap(ses, d) for d in all_drivers}
```

Add:
```python
compound_by_driver = {d: _compound_by_lap(ses, d) for d in all_drivers}
```

Then in the `condense_lap(...)` call inside the `for ln in our_laps:` loop (around line 471), add:

```python
        nl = condense_lap(
            our_samples[ln],
            rival_samples_n,
            lap=ln,
            total_laps=total_laps,
            data_mode="REPLAY",
            prev_soc_mj=prev_soc,
            gap_to_car_ahead_s=(None if we_lead else gap_usable),
            gap_to_car_behind_s=(gap_usable if we_lead else None),
            energy_model=model,
            rival_sector_baseline_s=laptime_baselines.get(rival_drv, {}).get(ln),
            throttle_baseline=thr_base.get(ln, all_thr.get(ln)),
            brake_baseline=brk_base.get(ln, all_brk.get(ln)),
            lap_status=o_status,
            rival_lap_status=r_status,
            strategic_rival=sri,
            source_detail=src,
            rival_compound=compound_by_driver.get(rival_drv, {}).get(ln),   # NEW
        )
```

- [ ] **Step 4: Run full suite**

```bash
pytest --tb=short -q
```

Expected: all green. (FastF1 tests use mocked sessions so compound extraction is exercised only in real-data paths; existing tests remain unaffected.)

- [ ] **Step 5: Commit**

```bash
git add app/data/fastf1_service.py
git commit -m "feat: extract rival tyre compound from FastF1 and thread to NormalizedLap"
```

---

### Task 3: Compound-stratified `RivalObservationBaseline`

**Files:**
- Modify: `rival_estimator.py`
- Create: `tests/test_compound_baseline.py`

This is the core change. `_CompoundAccumulator` is a private dataclass mirroring the 4-channel online statistics of `RivalObservationBaseline`. It lives at the top of `rival_estimator.py` (before `RivalObservationBaseline`). The existing baseline's pooled accumulators remain unchanged; a `_cpd: dict` maps compound string → `_CompoundAccumulator`. `update()` and `z_score()` gain a `compound: str = "UNKNOWN"` kwarg — fully backward compatible with every existing call site.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compound_baseline.py`:

```python
"""Tests for compound-stratified RivalObservationBaseline."""
import math
import pytest
from rival_estimator import RivalObservationBaseline


class _Obs:
    """Minimal stand-in for RivalObservation."""
    def __init__(self, speed=300.0, clip=0.7, accel=1.0, sector=0.0):
        self.terminal_speed_kmh = speed
        self.clipping_point_fraction = clip
        self.corner_exit_accel_g = accel
        self.sector_delta_s = sector


def _feed(baseline, obs, compound, n=5):
    """Feed n identical observations into the baseline."""
    for _ in range(n):
        baseline.update(obs, compound=compound)


def test_compound_fallback_to_pooled_when_insufficient():
    """When compound-specific count < min_obs, z_score falls back to pooled stats."""
    bl = RivalObservationBaseline(min_obs=5)
    obs_soft = _Obs(speed=310.0)
    obs_hard = _Obs(speed=295.0)

    # Feed 6 SOFT laps (qualifies for compound-specific stats)
    _feed(bl, obs_soft, "SOFT", 6)
    # Feed 3 HARD laps (below min_obs — should fall back to pooled)
    _feed(bl, obs_hard, "HARD", 3)

    # Z-score for HARD should use pooled baseline (not HARD-specific)
    z = bl.z_score(_Obs(speed=300.0), compound="HARD")
    assert isinstance(z, tuple) and len(z) == 4
    # Pooled mean ≈ (6*310 + 3*295)/9 ≈ 305; Z for 300 should be slightly negative
    assert z[0] < 0.0


def test_compound_specific_stats_used_when_ready():
    """When compound has >= min_obs, z_score uses compound-specific baseline."""
    bl = RivalObservationBaseline(min_obs=5)
    obs_soft = _Obs(speed=310.0)

    _feed(bl, obs_soft, "SOFT", 6)  # compound-specific ready

    # Z-score for exactly the mean speed on SOFT should be near zero
    z = bl.z_score(_Obs(speed=310.0), compound="SOFT")
    assert abs(z[0]) < 0.1


def test_different_compounds_produce_different_z_scores():
    """Laps on SOFT vs HARD produce different Z-scores (different reference distributions)."""
    bl = RivalObservationBaseline(min_obs=5)

    # SOFT: mean speed 320 kmh
    for _ in range(6):
        bl.update(_Obs(speed=320.0), compound="SOFT")

    # HARD: mean speed 300 kmh
    for _ in range(6):
        bl.update(_Obs(speed=300.0), compound="HARD")

    # Same observation: 310 kmh
    z_soft = bl.z_score(_Obs(speed=310.0), compound="SOFT")
    z_hard = bl.z_score(_Obs(speed=310.0), compound="HARD")

    # 310 is BELOW mean for SOFT → negative Z
    assert z_soft[0] < 0.0
    # 310 is ABOVE mean for HARD → positive Z
    assert z_hard[0] > 0.0


def test_unknown_compound_uses_pooled():
    """compound='UNKNOWN' always falls through to pooled baseline."""
    bl = RivalObservationBaseline(min_obs=5)
    obs = _Obs(speed=300.0)

    for _ in range(10):
        bl.update(obs, compound="UNKNOWN")

    # Z-score for the mean should be near zero regardless of key
    z = bl.z_score(_Obs(speed=300.0), compound="UNKNOWN")
    assert abs(z[0]) < 0.1


def test_no_compound_arg_backward_compatible():
    """Calling update/z_score without compound kwarg still works (defaults to UNKNOWN)."""
    bl = RivalObservationBaseline(min_obs=5)
    obs = _Obs(speed=300.0)

    for _ in range(6):
        bl.update(obs)  # no compound arg

    z = bl.z_score(obs)  # no compound arg
    assert isinstance(z, tuple) and len(z) == 4


def test_pooled_stats_unaffected_by_compound_path():
    """Pooled accumulator receives all observations regardless of compound."""
    bl = RivalObservationBaseline(min_obs=5)

    for _ in range(3):
        bl.update(_Obs(speed=300.0), compound="SOFT")
    for _ in range(3):
        bl.update(_Obs(speed=300.0), compound="HARD")

    assert bl.n == 6  # pooled n counts all laps
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_compound_baseline.py -v
```

Expected: FAIL — `update()` and `z_score()` don't accept `compound` kwarg yet.

- [ ] **Step 3: Add `_CompoundAccumulator` to `rival_estimator.py`**

In `rival_estimator.py`, after the `_norm_cdf` / import block (around line 30, before `RivalObservationBaseline`), add:

```python
@dataclass
class _CompoundAccumulator:
    """Online statistics for one tyre-compound stratum inside RivalObservationBaseline."""
    n: int = 0
    _sum_sp: float = 0.0;  _sum_sq_sp: float = 0.0
    _sum_cl: float = 0.0;  _sum_sq_cl: float = 0.0
    _sum_ac: float = 0.0;  _sum_sq_ac: float = 0.0
    _sum_se: float = 0.0;  _sum_sq_se: float = 0.0

    def add(self, sp: float, cl: float, ac: float, se: float) -> None:
        self.n += 1
        self._sum_sp += sp;  self._sum_sq_sp += sp * sp
        self._sum_cl += cl;  self._sum_sq_cl += cl * cl
        self._sum_ac += ac;  self._sum_sq_ac += ac * ac
        self._sum_se += se;  self._sum_sq_se += se * se

    def _stat(self, s: float, sq: float) -> tuple:
        if self.n < 2:
            return (s / max(1, self.n), 1.0)
        mean = s / self.n
        var = max(0.0, sq / self.n - mean * mean) * self.n / (self.n - 1)
        return (mean, max(math.sqrt(var), 1e-4))

    def z_scores(self, sp: float, cl: float, ac: float, se: float) -> tuple:
        ms, ss = self._stat(self._sum_sp, self._sum_sq_sp)
        mc, sc = self._stat(self._sum_cl, self._sum_sq_cl)
        ma, sa = self._stat(self._sum_ac, self._sum_sq_ac)
        me, se2 = self._stat(self._sum_se, self._sum_sq_se)
        return (
            (sp - ms) / ss,
            (cl - mc) / sc,
            (ac - ma) / sa,
            (se - me) / se2,
        )
```

Note: `_CompoundAccumulator` uses `dataclass` fields with default values. Because Python dataclasses require non-default fields before default ones, and all fields here have defaults, this works fine.

- [ ] **Step 4: Extend `RivalObservationBaseline`**

In `rival_estimator.py`, in the `RivalObservationBaseline` dataclass (around line 36–119), add one new field after all existing `_sum_*` fields:

```python
    _cpd: dict = field(default_factory=dict, init=False, repr=False)
```

Then change `update()` to accept and forward compound:

```python
    def update(self, obs, compound: str = "UNKNOWN") -> None:
        """Fold one observation into running stats. Call BEFORE z_score (causal)."""
        sp = float(obs.terminal_speed_kmh)
        cl = float(obs.clipping_point_fraction)
        ac = float(obs.corner_exit_accel_g)
        se = float(obs.sector_delta_s)

        # Update pooled accumulators (unchanged)
        self._n += 1
        self._sum_speed += sp; self._sum_sq_speed += sp * sp
        self._sum_clip += cl;  self._sum_sq_clip += cl * cl
        self._sum_accel += ac; self._sum_sq_accel += ac * ac
        self._sum_sector += se; self._sum_sq_sector += se * se

        # Update compound-specific accumulator
        key = compound.upper()
        if key not in self._cpd:
            self._cpd[key] = _CompoundAccumulator()
        self._cpd[key].add(sp, cl, ac, se)
```

Then change `z_score()` to prefer compound-specific stats when ready:

```python
    def z_score(self, obs, compound: str = "UNKNOWN") -> tuple:
        """Return (z_speed, z_clip, z_accel, z_sector). Only call when is_ready.

        Uses compound-specific baseline when that compound has >= min_obs observations.
        Falls back to the pooled (all-compound) baseline otherwise.
        """
        sp = float(obs.terminal_speed_kmh)
        cl = float(obs.clipping_point_fraction)
        ac = float(obs.corner_exit_accel_g)
        se = float(obs.sector_delta_s)

        key = compound.upper()
        acc = self._cpd.get(key)
        if acc is not None and acc.n >= self.min_obs:
            return acc.z_scores(sp, cl, ac, se)

        # Pooled fallback (existing behaviour)
        return (
            (sp - self.mean_speed) / self.std_speed,
            (cl - self.mean_clip) / self.std_clip,
            (ac - self.mean_accel) / self.std_accel,
            (se - self.mean_sector) / self.std_sector,
        )
```

- [ ] **Step 5: Extend `RivalStateEstimator.update()` to accept `compound`**

In `rival_estimator.py`, change the method signature at line 342:

```python
    def update(self, observation, compound: str = "UNKNOWN") -> None:
```

Inside `update()`, find the two calls to `self._baseline.update()` and `self._baseline.z_score()`:

```python
        # Update baseline BEFORE using it for Z-scoring (strictly causal)
        self._baseline.update(observation, compound=compound)
```

And:

```python
        if self._baseline.is_ready:
            # Z-score model: normalize against rival's own running distribution
            z_sp, z_cl, z_ac, z_se = self._baseline.z_score(observation, compound=compound)
```

The `else` branch (pre-baseline fallback, around line 410) does NOT use z_score — no change needed there.

- [ ] **Step 6: Run the new compound tests**

```bash
pytest tests/test_compound_baseline.py -v
```

Expected: all 6 PASS.

- [ ] **Step 7: Run full suite**

```bash
pytest --tb=short -q
```

Expected: all green. All existing tests call `update(obs)` and `z_score(obs)` without `compound` — they receive the default `"UNKNOWN"`, which pools everything under one key, identical to the old behaviour.

- [ ] **Step 8: Commit**

```bash
git add rival_estimator.py tests/test_compound_baseline.py
git commit -m "feat: compound-stratified RivalObservationBaseline with pooled fallback"
```

---

### Task 4: Thread compound through engine and training script

**Files:**
- Modify: `app/decision/engine.py`
- Modify: `scripts/collect_and_train_rival_model.py`

- [ ] **Step 1: Update `engine.py` — pass compound to `est.update()`**

In `app/decision/engine.py`, inside `_thread_state()`, find the block that calls `est.update(obs)` (around the `if obs is not None:` block). The full block currently is:

```python
        obs = to_rival_observation(nl)  # Extract 4 observables (or None)
        if obs is not None:
            est.predict()
            est.update(obs)
        elif est.observation_count > 0:
            # rival lap was pit/out/invalid — predict but do NOT update
            est.predict()
```

Change `est.update(obs)` to:

```python
        obs = to_rival_observation(nl)  # Extract 4 observables (or None)
        if obs is not None:
            est.predict()
            est.update(obs, compound=nl.rival_compound or "UNKNOWN")
        elif est.observation_count > 0:
            # rival lap was pit/out/invalid — predict but do NOT update
            est.predict()
```

No test file changes needed — the existing integration tests exercise this path; if compound is `None` (synthetic laps have no compound), it correctly defaults to `"UNKNOWN"` which uses pooled baseline — same as before.

- [ ] **Step 2: Verify existing integration tests still pass**

```bash
pytest tests/test_rival_estimate_fields.py tests/test_rival_ml_integration.py tests/test_rival_no_leakage.py -v
```

Expected: all green.

- [ ] **Step 3: Update `collect_and_train_rival_model.py`**

In `scripts/collect_and_train_rival_model.py`, inside `collect_features()`, find the two baseline calls:

```python
            obs = _make_obs_adapter(nl)
            baseline.update(obs)

            if not baseline.is_ready:
                continue  # warm-up; Z-scores unreliable before min_obs

            z_sp, z_cl, z_ac, z_se = baseline.z_score(obs)
```

Change to:

```python
            obs = _make_obs_adapter(nl)
            cpd = nl.rival_compound or "UNKNOWN"
            baseline.update(obs, compound=cpd)

            if not baseline.is_ready:
                continue  # warm-up; Z-scores unreliable before min_obs

            z_sp, z_cl, z_ac, z_se = baseline.z_score(obs, compound=cpd)
```

- [ ] **Step 4: Run full suite one final time**

```bash
pytest --tb=short -q
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add app/decision/engine.py scripts/collect_and_train_rival_model.py
git commit -m "feat: thread rival tyre compound through engine and training script to compound-stratified baseline"
```

---

## Self-Review

**Spec coverage:**
- ✅ `rival_compound` added to `NormalizedLap` (Task 1)
- ✅ FastF1 `Compound` column extracted for fixed-rival path (Task 2, Step 2)
- ✅ FastF1 `Compound` column extracted for dynamic-rival path (Task 2, Step 3)
- ✅ `_CompoundAccumulator` + stratified baseline in `RivalObservationBaseline` (Task 3)
- ✅ `RivalStateEstimator.update()` accepts compound (Task 3, Step 5)
- ✅ Engine passes compound to estimator (Task 4, Step 1)
- ✅ Training script uses compound (Task 4, Step 3)
- ✅ Pooled fallback when compound-specific count < min_obs (Task 3, Step 4)
- ✅ Backward compatible: all existing call sites omitting `compound` default to "UNKNOWN" pooled behaviour

**Placeholder scan:** None found.

**Type consistency:**
- `compound: str = "UNKNOWN"` is used consistently across `RivalObservationBaseline.update()`, `z_score()`, `RivalStateEstimator.update()`, engine, and training script.
- `_CompoundAccumulator.z_scores()` returns a plain `tuple` — same type as the pooled fallback path in `z_score()`.
- `nl.rival_compound` is `Optional[str]`; everywhere it's used, `nl.rival_compound or "UNKNOWN"` coerces `None` to the safe default.
