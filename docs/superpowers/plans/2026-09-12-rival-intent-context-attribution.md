# Rival Intent + Cause Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the 4 missing intelligence fields to the ChronoPace decision snapshot: `cause_attribution` (tyre/energy/traffic_aero/other percentages in `ContextAttributionBlock`), `RivalIntentBlock` (intent + intent_confidence + response_capability + trap_probability + intent_evidence) — all derived from the existing particle-filter posterior and window trends without inventing new sensors.

**Architecture:** Both additions are pure computed derivations from already-available data (`RivalSocEstimate.state_probs`, `p_defend`, `WindowBlock` trends, `active_aero_mode`). They are wired as new helpers `_compute_cause_attribution()` and `_build_rival_intent()` in `app/decision/engine.py`, surfaced through `ContextAttributionBlock` (extended) and a new `RivalIntentBlock` (added as `rival_intent: Optional[RivalIntentBlock]` on `DecisionSnapshot`). All values are labeled MODEL_ASSUMPTION / MODELED — NOT MEASURED.

**Tech Stack:** Python 3.12+, pydantic v2, pytest, FastAPI TestClient. Zero new dependencies.

---

## Existing infrastructure (do NOT modify)

| Component | File | What it does |
|-----------|------|--------------|
| `RivalSocEstimate` | `rival_estimator.py:309` | Particle-filter posterior: `mean_soc_mj`, `std_soc_mj`, `state_probs` (low/medium/high), `evidence_quality`, `baseline_ready` |
| `p_defend` | `app/decision/engine.py:295` | Deterministic P(rival defends): SoC-fraction × gap-closing × uncertainty pull |
| `WindowBlock` | `app/decision/snapshot.py:120` | `rival_sector_delta_trend`, `rival_terminal_speed_trend`, `closing` |
| `ContextAttributionBlock` | `app/decision/snapshot.py:250` | `rival_tyre_compound`, `compound_baseline_active`, `observed_sector_delta_s`, `active_aero_mode`, `residual_evidence_confidence` |
| `_build_context_attribution()` | `app/decision/engine.py:668` | Builds `ContextAttributionBlock` from `nl` and `rival_soc_estimate` |
| `_assemble()` | `app/decision/engine.py:765` | Assembles `DecisionSnapshot`; calls counterfactual + context_attribution builders at line 934–938 |
| `DecisionSnapshot` | `app/decision/snapshot.py:335` | API contract; `context_attribution` is already wired in |

---

## File map

**Modified files:**
- `app/decision/snapshot.py` — add `cause_attribution` field to `ContextAttributionBlock`; add `RivalIntentBlock` class; add `rival_intent: Optional[RivalIntentBlock]` to `DecisionSnapshot`
- `app/decision/engine.py` — update `_build_context_attribution()` to compute `cause_attribution`; add `_build_rival_intent()` helper; wire it into `_assemble()`
- `tests/test_context_attribution.py` — add tests for `cause_attribution` field
- `README.md` — add Rival Intent, Response Capability, Trap Probability sections

**New files:**
- `tests/test_rival_intent.py` — tests for `rival_intent`, `response_capability`, `trap_probability`

---

## Task 1: Add `cause_attribution` to `ContextAttributionBlock`

**Files:**
- Modify: `app/decision/snapshot.py` (ContextAttributionBlock class, lines 250–289)
- Modify: `app/decision/engine.py` (_build_context_attribution function, lines 668–717)
- Modify: `tests/test_context_attribution.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_context_attribution.py`:

```python
def test_cause_attribution_present():
    """cause_attribution dict must be present in context_attribution block."""
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    assert ca is not None, "context_attribution is None"
    assert "cause_attribution" in ca, f"cause_attribution missing from context_attribution: {list(ca.keys())}"


def test_cause_attribution_sums_to_one():
    """cause_attribution values must sum to 1.0 (± floating point)."""
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is None or ca.get("cause_attribution") is None:
        return
    d = ca["cause_attribution"]
    total = sum(d.values())
    assert abs(total - 1.0) < 0.01, f"cause_attribution values should sum to 1.0, got {total}: {d}"


def test_cause_attribution_has_correct_keys():
    """cause_attribution must have tyre, energy, traffic_aero, other keys."""
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is None or ca.get("cause_attribution") is None:
        return
    d = ca["cause_attribution"]
    required_keys = {"tyre", "energy", "traffic_aero", "other"}
    assert required_keys == set(d.keys()), f"Expected keys {required_keys}, got {set(d.keys())}"


def test_cause_attribution_values_non_negative():
    """All cause_attribution values must be >= 0."""
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is None or ca.get("cause_attribution") is None:
        return
    d = ca["cause_attribution"]
    for k, v in d.items():
        assert v >= 0, f"cause_attribution[{k}] = {v} is negative"


def test_cause_attribution_all_scenarios():
    """cause_attribution must be present for all scenarios A–E."""
    for scenario in ("A", "B", "C", "D", "E"):
        snap = _snapshot(scenario)
        ca = snap.get("context_attribution")
        assert ca is not None, f"context_attribution is None for scenario {scenario}"
        assert "cause_attribution" in ca, f"cause_attribution missing for scenario {scenario}"
        d = ca["cause_attribution"]
        assert abs(sum(d.values()) - 1.0) < 0.01, f"Scenario {scenario}: values don't sum to 1.0: {d}"
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/test_context_attribution.py -k "cause_attribution" -v 2>&1 | tail -15
```

Expected: All 5 new tests fail (key missing from response).

- [ ] **Step 3: Add `cause_attribution` field to `ContextAttributionBlock` in `snapshot.py`**

Read `app/decision/snapshot.py` first. Find `ContextAttributionBlock` (around line 250). Add `cause_attribution` as the last field, before the closing of the class:

The current end of `ContextAttributionBlock` is:
```python
    residual_evidence_confidence: str = Field(
        "UNAVAILABLE",
        description="HIGH | MEDIUM | LOW | UNAVAILABLE — confidence that the residual "
        "(after context attribution) reflects rival energy management."
    )
```

Add immediately after:
```python
    cause_attribution: Optional[dict] = Field(
        None,
        description=(
            "Estimated evidence attribution for the rival's observed pace delta. "
            "{'tyre': float, 'energy': float, 'traffic_aero': float, 'other': float} — "
            "values sum to 1.0. "
            "Modelled proportions, NOT causal certainty. MODEL_ASSUMPTION."
        ),
    )
```

- [ ] **Step 4: Add the `_compute_cause_attribution` helper to `engine.py`**

Read `app/decision/engine.py`. Insert the new helper immediately before `_build_context_attribution` (currently at line 668):

```python
def _compute_cause_attribution(
    compound_baseline_active: bool,
    active_aero_mode: str,
    residual_evidence_confidence: str,
) -> dict:
    """
    Heuristic evidence attribution: estimated contribution of each contextual factor
    to the observed rival pace delta.

    MODEL_ASSUMPTION — not causal certainty. Modelled proportions of evidence weight:
    - tyre: tyre compound always contributes; more when compound-stratified baseline active
    - energy: scaled by how strong the residual energy evidence signal is
    - traffic_aero: elevated when in overtake-eligible (close-following) zone
    - other: remainder (weather, circuit baseline, driver behaviour, unattributed)

    Returns a dict summing to 1.0.
    """
    tyre = 0.35 + (0.15 if compound_baseline_active else 0.0)
    traffic_aero = 0.12 if active_aero_mode == "OVERTAKE_ELIGIBLE" else 0.05
    energy = {
        "HIGH": 0.35,
        "MEDIUM": 0.25,
        "LOW": 0.15,
        "UNAVAILABLE": 0.10,
    }.get(residual_evidence_confidence, 0.10)
    other = max(0.0, 1.0 - tyre - energy - traffic_aero)
    total = tyre + energy + traffic_aero + other
    return {
        "tyre": round(tyre / total, 3),
        "energy": round(energy / total, 3),
        "traffic_aero": round(traffic_aero / total, 3),
        "other": round(other / total, 3),
    }
```

- [ ] **Step 5: Update `_build_context_attribution` to include `cause_attribution`**

In `_build_context_attribution` (currently at line 668), the current `return ContextAttributionBlock(...)` at lines 708–715 is:

```python
        return ContextAttributionBlock(
            rival_tyre_compound=rival_compound,
            compound_baseline_active=compound_baseline_active,
            observed_sector_delta_s=nl.rival_sector_delta_s,
            context_explained_note=note,
            active_aero_mode=active_aero,
            residual_evidence_confidence=residual_confidence,
        )
```

Replace it with:

```python
        cause_attr = _compute_cause_attribution(
            compound_baseline_active=compound_baseline_active,
            active_aero_mode=active_aero,
            residual_evidence_confidence=residual_confidence,
        )
        return ContextAttributionBlock(
            rival_tyre_compound=rival_compound,
            compound_baseline_active=compound_baseline_active,
            observed_sector_delta_s=nl.rival_sector_delta_s,
            context_explained_note=note,
            active_aero_mode=active_aero,
            residual_evidence_confidence=residual_confidence,
            cause_attribution=cause_attr,
        )
```

- [ ] **Step 6: Run tests — must all pass**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/test_context_attribution.py -v 2>&1 | tail -20
```

Expected: All 12 tests pass (7 original + 5 new).

- [ ] **Step 7: Commit**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
git add app/decision/snapshot.py app/decision/engine.py tests/test_context_attribution.py
git commit -m "feat: add cause_attribution dict to ContextAttributionBlock (tyre/energy/traffic_aero/other)"
```

---

## Task 2: Add `RivalIntentBlock` to DecisionSnapshot

**Files:**
- Modify: `app/decision/snapshot.py` — new `RivalIntentBlock` class; add `rival_intent` field to `DecisionSnapshot`
- Modify: `app/decision/engine.py` — add `_build_rival_intent()` helper; wire into `_assemble()`
- Create: `tests/test_rival_intent.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rival_intent.py`:

```python
"""
Tests for RivalIntentBlock in DecisionSnapshot.

Covers:
- rival_intent key present in API response
- intent is one of the 5 valid values
- response_capability is one of 3 valid values
- trap_probability is 0-1
- intent_confidence is 0-1
- values derived from state, not hardcoded
- intent_evidence is a non-empty string
- all 5 scenarios produce valid rival_intent blocks
- MODELED / NOT MEASURED labeling in intent_evidence
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_INTENTS = {"DEPLOYING", "CONSERVING", "HARVESTING", "DEFENDING", "UNCERTAIN"}
VALID_CAPABILITIES = {"CAN_COUNTER", "CANNOT_COUNTER", "UNCERTAIN"}


def _snap(scenario: str = "B", seed: int = 42) -> dict:
    resp = client.post(
        "/api/v1/decision",
        json={"source": "synthetic", "scenario": scenario, "seed": seed},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_rival_intent_key_present():
    """rival_intent key must exist in the API response."""
    snap = _snap("B")
    assert "rival_intent" in snap, f"rival_intent key missing. Keys: {list(snap.keys())}"


def test_rival_intent_schema():
    """rival_intent block must have all required fields with correct types."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    assert ri is not None, "rival_intent is None — expected a block for synthetic scenario B"
    assert "intent" in ri
    assert "intent_confidence" in ri
    assert "response_capability" in ri
    assert "trap_probability" in ri
    assert "intent_evidence" in ri


def test_intent_is_valid_value():
    """intent must be one of the 5 canonical values."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    assert ri["intent"] in VALID_INTENTS, f"Invalid intent: {ri['intent']}"


def test_intent_confidence_in_range():
    """intent_confidence must be 0.0 – 1.0."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    conf = ri["intent_confidence"]
    assert 0.0 <= conf <= 1.0, f"intent_confidence out of range: {conf}"


def test_response_capability_is_valid():
    """response_capability must be CAN_COUNTER, CANNOT_COUNTER, or UNCERTAIN."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    assert ri["response_capability"] in VALID_CAPABILITIES, \
        f"Invalid response_capability: {ri['response_capability']}"


def test_trap_probability_in_range():
    """trap_probability must be 0.0 – 1.0."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    tp = ri["trap_probability"]
    assert 0.0 <= tp <= 1.0, f"trap_probability out of range: {tp}"


def test_intent_evidence_is_non_empty_string():
    """intent_evidence must be a non-empty string."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    ev = ri["intent_evidence"]
    assert isinstance(ev, str) and len(ev) > 10, f"intent_evidence too short: {ev!r}"


def test_intent_evidence_contains_modeled_label():
    """intent_evidence must say MODELED — NOT MEASURED."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    assert "MODELED" in ri["intent_evidence"].upper(), \
        f"intent_evidence missing 'MODELED' label: {ri['intent_evidence']}"


def test_all_scenarios_produce_rival_intent():
    """All 5 synthetic scenarios must produce a rival_intent block."""
    for scenario in ("A", "B", "C", "D", "E"):
        snap = _snap(scenario)
        ri = snap.get("rival_intent")
        assert ri is not None, f"rival_intent is None for scenario {scenario}"
        assert ri["intent"] in VALID_INTENTS, f"Scenario {scenario}: invalid intent {ri['intent']}"
        assert ri["response_capability"] in VALID_CAPABILITIES
        assert 0.0 <= ri["trap_probability"] <= 1.0
        assert 0.0 <= ri["intent_confidence"] <= 1.0


def test_scenario_c_low_soc_cannot_counter():
    """
    Scenario C has very low rival SoC (1.8 MJ).
    With high p_low, response_capability should be CANNOT_COUNTER or UNCERTAIN.
    """
    snap = _snap("C")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    assert ri["response_capability"] in ("CANNOT_COUNTER", "UNCERTAIN"), \
        f"Scenario C (low SoC) should give CANNOT_COUNTER or UNCERTAIN, got {ri['response_capability']}"


def test_rival_intent_deterministic():
    """Same scenario + seed must produce identical rival_intent."""
    r1 = _snap("B", seed=42)
    r2 = _snap("B", seed=42)
    ri1 = r1.get("rival_intent")
    ri2 = r2.get("rival_intent")
    assert ri1 is not None and ri2 is not None
    assert ri1["intent"] == ri2["intent"]
    assert ri1["intent_confidence"] == ri2["intent_confidence"]
    assert ri1["response_capability"] == ri2["response_capability"]
    assert ri1["trap_probability"] == ri2["trap_probability"]


def test_rival_intent_not_in_rival_block():
    """
    rival_intent should be a top-level key in the snapshot, NOT nested inside rival.
    The rival block already has p_defend; rival_intent is its own block.
    """
    snap = _snap("B")
    rival = snap.get("rival", {})
    assert "rival_intent" not in rival, "rival_intent should not be inside the rival block"
    assert "intent" not in rival, "intent should not be a field inside the rival block"
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/test_rival_intent.py -v 2>&1 | tail -20
```

Expected: All tests fail (`rival_intent` key missing from API response).

- [ ] **Step 3: Add `RivalIntentBlock` to `snapshot.py`**

Read `app/decision/snapshot.py`. Find `class CounterfactualBlock` (around line 222). Insert `RivalIntentBlock` **before** `CounterfactualBlock` (it's logically a rival-inference block, so place it after `RivalBlock` definitions but before `Counterfactual`):

```python
class RivalIntentBlock(BaseModel):
    """
    Inferred rival strategic intent and response capability.

    Derived from the particle-filter SoC posterior, p_defend, and observable
    performance trends. MODELED — NOT MEASURED. This is probabilistic inference,
    not a measurement or claim of access to private team strategy.

    Intent values:
      DEPLOYING  — rival appears to be actively spending electrical energy
      CONSERVING — rival appears to be managing/saving energy
      HARVESTING — rival appears to be reducing pace to recover energy
      DEFENDING  — rival posture suggests active defence (see p_defend)
      UNCERTAIN  — insufficient evidence to classify

    Response capability:
      CAN_COUNTER    — posterior SoC high enough to respond to an attack
      CANNOT_COUNTER — posterior SoC too low for a credible counter-deployment
      UNCERTAIN      — insufficient evidence
    """
    intent: str = Field(
        "UNCERTAIN",
        description="DEPLOYING | CONSERVING | HARVESTING | DEFENDING | UNCERTAIN",
    )
    intent_confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Confidence in the intent classification (0–1). MODELED.",
    )
    response_capability: str = Field(
        "UNCERTAIN",
        description="CAN_COUNTER | CANNOT_COUNTER | UNCERTAIN — can rival respond this lap?",
    )
    trap_probability: float = Field(
        0.0, ge=0.0, le=1.0,
        description=(
            "P(rival is intentionally conserving/appearing slow to bait a counter. "
            "High apparent opportunity ≠ guaranteed opportunity. "
            "MODEL_ASSUMPTION — probabilistic inference, NOT causal certainty."
        ),
    )
    intent_evidence: str = Field(
        "",
        description=(
            "Human-readable evidence trace for the intent inference. "
            "MODEL_ASSUMPTION — never an accuracy claim."
        ),
    )
```

Then in `class DecisionSnapshot(BaseModel)`, add after `counterfactual`:

```python
    rival_intent: Optional[RivalIntentBlock] = Field(
        None,
        description=(
            "Inferred rival strategic intent, response capability, and trap probability. "
            "MODELED — NOT MEASURED. None when rival data is unavailable."
        ),
    )
```

- [ ] **Step 4: Add `_build_rival_intent()` to `engine.py`**

Insert after `_build_counterfactual` (currently ending around line 762) and before `_assemble` (line 765):

```python
def _build_rival_intent(rival_soc_estimate, p_defend: float, window_block) -> Optional[RivalIntentBlock]:
    """
    Infer rival strategic intent from particle-filter posterior and window trends.

    MODELED — NOT MEASURED. All values are probabilistic inferences from observable
    kinematic features. No access to actual rival car systems or team strategy.
    """
    try:
        est = rival_soc_estimate
        mean_soc = est.mean_soc_mj
        state_probs = est.state_probs or {}
        p_low = state_probs.get("low", 0.0)
        p_high = state_probs.get("high", 0.0)

        sector_trend = getattr(window_block, "rival_sector_delta_trend", None)  # +ve = slowing
        closing = getattr(window_block, "closing", False)

        # Intent classification — ordered from most specific to least
        if p_defend > 0.55:
            intent = "DEFENDING"
            intent_conf = float(min(1.0, p_defend))
        elif sector_trend is not None and sector_trend < -0.08 and p_high > 0.35:
            # Getting faster + high-energy posterior → likely deploying
            intent = "DEPLOYING"
            intent_conf = float(min(1.0, p_high * 1.4))
        elif sector_trend is not None and sector_trend > 0.12 and mean_soc < 4.5 and p_low > 0.25:
            # Slowing + lower-energy posterior → conserving
            intent = "CONSERVING"
            intent_conf = float(min(1.0, p_low * 1.3 + 0.2))
        elif sector_trend is not None and sector_trend > 0.10 and mean_soc >= 4.5:
            # Slowing but has energy → harvesting (deliberate energy recovery)
            intent = "HARVESTING"
            intent_conf = 0.40
        else:
            intent = "UNCERTAIN"
            intent_conf = float(max(0.15, 0.5 - est.std_soc_mj * 0.1))

        # Response capability: posterior SoC high enough for a counter-deployment?
        if p_high > 0.50 or mean_soc > 5.5:
            response_capability = "CAN_COUNTER"
        elif p_low > 0.55 or mean_soc < 2.5:
            response_capability = "CANNOT_COUNTER"
        else:
            response_capability = "UNCERTAIN"

        # Trap probability: rival appearing slow/conserving but posterior SoC still high
        # → could be deliberately baiting our attack
        trap_prob = 0.0
        if intent in ("CONSERVING", "HARVESTING") and p_high > 0.25:
            trap_prob = float(min(0.75, p_high * 1.5))
        elif p_defend > 0.40 and mean_soc > 4.0:
            trap_prob = float(min(0.55, p_defend * 1.1))

        trend_str = "N/A" if sector_trend is None else f"{sector_trend:.3f} s/lap"
        evidence = (
            f"SoC posterior: mean={mean_soc:.2f} MJ, p_low={p_low:.2f}, p_high={p_high:.2f}; "
            f"sector_delta_trend={trend_str}; p_defend={p_defend:.2f}; closing={closing}. "
            "MODELED — NOT MEASURED."
        )

        return RivalIntentBlock(
            intent=intent,
            intent_confidence=round(intent_conf, 3),
            response_capability=response_capability,
            trap_probability=round(trap_prob, 3),
            intent_evidence=evidence,
        )
    except Exception:
        return None
```

Also add `RivalIntentBlock` to the import block at the top of `engine.py`. Find the existing import line that imports snapshot types:

```bash
grep -n "from app.decision.snapshot import" /Users/zain/TrackShift-26/ChronoPace-Backend/app/decision/engine.py | head -5
```

Add `RivalIntentBlock` to that import.

- [ ] **Step 5: Wire `_build_rival_intent()` into `_assemble()`**

In `_assemble()`, find the block after `counterfactual` and `context_attribution` are built (around line 934–938). Add immediately after `context_attribution`:

```python
    rival_intent_block = (
        _build_rival_intent(
            rival_soc_estimate=ctx.rival.estimate,
            p_defend=ctx.rival.p_defend,
            window_block=window_block,
        )
        if ctx.rival is not None
        else None
    )
```

Then in the `return DecisionSnapshot(...)` call, add the new field:

```python
        rival_intent=rival_intent_block,
```

(Add it after `context_attribution=context_attribution`.)

- [ ] **Step 6: Run rival intent tests**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/test_rival_intent.py -v --tb=short 2>&1 | tail -25
```

Expected: All 13 tests pass.

- [ ] **Step 7: Run full context attribution tests to ensure no regression**

```bash
python -m pytest tests/test_context_attribution.py tests/test_rival_intent.py -v 2>&1 | tail -15
```

Expected: All 25 tests pass.

- [ ] **Step 8: Commit**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
git add app/decision/snapshot.py app/decision/engine.py tests/test_rival_intent.py
git commit -m "feat: add RivalIntentBlock (intent, response_capability, trap_probability) to DecisionSnapshot"
```

---

## Task 3: Full Regression + API Verification

**Files:** Existing tests only. README update.

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/ -q --tb=short 2>&1 | tail -10
```

Expected: 380+ passed, 60 skipped, 0 failed.

If the `deterministic_dict()` golden snapshot tests fail:

```bash
python -m pytest tests/ -q --tb=short 2>&1 | grep FAILED | head -20
```

Golden snapshot tests may fail because `DecisionSnapshot` has new fields. These are stored in `tests/test_golden*.py` or similar. If they fail, update the expected snapshots:

```bash
# Find golden tests
grep -rn "deterministic_dict\|golden" /Users/zain/TrackShift-26/ChronoPace-Backend/tests/ --include="*.py" | head -10
```

The `DecisionSnapshot.deterministic_dict()` method blanks `generated_at` and `narrative`. It does NOT snapshot every field — so adding new optional fields (defaulting to `None`) should not break golden tests. But verify.

If any golden test breaks because `rival_intent` is now in the output, update those golden fixtures to include `rival_intent: null` (or the computed value for the scenario).

- [ ] **Step 2: Live server verification**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
uvicorn app.main:app --host 127.0.0.1 --port 8767 --log-level warning &
sleep 4

curl -s -X POST "http://127.0.0.1:8767/api/v1/decision" \
  -H "Content-Type: application/json" \
  -d '{"source":"synthetic","scenario":"B","seed":42}' | \
  python -c "
import json, sys
d = json.load(sys.stdin)
ri = d.get('rival_intent')
ca = d.get('context_attribution', {})
print('rival_intent:', ri['intent'] if ri else 'MISSING')
print('intent_confidence:', ri['intent_confidence'] if ri else 'MISSING')
print('response_capability:', ri['response_capability'] if ri else 'MISSING')
print('trap_probability:', ri['trap_probability'] if ri else 'MISSING')
print('cause_attribution:', ca.get('cause_attribution', 'MISSING'))
"

pkill -f "uvicorn app.main:app" || true
```

Expected output:
```
rival_intent: <one of DEPLOYING/CONSERVING/HARVESTING/DEFENDING/UNCERTAIN>
intent_confidence: <0.0–1.0>
response_capability: <CAN_COUNTER/CANNOT_COUNTER/UNCERTAIN>
trap_probability: <0.0–1.0>
cause_attribution: {'tyre': ..., 'energy': ..., 'traffic_aero': ..., 'other': ...}
```

- [ ] **Step 3: Verify schema endpoint**

```bash
curl -s http://127.0.0.1:8767/openapi.json | python -c "
import json, sys
spec = json.load(sys.stdin)
schemas = spec.get('components', {}).get('schemas', {})
print('RivalIntentBlock in schema:', 'RivalIntentBlock' in schemas)
print('rival_intent in DecisionSnapshot:', 'rival_intent' in schemas.get('DecisionSnapshot', {}).get('properties', {}))
print('cause_attribution in ContextAttributionBlock:', 'cause_attribution' in schemas.get('ContextAttributionBlock', {}).get('properties', {}))
" 2>/dev/null || echo "Run server first"
```

Expected:
```
RivalIntentBlock in schema: True
rival_intent in DecisionSnapshot: True
cause_attribution in ContextAttributionBlock: True
```

- [ ] **Step 4: Update README**

Read `README.md` first to find appropriate section. Add the following content under the "Architecture" or "Intelligence Layers" section, after the existing Context Attribution content:

```markdown
### Rival Intent Inference

ChronoPace infers the rival's **strategic intent** from the particle-filter SoC posterior, observable performance trends, and tactical posture. Possible intent states:

| Intent | Signal |
|--------|--------|
| `DEPLOYING` | Sector delta trending negative + high-energy posterior |
| `CONSERVING` | Sector delta trending positive + lower-energy posterior |
| `HARVESTING` | Sector delta trending positive + sufficient energy (deliberate recovery) |
| `DEFENDING` | P(defend) elevated — rival posture consistent with active defence |
| `UNCERTAIN` | Insufficient evidence to classify |

**Response Capability** answers: *can the rival deploy a counter-attack this lap?*
- `CAN_COUNTER` — posterior SoC high enough for credible counter-deployment
- `CANNOT_COUNTER` — posterior SoC too low
- `UNCERTAIN` — evidence insufficient

**Trap Probability** (`trap_probability: 0.0–1.0`): probability the rival is *intentionally* conserving/appearing slow to bait a counter-attack. High apparent opportunity ≠ guaranteed opportunity.

> **Important:** These are probabilistic inferences from observable kinematic signals. ChronoPace does not access rival car telemetry, team radio, or strategy systems. All values are **MODELED — NOT MEASURED**.

### Cause Attribution

For each lap's observed rival pace delta, `context_attribution.cause_attribution` breaks down the estimated evidence weight:

```json
{
  "tyre": 0.54,
  "energy": 0.31,
  "traffic_aero": 0.09,
  "other": 0.06
}
```

These are **modelled evidence proportions**, not causal percentages. Tyre context is normalised away before feeding the particle filter (via compound-stratified Z-score baseline).
```

- [ ] **Step 5: Final commit**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
git add README.md
git commit -m "docs: add rival intent, response capability, trap probability, cause attribution to README"
```

---

## Self-Review

### Spec coverage

| Spec Requirement | Task |
|-----------------|------|
| Phase 4: `cause_attribution` dict (tyre/energy/traffic_aero/other) | Task 1 |
| Phase 8: rival_intent (DEPLOYING/CONSERVING/HARVESTING/DEFENDING/UNCERTAIN) | Task 2 |
| Phase 8: response_capability (CAN_COUNTER/CANNOT_COUNTER/UNCERTAIN) | Task 2 |
| Phase 9: trap_probability float 0–1 with explanation | Task 2 |
| Phase 17: rival_intent structured block in DecisionSnapshot | Task 2 |
| Phase 22: tests for context attribution | Task 1 |
| Phase 22: tests for intent, response capability, trap probability | Task 2 |
| Phase 26: README updated with all 4 new features | Task 3 |

### Already implemented — NOT in this plan

The following were already complete before this plan:
- All 5 deployment modes (CONSERVE/BALANCED/ARM/USE/PUSH)
- Particle-filter rival energy estimator
- Context attribution (tyre compound + active aero mode)
- Opportunity horizon with per-strategy ranked array
- Regulatory gate (before Monte Carlo)
- Monte Carlo planner (10,000 rollouts, seeded)
- Confidence/significance gate
- Counterfactual block
- LLM narrator (snapshot-only, no decision authority)
- Validation framework + `GET /api/v1/validation/summary`
- WebSocket endpoint
- No DRS in 2026 production code
- 367 passing tests

### Placeholder scan

None — all steps include exact file paths, exact code, exact commands.

### Type consistency

- `RivalIntentBlock` defined in Task 2 Step 3 (snapshot.py)
- `_build_rival_intent()` returns `Optional[RivalIntentBlock]` ✓
- `_build_rival_intent()` imported in engine.py via existing snapshot import ✓
- `cause_attribution: Optional[dict]` — plain dict, no new type ✓
- `_compute_cause_attribution()` returns `dict` with keys tyre/energy/traffic_aero/other ✓
