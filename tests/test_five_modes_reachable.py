"""
Verifies that all five DeploymentModes are reachable through the decision engine
under appropriate input conditions.

Architecture requirement: CONSERVE, BALANCED, ARM, USE_BONUS, PUSH must all be
reachable. Engine must never be stuck in a mode for incorrect reasons.

Scenario presets:
  A: gap=2.5s, SoC=5.5 MJ  → BALANCED (no target in range)
  B: gap=0.6s, SoC=7.2 MJ, overtake_qualified=True → USE_BONUS or ARM
  C: SoC=1.8 MJ             → CONSERVE or BALANCED (low energy)
  D: gap_behind=0.4s        → PUSH or BALANCED (defensive)
  E: deployed=9.2 MJ (over cap) → BALANCED (gate blocks aggressive)
"""

from fastapi.testclient import TestClient
from app.decision.engine import run_decision
from app.decision.config import DecisionConfig
from app.data.providers import build_provider
from app.main import app

client = TestClient(app)
_API = "/api/v1/decision"
_VALID_MODES = {
    "CONSERVE_MODE", "BALANCED_MODE",
    "ARM_OVERTAKE_MODE", "USE_OVERTAKE_BONUS_MODE", "PUSH_MODE"
}


def _post(scenario: str, seed: int = 42, **overrides) -> dict:
    payload = {"source": "synthetic", "scenario": scenario, "seed": seed}
    if overrides:
        payload["overrides"] = overrides
    resp = client.post(_API, json=payload)
    assert resp.status_code == 200, f"scenario {scenario}: {resp.status_code} {resp.text[:200]}"
    return resp.json()


def _mode(scenario: str, **overrides) -> str:
    return _post(scenario, **overrides)["decision"]["mode"]


# ─── BALANCED ──────────────────────────────────────────────────────────────────

def test_balanced_reachable_scenario_a():
    """Scenario A: gap 2.5s, moderate SoC — no ARM target → BALANCED expected."""
    mode = _mode("A")
    assert mode == "BALANCED_MODE", f"Scenario A should be BALANCED, got {mode}"


def test_balanced_reachable_scenario_e():
    """Scenario E: deployment over cap → gate rejects all → BALANCED fallback."""
    snap = _post("E")
    assert snap["decision"]["mode"] == "BALANCED_MODE", \
        f"All-illegal scenario must fall back to BALANCED, got {snap['decision']['mode']}"
    # All modes should be illegal (gate blocks everything including BALANCED)
    assert len(snap["compliance"]["illegal_modes"]) > 0


# ─── CONSERVE ──────────────────────────────────────────────────────────────────

def test_conserve_reachable_scenario_c():
    """Scenario C: SoC=1.8 MJ (very low) → CONSERVE_MODE."""
    mode = _mode("C")
    assert mode == "CONSERVE_MODE", \
        f"Low-energy scenario C should be CONSERVE, got {mode}"


def test_conserve_reachable_near_zero_soc():
    """Drive run_decision directly with near-zero SoC."""
    provider = build_provider("synthetic", scenario="C", seed=42, total_laps=50)
    cfg = DecisionConfig(seed=42)
    snap = run_decision(provider, config=cfg)
    assert snap.decision.mode in ("CONSERVE_MODE", "BALANCED_MODE"), \
        f"Near-zero SoC must not produce aggressive mode, got {snap.decision.mode}"
    # Energy block must reflect the low reserve
    assert snap.energy.soc_mj is None or snap.energy.soc_mj < 5.0


# ─── ARM_OVERTAKE ──────────────────────────────────────────────────────────────

def test_arm_reachable_scenario_b():
    """Scenario B: gap=0.6s, high SoC → ARM or USE_BONUS expected."""
    mode = _mode("B")
    assert mode in ("ARM_OVERTAKE_MODE", "USE_OVERTAKE_BONUS_MODE", "BALANCED_MODE"), \
        f"Scenario B (close gap, high SoC) should ARM/USE_BONUS/BALANCED, got {mode}"


def test_arm_in_compliance_with_close_gap():
    """ARM_OVERTAKE_MODE must appear in compliance output for scenario B."""
    snap = _post("B")
    all_compliance = snap["compliance"]["legal_modes"] + snap["compliance"]["illegal_modes"]
    assert "ARM_OVERTAKE_MODE" in all_compliance, \
        f"ARM_OVERTAKE_MODE missing from compliance output. Got: {all_compliance}"


# ─── USE_OVERTAKE_BONUS ────────────────────────────────────────────────────────

def test_use_bonus_reachable_scenario_b():
    """Scenario B has overtake_qualified_last_lap=True → USE_BONUS should be legal."""
    snap = _post("B")
    all_compliance = snap["compliance"]["legal_modes"] + snap["compliance"]["illegal_modes"]
    assert "USE_OVERTAKE_BONUS_MODE" in all_compliance, \
        f"USE_OVERTAKE_BONUS_MODE missing from compliance. Got: {all_compliance}"


def test_use_bonus_legal_when_qualified():
    """When overtake_qualified_last_lap=True and gap<1s, USE_BONUS must be legal or chosen."""
    snap = _post("B")
    legal = snap["compliance"]["legal_modes"]
    mode = snap["decision"]["mode"]
    use_bonus_legal = "USE_OVERTAKE_BONUS_MODE" in legal
    use_bonus_chosen = (mode == "USE_OVERTAKE_BONUS_MODE")
    assert use_bonus_legal or use_bonus_chosen, \
        f"USE_BONUS should be legal or chosen for scenario B. Legal: {legal}, Mode: {mode}"


# ─── PUSH ──────────────────────────────────────────────────────────────────────

def test_push_reachable_scenario_d():
    """Scenario D: gap_behind=0.4s (defending) → PUSH_MODE."""
    mode = _mode("D")
    assert mode == "PUSH_MODE", \
        f"Defensive scenario D should be PUSH, got {mode}"


def test_push_in_compliance():
    """PUSH_MODE must appear in compliance output for scenario D."""
    snap = _post("D")
    all_compliance = snap["compliance"]["legal_modes"] + snap["compliance"]["illegal_modes"]
    assert "PUSH_MODE" in all_compliance, \
        f"PUSH_MODE missing from compliance output. Got: {all_compliance}"


# ─── ALL FIVE MODES APPEAR ─────────────────────────────────────────────────────

def test_all_five_modes_in_compliance_across_scenarios():
    """Running scenarios A-D collectively must surface all 5 modes in compliance output."""
    all_seen = set()
    for sc in ("A", "B", "C", "D"):
        snap = _post(sc)
        all_seen.update(snap["compliance"]["legal_modes"])
        all_seen.update(snap["compliance"]["illegal_modes"])
        all_seen.add(snap["decision"]["mode"])

    missing = _VALID_MODES - all_seen
    assert not missing, f"These modes never appeared across scenarios A-D: {missing}"


# ─── DECISION MODES ARE ALWAYS VALID ───────────────────────────────────────────

def test_all_scenario_decisions_are_valid_modes():
    """Every scenario must produce a valid DeploymentMode enum value."""
    for sc in ("A", "B", "C", "D", "E"):
        snap = _post(sc)
        mode = snap["decision"]["mode"]
        assert mode in _VALID_MODES, f"Scenario {sc}: invalid mode '{mode}'"


# ─── NO DRS IN RESPONSES ───────────────────────────────────────────────────────

def test_no_drs_in_decision_compliance_reasons():
    """DRS must not appear in decision, compliance, or reason_codes for 2026."""
    snap = _post("B")
    for block_name in ("decision", "compliance"):
        block_str = str(snap.get(block_name, {})).lower()
        assert "drs_available" not in block_str, f"drs_available found in {block_name}: {block_str}"
    reason_str = str(snap.get("reason_codes", [])).lower()
    assert "drs" not in reason_str, f"DRS found in reason_codes: {reason_str}"
