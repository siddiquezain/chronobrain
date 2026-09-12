"""
The provenance catalogue must stay in sync with the real GateConfig values, and
must never silently promote a modelling assumption to VERIFIED_FIA.
"""

from dataclasses import fields

from rule_gate import GateConfig
from app.regulation import Provenance, REGULATORY_CONSTANTS


def test_catalogued_values_match_gate_config():
    gc = GateConfig()
    for name, entry in REGULATORY_CONSTANTS.items():
        if entry.gate_config_field is None:
            continue
        assert hasattr(gc, entry.gate_config_field), f"unknown GateConfig field {name}"
        assert getattr(gc, entry.gate_config_field) == entry.value, (
            f"provenance catalogue value for {name} ({entry.value}) != "
            f"GateConfig.{entry.gate_config_field} ({getattr(gc, entry.gate_config_field)})"
        )


def test_every_numeric_gate_constant_is_catalogued():
    gc = GateConfig()
    catalogued = {e.gate_config_field for e in REGULATORY_CONSTANTS.values()}
    for f in fields(gc):
        assert f.name in catalogued, f"GateConfig.{f.name} has no provenance entry"


def test_verified_constants_have_a_source():
    for name, entry in REGULATORY_CONSTANTS.items():
        if entry.provenance is Provenance.VERIFIED_FIA:
            assert entry.source, f"{name} is VERIFIED_FIA but cites no source"


def test_deployment_and_swing_caps_are_flagged_for_verification():
    """These two drive legality and are not literal published clauses — must be flagged."""
    assert REGULATORY_CONSTANTS["max_deployment_per_lap_mj"].needs_verification
    assert REGULATORY_CONSTANTS["max_delta_soc_mj"].needs_verification
