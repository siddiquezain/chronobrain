"""
app.regulation — provenance catalogue for every regulatory constant ChronoPace uses.

The *values* live in `rule_gate.GateConfig` (Stack A, unchanged). This package
does not redefine them; it classifies each one so nothing gets presented to a
judge as an official FIA limit when it is actually a ChronoPace modelling choice.

    VERIFIED_FIA      published 2026 F1 regulation / official F1 communication
    MODEL_ASSUMPTION  a ChronoPace modelling choice, possibly grounded in public
                      figures but not a hard rule as applied here
    DEMO_CONSTANT     a value chosen only to make the demo scenarios behave

See docs/regulation.md for the human-readable table and sources.
`test_regulation_provenance.py` asserts this catalogue stays in sync with GateConfig.
"""

from app.regulation.constants import (
    Provenance,
    REGULATORY_CONSTANTS,
    RegConstant,
    provenance_of,
)

__all__ = ["Provenance", "REGULATORY_CONSTANTS", "RegConstant", "provenance_of"]
