"""Regulatory check result model."""

from typing import Optional

from pydantic import BaseModel


class RegulatoryCheck(BaseModel):
    legal: bool
    violations: list[str]
    constraints_checked: list[str]
    mode: str
