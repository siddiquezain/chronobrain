"""
recv_validation.py — Rival Energy Cross-Validation (RECV) harness.

IMPORTANT: RECV does NOT compare against actual FIA battery state-of-charge.
Public F1 telemetry does not expose actual battery SoC. ChronoPace therefore
validates hidden-state inference against independent energy reconstructions
(NormalizedLap.our_soc_mj from ReplayEnergyModel) and evaluates predictive
validity on held-out historical telemetry.

The reference is:
    B's NormalizedLap.our_soc_mj  (ChronoPace ReplayEnergyModel modeled state)

NOT:
    Actual FIA measured battery SoC (unavailable in public telemetry)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RecvCheckpoint:
    """Single comparison: A's estimate of B vs B's ChronoPace reference."""
    race: str
    session: str
    lap: int
    observer_driver: str
    rival_driver: str
    predicted_energy_mj: float
    predicted_std_mj: float
    reference_energy_mj: float
    absolute_error_mj: float
    signed_error_mj: float
    confidence: float
    evidence_quality: str
    posterior_health: str
    baseline_ready: bool
    n_observations: int


@dataclass
class RecvMetrics:
    """Aggregate metrics over a set of checkpoints."""
    n_checkpoints: int
    mae_mj: float
    rmse_mj: float
    bias_mj: float
    median_abs_error_mj: float
    p75_abs_error_mj: float
    p90_abs_error_mj: float
    coverage_68pct: float   # fraction where |error| < 1*std
    coverage_95pct: float   # fraction where |error| < 2*std


@dataclass
class RecvReport:
    """Full RECV report: aggregate + breakdowns."""
    label: str
    reference_description: str = (
        "ChronoPace ReplayEnergyModel modeled energy state — "
        "an independent reconstruction from throttle/brake telemetry, "
        "NOT actual FIA measured battery SoC (unavailable in public telemetry)."
    )
    n_checkpoints: int = 0
    mae_mj: float = float("nan")
    rmse_mj: float = float("nan")
    bias_mj: float = float("nan")
    median_abs_error_mj: float = float("nan")
    p75_abs_error_mj: float = float("nan")
    p90_abs_error_mj: float = float("nan")
    coverage_68pct: float = float("nan")
    coverage_95pct: float = float("nan")
    by_evidence_quality: Dict[str, RecvMetrics] = field(default_factory=dict)
    by_race: Dict[str, RecvMetrics] = field(default_factory=dict)
    checkpoints: List[RecvCheckpoint] = field(default_factory=list)


def _metrics(checkpoints: List[RecvCheckpoint]) -> RecvMetrics:
    if not checkpoints:
        return RecvMetrics(0, float("nan"), float("nan"), float("nan"),
                           float("nan"), float("nan"), float("nan"),
                           float("nan"), float("nan"))
    errs = np.array([c.absolute_error_mj for c in checkpoints])
    signed = np.array([c.signed_error_mj for c in checkpoints])
    stds = np.array([c.predicted_std_mj for c in checkpoints])
    abs_errs = np.abs(signed)
    coverage_68 = float(np.mean(abs_errs < stds))
    coverage_95 = float(np.mean(abs_errs < 2 * stds))
    return RecvMetrics(
        n_checkpoints=len(checkpoints),
        mae_mj=round(float(np.mean(errs)), 4),
        rmse_mj=round(float(np.sqrt(np.mean(signed**2))), 4),
        bias_mj=round(float(np.mean(signed)), 4),
        median_abs_error_mj=round(float(np.median(errs)), 4),
        p75_abs_error_mj=round(float(np.percentile(errs, 75)), 4),
        p90_abs_error_mj=round(float(np.percentile(errs, 90)), 4),
        coverage_68pct=round(coverage_68, 4),
        coverage_95pct=round(coverage_95, 4),
    )


def compute_recv_report(
    checkpoints: List[RecvCheckpoint],
    *,
    label: str = "recv",
) -> RecvReport:
    if not checkpoints:
        return RecvReport(label=label, n_checkpoints=0)
    overall = _metrics(checkpoints)
    by_eq: Dict[str, List[RecvCheckpoint]] = {}
    by_race: Dict[str, List[RecvCheckpoint]] = {}
    for c in checkpoints:
        by_eq.setdefault(c.evidence_quality, []).append(c)
        by_race.setdefault(c.race, []).append(c)
    return RecvReport(
        label=label,
        n_checkpoints=overall.n_checkpoints,
        mae_mj=overall.mae_mj,
        rmse_mj=overall.rmse_mj,
        bias_mj=overall.bias_mj,
        median_abs_error_mj=overall.median_abs_error_mj,
        p75_abs_error_mj=overall.p75_abs_error_mj,
        p90_abs_error_mj=overall.p90_abs_error_mj,
        coverage_68pct=overall.coverage_68pct,
        coverage_95pct=overall.coverage_95pct,
        by_evidence_quality={k: _metrics(v) for k, v in by_eq.items()},
        by_race={k: _metrics(v) for k, v in by_race.items()},
        checkpoints=checkpoints,
    )


def recv_baselines(
    checkpoints: List[RecvCheckpoint],
    prior_mean_mj: float = 4.5,
) -> Dict[str, RecvReport]:
    """
    Baseline RECV reports for comparison with the particle filter.
    naive_prior: always predict 4.5 MJ, std = 2.6 MJ (uniform prior std)
    persistence: repeat previous predicted value for all subsequent checkpoints
    """
    if not checkpoints:
        return {"naive_prior": RecvReport(label="naive_prior"),
                "persistence": RecvReport(label="persistence")}

    naive = []
    for c in checkpoints:
        err = abs(prior_mean_mj - c.reference_energy_mj)
        naive.append(RecvCheckpoint(
            race=c.race, session=c.session, lap=c.lap,
            observer_driver=c.observer_driver, rival_driver=c.rival_driver,
            predicted_energy_mj=prior_mean_mj,
            predicted_std_mj=2.6,
            reference_energy_mj=c.reference_energy_mj,
            absolute_error_mj=err,
            signed_error_mj=prior_mean_mj - c.reference_energy_mj,
            confidence=0.0, evidence_quality="none",
            posterior_health="insufficient_data", baseline_ready=False,
            n_observations=0,
        ))

    persist = []
    prev_pred = checkpoints[0].predicted_energy_mj
    for c in checkpoints:
        err = abs(prev_pred - c.reference_energy_mj)
        persist.append(RecvCheckpoint(
            race=c.race, session=c.session, lap=c.lap,
            observer_driver=c.observer_driver, rival_driver=c.rival_driver,
            predicted_energy_mj=prev_pred,
            predicted_std_mj=c.predicted_std_mj,
            reference_energy_mj=c.reference_energy_mj,
            absolute_error_mj=err,
            signed_error_mj=prev_pred - c.reference_energy_mj,
            confidence=c.confidence, evidence_quality="persistence",
            posterior_health="unknown", baseline_ready=c.baseline_ready,
            n_observations=c.n_observations,
        ))
        prev_pred = c.predicted_energy_mj

    return {
        "naive_prior": compute_recv_report(naive, label="naive_prior"),
        "persistence": compute_recv_report(persist, label="persistence"),
    }
