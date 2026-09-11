"""
providers.py — sources that yield `NormalizedLap`s.

    TelemetryProvider              (ABC — .laps() -> list[NormalizedLap])
        |
        +-- SyntheticProvider      wraps root telemetry_simulator.TelemetrySimulator
        +-- ReplayProvider         a pre-normalized lap list (FastF1 or a fixture)

`build_provider(...)` is the one factory the API / CLI call. FastF1 is imported
lazily inside `fastf1_service`; `FASTF1_AVAILABLE` says whether that import worked
so callers can fail with a clear message instead of an ImportError deep in a stack.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional

import numpy as np

from app.data.samples import NormalizedLap
from telemetry_simulator import SCENARIO_PRESETS, TelemetrySimulator

try:  # pragma: no cover - trivial import guard
    import fastf1 as _fastf1  # noqa: F401

    FASTF1_AVAILABLE = True
except Exception:  # ImportError, or fastf1 present but broken
    FASTF1_AVAILABLE = False


class TelemetryProvider(abc.ABC):
    """A finite, replayable sequence of NormalizedLaps. Deterministic given its config."""

    data_mode: str = "SYNTHETIC"

    @abc.abstractmethod
    def laps(self) -> List[NormalizedLap]:
        """Return every lap, lap 1 first. Cheap to call repeatedly."""

    @abc.abstractmethod
    def describe(self) -> str:
        """Human-readable source label, e.g. 'synthetic scenario B, seed 42'."""


class SyntheticProvider(TelemetryProvider):
    """
    Wraps the deterministic root `TelemetrySimulator`. The simulator's SoC is the
    ground truth of its synthetic world, so `energy_is_modeled=False` here.

    `preset_overrides` lets the interactive demo change race inputs (our SoC, the
    HIDDEN rival SoC, gap, rival terminal speed, deployment, telemetry noise). The
    rival SoC override sets only the simulator's hidden state — the estimator still
    has to infer it from the generated telemetry.

    `rival_obs_dropout` drops the rival observation on a deterministic fraction of
    laps (freshness / data-quality demo). `ground_truth_rival_soc_by_lap` is
    populated by `laps()` and is DEMO-ONLY — the decision engine never sees it.
    """

    data_mode = "SYNTHETIC"

    def __init__(
        self,
        scenario: str = "B",
        seed: int = 42,
        total_laps: int = 50,
        preset_overrides: Optional[dict] = None,
        rival_obs_dropout: float = 0.0,
        telemetry_glitch: bool = False,
    ):
        if scenario not in SCENARIO_PRESETS:
            raise ValueError(
                f"Unknown scenario {scenario!r}; choose from {sorted(SCENARIO_PRESETS)}"
            )
        self.scenario = scenario
        self.seed = seed
        self.total_laps = total_laps
        self.preset_overrides = dict(preset_overrides or {})
        self.rival_obs_dropout = max(0.0, min(1.0, float(rival_obs_dropout)))
        # Deterministically corrupt the feed (missing laps + an out-of-range value)
        # so the Data Quality Gate degrades and the Confidence Gate abstains.
        self.telemetry_glitch = bool(telemetry_glitch)
        self.ground_truth_rival_soc_by_lap: Dict[int, float] = {}

    def laps(self) -> List[NormalizedLap]:
        sim = TelemetrySimulator(
            scenario=self.scenario, seed=self.seed, total_laps=self.total_laps,
            preset_overrides=self.preset_overrides,
        )
        seq = sim.generate_sequence(self.total_laps)
        self.ground_truth_rival_soc_by_lap = {
            i + 1: gt for i, gt in enumerate(sim.rival_soc_ground_truth)
        }
        drop_rng = np.random.default_rng(self.seed + 777)
        # telemetry_glitch: a fixed, deterministic corruption pattern — skip some
        # interior laps (missing lap numbers) and push several speeds out of range.
        # Laps {12, 18, 24, 30} are corrupted so a demo can target one of them.
        glitch_missing = {7, 11, 16, 21} if self.telemetry_glitch else set()
        glitch_bad_speed = {12, 18, 24, 30} if self.telemetry_glitch else set()
        out: List[NormalizedLap] = []
        for tel, obs in seq:
            if tel.lap_number in glitch_missing:
                continue
            drop = self.rival_obs_dropout > 0.0 and float(drop_rng.random()) < self.rival_obs_dropout
            speed = 940.0 if tel.lap_number in glitch_bad_speed else tel.speed_kmh
            out.append(
                NormalizedLap(
                    lap=tel.lap_number,
                    total_laps=tel.total_laps,
                    data_mode="SYNTHETIC",
                    our_speed_kmh=speed,
                    our_soc_mj=tel.current_soc_mj,
                    our_lap_start_soc_mj=tel.lap_start_soc_mj,
                    our_lap_energy_deployed_mj=tel.lap_energy_deployed_mj,
                    gap_to_car_ahead_s=tel.gap_to_car_ahead_s,
                    gap_to_car_behind_s=tel.gap_to_car_behind_s,
                    position=None,
                    sector=None,
                    drs_available=(
                        tel.gap_to_car_ahead_s is not None
                        and tel.gap_to_car_ahead_s < 1.0
                    ),
                    rival_terminal_speed_kmh=None if drop else obs.terminal_speed_kmh,
                    rival_clipping_point_fraction=None if drop else obs.clipping_point_fraction,
                    rival_corner_exit_accel_g=None if drop else obs.corner_exit_accel_g,
                    rival_sector_delta_s=None if drop else obs.sector_delta_s,
                    energy_is_modeled=False,
                    raw_sample_count=0,
                    source_detail=self.describe(),
                )
            )
        return out

    def describe(self) -> str:
        extra = ""
        if self.preset_overrides:
            extra = " + overrides(" + ",".join(sorted(self.preset_overrides)) + ")"
        return f"synthetic scenario {self.scenario}, seed {self.seed}, {self.total_laps} laps{extra}"


class ReplayProvider(TelemetryProvider):
    """
    A pre-normalized lap list — produced by `fastf1_service.load_replay()` or a
    test fixture. The engine cannot tell this apart from SyntheticProvider beyond
    `data_mode` and the `energy_is_modeled` flag on each lap.
    """

    data_mode = "REPLAY"

    def __init__(self, normalized_laps: List[NormalizedLap], label: str = "replay"):
        if not normalized_laps:
            raise ValueError("ReplayProvider needs at least one NormalizedLap")
        self._laps = list(normalized_laps)
        self._label = label

    def laps(self) -> List[NormalizedLap]:
        return list(self._laps)

    def describe(self) -> str:
        return f"{self._label} ({len(self._laps)} laps)"


def build_provider(
    source: str = "synthetic",
    *,
    scenario: str = "B",
    seed: int = 42,
    total_laps: int = 50,
    preset_overrides: Optional[dict] = None,
    rival_obs_dropout: float = 0.0,
    telemetry_glitch: bool = False,
    fastf1_kwargs: Optional[dict] = None,
) -> TelemetryProvider:
    """
    source='synthetic' -> SyntheticProvider (with optional preset_overrides)
    source='fastf1'    -> ReplayProvider over fastf1_service.load_replay(**fastf1_kwargs)

    The same downstream engine consumes whichever comes back.
    """
    if source == "synthetic":
        return SyntheticProvider(
            scenario=scenario, seed=seed, total_laps=total_laps,
            preset_overrides=preset_overrides, rival_obs_dropout=rival_obs_dropout,
            telemetry_glitch=telemetry_glitch,
        )

    if source == "fastf1":
        if not FASTF1_AVAILABLE:
            raise RuntimeError(
                "source='fastf1' requested but the `fastf1` package is not installed. "
                "Run `pip install fastf1`, or use source='synthetic'."
            )
        from app.data.fastf1_service import load_replay

        laps = load_replay(**(fastf1_kwargs or {}))
        return ReplayProvider(laps, label=(fastf1_kwargs or {}).get("label", "fastf1 replay"))

    raise ValueError(f"Unknown telemetry source {source!r}; use 'synthetic' or 'fastf1'.")
