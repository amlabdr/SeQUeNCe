"""Continuous fault scenarios for polarization-BSM monitoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import DatasetConfig, FAULT_LABELS


@dataclass(frozen=True)
class BSMScenario:
    episode_id: str
    fault_class: str
    seed: int
    onset_s: float
    ramp_duration_s: float
    target: float
    affected_source: str = ""
    affected_arm: str = ""
    affected_detector: str = ""

    def progress(self, time_s: float) -> float:
        if self.fault_class == "healthy" or time_s < self.onset_s:
            return 0.0
        if self.ramp_duration_s <= 0:
            return 1.0
        return float(np.clip((time_s - self.onset_s) / self.ramp_duration_s, 0.0, 1.0))

    def metadata(self) -> dict:
        return asdict(self)


def _signed(rng, low, high):
    return (-1.0 if rng.random() < 0.5 else 1.0) * float(rng.uniform(low, high))


def build_scenario(label: str, episode_index: int, cfg: DatasetConfig, seed: int) -> BSMScenario:
    if label not in FAULT_LABELS:
        raise ValueError(f"Unknown fault class {label!r}")
    rng = np.random.default_rng(seed)
    duration = cfg.generation.episode_duration_s
    if label == "healthy":
        onset = duration + 1.0
        ramp = 0.0
    else:
        onset = float(rng.uniform(
            cfg.generation.onset_min_fraction * duration,
            cfg.generation.onset_max_fraction * duration,
        ))
        ramp = float(rng.uniform(
            cfg.generation.ramp_duration_min_s,
            cfg.generation.ramp_duration_max_s,
        ))

    f = cfg.faults
    targets = {
        "healthy": lambda: 0.0,
        "temperature_change": lambda: _signed(rng, f.temperature_shift_c_min, f.temperature_shift_c_max),
        "polarization_drift": lambda: float(rng.uniform(f.polarization_twist_rad_per_m_min, f.polarization_twist_rad_per_m_max)),
        "spectral_detuning": lambda: _signed(rng, f.spectral_detuning_nm_min, f.spectral_detuning_nm_max),
        "raman_noise": lambda: float(rng.uniform(f.raman_extra_power_mw_min, f.raman_extra_power_mw_max)),
        "attenuation_loss": lambda: float(rng.uniform(f.attenuation_extra_db_per_m_min, f.attenuation_extra_db_per_m_max)),
        "source_brightness_loss": lambda: float(rng.uniform(f.source_brightness_factor_min, f.source_brightness_factor_max)),
        "sync_issue": lambda: float(rng.uniform(f.sync_jitter_ps_min, f.sync_jitter_ps_max)),
        "source_clock_drift": lambda: _signed(rng, f.source_clock_drift_ps_per_s_min, f.source_clock_drift_ps_per_s_max),
    }
    source_faults = {"spectral_detuning", "source_brightness_loss", "source_clock_drift"}
    arm_faults = {"temperature_change", "polarization_drift", "raman_noise", "attenuation_loss"}
    return BSMScenario(
        episode_id=f"{label}_{episode_index:05d}",
        fault_class=label,
        seed=seed,
        onset_s=onset,
        ramp_duration_s=ramp,
        target=targets[label](),
        affected_source=str(rng.choice(("A", "B"))) if label in source_faults else "",
        affected_arm=str(rng.choice(("A", "B"))) if label in arm_faults else "",
        affected_detector=str(rng.choice(("D3H", "D3V", "D4H", "D4V"))) if label == "sync_issue" else "",
    )
