"""Fault scenario generation for BSM-only HOM diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import FAULT_LABELS, DatasetConfig


@dataclass(frozen=True)
class WindowState:
    window_index: int
    fault_active: bool
    fault_progress: float
    true_temperature_shift_c: float = 0.0
    true_polarization_twist_rad_per_m: float = 0.0
    true_polarization_bend_radius_m: float = 0.0
    true_spectral_detuning_nm: float = 0.0
    true_raman_power_mw: float = 0.0
    true_extra_loss_db_per_m_a: float = 0.0
    true_extra_loss_db_per_m_b: float = 0.0
    true_source_brightness_factor_a: float = 1.0
    true_source_brightness_factor_b: float = 1.0
    true_sync_jitter_ps: float = 0.0
    true_source_clock_drift_ps_per_s: float = 0.0
    true_source_clock_offset_ps: float = 0.0
    faulty_source: str = ""
    faulty_arm: str = ""
    faulty_bsm_detector: str = ""


@dataclass(frozen=True)
class BSMDiagnosisScenario:
    episode_id: str
    setup_id: str
    fault_class: str
    seed: int
    fault_onset_window: int
    target_fault_level: float
    faulty_source: str
    faulty_arm: str
    faulty_bsm_detector: str
    windows: tuple[WindowState, ...]

    def to_metadata(self) -> dict:
        return asdict(self)


def _signed_target(rng: np.random.Generator, low: float, high: float) -> float:
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return sign * float(rng.uniform(low, high))


def build_scenario(label: str, episode_index: int, cfg: DatasetConfig, seed: int) -> BSMDiagnosisScenario:
    if label not in FAULT_LABELS:
        raise ValueError(f"Unsupported fault label {label!r}")

    rng = np.random.default_rng(int(seed))
    windows = cfg.generation.windows_per_episode
    if label == "healthy":
        onset = windows + 1
    else:
        onset = int(rng.integers(0, windows))

    source_faults = {"source_brightness_loss", "spectral_detuning", "source_clock_drift"}
    arm_faults = {"attenuation_loss", "polarization_drift", "raman_noise", "temperature_drift"}
    detector_faults = {"sync_issue"}
    faulty_source = str(rng.choice(["A", "B"])) if label in source_faults else ""
    faulty_arm = str(rng.choice(["A", "B"])) if label in arm_faults else ""
    faulty_bsm_detector = str(rng.choice(["BSM1", "BSM2"])) if label in detector_faults else ""
    target = 0.0
    states: list[WindowState] = []

    if label == "temperature_drift":
        target = _signed_target(
            rng,
            cfg.faults.temperature_shift_c_min,
            cfg.faults.temperature_shift_c_max,
        )
    elif label == "polarization_drift":
        target = float(rng.uniform(cfg.faults.polarization_twist_rad_per_m_min, cfg.faults.polarization_twist_rad_per_m_max))
    elif label == "spectral_detuning":
        target = _signed_target(rng, cfg.faults.spectral_detuning_nm_min, cfg.faults.spectral_detuning_nm_max)
    elif label == "raman_noise":
        target = float(rng.uniform(cfg.faults.raman_power_mw_min, cfg.faults.raman_power_mw_max))
    elif label == "attenuation_loss":
        target = float(rng.uniform(cfg.faults.attenuation_extra_db_per_m_min, cfg.faults.attenuation_extra_db_per_m_max))
    elif label == "source_brightness_loss":
        target = float(rng.uniform(cfg.faults.source_brightness_factor_min, cfg.faults.source_brightness_factor_max))
    elif label == "sync_issue":
        target = float(rng.uniform(cfg.faults.sync_jitter_ps_min, cfg.faults.sync_jitter_ps_max))
    elif label == "source_clock_drift":
        target = _signed_target(
            rng,
            cfg.faults.source_clock_drift_ps_per_s_min,
            cfg.faults.source_clock_drift_ps_per_s_max,
        )

    for window_index in range(windows):
        progress = 1.0 if window_index >= onset else 0.0
        active = progress > 0.0
        active_sim_s = max(0.0, (window_index - onset + 1) * cfg.generation.window_duration_s) if active else 0.0

        kwargs = {
            "window_index": window_index,
            "fault_active": active,
            "fault_progress": progress,
            "faulty_source": faulty_source,
            "faulty_arm": faulty_arm,
            "faulty_bsm_detector": faulty_bsm_detector if label == "sync_issue" else "",
        }

        if label == "temperature_drift":
            temp = target * progress
            kwargs["true_temperature_shift_c"] = temp
        elif label == "polarization_drift":
            kwargs["true_polarization_twist_rad_per_m"] = target * progress
            kwargs["true_polarization_bend_radius_m"] = (
                cfg.faults.polarization_bend_radius_m_max
                - progress * (cfg.faults.polarization_bend_radius_m_max - cfg.faults.polarization_bend_radius_m_min)
            )
        elif label == "spectral_detuning":
            kwargs["true_spectral_detuning_nm"] = target * progress
        elif label == "raman_noise":
            kwargs["true_raman_power_mw"] = target * progress
        elif label == "attenuation_loss":
            if faulty_arm == "A":
                kwargs["true_extra_loss_db_per_m_a"] = target * progress
            else:
                kwargs["true_extra_loss_db_per_m_b"] = target * progress
        elif label == "source_brightness_loss":
            factor = 1.0 - progress * (1.0 - target)
            if faulty_source == "A":
                kwargs["true_source_brightness_factor_a"] = factor
            else:
                kwargs["true_source_brightness_factor_b"] = factor
        elif label == "sync_issue":
            kwargs["true_sync_jitter_ps"] = target * progress
        elif label == "source_clock_drift":
            kwargs["faulty_source"] = faulty_source
            kwargs["true_source_clock_drift_ps_per_s"] = target * progress
            kwargs["true_source_clock_offset_ps"] = target * active_sim_s

        states.append(WindowState(**kwargs))

    return BSMDiagnosisScenario(
        episode_id=f"{label}_{episode_index:05d}",
        setup_id=cfg.setup.setup_id,
        fault_class=label,
        seed=int(seed),
        fault_onset_window=int(onset),
        target_fault_level=float(target),
        faulty_source=faulty_source,
        faulty_arm=faulty_arm,
        faulty_bsm_detector=faulty_bsm_detector,
        windows=tuple(states),
    )
