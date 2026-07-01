"""INI configuration for continuous polarization-BSM datasets."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


FAULT_LABELS = (
    "healthy",
    "temperature_change",
    "polarization_drift",
    "spectral_detuning",
    "raman_noise",
    "attenuation_loss",
    "source_brightness_loss",
    "sync_issue",
    "source_clock_drift",
)


@dataclass(frozen=True)
class GenerationConfig:
    dataset_name: str = "polarization_bsm_dataset"
    dataset_size: int = 100
    episode_duration_s: float = 60.0
    window_duration_s: float = 1.0
    fault_update_interval_s: float = 0.1
    onset_min_fraction: float = 0.15
    onset_max_fraction: float = 0.65
    ramp_duration_min_s: float = 1.0
    ramp_duration_max_s: float = 10.0
    base_seed: int = 120_000
    workers: int = 1
    resume: bool = True
    write_raw_streams: bool = False

    @property
    def windows_per_episode(self) -> int:
        return int(round(self.episode_duration_s / self.window_duration_s))


@dataclass(frozen=True)
class SetupConfig:
    setup_id: str = "polarization_bsm_40km"
    source_frequency_hz: float = 1_000_000.0
    mean_photon_num: float = 0.01
    source_bandwidth_nm: float = 0.1
    photon_statistics: str = "thermal"
    use_sparse_emission: bool = True
    emission_chunk_pulses: int = 25_000
    bell_state_a: str = "psi-"
    bell_state_b: str = "psi-"
    signal_wavelength_nm_a: float = 1550.0
    signal_wavelength_nm_b: float = 1550.0
    sink_wavelength_nm_a: float = 1550.0
    sink_wavelength_nm_b: float = 1550.0
    signal_length_m_a: float = 20_000.0
    signal_length_m_b: float = 20_000.0
    attenuation_db_per_m: float = 0.0002
    baseline_raman_power_mw: float = 0.02
    detector_efficiency: float = 0.95
    detector_jitter_ps: float = 20.0
    detector_dark_hz: float = 50.0
    matching_window_ps: int = 1200
    coincidence_window_ps: int = 120
    pbs_fidelity: float = 1.0
    pbs_mismeasure_prob: float = 0.0


@dataclass(frozen=True)
class FeatureConfig:
    histogram_range_ps: int = 5000
    histogram_bin_width_ps: int = 20


@dataclass(frozen=True)
class FaultConfig:
    temperature_shift_c_min: float = 2.0
    temperature_shift_c_max: float = 20.0
    polarization_twist_rad_per_m_min: float = 0.02
    polarization_twist_rad_per_m_max: float = 0.20
    polarization_bend_radius_m_min: float = 0.05
    polarization_bend_radius_m_max: float = 0.50
    spectral_detuning_nm_min: float = 0.01
    spectral_detuning_nm_max: float = 0.20
    raman_extra_power_mw_min: float = 0.02
    raman_extra_power_mw_max: float = 0.50
    attenuation_extra_db_per_m_min: float = 0.00002
    attenuation_extra_db_per_m_max: float = 0.00020
    source_brightness_factor_min: float = 0.20
    source_brightness_factor_max: float = 0.75
    sync_jitter_ps_min: float = 100.0
    sync_jitter_ps_max: float = 5000.0
    source_clock_drift_ps_per_s_min: float = 20.0
    source_clock_drift_ps_per_s_max: float = 1000.0


@dataclass(frozen=True)
class SlurmConfig:
    target: str = "local"
    host: str = "blackbird.nist.gov"
    submit: bool = False
    sync_project: bool = True
    remote_repo: str = "auto"
    plink_profile: str = "blackbird"
    plink_user: str = ""
    plink_path: str = r"C:\Program Files\PuTTY\plink.exe"
    pscp_path: str = r"C:\Program Files\PuTTY\pscp.exe"
    partition: str = "batch"
    time_limit: str = "08:00:00"
    nodes: int = 1
    cpus_per_node: int = 1
    python_cmd: str = "/home/ana35/.venvs/sequence-swapping/bin/python"
    module_load: str = "miniforge"
    wait_for_completion: bool = False
    collect_outputs: bool = False
    poll_seconds: int = 60


@dataclass(frozen=True)
class DatasetConfig:
    generation: GenerationConfig
    setup: SetupConfig
    features: FeatureConfig
    faults: FaultConfig
    slurm: SlurmConfig


def _value(text: str):
    value = text.strip()
    if value.lower() in {"true", "yes", "on"}:
        return True
    if value.lower() in {"false", "no", "off"}:
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_config(path: Path) -> DatasetConfig:
    parser = configparser.ConfigParser(
        interpolation=None, inline_comment_prefixes=(";", "#")
    )
    parser.read(path, encoding="utf-8-sig")
    values = {
        key: _value(value)
        for section in parser.sections()
        for key, value in parser.items(section)
    }

    classes = (GenerationConfig, SetupConfig, FeatureConfig, FaultConfig, SlurmConfig)
    known = set().union(*(set(cls.__dataclass_fields__) for cls in classes))
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")

    def build(cls):
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})

    config = DatasetConfig(*(build(cls) for cls in classes))
    if config.generation.windows_per_episode < 1:
        raise ValueError("episode_duration_s must contain at least one window")
    if not np_isclose_multiple(
        config.generation.episode_duration_s, config.generation.window_duration_s
    ):
        raise ValueError("episode_duration_s must be a multiple of window_duration_s")
    return config


def np_isclose_multiple(total: float, step: float) -> bool:
    if step <= 0:
        return False
    return abs(total / step - round(total / step)) < 1e-9
