"""Configuration helpers for HOM-only AI diagnosis dataset generation."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FAULT_LABELS = [
    "healthy",
    "temperature_change",
    "polarization_drift",
    "spectral_detuning",
    "raman_noise",
    "attenuation_loss",
    "source_brightness_loss",
    "sync_issue",
    "source_clock_drift",
]


@dataclass(frozen=True)
class HistogramConfig:
    range_ps: int = 5_000
    bin_width_ps: int = 20
    coincidence_window_ps: int = 120


@dataclass(frozen=True)
class SetupConfig:
    setup_id: str = "hom_reference_40km"
    source_frequency_hz: float = 80_000_000.0
    mean_photon_num: float = 0.01
    source_bandwidth_nm: float = 0.04
    lambda_a_nm: float = 1550.0
    lambda_b_nm: float = 1550.0
    lambda_idler_a_nm: float = 1550.0
    lambda_idler_b_nm: float = 1550.0
    photon_statistics: str = "thermal"
    use_sparse_emission: bool = True
    bell_state_a: str = "psi-"
    bell_state_b: str = "psi-"
    arm_length_m_a: float = 20_000.0
    arm_length_m_b: float = 20_000.0
    herald_length_m_a: float = 1.0
    herald_length_m_b: float = 1.0
    attenuation_db_per_m: float = 0.0002
    baseline_raman_power_mw: float = 0.02
    detector_eff_hom: float = 1.0
    detector_jitter_ps: float = 0.0
    detector_dark_hz_hom: float = 0.0
    hom_match_window_ps: int = 1200
    emission_chunk_pulses: int = 25_000


@dataclass(frozen=True)
class GenerationConfig:
    dataset_name: str = "hom_ai_sample"
    dataset_size: int = 18
    episode_duration_s: float = 10.0
    window_duration_s: float = 1.0
    base_seed: int = 91_000
    workers: int = 1
    write_raw_streams: bool = False
    resume: bool = True

    @property
    def windows_per_episode(self) -> int:
        return max(1, int(round(self.episode_duration_s / self.window_duration_s)))


@dataclass(frozen=True)
class FaultRanges:
    temperature_shift_c_min: float = 8.0
    temperature_shift_c_max: float = 35.0
    polarization_twist_rad_per_m_min: float = 0.02
    polarization_twist_rad_per_m_max: float = 0.20
    polarization_bend_radius_m_min: float = 0.05
    polarization_bend_radius_m_max: float = 0.50
    spectral_detuning_nm_min: float = 0.05
    spectral_detuning_nm_max: float = 0.40
    raman_power_mw_min: float = 0.02
    raman_power_mw_max: float = 0.50
    attenuation_extra_db_per_m_min: float = 0.00002
    attenuation_extra_db_per_m_max: float = 0.00020
    source_brightness_factor_min: float = 0.20
    source_brightness_factor_max: float = 0.75
    sync_jitter_ps_min: float = 500.0
    sync_jitter_ps_max: float = 12_000.0
    source_clock_drift_ps_per_s_min: float = 50.0
    source_clock_drift_ps_per_s_max: float = 2_000.0


@dataclass(frozen=True)
class DatasetConfig:
    setup: SetupConfig
    generation: GenerationConfig
    histogram: HistogramConfig
    faults: FaultRanges
    slurm: SlurmConfig


@dataclass(frozen=True)
class SlurmConfig:
    host: str = "blackbird.nist.gov"
    target: str = "local"
    submit: bool = False
    sync_project: bool = True
    remote_repo: str = "auto"
    plink_profile: str = ""
    plink_user: str = ""
    plink_path: str = r"C:\Program Files\PuTTY\plink.exe"
    pscp_path: str = r"C:\Program Files\PuTTY\pscp.exe"
    output_dir: str = ""
    partition: str = "batch"
    time_limit: str = "08:00:00"
    nodes: int = 1
    cpus_per_node: int = 1
    shards_per_node: int = 1
    array_tasks: int = 1
    python_cmd: str = "/home/ana35/.venvs/sequence-hom/bin/python"
    module_load: str = "miniforge"
    poll_seconds: int = 60
    wait_for_completion: bool = False
    collect_outputs: bool = False
    clean_old_logs: bool = True


def default_output_root() -> Path:
    return Path(__file__).resolve().parent / "generated"


def _parse_value(value: str) -> Any:
    text = value.strip()
    lower = text.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _load_ini(path: Path) -> dict[str, Any]:
    parser = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=("#", ";"),
        empty_lines_in_values=False,
    )
    parser.optionxform = str
    parser.read(path, encoding="utf-8-sig")
    data: dict[str, Any] = {}
    for section in parser.sections():
        for key, value in parser.items(section):
            data[key] = _parse_value(value)
    return data


def _build_dataclass(cls, data: dict[str, Any]):
    valid = set(cls.__dataclass_fields__)
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_dataset_config(path: Path) -> DatasetConfig:
    data = _load_ini(path)
    known = (
        set(SetupConfig.__dataclass_fields__)
        | set(GenerationConfig.__dataclass_fields__)
        | set(HistogramConfig.__dataclass_fields__)
        | set(FaultRanges.__dataclass_fields__)
        | set(SlurmConfig.__dataclass_fields__)
    )
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {unknown}")
    return DatasetConfig(
        setup=_build_dataclass(SetupConfig, data),
        generation=_build_dataclass(GenerationConfig, data),
        histogram=_build_dataclass(HistogramConfig, data),
        faults=_build_dataclass(FaultRanges, data),
        slurm=_build_dataclass(SlurmConfig, data),
    )
