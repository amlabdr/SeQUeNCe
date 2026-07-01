"""INI configuration for full swapping batch validation."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

from .full_simulator import FullSwappingConfig


@dataclass(frozen=True)
class BatchConfig:
    experiment_name: str = "full_swapping_validation"
    runs: int = 10
    workers: int = 1
    resume: bool = True
    output_dir: str = ""


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
    python_cmd: str = "/home/ana35/.venvs/sequence-hom/bin/python"
    module_load: str = "miniforge"
    wait_for_completion: bool = False
    collect_outputs: bool = False
    poll_seconds: int = 60
    clean_old_logs: bool = True


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


def load_config(path: Path) -> tuple[FullSwappingConfig, BatchConfig, SlurmConfig]:
    parser = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=(";", "#"),
    )
    parser.read(path, encoding="utf-8-sig")
    sections = {
        name: {key: _value(value) for key, value in parser.items(name)}
        for name in parser.sections()
    }

    def build(cls, section):
        valid = set(cls.__dataclass_fields__)
        unknown = set(section) - valid
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
        return cls(**section)

    return (
        build(FullSwappingConfig, sections.get("simulation", {})),
        build(BatchConfig, sections.get("batch", {})),
        build(SlurmConfig, sections.get("slurm", {})),
    )
