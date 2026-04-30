"""Configuration helpers for the diagnosis dataset generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMPAIRMENT_LABELS = [
    "normal",
    "synchronization_issue",
    "polarization_drift",
    "raman_noise",
    "loss",
    "peak_shift_temperature",
]


@dataclass(frozen=True)
class HistogramConfig:
    """Coincidence-histogram extraction settings."""

    range_ps: int
    bin_width_ps: int
    coincidence_window_ps: int


@dataclass(frozen=True)
class GenerationPreset:
    """Top-level dataset-size preset."""

    name: str
    samples_per_label: int
    episode_duration_s: float
    window_duration_s: float
    source_frequency_hz: float
    visibility_pulses: int
    base_seed: int
    temperature_episode_span_s: float
    visibility_angles_deg: tuple[float, ...]
    visibility_measurement_stride_windows: int
    histogram: HistogramConfig

    @property
    def windows_per_sample(self) -> int:
        return int(round(self.episode_duration_s / self.window_duration_s))

    @property
    def histogram_pulses(self) -> int:
        return int(round(self.source_frequency_hz * self.window_duration_s))


def get_preset(mode: str) -> GenerationPreset:
    """Return a generation preset by name."""

    normalized = mode.lower().strip()
    if normalized == "sample":
        return GenerationPreset(
            name="sample",
            samples_per_label=2,
            episode_duration_s=20.0,
            window_duration_s=1.0,
            source_frequency_hz=10_000.0,
            visibility_pulses=6_000,
            base_seed=61_000,
            temperature_episode_span_s=86_400.0,
            visibility_angles_deg=(0.0, 22.5, 45.0, 67.5, 90.0),
            visibility_measurement_stride_windows=10,
            histogram=HistogramConfig(range_ps=600_000, bin_width_ps=50, coincidence_window_ps=2_000),
        )
    if normalized == "full":
        return GenerationPreset(
            name="full",
            samples_per_label=60,
            episode_duration_s=30.0,
            window_duration_s=1.0,
            source_frequency_hz=20_000.0,
            visibility_pulses=10_000,
            base_seed=161_000,
            temperature_episode_span_s=86_400.0,
            visibility_angles_deg=(0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0),
            visibility_measurement_stride_windows=5,
            histogram=HistogramConfig(range_ps=800_000, bin_width_ps=50, coincidence_window_ps=2_000),
        )
    raise ValueError(f"Unknown mode '{mode}'. Use 'sample' or 'full'.")


def default_output_root() -> Path:
    """Return the default output directory."""

    return Path(__file__).resolve().parent / "generated"
