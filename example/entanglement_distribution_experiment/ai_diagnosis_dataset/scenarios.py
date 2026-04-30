"""Scenario sampling for the diagnosis dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from config import IMPAIRMENT_LABELS, GenerationPreset


@dataclass(frozen=True)
class ArmStaticConfig:
    """Static arm properties for one receiver path."""

    length_m: float
    base_attenuation_db_per_m: float
    detector_efficiency: float
    detector_dark_count_hz: float
    source_bandwidth_nm: float
    classical_wavelength_nm: float


@dataclass(frozen=True)
class WindowState:
    """Hidden state for one monitoring window."""

    timing_offset_ps: int
    detector_jitter_ps: float
    analyzer_rotation_error_deg_a: float
    analyzer_rotation_error_deg_b: float
    arm_a_temperature_c: float
    arm_b_temperature_c: float
    arm_a_core_ellipticity: float
    arm_b_core_ellipticity: float
    arm_a_bend_radius_m: float
    arm_b_bend_radius_m: float
    arm_a_twist_rate_rad_per_m: float
    arm_b_twist_rate_rad_per_m: float
    arm_a_extra_attenuation_db_per_m: float
    arm_b_extra_attenuation_db_per_m: float
    arm_a_classical_power_mw: float
    arm_b_classical_power_mw: float
    synchronization_offset_ps: int
    synchronization_jitter_ps: float
    clock_skew_ppm: float


@dataclass(frozen=True)
class SequenceScenario:
    """One labeled time-series sample."""

    sample_id: str
    impairment_label: str
    seed: int
    source_frequency_hz: float
    mean_photon_num: float
    fault_onset_window: int
    trajectory_variant: str
    physical_window_spacing_s: float
    arm_a: ArmStaticConfig
    arm_b: ArmStaticConfig
    windows: tuple[WindowState, ...]

    def to_metadata(self) -> dict:
        """Return JSON-friendly metadata."""

        return asdict(self)


def _ramp(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [float(end)]
    return np.linspace(start, end, count).astype(float).tolist()


def _int_ramp(start: int, end: int, count: int) -> list[int]:
    if count <= 1:
        return [int(end)]
    return [int(round(x)) for x in np.linspace(start, end, count)]


def _fault_profile(
    rng: np.random.Generator,
    *,
    start: float,
    target: float,
    steps: int,
    min_value: float | None = None,
    max_value: float | None = None,
    noise_scale: float = 0.0,
    recovery_fraction_range: tuple[float, float] = (0.15, 0.45),
    oscillation_fraction_range: tuple[float, float] = (0.05, 0.18),
) -> list[float]:
    """Create a non-monotonic fault trajectory.

    The profile ramps from baseline into a degraded regime, then partially recovers
    and oscillates with bounded noise instead of remaining strictly monotonic.
    """

    if steps <= 1:
        value = float(target)
        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
        return [value]

    attack_steps = max(2, min(steps - 1, int(round(steps * rng.uniform(0.35, 0.6)))))
    hold_steps = max(0, steps - attack_steps)

    attack = np.linspace(start, target, attack_steps, dtype=float)

    degradation = target - start
    recovery_fraction = float(rng.uniform(*recovery_fraction_range))
    settle = target - degradation * recovery_fraction
    if hold_steps > 0:
        hold = np.linspace(target, settle, hold_steps, dtype=float)
        base = np.concatenate([attack, hold])
    else:
        base = attack

    amplitude = max(abs(degradation) * float(rng.uniform(*oscillation_fraction_range)), noise_scale * 2.0)
    cycles = float(rng.uniform(1.25, 3.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    t = np.linspace(0.0, 1.0, steps, dtype=float)
    oscillation = amplitude * np.sin(2.0 * np.pi * cycles * t + phase)

    walk = np.cumsum(rng.normal(0.0, noise_scale, size=steps)).astype(float)
    walk = walk - walk[0]
    walk = walk * np.linspace(0.0, 1.0, steps, dtype=float)

    values = base[:steps] + oscillation + walk
    if min_value is not None:
        values = np.maximum(values, min_value)
    if max_value is not None:
        values = np.minimum(values, max_value)
    return values.astype(float).tolist()


def _fault_profile_int(
    rng: np.random.Generator,
    *,
    start: int,
    target: int,
    steps: int,
    min_value: int | None = None,
    max_value: int | None = None,
    noise_scale: float = 0.0,
    recovery_fraction_range: tuple[float, float] = (0.15, 0.45),
    oscillation_fraction_range: tuple[float, float] = (0.05, 0.18),
) -> list[int]:
    values = _fault_profile(
        rng,
        start=float(start),
        target=float(target),
        steps=steps,
        min_value=None if min_value is None else float(min_value),
        max_value=None if max_value is None else float(max_value),
        noise_scale=noise_scale,
        recovery_fraction_range=recovery_fraction_range,
        oscillation_fraction_range=oscillation_fraction_range,
    )
    return [int(round(value)) for value in values]


def _window(
    timing_offset_ps: int = 0,
    detector_jitter_ps: float = 40.0,
    analyzer_rotation_error_deg_a: float = 0.0,
    analyzer_rotation_error_deg_b: float = 0.0,
    arm_a_temperature_c: float = 20.0,
    arm_b_temperature_c: float = 20.0,
    arm_a_core_ellipticity: float = 1.0,
    arm_b_core_ellipticity: float = 1.0,
    arm_a_bend_radius_m: float = 0.0,
    arm_b_bend_radius_m: float = 0.0,
    arm_a_twist_rate_rad_per_m: float = 0.0,
    arm_b_twist_rate_rad_per_m: float = 0.0,
    arm_a_extra_attenuation_db_per_m: float = 0.0,
    arm_b_extra_attenuation_db_per_m: float = 0.0,
    arm_a_classical_power_mw: float = 0.0,
    arm_b_classical_power_mw: float = 0.0,
    synchronization_offset_ps: int = 0,
    synchronization_jitter_ps: float = 0.0,
    clock_skew_ppm: float = 0.0,
) -> WindowState:
    return WindowState(
        timing_offset_ps=int(timing_offset_ps),
        detector_jitter_ps=float(detector_jitter_ps),
        analyzer_rotation_error_deg_a=float(analyzer_rotation_error_deg_a),
        analyzer_rotation_error_deg_b=float(analyzer_rotation_error_deg_b),
        arm_a_temperature_c=float(arm_a_temperature_c),
        arm_b_temperature_c=float(arm_b_temperature_c),
        arm_a_core_ellipticity=float(arm_a_core_ellipticity),
        arm_b_core_ellipticity=float(arm_b_core_ellipticity),
        arm_a_bend_radius_m=float(arm_a_bend_radius_m),
        arm_b_bend_radius_m=float(arm_b_bend_radius_m),
        arm_a_twist_rate_rad_per_m=float(arm_a_twist_rate_rad_per_m),
        arm_b_twist_rate_rad_per_m=float(arm_b_twist_rate_rad_per_m),
        arm_a_extra_attenuation_db_per_m=float(arm_a_extra_attenuation_db_per_m),
        arm_b_extra_attenuation_db_per_m=float(arm_b_extra_attenuation_db_per_m),
        arm_a_classical_power_mw=float(arm_a_classical_power_mw),
        arm_b_classical_power_mw=float(arm_b_classical_power_mw),
        synchronization_offset_ps=int(synchronization_offset_ps),
        synchronization_jitter_ps=float(synchronization_jitter_ps),
        clock_skew_ppm=float(clock_skew_ppm),
    )


def _window_with_updates(base: WindowState, **changes) -> WindowState:
    data = asdict(base)
    data.update(changes)
    return WindowState(**data)


def _base_arm(rng: np.random.Generator) -> ArmStaticConfig:
    return ArmStaticConfig(
        length_m=float(rng.uniform(8_000, 22_000)),
        base_attenuation_db_per_m=float(rng.uniform(0.00018, 0.00024)),
        detector_efficiency=float(rng.uniform(0.90, 0.98)),
        detector_dark_count_hz=float(rng.uniform(5.0, 25.0)),
        source_bandwidth_nm=float(rng.uniform(0.2, 0.6)),
        classical_wavelength_nm=float(rng.choice([1270.0, 1310.0, 1490.0])),
    )


def _make_normal_windows(rng: np.random.Generator, count: int) -> list[WindowState]:
    return [
        _window(
            timing_offset_ps=int(rng.integers(-150, 151)),
            detector_jitter_ps=float(rng.uniform(25.0, 60.0)),
            analyzer_rotation_error_deg_a=0.0,
            analyzer_rotation_error_deg_b=0.0,
            arm_a_temperature_c=float(rng.uniform(19.9, 20.1)),
            arm_b_temperature_c=float(rng.uniform(19.9, 20.1)),
            arm_a_core_ellipticity=1.0,
            arm_b_core_ellipticity=1.0,
            arm_a_bend_radius_m=0.0,
            arm_b_bend_radius_m=0.0,
            arm_a_twist_rate_rad_per_m=0.0,
            arm_b_twist_rate_rad_per_m=0.0,
            arm_a_extra_attenuation_db_per_m=float(rng.uniform(0.0, 0.00001)),
            arm_b_extra_attenuation_db_per_m=float(rng.uniform(0.0, 0.00001)),
            arm_a_classical_power_mw=0.0,
            arm_b_classical_power_mw=0.0,
            synchronization_offset_ps=int(rng.integers(-100, 101)),
            synchronization_jitter_ps=float(rng.uniform(0.0, 40.0)),
            clock_skew_ppm=float(rng.uniform(-0.0002, 0.0002)),
        )
        for _ in range(count)
    ]


def _combine_windows(normal_prefix: list[WindowState], fault_suffix: list[WindowState]) -> tuple[WindowState, ...]:
    return tuple(normal_prefix + fault_suffix)


def _standard_physical_window_spacing_s(preset: GenerationPreset) -> float:
    return float(preset.window_duration_s)


def _temperature_physical_window_spacing_s(preset: GenerationPreset) -> float:
    if preset.windows_per_sample <= 1:
        return float(preset.temperature_episode_span_s)
    return float(preset.temperature_episode_span_s / (preset.windows_per_sample - 1))


TRAJECTORY_VARIANTS = {
    "synchronization_issue": ("progressive", "recovering", "oscillatory", "intermittent"),
    "polarization_drift": ("progressive", "recovering", "oscillatory", "intermittent"),
    "raman_noise": ("progressive", "recovering", "oscillatory", "intermittent"),
    "loss": ("progressive", "recovering", "oscillatory", "intermittent"),
    "peak_shift_temperature": ("progressive", "recovering", "oscillatory", "intermittent"),
}


def _trajectory_variant(label: str, item_index: int) -> str:
    variants = TRAJECTORY_VARIANTS.get(label, ("progressive",))
    return str(variants[item_index % len(variants)])


def _recovery_target(start: float, target: float, fraction: float) -> float:
    return float(target + (start - target) * fraction)


def _profile_values(
    rng: np.random.Generator,
    *,
    variant: str,
    start: float,
    target: float,
    steps: int,
    min_value: float | None = None,
    max_value: float | None = None,
    noise_scale: float = 0.0,
) -> list[float]:
    if variant == "progressive":
        return _fault_profile(
            rng,
            start=start,
            target=target,
            steps=steps,
            min_value=min_value,
            max_value=max_value,
            noise_scale=noise_scale,
            recovery_fraction_range=(0.05, 0.20),
            oscillation_fraction_range=(0.03, 0.10),
        )
    if variant == "recovering":
        values = _fault_profile(
            rng,
            start=start,
            target=target,
            steps=steps,
            min_value=min_value,
            max_value=max_value,
            noise_scale=noise_scale,
            recovery_fraction_range=(0.45, 0.80),
            oscillation_fraction_range=(0.02, 0.08),
        )
        return values
    if variant == "oscillatory":
        return _fault_profile(
            rng,
            start=start,
            target=target,
            steps=steps,
            min_value=min_value,
            max_value=max_value,
            noise_scale=max(noise_scale, abs(target - start) * 0.02),
            recovery_fraction_range=(0.15, 0.45),
            oscillation_fraction_range=(0.18, 0.35),
        )
    if variant == "intermittent":
        if steps <= 1:
            return [float(target)]
        values: list[float] = []
        low = float(start)
        high = float(target)
        degraded = False
        remaining = steps
        current = low
        while remaining > 0:
            segment = min(remaining, int(rng.integers(2, 5)))
            next_level = high if not degraded else _recovery_target(low, high, float(rng.uniform(0.65, 0.9)))
            seg = np.linspace(current, next_level, segment, dtype=float).tolist()
            values.extend(seg)
            current = next_level
            degraded = not degraded
            remaining -= segment
        arr = np.asarray(values[:steps], dtype=float)
        arr = arr + rng.normal(0.0, noise_scale, size=arr.shape[0])
        if min_value is not None:
            arr = np.maximum(arr, min_value)
        if max_value is not None:
            arr = np.minimum(arr, max_value)
        return arr.astype(float).tolist()
    raise ValueError(f"Unknown trajectory variant '{variant}'")


def _profile_values_int(
    rng: np.random.Generator,
    *,
    variant: str,
    start: int,
    target: int,
    steps: int,
    min_value: int | None = None,
    max_value: int | None = None,
    noise_scale: float = 0.0,
) -> list[int]:
    values = _profile_values(
        rng,
        variant=variant,
        start=float(start),
        target=float(target),
        steps=steps,
        min_value=None if min_value is None else float(min_value),
        max_value=None if max_value is None else float(max_value),
        noise_scale=noise_scale,
    )
    return [int(round(value)) for value in values]


def _common_scenario(sample_id: str, label: str, seed: int, preset: GenerationPreset) -> tuple[np.random.Generator, int, ArmStaticConfig, ArmStaticConfig]:
    rng = np.random.default_rng(seed)
    onset = max(3, preset.windows_per_sample // 3)
    return rng, onset, _base_arm(rng), _base_arm(rng)


def _make_normal(sample_id: str, seed: int, preset: GenerationPreset) -> SequenceScenario:
    rng, onset, arm_a, arm_b = _common_scenario(sample_id, "normal", seed, preset)
    windows = tuple(_make_normal_windows(rng, preset.windows_per_sample))
    return SequenceScenario(
        sample_id=sample_id,
        impairment_label="normal",
        seed=seed,
        source_frequency_hz=preset.source_frequency_hz,
        mean_photon_num=float(rng.uniform(0.04, 0.10)),
        fault_onset_window=onset,
        trajectory_variant="healthy_baseline",
        physical_window_spacing_s=_standard_physical_window_spacing_s(preset),
        arm_a=arm_a,
        arm_b=arm_b,
        windows=windows,
    )


def _make_synchronization_issue(sample_id: str, seed: int, preset: GenerationPreset) -> SequenceScenario:
    rng, onset, arm_a, arm_b = _common_scenario(sample_id, "synchronization_issue", seed, preset)
    variant = _trajectory_variant("synchronization_issue", int(sample_id.rsplit("_", 1)[1]))
    baseline = _make_normal_windows(rng, preset.windows_per_sample)
    prefix = baseline[:onset]
    steps = preset.windows_per_sample - onset
    final_offset = int(rng.integers(10_000, 80_000)) * (-1 if rng.random() < 0.5 else 1)
    offsets = _profile_values_int(
        rng,
        variant=variant,
        start=int(prefix[-1].synchronization_offset_ps if prefix else 0),
        target=final_offset,
        steps=steps,
        min_value=min(-90_000, final_offset),
        max_value=max(90_000, final_offset),
        noise_scale=250.0,
    )
    sync_jitters = _profile_values(
        rng,
        variant=variant,
        start=float(rng.uniform(150.0, 500.0)),
        target=float(rng.uniform(2_000.0, 12_000.0)),
        steps=steps,
        min_value=50.0,
        max_value=15_000.0,
        noise_scale=40.0,
    )
    skews = _profile_values(
        rng,
        variant=variant,
        start=float(rng.uniform(0.003, 0.01)),
        target=float(rng.uniform(0.03, 0.18)),
        steps=steps,
        min_value=0.001,
        max_value=0.25,
        noise_scale=0.0015,
    )
    skew_sign = -1.0 if rng.random() < 0.5 else 1.0
    suffix = [
        _window_with_updates(
            baseline[onset + idx],
            synchronization_offset_ps=offsets[idx],
            synchronization_jitter_ps=sync_jitters[idx],
            clock_skew_ppm=skew_sign * skews[idx],
        )
        for idx in range(steps)
    ]
    return SequenceScenario(
        sample_id=sample_id,
        impairment_label="synchronization_issue",
        seed=seed,
        source_frequency_hz=preset.source_frequency_hz,
        mean_photon_num=float(rng.uniform(0.04, 0.10)),
        fault_onset_window=onset,
        trajectory_variant=variant,
        physical_window_spacing_s=_standard_physical_window_spacing_s(preset),
        arm_a=arm_a,
        arm_b=arm_b,
        windows=_combine_windows(prefix, suffix),
    )


def _make_polarization_drift(sample_id: str, seed: int, preset: GenerationPreset) -> SequenceScenario:
    rng, onset, arm_a, arm_b = _common_scenario(sample_id, "polarization_drift", seed, preset)
    variant = _trajectory_variant("polarization_drift", int(sample_id.rsplit("_", 1)[1]))
    baseline = _make_normal_windows(rng, preset.windows_per_sample)
    prefix = baseline[:onset]
    steps = preset.windows_per_sample - onset
    target_a = float(rng.uniform(4.0, 10.0)) * (-1 if rng.random() < 0.5 else 1)
    target_b = float(rng.uniform(4.0, 10.0)) * (-1 if rng.random() < 0.5 else 1)
    drift_a = _profile_values(rng, variant=variant, start=0.0, target=target_a, steps=steps, min_value=-15.0, max_value=15.0, noise_scale=0.12)
    drift_b = _profile_values(rng, variant=variant, start=0.0, target=target_b, steps=steps, min_value=-15.0, max_value=15.0, noise_scale=0.12)
    ellipticity_a = _profile_values(rng, variant=variant, start=1.0, target=float(rng.uniform(1.002, 1.006)), steps=steps, min_value=1.0, max_value=1.01, noise_scale=0.00008)
    ellipticity_b = _profile_values(rng, variant=variant, start=1.0, target=float(rng.uniform(1.002, 1.006)), steps=steps, min_value=1.0, max_value=1.01, noise_scale=0.00008)
    suffix = [
        _window_with_updates(
            baseline[onset + idx],
            analyzer_rotation_error_deg_a=drift_a[idx],
            analyzer_rotation_error_deg_b=drift_b[idx],
            arm_a_core_ellipticity=ellipticity_a[idx],
            arm_b_core_ellipticity=ellipticity_b[idx],
            arm_a_bend_radius_m=float(rng.uniform(5.0, 18.0)),
            arm_b_bend_radius_m=float(rng.uniform(5.0, 18.0)),
            arm_a_twist_rate_rad_per_m=float(rng.uniform(0.15, 0.9)),
            arm_b_twist_rate_rad_per_m=float(rng.uniform(0.15, 0.9)),
        )
        for idx in range(steps)
    ]
    return SequenceScenario(
        sample_id=sample_id,
        impairment_label="polarization_drift",
        seed=seed,
        source_frequency_hz=preset.source_frequency_hz,
        mean_photon_num=float(rng.uniform(0.04, 0.10)),
        fault_onset_window=onset,
        trajectory_variant=variant,
        physical_window_spacing_s=_standard_physical_window_spacing_s(preset),
        arm_a=arm_a,
        arm_b=arm_b,
        windows=_combine_windows(prefix, suffix),
    )


def _make_raman_noise(sample_id: str, seed: int, preset: GenerationPreset) -> SequenceScenario:
    rng, onset, arm_a, arm_b = _common_scenario(sample_id, "raman_noise", seed, preset)
    variant = _trajectory_variant("raman_noise", int(sample_id.rsplit("_", 1)[1]))
    wavelength = float(rng.choice([1270.0, 1310.0]))
    arm_a = ArmStaticConfig(**{**asdict(arm_a), "classical_wavelength_nm": wavelength})
    arm_b = ArmStaticConfig(**{**asdict(arm_b), "classical_wavelength_nm": wavelength})
    baseline = _make_normal_windows(rng, preset.windows_per_sample)
    prefix = baseline[:onset]
    steps = preset.windows_per_sample - onset
    powers_a = _profile_values(rng, variant=variant, start=0.0, target=float(rng.uniform(0.002, 0.02)), steps=steps, min_value=0.0, max_value=0.03, noise_scale=0.0002)
    powers_b = _profile_values(rng, variant=variant, start=0.0, target=float(rng.uniform(0.002, 0.02)), steps=steps, min_value=0.0, max_value=0.03, noise_scale=0.0002)
    suffix = [
        _window_with_updates(
            baseline[onset + idx],
            arm_a_classical_power_mw=powers_a[idx],
            arm_b_classical_power_mw=powers_b[idx],
        )
        for idx in range(steps)
    ]
    return SequenceScenario(
        sample_id=sample_id,
        impairment_label="raman_noise",
        seed=seed,
        source_frequency_hz=preset.source_frequency_hz,
        mean_photon_num=float(rng.uniform(0.06, 0.12)),
        fault_onset_window=onset,
        trajectory_variant=variant,
        physical_window_spacing_s=_standard_physical_window_spacing_s(preset),
        arm_a=arm_a,
        arm_b=arm_b,
        windows=_combine_windows(prefix, suffix),
    )


def _make_loss(sample_id: str, seed: int, preset: GenerationPreset) -> SequenceScenario:
    rng, onset, arm_a, arm_b = _common_scenario(sample_id, "loss", seed, preset)
    variant = _trajectory_variant("loss", int(sample_id.rsplit("_", 1)[1]))
    baseline = _make_normal_windows(rng, preset.windows_per_sample)
    prefix = baseline[:onset]
    steps = preset.windows_per_sample - onset
    loss_a = _profile_values(rng, variant=variant, start=0.0, target=float(rng.uniform(0.0002, 0.0008)), steps=steps, min_value=0.0, max_value=0.001, noise_scale=0.00001)
    loss_b = _profile_values(rng, variant=variant, start=0.0, target=float(rng.uniform(0.0002, 0.0008)), steps=steps, min_value=0.0, max_value=0.001, noise_scale=0.00001)
    suffix = [
        _window_with_updates(
            baseline[onset + idx],
            arm_a_extra_attenuation_db_per_m=loss_a[idx],
            arm_b_extra_attenuation_db_per_m=loss_b[idx],
        )
        for idx in range(steps)
    ]
    return SequenceScenario(
        sample_id=sample_id,
        impairment_label="loss",
        seed=seed,
        source_frequency_hz=preset.source_frequency_hz,
        mean_photon_num=float(rng.uniform(0.05, 0.12)),
        fault_onset_window=onset,
        trajectory_variant=variant,
        physical_window_spacing_s=_standard_physical_window_spacing_s(preset),
        arm_a=arm_a,
        arm_b=arm_b,
        windows=_combine_windows(prefix, suffix),
    )


def _make_peak_shift_temperature(sample_id: str, seed: int, preset: GenerationPreset) -> SequenceScenario:
    rng, onset, arm_a, arm_b = _common_scenario(sample_id, "peak_shift_temperature", seed, preset)
    variant = _trajectory_variant("peak_shift_temperature", int(sample_id.rsplit("_", 1)[1]))
    baseline = _make_normal_windows(rng, preset.windows_per_sample)
    prefix = baseline[:onset]
    steps = preset.windows_per_sample - onset
    cool_arm_a = rng.random() < 0.5
    base_cold_temps = [baseline[onset + idx].arm_a_temperature_c if cool_arm_a else baseline[onset + idx].arm_b_temperature_c for idx in range(steps)]
    target_cold_temps = _profile_values(
        rng,
        variant=variant,
        start=base_cold_temps[0],
        target=float(rng.uniform(0.0, 6.0)),
        steps=steps,
        min_value=0.0,
        max_value=22.0,
        noise_scale=0.12,
    )
    delta_temp = np.asarray(base_cold_temps, dtype=float) - np.asarray(target_cold_temps, dtype=float)
    peak_sign = -1 if cool_arm_a else 1
    offsets = [int(round(peak_sign * max(0.0, dt) * rng.uniform(120.0, 260.0))) for dt in delta_temp]
    suffix = [
        _window_with_updates(
            baseline[onset + idx],
            timing_offset_ps=offsets[idx],
            arm_a_temperature_c=target_cold_temps[idx] if cool_arm_a else baseline[onset + idx].arm_a_temperature_c,
            arm_b_temperature_c=baseline[onset + idx].arm_b_temperature_c if cool_arm_a else target_cold_temps[idx],
        )
        for idx in range(steps)
    ]
    return SequenceScenario(
        sample_id=sample_id,
        impairment_label="peak_shift_temperature",
        seed=seed,
        source_frequency_hz=preset.source_frequency_hz,
        mean_photon_num=float(rng.uniform(0.04, 0.10)),
        fault_onset_window=onset,
        trajectory_variant=variant,
        physical_window_spacing_s=_temperature_physical_window_spacing_s(preset),
        arm_a=arm_a,
        arm_b=arm_b,
        windows=_combine_windows(prefix, suffix),
    )


BUILDERS = {
    "normal": _make_normal,
    "synchronization_issue": _make_synchronization_issue,
    "polarization_drift": _make_polarization_drift,
    "raman_noise": _make_raman_noise,
    "loss": _make_loss,
    "peak_shift_temperature": _make_peak_shift_temperature,
}


def build_scenario(label: str, item_index: int, preset: GenerationPreset, seed: int | None = None) -> SequenceScenario:
    """Create one scenario for a specific label and index."""

    if label not in BUILDERS:
        raise ValueError(f"Unknown impairment label '{label}'")
    scenario_seed = int(seed if seed is not None else preset.base_seed + IMPAIRMENT_LABELS.index(label) * 10_000 + item_index)
    sample_id = f"{label}_{item_index:03d}"
    return BUILDERS[label](sample_id, scenario_seed, preset)


def build_scenarios(preset: GenerationPreset) -> list[SequenceScenario]:
    """Create all scenarios for a preset."""

    items: list[SequenceScenario] = []
    for label_index, label in enumerate(IMPAIRMENT_LABELS):
        for item_index in range(preset.samples_per_label):
            seed = preset.base_seed + label_index * 10_000 + item_index
            items.append(build_scenario(label, item_index, preset, seed=seed))
    return items
