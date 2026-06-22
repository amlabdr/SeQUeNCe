"""BSM-only episode simulation for AI diagnosis datasets."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from sequence.components.fiber_quantum_channel import FiberSpec

from example.hom_swapping_validation.simulator import (
    CoincidenceConfig,
    SequenceHOMConfig,
    run_hom_sequence_delay_scan,
)

from .config import DatasetConfig
from .features import add_temporal_derived_features, bsm_observable_features
from .scenarios import BSMDiagnosisScenario, WindowState


def _apply_clock_desynchronization(
    timestamps_ps,
    *,
    jitter_ps: float,
    rng: np.random.Generator,
) -> list[int]:
    """Apply detector-clock jitter to one BSM timestamp stream.

    This intentionally changes the observable BSM timestamps. It is not a
    correction and it is not passed to coincidence matching as a known offset.
    """
    if len(timestamps_ps) == 0:
        return []
    transformed = np.asarray(timestamps_ps, dtype=np.float64)
    if jitter_ps > 0:
        transformed = transformed + rng.normal(loc=0.0, scale=float(jitter_ps), size=transformed.shape[0])
    out = [int(round(item)) for item in transformed.tolist()]
    out.sort()
    return out


def _raman_power_for_arm(window: WindowState, cfg: DatasetConfig, arm: str) -> float:
    power = max(0.0, float(cfg.setup.baseline_raman_power_mw))
    if arm == window.faulty_arm:
        power += max(0.0, float(window.true_raman_power_mw))
    return power


def _fiber_spec_for_window(base_lambda_nm: float, window: WindowState, cfg: DatasetConfig, arm: str) -> FiberSpec:
    temperature = 20.0
    twist = 0.0
    bend = 0.0
    classical_power_mw = _raman_power_for_arm(window, cfg, arm)
    classical_enabled = classical_power_mw > 0.0

    if arm == window.faulty_arm:
        temperature += float(window.true_temperature_shift_c)

    if arm == window.faulty_arm:
        if window.true_polarization_twist_rad_per_m > 0:
            twist = float(window.true_polarization_twist_rad_per_m)
            bend = float(window.true_polarization_bend_radius_m)

    return FiberSpec(
        wavelength_m=float(base_lambda_nm) * 1e-9,
        temperature_C=float(temperature),
        core_ellipticity=1.0,
        bend_radius_m=float(bend),
        twist_rate_rad_per_m=float(twist),
        classical_coexist_enabled=bool(classical_enabled),
        classical_wavelength_nm=1270.0,
        classical_power_mW=float(classical_power_mw),
        quantum_wavelength_nm=float(base_lambda_nm),
    )


def _window_sequence_config(window: WindowState, cfg: DatasetConfig, seed: int) -> SequenceHOMConfig:
    setup = cfg.setup
    pulses = max(1, int(round(cfg.generation.window_duration_s * setup.source_frequency_hz)))

    mean_a = setup.mean_photon_num * float(window.true_source_brightness_factor_a)
    mean_b = setup.mean_photon_num * float(window.true_source_brightness_factor_b)
    loss_a = setup.attenuation_db_per_m + float(window.true_extra_loss_db_per_m_a)
    loss_b = setup.attenuation_db_per_m + float(window.true_extra_loss_db_per_m_b)
    lambda_a = setup.lambda_a_nm
    lambda_b = setup.lambda_b_nm
    if window.faulty_source == "A":
        lambda_a += float(window.true_spectral_detuning_nm)
    elif window.faulty_source == "B":
        lambda_b += float(window.true_spectral_detuning_nm)

    return SequenceHOMConfig(
        pulses_per_delay=pulses,
        emission_chunk_pulses=int(setup.emission_chunk_pulses),
        source_frequency_hz=float(setup.source_frequency_hz),
        mean_photon_num=float(setup.mean_photon_num),
        mean_photon_num_a=float(mean_a),
        mean_photon_num_b=float(mean_b),
        source_bandwidth_nm=float(setup.source_bandwidth_nm),
        wavelengths_nm_a=(float(lambda_a), float(setup.lambda_idler_a_nm)),
        wavelengths_nm_b=(float(lambda_b), float(setup.lambda_idler_b_nm)),
        photon_statistics=str(setup.photon_statistics),
        use_sparse_emission=bool(setup.use_sparse_emission),
        source_bell_state_a=str(setup.bell_state_a),
        source_bell_state_b=str(setup.bell_state_b),
        arm_length_m_a=float(setup.arm_length_m_a),
        arm_length_m_b=float(setup.arm_length_m_b),
        herald_length_m_a=float(setup.herald_length_m_a),
        herald_length_m_b=float(setup.herald_length_m_b),
        attenuation_db_per_m=float(setup.attenuation_db_per_m),
        attenuation_db_per_m_signal_a=float(loss_a),
        attenuation_db_per_m_signal_b=float(loss_b),
        attenuation_db_per_m_herald_a=float(setup.attenuation_db_per_m),
        attenuation_db_per_m_herald_b=float(setup.attenuation_db_per_m),
        detector_eff_bsm=float(setup.detector_eff_bsm),
        detector_eff_herald=1.0,
        detector_jitter_ps=float(setup.detector_jitter_ps),
        detector_dark_hz_bsm=float(setup.detector_dark_hz_bsm),
        detector_dark_hz_herald=0.0,
        hom_match_window_ps=int(setup.hom_match_window_ps),
        hom_coinc_window_ps=int(cfg.histogram.coincidence_window_ps),
        herald_mode="bsm_only",
        extra_overlap_scale=1.0,
        use_signal_rotators=False,
        fiber_spec_signal_a=_fiber_spec_for_window(lambda_a, window, cfg, "A"),
        fiber_spec_signal_b=_fiber_spec_for_window(lambda_b, window, cfg, "B"),
        fiber_spec_herald_a=FiberSpec(wavelength_m=float(setup.lambda_idler_a_nm) * 1e-9),
        fiber_spec_herald_b=FiberSpec(wavelength_m=float(setup.lambda_idler_b_nm) * 1e-9),
        seed=int(seed),
    )


def simulate_window(
    scenario: BSMDiagnosisScenario,
    window: WindowState,
    cfg: DatasetConfig,
    seed: int,
    include_raw: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    sim_cfg = _window_sequence_config(window, cfg, seed)
    # Temperature drift is expressed through the fiber state. Source-clock
    # drift is a real relative source-timing walk-off before the BSM, not a
    # coincidence-analysis correction.
    delay_ps = 0
    if window.true_source_clock_offset_ps != 0:
        # HOMInterferenceNode delays arm B. A source-A timing walk-off is the
        # equivalent relative opposite delay.
        sign = -1.0 if window.faulty_source == "A" else 1.0
        delay_ps = int(round(sign * float(window.true_source_clock_offset_ps)))
    df, streams = run_hom_sequence_delay_scan(
        delays_ps=[delay_ps],
        cfg=sim_cfg,
        bsm_cfg=CoincidenceConfig(window_ps=int(cfg.histogram.coincidence_window_ps), offset_ps=0),
        store_streams=True,
    )
    bsm1 = streams[delay_ps]["BSM1_ps"]
    bsm2 = streams[delay_ps]["BSM2_ps"]
    sync_rng = np.random.default_rng(int(seed) + 7919)
    if window.faulty_bsm_detector == "BSM1":
        bsm1_observed = _apply_clock_desynchronization(
            bsm1,
            jitter_ps=float(window.true_sync_jitter_ps),
            rng=sync_rng,
        )
        bsm2_observed = bsm2
    else:
        bsm1_observed = bsm1
        bsm2_observed = _apply_clock_desynchronization(
            bsm2,
            jitter_ps=float(window.true_sync_jitter_ps),
            rng=sync_rng,
        )
    features, raw_hist = bsm_observable_features(
        bsm1_ps=bsm1_observed,
        bsm2_ps=bsm2_observed,
        integration_time_s=float(cfg.generation.window_duration_s),
        histogram_range_ps=int(cfg.histogram.range_ps),
        histogram_bin_width_ps=int(cfg.histogram.bin_width_ps),
        coincidence_window_ps=int(cfg.histogram.coincidence_window_ps),
        bsm_offset_ps=0,
        bsm2_timestamp_offset_ps=0,
    )

    row = {
        "episode_id": scenario.episode_id,
        "setup_id": scenario.setup_id,
        "fault_class": scenario.fault_class,
        "window_index": int(window.window_index),
        "window_start_s": float(window.window_index * cfg.generation.window_duration_s),
        "window_end_s": float((window.window_index + 1) * cfg.generation.window_duration_s),
        "window_duration_s": float(cfg.generation.window_duration_s),
        "fault_active": int(window.fault_active),
        **features,
    }

    truth = {
        "episode_id": scenario.episode_id,
        "window_index": int(window.window_index),
        "seed": int(seed),
        "fault_class": scenario.fault_class,
        "fault_onset_window": int(scenario.fault_onset_window),
        "target_fault_level": float(scenario.target_fault_level),
        **{
            key: value
            for key, value in asdict(window).items()
            if key not in {"faulty_source", "faulty_arm", "faulty_bsm_detector"}
        },
        "sim_delay_ps": int(delay_ps),
        "sim_mean_photon_num_a": float(sim_cfg.mean_photon_num_a or sim_cfg.mean_photon_num),
        "sim_mean_photon_num_b": float(sim_cfg.mean_photon_num_b or sim_cfg.mean_photon_num),
        "sim_attenuation_db_per_m_signal_a": float(sim_cfg.attenuation_db_per_m_signal_a or sim_cfg.attenuation_db_per_m),
        "sim_attenuation_db_per_m_signal_b": float(sim_cfg.attenuation_db_per_m_signal_b or sim_cfg.attenuation_db_per_m),
        "sim_lambda_a_nm": float(sim_cfg.wavelengths_nm_a[0]),
        "sim_lambda_b_nm": float(sim_cfg.wavelengths_nm_b[0]),
        "sim_raman_power_mw_a": float(_raman_power_for_arm(window, cfg, "A")),
        "sim_raman_power_mw_b": float(_raman_power_for_arm(window, cfg, "B")),
        "sim_raman_noise_rate_total_hz": float(df["raman_noise_rate_total_hz"].iloc[0]),
    }

    raw = None
    if include_raw:
        raw = {
            "episode_id": scenario.episode_id,
            "window_index": int(window.window_index),
            "histogram": raw_hist,
            "bsm1_ps": [int(v) for v in bsm1_observed],
            "bsm2_ps": [int(v) for v in bsm2_observed],
        }
    return row, truth, raw


def simulate_episode(
    scenario: BSMDiagnosisScenario,
    cfg: DatasetConfig,
    include_raw: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    raw_windows: list[dict[str, Any]] = []
    for window in scenario.windows:
        seed = int(scenario.seed + 1009 * int(window.window_index))
        row, truth, raw = simulate_window(scenario, window, cfg, seed=seed, include_raw=include_raw)
        rows.append(row)
        truth_rows.append(truth)
        if raw is not None:
            raw_windows.append(raw)
    rows = add_temporal_derived_features(rows)
    raw_episode = {
        "scenario": scenario.to_metadata(),
        "windows": raw_windows,
    } if include_raw else None
    return rows, truth_rows, raw_episode
