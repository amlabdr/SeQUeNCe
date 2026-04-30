"""Continuous-episode simulation runners for the AI diagnosis dataset."""

from __future__ import annotations

from dataclasses import asdict
from math import radians

import numpy as np

from config import GenerationPreset
from features import coincidence_histogram
from scenarios import ArmStaticConfig, SequenceScenario, WindowState
from sequence.components.fiber_quantum_channel import FiberSection, FiberSpec, fiberQuantumChannel
from sequence.kernel.entity import Entity
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.topology.optical_nodes import PolarizationAnalyzerNode, SpdcSourceNode

PS_PER_S = 10**12


class EpisodeController(Entity):
    """Controller that applies window states and schedules Raman noise within one episode timeline."""

    def __init__(
        self,
        name: str,
        timeline: Timeline,
        scenario: SequenceScenario,
        preset: GenerationPreset,
        source: SpdcSourceNode,
        analyzer_a: PolarizationAnalyzerNode,
        analyzer_b: PolarizationAnalyzerNode,
        channel_a: fiberQuantumChannel,
        channel_b: fiberQuantumChannel,
    ) -> None:
        super().__init__(name, timeline)
        self.scenario = scenario
        self.preset = preset
        self.source = source
        self.analyzer_a = analyzer_a
        self.analyzer_b = analyzer_b
        self.channel_a = channel_a
        self.channel_b = channel_b
        self.window_offset_ps: dict[int, int] = {}
        self.current_window: int | None = None

    def init(self) -> None:
        pass

    def apply_window_state(self, window_index: int) -> None:
        window = self.scenario.windows[window_index]
        self.current_window = window_index
        self._apply_channel_state(self.channel_a, self.scenario.arm_a, window, "a")
        self._apply_channel_state(self.channel_b, self.scenario.arm_b, window, "b")
        delay_diff_ps = int(round((self.channel_b.base_group_delay_s - self.channel_a.base_group_delay_s) * 1e12))
        self.window_offset_ps[window_index] = -delay_diff_ps
        self.analyzer_a.set_rotation_angle(radians(window.analyzer_rotation_error_deg_a))
        self.analyzer_b.set_rotation_angle(radians(window.analyzer_rotation_error_deg_b))

        window_start_ps = int(round(window_index * self.preset.window_duration_s * PS_PER_S))
        window_end_ps = int(round((window_index + 1) * self.preset.window_duration_s * PS_PER_S))
        self._schedule_raman_noise_for_window(self.channel_a, window_start_ps, window_end_ps)
        self._schedule_raman_noise_for_window(self.channel_b, window_start_ps, window_end_ps)

    def set_visibility_angle(self, angle_deg: float) -> None:
        if self.current_window is None:
            base_a = 0.0
            base_b = 0.0
        else:
            window = self.scenario.windows[self.current_window]
            base_a = radians(window.analyzer_rotation_error_deg_a)
            base_b = radians(window.analyzer_rotation_error_deg_b)
        self.analyzer_a.set_rotation_angle(base_a)
        self.analyzer_b.set_rotation_angle(base_b + radians(angle_deg))

    def _apply_channel_state(self, channel, arm: ArmStaticConfig, window: WindowState, side: str) -> None:
        spec = channel.sections[0].spec
        if side == "a":
            spec.temperature_C = float(window.arm_a_temperature_c)
            spec.core_ellipticity = float(window.arm_a_core_ellipticity)
            spec.bend_radius_m = float(window.arm_a_bend_radius_m)
            spec.twist_rate_rad_per_m = float(window.arm_a_twist_rate_rad_per_m)
            spec.classical_power_mW = float(window.arm_a_classical_power_mw)
            classical_enabled = window.arm_a_classical_power_mw > 0
            extra_att = float(window.arm_a_extra_attenuation_db_per_m)
        else:
            spec.temperature_C = float(window.arm_b_temperature_c)
            spec.core_ellipticity = float(window.arm_b_core_ellipticity)
            spec.bend_radius_m = float(window.arm_b_bend_radius_m)
            spec.twist_rate_rad_per_m = float(window.arm_b_twist_rate_rad_per_m)
            spec.classical_power_mW = float(window.arm_b_classical_power_mw)
            classical_enabled = window.arm_b_classical_power_mw > 0
            extra_att = float(window.arm_b_extra_attenuation_db_per_m)

        spec.classical_coexist_enabled = bool(classical_enabled)
        channel.attenuation = float(arm.base_attenuation_db_per_m + extra_att)
        channel.loss = 1 - 10 ** (channel.distance * channel.attenuation / -10)
        channel._compute_link_model()
        if channel._is_classical_coexist_enabled():
            fs, bs = channel._compute_raman_noise_rate()
            channel.raman_noise_rate_FS_Hz = fs
            channel.raman_noise_rate_BS_Hz = bs
            channel.raman_noise_rate_Hz = fs + bs
            channel.noise_enabled = channel.raman_noise_rate_Hz > 0
        else:
            channel.raman_noise_rate_FS_Hz = 0.0
            channel.raman_noise_rate_BS_Hz = 0.0
            channel.raman_noise_rate_Hz = 0.0
            channel.noise_enabled = False

    def _schedule_raman_noise_for_window(self, channel, start_ps: int, end_ps: int) -> None:
        rate_hz = float(getattr(channel, "raman_noise_rate_Hz", 0.0))
        if rate_hz <= 0 or end_ps <= start_ps:
            return

        rng = self.scenario.arm_a if False else None
        duration_s = (end_ps - start_ps) * 1e-12
        count = int(self.source.get_generator().poisson(rate_hz * duration_s))
        if count <= 0:
            return

        draws = np.sort(self.source.get_generator().random(count))
        for draw in draws:
            event_time = int(start_ps + draw * (end_ps - start_ps))
            self.timeline.schedule(Event(event_time, Process(channel.receiver, "receive_noise_photon", [])))


def _build_spec(arm: ArmStaticConfig) -> FiberSpec:
    return FiberSpec(
        temperature_C=20.0,
        classical_wavelength_nm=arm.classical_wavelength_nm,
        quantum_wavelength_nm=1550.0,
        quantum_bandwidth_Hz=100e9,
        classical_coexist_enabled=False,
        classical_power_mW=0.0,
    )


def _build_channel(name: str, timeline: Timeline, arm: ArmStaticConfig) -> fiberQuantumChannel:
    return fiberQuantumChannel(
        name=name,
        timeline=timeline,
        attenuation=arm.base_attenuation_db_per_m,
        distance=arm.length_m,
        sections=[FiberSection(arm.length_m, _build_spec(arm))],
    )


def _filter_times(times: list[int], start_ps: int, end_ps: int) -> list[int]:
    return [int(t) for t in times if start_ps <= int(t) < end_ps]


def _apply_clock_desynchronization(
    timestamps_ps: list[int],
    *,
    window_start_ps: int,
    offset_ps: int,
    jitter_ps: float,
    clock_skew_ppm: float,
    rng: np.random.Generator,
) -> list[int]:
    if not timestamps_ps:
        return []
    values = np.asarray(timestamps_ps, dtype=np.float64)
    relative = values - float(window_start_ps)
    skew_scale = 1.0 + float(clock_skew_ppm) * 1e-6
    transformed = float(window_start_ps) + relative * skew_scale + float(offset_ps)
    if jitter_ps > 0:
        transformed = transformed + rng.normal(loc=0.0, scale=float(jitter_ps), size=values.shape[0])
    return [int(round(item)) for item in transformed.tolist()]


def _pair_stats(
    timestamps_a_ps: list[int],
    timestamps_b_ps: list[int],
    preset: GenerationPreset,
    integration_time_s: float,
    offset_b_ps: int,
    fixed_peak_position_ps: float | None = None,
) -> tuple[float, int, dict]:
    features, raw = coincidence_histogram(
        timestamps_a_ps=timestamps_a_ps,
        timestamps_b_ps=timestamps_b_ps,
        histogram_range_ps=preset.histogram.range_ps,
        bin_width_ps=preset.histogram.bin_width_ps,
        coincidence_window_ps=preset.histogram.coincidence_window_ps,
        integration_time_s=integration_time_s,
        offset_b_ps=offset_b_ps,
        fixed_peak_position_ps=fixed_peak_position_ps,
    )
    return float(features.coincidence_rate_hz), int(features.coincidences_count), {
        "peak_position_ps": float(features.peak_position_ps),
        "peak_width_ps": float(features.peak_width_ps),
        "peak_height": float(features.peak_height),
        "peak_snr": float(features.peak_snr),
        "accidental_coincidences_count_estimate": float(features.accidental_coincidences_count_estimate),
        "accidental_coincidence_rate_hz": float(features.accidental_coincidence_rate_hz),
        "car": float(features.car),
        "coincidence_rate_hz": float(features.coincidence_rate_hz),
        "coincidences_count": int(features.coincidences_count),
        "histogram": raw,
    }


def _safe_corr(pp: int, pm: int, mp: int, mm: int) -> float:
    denom = pp + pm + mp + mm
    if denom <= 0:
        return 0.0
    return float((pp + mm - pm - mp) / denom)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _accidental_rate_from_singles(
    singles_rate_a_hz: float,
    singles_rate_b_hz: float,
    coincidence_window_ps: int,
) -> float:
    return float(singles_rate_a_hz * singles_rate_b_hz * coincidence_window_ps * 1e-12)


def _visibility_from_trace(trace: list[dict[str, float]]) -> float | None:
    if not trace:
        return None
    values = np.asarray([point["hh_coincidence_rate_hz"] for point in trace], dtype=float)
    max_val = float(np.max(values))
    min_val = float(np.min(values))
    if max_val + min_val <= 0:
        return 0.0
    return float((max_val - min_val) / (max_val + min_val))


def _active_window(window_index: int, preset: GenerationPreset) -> bool:
    return window_index % preset.visibility_measurement_stride_windows == 0


def _annotate_temporal_features(rows: list[dict], rolling_window: int = 3) -> None:
    for idx, row in enumerate(rows):
        prev = rows[idx - 1] if idx > 0 else None
        for key, new_key in (
            ("peak_position_ps", "delta_peak_position_ps"),
            ("coincidence_rate_hz", "delta_coincidence_rate_hz"),
            ("singles_rate_a_hz", "delta_singles_rate_a_hz"),
            ("singles_rate_b_hz", "delta_singles_rate_b_hz"),
            ("zz_correlation", "delta_zz_correlation"),
        ):
            row[new_key] = 0.0 if prev is None else float(row[key] - prev[key])

        start = max(0, idx - rolling_window + 1)
        history = rows[start : idx + 1]
        for key, new_key in (
            ("peak_position_ps", f"rolling_peak_position_ps_mean_{rolling_window}"),
            ("coincidence_rate_hz", f"rolling_coincidence_rate_hz_mean_{rolling_window}"),
            ("singles_rate_a_hz", f"rolling_singles_rate_a_hz_mean_{rolling_window}"),
            ("singles_rate_b_hz", f"rolling_singles_rate_b_hz_mean_{rolling_window}"),
            ("zz_correlation", f"rolling_zz_correlation_mean_{rolling_window}"),
        ):
            row[new_key] = float(np.mean([item[key] for item in history]))


def _schedule_episode(
    timeline: Timeline,
    controller: EpisodeController,
    source: SpdcSourceNode,
    preset: GenerationPreset,
) -> list[dict]:
    window_specs: list[dict] = []
    period_ps = int(round(preset.window_duration_s * PS_PER_S))
    pulse_period_ps = int(round(PS_PER_S / preset.source_frequency_hz))

    for window_index in range(preset.windows_per_sample):
        start_ps = window_index * period_ps
        end_ps = start_ps + period_ps
        timeline.schedule(Event(start_ps, Process(controller, "apply_window_state", [window_index])))

        active = _active_window(window_index, preset)
        if not active:
            timeline.schedule(Event(start_ps, Process(source, "emit", [preset.histogram_pulses])))
            window_specs.append(
                {
                    "window_index": window_index,
                    "window_start_ps": start_ps,
                    "window_end_ps": end_ps,
                    "passive_start_ps": start_ps,
                    "passive_end_ps": end_ps,
                    "active_segments": [],
                }
            )
            continue

        active_total_pulses = min(preset.visibility_pulses, preset.histogram_pulses)
        passive_pulses = max(0, preset.histogram_pulses - active_total_pulses)
        passive_duration_ps = passive_pulses * pulse_period_ps
        timeline.schedule(Event(start_ps, Process(source, "emit", [passive_pulses])))

        active_segments = []
        active_start_ps = start_ps + passive_duration_ps
        segment_count = len(preset.visibility_angles_deg)
        pulses_per_angle = max(1, int(round(active_total_pulses / max(segment_count, 1))))

        for idx, angle_deg in enumerate(preset.visibility_angles_deg):
            seg_start_ps = active_start_ps + idx * pulses_per_angle * pulse_period_ps
            seg_end_ps = min(end_ps, seg_start_ps + pulses_per_angle * pulse_period_ps)
            timeline.schedule(Event(seg_start_ps, Process(controller, "set_visibility_angle", [float(angle_deg)])))
            timeline.schedule(Event(seg_start_ps, Process(source, "emit", [pulses_per_angle])))
            active_segments.append(
                {
                    "angle_deg": float(angle_deg),
                    "start_ps": seg_start_ps,
                    "end_ps": seg_end_ps,
                }
            )

        window_specs.append(
            {
                "window_index": window_index,
                "window_start_ps": start_ps,
                "window_end_ps": end_ps,
                "passive_start_ps": start_ps,
                "passive_end_ps": active_start_ps,
                "active_segments": active_segments,
            }
        )

    return window_specs


def _schedule_window(
    timeline: Timeline,
    controller: EpisodeController,
    source: SpdcSourceNode,
    preset: GenerationPreset,
    window_index: int,
) -> dict:
    """Schedule one monitoring window on the existing timeline."""

    period_ps = int(round(preset.window_duration_s * PS_PER_S))
    pulse_period_ps = int(round(PS_PER_S / preset.source_frequency_hz))
    start_ps = window_index * period_ps
    end_ps = start_ps + period_ps

    controller.apply_window_state(window_index)
    active = _active_window(window_index, preset)

    if not active:
        timeline.schedule(Event(start_ps, Process(source, "emit", [preset.histogram_pulses])))
        return {
            "window_index": window_index,
            "window_start_ps": start_ps,
            "window_end_ps": end_ps,
            "passive_start_ps": start_ps,
            "passive_end_ps": end_ps,
            "active_segments": [],
        }

    active_total_pulses = min(preset.visibility_pulses, preset.histogram_pulses)
    passive_pulses = max(0, preset.histogram_pulses - active_total_pulses)
    passive_duration_ps = passive_pulses * pulse_period_ps
    timeline.schedule(Event(start_ps, Process(source, "emit", [passive_pulses])))

    active_segments = []
    active_start_ps = start_ps + passive_duration_ps
    segment_count = len(preset.visibility_angles_deg)
    pulses_per_angle = max(1, int(round(active_total_pulses / max(segment_count, 1))))

    for idx, angle_deg in enumerate(preset.visibility_angles_deg):
        seg_start_ps = active_start_ps + idx * pulses_per_angle * pulse_period_ps
        seg_end_ps = min(end_ps, seg_start_ps + pulses_per_angle * pulse_period_ps)
        timeline.schedule(Event(seg_start_ps, Process(controller, "set_visibility_angle", [float(angle_deg)])))
        timeline.schedule(Event(seg_start_ps, Process(source, "emit", [pulses_per_angle])))
        active_segments.append(
            {
                "angle_deg": float(angle_deg),
                "start_ps": seg_start_ps,
                "end_ps": seg_end_ps,
            }
        )

    return {
        "window_index": window_index,
        "window_start_ps": start_ps,
        "window_end_ps": end_ps,
        "passive_start_ps": start_ps,
        "passive_end_ps": active_start_ps,
        "active_segments": active_segments,
    }


def simulate_sequence(scenario: SequenceScenario, preset: GenerationPreset, include_raw: bool = True) -> tuple[list[dict], dict | None]:
    """Run one continuous episode and aggregate receiver observables by window."""

    episode_stop_ps = int(round(preset.episode_duration_s * PS_PER_S)) + int(2e9)
    timeline = Timeline(stop_time=episode_stop_ps)

    source = SpdcSourceNode(
        "source",
        timeline,
        {
            "frequency": scenario.source_frequency_hz,
            "mean_photon_num": scenario.mean_photon_num,
            "bandwidth": 0.5 * (scenario.arm_a.source_bandwidth_nm + scenario.arm_b.source_bandwidth_nm),
            "bell_state": "phi+",
        },
    )
    analyzer_a = PolarizationAnalyzerNode(
        "analyzer_a",
        timeline,
        {
            "mode": "hwp_only",
            "rotation_angle": 0.0,
            "detector_efficiency": scenario.arm_a.detector_efficiency,
            "dark_count": scenario.arm_a.detector_dark_count_hz,
            "pbs_fidelity": 1.0,
            "hwp_fidelity": 1.0,
            "mismeasure_prob": 0.0,
        },
    )
    analyzer_b = PolarizationAnalyzerNode(
        "analyzer_b",
        timeline,
        {
            "mode": "hwp_only",
            "rotation_angle": 0.0,
            "detector_efficiency": scenario.arm_b.detector_efficiency,
            "dark_count": scenario.arm_b.detector_dark_count_hz,
            "pbs_fidelity": 1.0,
            "hwp_fidelity": 1.0,
            "mismeasure_prob": 0.0,
        },
    )
    source.set_seed(scenario.seed)
    analyzer_a.set_seed(scenario.seed + 1)
    analyzer_b.set_seed(scenario.seed + 2)

    channel_a = _build_channel("qc_a", timeline, scenario.arm_a)
    channel_b = _build_channel("qc_b", timeline, scenario.arm_b)
    channel_a.set_ends(source, analyzer_a.name)
    channel_b.set_ends(source, analyzer_b.name)

    controller = EpisodeController("episode_controller", timeline, scenario, preset, source, analyzer_a, analyzer_b, channel_a, channel_b)

    timeline.init()

    rows: list[dict] = []
    raw_windows: list[dict] | None = [] if include_raw else None

    for window_index in range(preset.windows_per_sample):
        spec = _schedule_window(timeline, controller, source, preset, window_index)
        timeline.stop_time = spec["window_end_ps"]
        timeline.run()

        clicks_a = [sorted(int(t) for t in channel) for channel in analyzer_a.get_photon_times()]
        clicks_b = [sorted(int(t) for t in channel) for channel in analyzer_b.get_photon_times()]

        idx = spec["window_index"]
        window = scenario.windows[idx]
        alignment_offset_ps = controller.window_offset_ps.get(idx, 0) + int(window.timing_offset_ps)
        passive_start_ps = spec["passive_start_ps"]
        passive_end_ps = spec["passive_end_ps"]
        passive_duration_s = max((passive_end_ps - passive_start_ps) * 1e-12, 1e-12)
        window_rng = np.random.default_rng(scenario.seed + idx * 104729 + 17)

        a0 = _filter_times(clicks_a[0], passive_start_ps, passive_end_ps)
        a1 = _filter_times(clicks_a[1], passive_start_ps, passive_end_ps)
        b0 = _apply_clock_desynchronization(
            _filter_times(clicks_b[0], passive_start_ps, passive_end_ps),
            window_start_ps=spec["window_start_ps"],
            offset_ps=int(window.synchronization_offset_ps),
            jitter_ps=float(window.synchronization_jitter_ps),
            clock_skew_ppm=float(window.clock_skew_ppm),
            rng=window_rng,
        )
        b1 = _apply_clock_desynchronization(
            _filter_times(clicks_b[1], passive_start_ps, passive_end_ps),
            window_start_ps=spec["window_start_ps"],
            offset_ps=int(window.synchronization_offset_ps),
            jitter_ps=float(window.synchronization_jitter_ps),
            clock_skew_ppm=float(window.clock_skew_ppm),
            rng=window_rng,
        )
        flat_a = sorted(a0 + a1)
        flat_b = sorted(b0 + b1)
        singles_rate_a_hz = float(len(flat_a) / passive_duration_s)
        singles_rate_b_hz = float(len(flat_b) / passive_duration_s)

        coincidence_rate_hz, coincidences_count, hist = _pair_stats(flat_a, flat_b, preset, passive_duration_s, alignment_offset_ps)
        peak_window_center_ps = float(hist["peak_position_ps"])
        hh_rate_z, hh_count_z, _ = _pair_stats(a0, b0, preset, passive_duration_s, alignment_offset_ps, fixed_peak_position_ps=peak_window_center_ps)
        hv_rate_z, hv_count_z, _ = _pair_stats(a0, b1, preset, passive_duration_s, alignment_offset_ps, fixed_peak_position_ps=peak_window_center_ps)
        vh_rate_z, vh_count_z, _ = _pair_stats(a1, b0, preset, passive_duration_s, alignment_offset_ps, fixed_peak_position_ps=peak_window_center_ps)
        vv_rate_z, vv_count_z, _ = _pair_stats(a1, b1, preset, passive_duration_s, alignment_offset_ps, fixed_peak_position_ps=peak_window_center_ps)

        pp_rate_x = pm_rate_x = mp_rate_x = mm_rate_x = None
        xx_correlation = None
        visibility = None
        visibility_trace: list[dict[str, float]] = []
        physical_window_start_s = float(idx * scenario.physical_window_spacing_s)
        physical_window_end_s = float(physical_window_start_s + preset.window_duration_s)

        if spec["active_segments"]:
            x_counts = {"pp": 0, "pm": 0, "mp": 0, "mm": 0}
            total_active_s = 0.0
            for segment in spec["active_segments"]:
                seg_start_ps = segment["start_ps"]
                seg_end_ps = segment["end_ps"]
                seg_duration_s = max((seg_end_ps - seg_start_ps) * 1e-12, 1e-12)
                total_active_s += seg_duration_s
                sa0 = _filter_times(clicks_a[0], seg_start_ps, seg_end_ps)
                sa1 = _filter_times(clicks_a[1], seg_start_ps, seg_end_ps)
                sb0 = _apply_clock_desynchronization(
                    _filter_times(clicks_b[0], seg_start_ps, seg_end_ps),
                    window_start_ps=spec["window_start_ps"],
                    offset_ps=int(window.synchronization_offset_ps),
                    jitter_ps=float(window.synchronization_jitter_ps),
                    clock_skew_ppm=float(window.clock_skew_ppm),
                    rng=window_rng,
                )
                sb1 = _apply_clock_desynchronization(
                    _filter_times(clicks_b[1], seg_start_ps, seg_end_ps),
                    window_start_ps=spec["window_start_ps"],
                    offset_ps=int(window.synchronization_offset_ps),
                    jitter_ps=float(window.synchronization_jitter_ps),
                    clock_skew_ppm=float(window.clock_skew_ppm),
                    rng=window_rng,
                )

                hh_rate_vis, hh_count_vis, _ = _pair_stats(sa0, sb0, preset, seg_duration_s, alignment_offset_ps)
                visibility_trace.append(
                    {
                        "angle_deg": float(segment["angle_deg"]),
                        "hh_coincidence_rate_hz": hh_rate_vis,
                        "hh_coincidences_count": int(hh_count_vis),
                    }
                )

                if abs(segment["angle_deg"] - 45.0) < 1e-9:
                    _, _, x_hist = _pair_stats(sorted(sa0 + sa1), sorted(sb0 + sb1), preset, seg_duration_s, alignment_offset_ps)
                    x_peak_window_center_ps = float(x_hist["peak_position_ps"])
                    pp_rate_x, x_counts["pp"], _ = _pair_stats(sa0, sb0, preset, seg_duration_s, alignment_offset_ps, fixed_peak_position_ps=x_peak_window_center_ps)
                    pm_rate_x, x_counts["pm"], _ = _pair_stats(sa0, sb1, preset, seg_duration_s, alignment_offset_ps, fixed_peak_position_ps=x_peak_window_center_ps)
                    mp_rate_x, x_counts["mp"], _ = _pair_stats(sa1, sb0, preset, seg_duration_s, alignment_offset_ps, fixed_peak_position_ps=x_peak_window_center_ps)
                    mm_rate_x, x_counts["mm"], _ = _pair_stats(sa1, sb1, preset, seg_duration_s, alignment_offset_ps, fixed_peak_position_ps=x_peak_window_center_ps)

            if sum(x_counts.values()) > 0:
                xx_correlation = _safe_corr(x_counts["pp"], x_counts["pm"], x_counts["mp"], x_counts["mm"])
            visibility = _visibility_from_trace(visibility_trace)

        row = {
            "sample_id": scenario.sample_id,
            "label": scenario.impairment_label,
            "window_index": idx,
            "timestamp_step": idx,
            "fault_active": int(idx >= scenario.fault_onset_window and scenario.impairment_label != "normal"),
            "physical_time_s": physical_window_start_s,
            "physical_time_h": physical_window_start_s / 3600.0,
            "physical_window_start_s": physical_window_start_s,
            "physical_window_end_s": physical_window_end_s,
            "physical_window_spacing_s": float(scenario.physical_window_spacing_s),
            "acquisition_window_duration_s": float(preset.window_duration_s),
            "window_start_s": float(spec["window_start_ps"] * 1e-12),
            "window_end_s": float(spec["window_end_ps"] * 1e-12),
            "arm_length_a_m": float(scenario.arm_a.length_m),
            "arm_length_b_m": float(scenario.arm_b.length_m),
            "peak_position_ps": float(hist["peak_position_ps"]),
            "peak_width_ps": float(hist["peak_width_ps"]),
            "peak_height": float(hist["peak_height"]),
            "peak_snr": float(hist["peak_snr"]),
            "accidental_coincidence_rate_hz": _accidental_rate_from_singles(singles_rate_a_hz, singles_rate_b_hz, preset.histogram.coincidence_window_ps),
            "car": _safe_ratio(
                coincidence_rate_hz,
                _accidental_rate_from_singles(singles_rate_a_hz, singles_rate_b_hz, preset.histogram.coincidence_window_ps),
            ),
            "coincidence_rate_hz": coincidence_rate_hz,
            "coincidences_count": coincidences_count,
            "singles_rate_a_hz": singles_rate_a_hz,
            "singles_rate_b_hz": singles_rate_b_hz,
            "coincidence_to_singles_a": _safe_ratio(coincidence_rate_hz, singles_rate_a_hz),
            "coincidence_to_singles_b": _safe_ratio(coincidence_rate_hz, singles_rate_b_hz),
            "hh_rate_z_hz": hh_rate_z,
            "hv_rate_z_hz": hv_rate_z,
            "vh_rate_z_hz": vh_rate_z,
            "vv_rate_z_hz": vv_rate_z,
            "zz_correlation": _safe_corr(hh_count_z, hv_count_z, vh_count_z, vv_count_z),
            "pp_rate_x_hz": pp_rate_x,
            "pm_rate_x_hz": pm_rate_x,
            "mp_rate_x_hz": mp_rate_x,
            "mm_rate_x_hz": mm_rate_x,
            "xx_correlation": xx_correlation,
            "xx_measured": int(bool(spec["active_segments"])),
            "link_temperature_a_c": float(window.arm_a_temperature_c),
            "link_temperature_b_c": float(window.arm_b_temperature_c),
            "visibility": visibility,
            "visibility_measured": int(bool(spec["active_segments"])),
        }
        rows.append(row)
        if include_raw and raw_windows is not None:
            raw_windows.append(
                {
                    "window_index": idx,
                    "hidden_state": asdict(window),
                    "window_spec": spec,
                    "observable_row": row,
                    "histogram": hist["histogram"],
                    "visibility_trace": visibility_trace,
                }
            )

    raw = None
    if include_raw and raw_windows is not None:
        raw = {
            "sample_metadata": scenario.to_metadata(),
            "observable_windows": raw_windows,
        }
    _annotate_temporal_features(rows)
    return rows, raw
