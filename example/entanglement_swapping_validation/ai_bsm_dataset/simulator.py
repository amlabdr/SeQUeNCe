"""One continuous SeQUeNCe timeline per polarization-BSM dataset episode."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from sequence.components.fiber_quantum_channel import FiberSection, FiberSpec, fiberQuantumChannel
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.topology.optical_nodes import (
    PhotonSinkNode,
    PolarizationBSMNode,
    SpdcSourceNode,
)
from sequence.utils.encoding import polarization

from example.hom_validation.simulator import _schedule_chunked_emission

from .config import DatasetConfig
from .features import DETECTORS, add_deltas, observable_features
from .scenarios import BSMScenario


def _fiber(name, timeline, length_m, attenuation, spec):
    return fiberQuantumChannel(
        name,
        timeline,
        attenuation=attenuation,
        distance=length_m,
        sections=[FiberSection(length_m=length_m, spec=spec)],
    )


def _fiber_spec(wavelength_nm: float, classical_power_mw: float) -> FiberSpec:
    return FiberSpec(
        wavelength_m=wavelength_nm * 1e-9,
        temperature_C=20.0,
        core_ellipticity=1.0,
        classical_coexist_enabled=classical_power_mw > 0,
        classical_wavelength_nm=1270.0,
        classical_power_mW=classical_power_mw,
        quantum_wavelength_nm=wavelength_nm,
    )


def _apply_sync_jitter(streams, detector, jitter_ps, rng):
    if not detector or jitter_ps <= 0:
        return streams
    values = np.asarray(streams[detector], dtype=float)
    values += rng.normal(0.0, jitter_ps, len(values))
    streams[detector] = sorted(max(0, int(round(value))) for value in values)
    return streams


class ContinuousEpisodeController:
    def __init__(
        self, timeline, cfg, scenario, source_a, source_b, channel_a, channel_b, bsm
    ):
        self.timeline = timeline
        self.cfg = cfg
        self.scenario = scenario
        self.source_a = source_a
        self.source_b = source_b
        self.channels = {"A": channel_a, "B": channel_b}
        self.bsm = bsm
        self.rows = []
        self.truth = []
        self.raw = []
        self.rng = np.random.default_rng(scenario.seed + 77_777)
        self.base_specs = {
            "A": channel_a.sections[0].spec,
            "B": channel_b.sections[0].spec,
        }
        self._last_channel_state = {"A": None, "B": None}

    def _state(self, time_s):
        progress = self.scenario.progress(time_s)
        active = progress > 0
        label = self.scenario.fault_class
        target = self.scenario.target
        return {
            "progress": progress,
            "active": active,
            "temperature_c": target * progress if label == "temperature_change" else 0.0,
            "twist": target * progress if label == "polarization_drift" else 0.0,
            "bend_radius": (
                self.cfg.faults.polarization_bend_radius_m_max
                - progress * (
                    self.cfg.faults.polarization_bend_radius_m_max
                    - self.cfg.faults.polarization_bend_radius_m_min
                )
            ) if label == "polarization_drift" and active else 0.0,
            "detuning_nm": target * progress if label == "spectral_detuning" else 0.0,
            "raman_extra_mw": target * progress if label == "raman_noise" else 0.0,
            "extra_loss": target * progress if label == "attenuation_loss" else 0.0,
            "brightness_factor": (
                1.0 - progress * (1.0 - target)
                if label == "source_brightness_loss" else 1.0
            ),
            "sync_jitter_ps": target * progress if label == "sync_issue" else 0.0,
            "clock_offset_ps": (
                target * max(0.0, time_s - self.scenario.onset_s) * progress
                if label == "source_clock_drift" else 0.0
            ),
        }

    def update_source_and_fiber(self, time_s):
        state = self._state(float(time_s))
        affected_arm = self.scenario.affected_arm
        affected_source = self.scenario.affected_source
        setup = self.cfg.setup

        for arm, channel in self.channels.items():
            affected = arm == affected_arm
            base = self.base_specs[arm]
            temperature = 20.0 + (state["temperature_c"] if affected else 0.0)
            twist = state["twist"] if affected else 0.0
            bend = state["bend_radius"] if affected else 0.0
            raman_power = (
                setup.baseline_raman_power_mw
                + (state["raman_extra_mw"] if affected else 0.0)
            )
            attenuation = setup.attenuation_db_per_m + (
                state["extra_loss"] if affected else 0.0
            )
            channel_state = (temperature, twist, bend, raman_power, attenuation)
            if channel_state == self._last_channel_state[arm]:
                continue
            spec = replace(
                base,
                temperature_C=temperature,
                twist_rate_rad_per_m=twist,
                bend_radius_m=bend,
                classical_power_mW=raman_power,
            )
            channel.sections[0].spec = spec
            channel.attenuation = attenuation
            channel.loss = 1.0 - 10.0 ** (
                -channel.distance * channel.attenuation / 10.0
            )
            channel._compute_link_model()
            fs, bs = channel._compute_raman_noise_rate()
            channel.raman_noise_rate_FS_Hz = fs
            channel.raman_noise_rate_BS_Hz = bs
            channel.raman_noise_rate_Hz = fs + bs
            channel.noise_enabled = channel.raman_noise_rate_Hz > 0
            self._last_channel_state[arm] = channel_state

        factor_a = state["brightness_factor"] if affected_source == "A" else 1.0
        factor_b = state["brightness_factor"] if affected_source == "B" else 1.0
        self.source_a.set_mean_photon_num(setup.mean_photon_num * factor_a)
        self.source_b.set_mean_photon_num(setup.mean_photon_num * factor_b)
        lambda_a = setup.signal_wavelength_nm_a + (
            state["detuning_nm"] if affected_source == "A" else 0.0
        )
        lambda_b = setup.signal_wavelength_nm_b + (
            state["detuning_nm"] if affected_source == "B" else 0.0
        )
        self.source_a.spdc.set_wavelength(lambda_a, setup.sink_wavelength_nm_a)
        self.source_b.spdc.set_wavelength(lambda_b, setup.sink_wavelength_nm_b)
        self.bsm.center_wavelength_arm0_nm = lambda_a
        self.bsm.center_wavelength_arm1_nm = lambda_b

    def update_bsm_clock(self, time_s):
        state = self._state(float(time_s))
        sign = -1.0 if self.scenario.affected_source == "A" else 1.0
        self.bsm.scan_delay_ps = int(round(sign * state["clock_offset_ps"]))

    def collect_window(self, window_index):
        streams = self.bsm.drain_click_times(include_dark_counts=True)
        midpoint_s = (window_index + 0.5) * self.cfg.generation.window_duration_s
        state = self._state(midpoint_s)
        streams = _apply_sync_jitter(
            streams,
            self.scenario.affected_detector,
            state["sync_jitter_ps"],
            self.rng,
        )
        features, histograms = observable_features(
            streams,
            duration_s=self.cfg.generation.window_duration_s,
            coincidence_window_ps=self.cfg.setup.coincidence_window_ps,
            histogram_range_ps=self.cfg.features.histogram_range_ps,
            histogram_bin_width_ps=self.cfg.features.histogram_bin_width_ps,
        )
        start_s = window_index * self.cfg.generation.window_duration_s
        end_s = start_s + self.cfg.generation.window_duration_s
        active_duration_s = max(0.0, end_s - max(start_s, self.scenario.onset_s))
        active_fraction = min(
            1.0, active_duration_s / self.cfg.generation.window_duration_s
        )
        self.rows.append({
            "episode_id": self.scenario.episode_id,
            "setup_id": self.cfg.setup.setup_id,
            "fault_class": self.scenario.fault_class,
            "window_index": window_index,
            "window_start_s": start_s,
            "window_end_s": end_s,
            "window_duration_s": self.cfg.generation.window_duration_s,
            "fault_active": int(active_fraction > 0),
            "fault_active_fraction": active_fraction,
            **features,
        })
        self.truth.append({
            "episode_id": self.scenario.episode_id,
            "window_index": window_index,
            "fault_class": self.scenario.fault_class,
            "fault_onset_s": self.scenario.onset_s,
            "fault_ramp_duration_s": self.scenario.ramp_duration_s,
            "target_fault_level": self.scenario.target,
            "fault_progress": state["progress"],
            **{f"true_{key}": value for key, value in state.items() if key not in {"active", "progress"}},
        })
        if self.cfg.generation.write_raw_streams:
            self.raw.append({
                "window_index": window_index,
                "streams": streams,
                "histograms": histograms,
            })


def simulate_episode(scenario: BSMScenario, cfg: DatasetConfig):
    setup = cfg.setup
    signal_spec_a = _fiber_spec(
        setup.signal_wavelength_nm_a, setup.baseline_raman_power_mw
    )
    signal_spec_b = _fiber_spec(
        setup.signal_wavelength_nm_b, setup.baseline_raman_power_mw
    )
    propagation_guard_ps = int(2e9)
    approximate_delay_ps = int(max(setup.signal_length_m_a, setup.signal_length_m_b) / 2e8 * 1e12)
    episode_ps = int(round(cfg.generation.episode_duration_s * 1e12))
    timeline = Timeline(stop_time=episode_ps + approximate_delay_ps + propagation_guard_ps)

    common = {
        "frequency": setup.source_frequency_hz,
        "mean_photon_num": setup.mean_photon_num,
        "bandwidth": setup.source_bandwidth_nm,
        "encoding": polarization,
        "photon_statistics": setup.photon_statistics,
        "use_sparse_emission": setup.use_sparse_emission,
    }
    source_a = SpdcSourceNode("source_a", timeline, {
        **common,
        "wavelengths": [setup.signal_wavelength_nm_a, setup.sink_wavelength_nm_a],
        "bell_state": setup.bell_state_a,
    })
    source_b = SpdcSourceNode("source_b", timeline, {
        **common,
        "wavelengths": [setup.signal_wavelength_nm_b, setup.sink_wavelength_nm_b],
        "bell_state": setup.bell_state_b,
    })
    bsm = PolarizationBSMNode("polarization_bsm", timeline, {
        "source_bandwidth_nm": setup.source_bandwidth_nm,
        "source_bandwidth_arm0_nm": setup.source_bandwidth_nm,
        "source_bandwidth_arm1_nm": setup.source_bandwidth_nm,
        "center_wavelength_arm0_nm": setup.signal_wavelength_nm_a,
        "center_wavelength_arm1_nm": setup.signal_wavelength_nm_b,
        "matching_window_ps": setup.matching_window_ps,
        "coincidence_window_ps": setup.coincidence_window_ps,
        "detector_efficiency": setup.detector_efficiency,
        "detector_jitter_ps": setup.detector_jitter_ps,
        "dark_count_rate_hz": setup.detector_dark_hz,
        "pbs_fidelity": setup.pbs_fidelity,
        "pbs_mismeasure_prob": setup.pbs_mismeasure_prob,
        "unmatched_policy": "single",
    })
    sink_a = PhotonSinkNode("sink_a", timeline)
    sink_b = PhotonSinkNode("sink_b", timeline)
    channel_a = _fiber("signal_a", timeline, setup.signal_length_m_a, setup.attenuation_db_per_m, signal_spec_a)
    channel_b = _fiber("signal_b", timeline, setup.signal_length_m_b, setup.attenuation_db_per_m, signal_spec_b)
    channel_a.set_ends(source_a, bsm.name)
    channel_b.set_ends(source_b, bsm.name)
    source_a.set_direct_receiver(1, sink_a)
    source_b.set_direct_receiver(1, sink_b)
    bsm.register_input(source_a.name, 0)
    bsm.register_input(source_b.name, 1)
    for offset, entity in enumerate((source_a, source_b, bsm, sink_a, sink_b), 1):
        entity.set_seed(scenario.seed + offset)
    timeline.init()

    controller = ContinuousEpisodeController(
        timeline, cfg, scenario, source_a, source_b, channel_a, channel_b, bsm
    )
    update_step = cfg.generation.fault_update_interval_s
    update_times = set(np.arange(
        0.0, cfg.generation.episode_duration_s, update_step
    ).tolist())
    if scenario.fault_class != "healthy":
        update_times.add(scenario.onset_s)
        update_times.add(min(
            cfg.generation.episode_duration_s,
            scenario.onset_s + scenario.ramp_duration_s,
        ))
    propagation_ps = int(round(max(channel_a.base_group_delay_s, channel_b.base_group_delay_s) * 1e12))
    for time_s in sorted(update_times):
        source_time_ps = int(round(time_s * 1e12))
        timeline.schedule(Event(source_time_ps, Process(
            controller, "update_source_and_fiber", [float(time_s)]
        )))
        timeline.schedule(Event(source_time_ps + propagation_ps, Process(
            controller, "update_bsm_clock", [float(time_s)]
        )))

    total_pulses = int(round(
        cfg.generation.episode_duration_s * setup.source_frequency_hz
    ))
    _schedule_chunked_emission(
        timeline, source_a, total_pulses, setup.emission_chunk_pulses
    )
    _schedule_chunked_emission(
        timeline, source_b, total_pulses, setup.emission_chunk_pulses
    )
    for window_index in range(cfg.generation.windows_per_episode):
        boundary_s = (window_index + 1) * cfg.generation.window_duration_s
        collect_ps = int(round(boundary_s * 1e12)) + propagation_ps + 1
        timeline.schedule(Event(collect_ps, Process(
            controller, "collect_window", [window_index]
        )))
    timeline.run()
    add_deltas(controller.rows)
    return controller.rows, controller.truth, (
        controller.raw if cfg.generation.write_raw_streams else None
    )
