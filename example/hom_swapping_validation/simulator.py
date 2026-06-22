from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.components.wave_plate import WavePlate
from sequence.components.fiber_quantum_channel import (
    FiberSection,
    FiberSpec,
    fiberQuantumChannel,
)
from sequence.topology.optical_nodes import HOMInterferenceNode, PolarizationAnalyzerNode, SpdcSourceNode
from sequence.topology.node import Node
from sequence.utils.encoding import polarization


@dataclass
class CoincidenceConfig:
    window_ps: int
    offset_ps: Optional[int] = 0


@dataclass
class SequenceHOMConfig:
    pulses_per_delay: int = 120_000
    emission_chunk_pulses: int = 25_000
    source_frequency_hz: float = 8e7
    mean_photon_num: float = 0.12
    mean_photon_num_a: Optional[float] = None
    mean_photon_num_b: Optional[float] = None
    source_bandwidth_nm: float = 0.6
    wavelengths_nm_a: Tuple[float, float] = (1550.0, 1550.0)
    wavelengths_nm_b: Tuple[float, float] = (1550.0, 1550.0)
    photon_statistics: str = "thermal"
    use_sparse_emission: bool = False
    source_bell_state_a: str = "psi-"
    source_bell_state_b: str = "psi-"
    arm_length_m_a: float = 20_000.0
    arm_length_m_b: float = 20_000.0
    herald_length_m_a: float = 1.0
    herald_length_m_b: float = 1.0
    detector_eff_bsm: float = 0.9
    detector_eff_herald: float = 0.9
    detector_jitter_ps: float = 20.0
    detector_dark_hz_bsm: float = 0.0
    detector_dark_hz_herald: float = 0.0
    hom_match_window_ps: int = 120
    hom_coinc_window_ps: int = 120
    herald_mode: str = "projected"
    herald_basis: str = "Z"
    herald_channel_a: int = 0
    herald_channel_b: int = 0
    extra_overlap_scale: float = 1.0
    attenuation_db_per_m: float = 0.00002
    attenuation_db_per_m_signal_a: Optional[float] = None
    attenuation_db_per_m_signal_b: Optional[float] = None
    attenuation_db_per_m_herald_a: Optional[float] = None
    attenuation_db_per_m_herald_b: Optional[float] = None
    fiber_spec_signal_a: FiberSpec = field(default_factory=FiberSpec)
    fiber_spec_signal_b: FiberSpec = field(default_factory=FiberSpec)
    fiber_spec_herald_a: FiberSpec = field(default_factory=FiberSpec)
    fiber_spec_herald_b: FiberSpec = field(default_factory=FiberSpec)
    seed: int = 1234
    pol_rotation_arm0_rad: float = 0.0
    pol_rotation_arm1_rad: float = 0.0
    use_signal_rotators: bool = True


def _sorted_array(values: Sequence[int], assume_sorted: bool = False) -> np.ndarray:
    if len(values) == 0:
        return np.asarray([], dtype=np.int64)
    arr = np.asarray(values, dtype=np.int64)
    if not assume_sorted:
        arr.sort()
    elif arr.size > 1 and np.any(arr[1:] < arr[:-1]):
        # Fallback safety: if caller claimed sorted but data is not monotonic.
        arr = np.sort(arr)
    return arr


def pairwise_coincidences(
    t_a_ps: Sequence[int],
    t_b_ps: Sequence[int],
    window_ps: int,
    offset_ps: int = 0,
    assume_sorted: bool = True,
) -> List[Tuple[int, int]]:
    """Greedy two-stream coincidence matching.

    A match is valid when abs((t_b - t_a) - offset_ps) <= window_ps.
    Returns a list of matched timestamp pairs.
    """
    a = _sorted_array(t_a_ps, assume_sorted=assume_sorted)
    b = _sorted_array(t_b_ps, assume_sorted=assume_sorted)
    pairs: List[Tuple[int, int]] = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        delta = int(b[j] - a[i] - offset_ps)
        if abs(delta) <= window_ps:
            pairs.append((int(a[i]), int(b[j])))
            i += 1
            j += 1
        elif delta < -window_ps:
            j += 1
        else:
            i += 1
    return pairs


def event_has_hit(
    stream_ps: Sequence[int],
    ref_time_ps: int,
    offset_ps: int,
    window_ps: int,
    assume_sorted: bool = True,
) -> bool:
    """Check if there exists a timestamp near ref_time_ps + offset_ps."""
    if len(stream_ps) == 0:
        return False
    target = int(ref_time_ps + offset_ps)
    arr = _sorted_array(stream_ps, assume_sorted=assume_sorted)
    idx = int(np.searchsorted(arr, target))
    left = max(0, idx - 2)
    right = min(len(arr), idx + 3)
    for k in range(left, right):
        if abs(int(arr[k]) - target) <= window_ps:
            return True
    return False


def bsm_twofold_events(
    bsm1_ps: Sequence[int],
    bsm2_ps: Sequence[int],
    cfg: CoincidenceConfig,
    assume_sorted: bool = True,
) -> List[Tuple[int, int, int]]:
    """Return matched BSM twofold events as (t1, t2, t_ref_midpoint)."""
    pairs = pairwise_coincidences(
        bsm1_ps,
        bsm2_ps,
        cfg.window_ps,
        cfg.offset_ps,
        assume_sorted=assume_sorted,
    )
    events = []
    for t1, t2 in pairs:
        events.append((t1, t2, int((t1 + t2) // 2)))
    return events


def fourfold_count_from_bsm(
    bsm_events: Sequence[Tuple[int, int, int]],
    stream_a_ps: Sequence[int],
    stream_b_ps: Sequence[int],
    rel_a: CoincidenceConfig,
    rel_b: CoincidenceConfig,
    assume_sorted: bool = True,
) -> int:
    """Count fourfold events by validating extra detector hits around BSM events."""
    total = 0
    for _, _, tref in bsm_events:
        has_a = event_has_hit(
            stream_a_ps, tref, rel_a.offset_ps, rel_a.window_ps, assume_sorted=assume_sorted
        )
        has_b = event_has_hit(
            stream_b_ps, tref, rel_b.offset_ps, rel_b.window_ps, assume_sorted=assume_sorted
        )
        if has_a and has_b:
            total += 1
    return total


def _records_to_pair_channel(records: Sequence[Sequence[dict]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for channel, channel_records in enumerate(records):
        for record in channel_records:
            pair_id = record.get("pair_id")
            if pair_id is not None:
                out[str(pair_id)] = int(channel)
    return out


@dataclass
class HeraldReadout:
    """Polarization-resolved herald detector output.

    For the Z-basis validation, channel 0 is H and channel 1 is V. For other
    analyzer bases, these fields still mean detector channel 0/1.
    """

    h_times: List[int]
    v_times: List[int]
    all_times: List[int]
    pair_channel: Dict[str, int]


def _read_herald_analyzer(analyzer: PolarizationAnalyzerNode) -> HeraldReadout:
    """Read timestamps and pair-id projection metadata from one herald analyzer."""
    channel_times = analyzer.get_photon_times()
    records = analyzer.get_detection_records()
    h_times, v_times = channel_times
    return HeraldReadout(
        h_times=h_times,
        v_times=v_times,
        all_times=sorted(h_times + v_times),
        pair_channel=_records_to_pair_channel(records),
    )


def _schedule_chunked_emission(timeline: Timeline, source: SpdcSourceNode, total_pulses: int, chunk_pulses: int) -> None:
    """Schedule SPDC emission in chunks instead of creating all photons at t=0."""
    total_pulses = int(total_pulses)
    chunk_pulses = max(1, int(chunk_pulses))
    period_ps = int(round(1e12 / float(source.spdc.frequency)))

    pulse_start = 0
    while pulse_start < total_pulses:
        pulses = min(chunk_pulses, total_pulses - pulse_start)
        start_time_ps = int(pulse_start * period_ps)
        timeline.schedule(Event(start_time_ps, Process(source, "emit", [int(pulses)])))
        pulse_start += pulses


class _WavePlateRelayPort:
    def __init__(self, owner: "_WavePlateRelayNode"):
        self.owner = owner

    def get(self, photon, **kwargs):
        self.owner.forward_photon(photon)


class _WavePlateRelayNode(Node):
    """Inline polarization rotator node using an actual WavePlate component."""

    def __init__(self, name: str, timeline: Timeline, plate_type: str = "HWP", angle_rad: float = 0.0):
        super().__init__(name, timeline)
        self.waveplate = WavePlate(
            f"{name}.waveplate",
            timeline,
            plate_type=plate_type,
            angle=float(angle_rad),
            fidelity=1.0,
            encoding_type=polarization,
        )
        self.add_component(self.waveplate)
        self.set_first_component(self.waveplate.name)
        self._port = _WavePlateRelayPort(self)
        self.waveplate.add_receiver(self._port)
        self._output_node = None

    def set_angle(self, angle_rad: float) -> None:
        self.waveplate.set_angle(float(angle_rad))

    def set_output_node(self, node: Node) -> None:
        self._output_node = node

    def forward_photon(self, photon) -> None:
        if self._output_node is not None:
            self._output_node.receive_qubit(self.name, photon)
            return
        if self.qchannels:
            dst = next(iter(self.qchannels.keys()))
            self.send_qubit(dst, photon)


class _PhotonSinkNode(Node):
    """Passive idler sink used when the run is BSM-only.

    The idler photons still travel through the configured fiber arms, but they
    are not analyzed or detected. This avoids state projection and prevents any
    fourfold/herald data from entering the operational output.
    """

    def __init__(self, name: str, timeline: Timeline):
        super().__init__(name, timeline)
        self.received = 0

    def receive_qubit(self, src: str, qubit) -> None:
        self.received += 1

    def get(self, photon, **kwargs) -> None:
        self.received += 1


def _build_fiber_channel(
    name: str,
    timeline: Timeline,
    distance_m: float,
    attenuation_db_per_m: float,
    spec: FiberSpec,
) -> fiberQuantumChannel:
    section = FiberSection(length_m=float(distance_m), spec=spec)
    return fiberQuantumChannel(
        name=name,
        timeline=timeline,
        attenuation=float(attenuation_db_per_m),
        distance=float(distance_m),
        sections=[section],
    )


def run_hom_sequence_delay_scan(
    delays_ps: Sequence[int],
    cfg: SequenceHOMConfig = SequenceHOMConfig(),
    bsm_cfg: CoincidenceConfig = CoincidenceConfig(window_ps=120, offset_ps=0),
    herald_a_rel: CoincidenceConfig = CoincidenceConfig(window_ps=300, offset_ps=0),
    herald_b_rel: CoincidenceConfig = CoincidenceConfig(window_ps=300, offset_ps=0),
    store_streams: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, np.ndarray]]]:
    """Full SeQUeNCe timeline simulation for HOM validation.

    Uses actual nodes/channels:
    - SpdcSourceNode A/B (thermal/poisson pair generation + sampled wavelengths)
    - fiberQuantumChannel on all 4 arms
    - HOMInterferenceNode at middle BSM
    - PolarizationAnalyzerNode on local idler arms for state-projecting heralds,
      or passive idler sinks when cfg.herald_mode == "bsm_only"
    """
    rows: List[Dict[str, float]] = []
    streams: Dict[int, Dict[str, np.ndarray]] = {}
    herald_mode = str(cfg.herald_mode).lower()
    if herald_mode not in {"projected", "bsm_only"}:
        raise ValueError("cfg.herald_mode must be 'projected' or 'bsm_only'")

    for idx, delay_ps in enumerate(delays_ps):
        stop_ps = int(np.ceil(cfg.pulses_per_delay * (1e12 / cfg.source_frequency_hz) + 5e9))
        timeline = Timeline(stop_time=stop_ps)

        source_a = SpdcSourceNode(
            "source_a",
            timeline,
            {
                "wavelengths": list(cfg.wavelengths_nm_a),
                "frequency": float(cfg.source_frequency_hz),
                "mean_photon_num": float(
                    cfg.mean_photon_num if cfg.mean_photon_num_a is None else cfg.mean_photon_num_a
                ),
                "bandwidth": float(cfg.source_bandwidth_nm),
                "encoding": polarization,
                "photon_statistics": cfg.photon_statistics,
                "use_sparse_emission": bool(cfg.use_sparse_emission),
                "bell_state": str(cfg.source_bell_state_a),
            },
        )
        source_b = SpdcSourceNode(
            "source_b",
            timeline,
            {
                "wavelengths": list(cfg.wavelengths_nm_b),
                "frequency": float(cfg.source_frequency_hz),
                "mean_photon_num": float(
                    cfg.mean_photon_num if cfg.mean_photon_num_b is None else cfg.mean_photon_num_b
                ),
                "bandwidth": float(cfg.source_bandwidth_nm),
                "encoding": polarization,
                "photon_statistics": cfg.photon_statistics,
                "use_sparse_emission": bool(cfg.use_sparse_emission),
                "bell_state": str(cfg.source_bell_state_b),
            },
        )

        hom = HOMInterferenceNode(
            "bsm_hom",
            timeline,
            {
                "scan_delay_ps": int(delay_ps),
                "delayed_arm": 1,
                "source_bandwidth_nm": float(cfg.source_bandwidth_nm),
                "center_wavelength_arm0_nm": float(cfg.wavelengths_nm_a[0]),
                "center_wavelength_arm1_nm": float(cfg.wavelengths_nm_b[0]),
                "matching_window_ps": int(cfg.hom_match_window_ps),
                "coincidence_window_ps": int(cfg.hom_coinc_window_ps),
                "detector_efficiency": float(cfg.detector_eff_bsm),
                "detector_jitter_ps": float(cfg.detector_jitter_ps),
                "dark_count_rate_hz": float(cfg.detector_dark_hz_bsm),
                "extra_overlap_scale": float(cfg.extra_overlap_scale),
                "unmatched_policy": "single",
            },
        )
        if herald_mode == "projected":
            herald_a = PolarizationAnalyzerNode(
                "herald_a",
                timeline,
                {
                    "mode": "hwp_qwp",
                    "basis": str(cfg.herald_basis),
                    "detector_efficiency": float(cfg.detector_eff_herald),
                    "dark_count": float(cfg.detector_dark_hz_herald),
                    "pbs_fidelity": 1.0,
                    "mismeasure_prob": 0.0,
                },
            )
            herald_b = PolarizationAnalyzerNode(
                "herald_b",
                timeline,
                {
                    "mode": "hwp_qwp",
                    "basis": str(cfg.herald_basis),
                    "detector_efficiency": float(cfg.detector_eff_herald),
                    "dark_count": float(cfg.detector_dark_hz_herald),
                    "pbs_fidelity": 1.0,
                    "mismeasure_prob": 0.0,
                },
            )
        else:
            herald_a = _PhotonSinkNode("herald_a_sink", timeline)
            herald_b = _PhotonSinkNode("herald_b_sink", timeline)

        use_signal_rotators = bool(cfg.use_signal_rotators)
        if use_signal_rotators:
            rot_a = _WavePlateRelayNode(
                "rot_a",
                timeline,
                plate_type="HWP",
                angle_rad=float(cfg.pol_rotation_arm0_rad) / 2.0,
            )
            rot_b = _WavePlateRelayNode(
                "rot_b",
                timeline,
                plate_type="HWP",
                angle_rad=float(cfg.pol_rotation_arm1_rad) / 2.0,
            )

        ch_sig_a = _build_fiber_channel(
            "qc_sig_a",
            timeline,
            cfg.arm_length_m_a,
            cfg.attenuation_db_per_m
            if cfg.attenuation_db_per_m_signal_a is None
            else cfg.attenuation_db_per_m_signal_a,
            cfg.fiber_spec_signal_a,
        )
        ch_sig_b = _build_fiber_channel(
            "qc_sig_b",
            timeline,
            cfg.arm_length_m_b,
            cfg.attenuation_db_per_m
            if cfg.attenuation_db_per_m_signal_b is None
            else cfg.attenuation_db_per_m_signal_b,
            cfg.fiber_spec_signal_b,
        )
        ch_herald_a = _build_fiber_channel(
            "qc_herald_a",
            timeline,
            cfg.herald_length_m_a,
            cfg.attenuation_db_per_m
            if cfg.attenuation_db_per_m_herald_a is None
            else cfg.attenuation_db_per_m_herald_a,
            cfg.fiber_spec_herald_a,
        )
        ch_herald_b = _build_fiber_channel(
            "qc_herald_b",
            timeline,
            cfg.herald_length_m_b,
            cfg.attenuation_db_per_m
            if cfg.attenuation_db_per_m_herald_b is None
            else cfg.attenuation_db_per_m_herald_b,
            cfg.fiber_spec_herald_b,
        )

        # signal = port0 -> one fiber -> optional waveplate relay -> BSM.
        # idler = port1 -> local herald analyzer.
        if use_signal_rotators:
            ch_sig_a.set_ends(source_a, rot_a.name)
            ch_sig_b.set_ends(source_b, rot_b.name)
            rot_a.set_output_node(hom)
            rot_b.set_output_node(hom)
            hom.register_input(rot_a.name, 0)
            hom.register_input(rot_b.name, 1)
        else:
            ch_sig_a.set_ends(source_a, hom.name)
            ch_sig_b.set_ends(source_b, hom.name)
            hom.register_input(source_a.name, 0)
            hom.register_input(source_b.name, 1)
        ch_herald_a.set_ends(source_a, herald_a.name)
        ch_herald_b.set_ends(source_b, herald_b.name)

        base_seed = int(cfg.seed + 97 * idx)
        source_a.set_seed(base_seed + 1)
        source_b.set_seed(base_seed + 2)
        hom.set_seed(base_seed + 3)
        if herald_mode == "projected":
            herald_a.set_seed(base_seed + 4)
            herald_b.set_seed(base_seed + 5)

        timeline.init()
        signal_a_delay_ps = int(round(ch_sig_a.base_group_delay_s * 1e12))
        signal_b_delay_ps = int(round(ch_sig_b.base_group_delay_s * 1e12))
        herald_a_delay_ps = int(round(ch_herald_a.base_group_delay_s * 1e12))
        herald_b_delay_ps = int(round(ch_herald_b.base_group_delay_s * 1e12))
        signal_a_ref_delay_ps = signal_a_delay_ps + (int(delay_ps) if hom.delayed_arm == 0 else 0)
        signal_b_ref_delay_ps = signal_b_delay_ps + (int(delay_ps) if hom.delayed_arm == 1 else 0)
        bsm_ref_delay_ps = max(signal_a_ref_delay_ps, signal_b_ref_delay_ps)
        rel_a = CoincidenceConfig(
            window_ps=int(herald_a_rel.window_ps),
            offset_ps=(
                herald_a_delay_ps - bsm_ref_delay_ps
                if herald_a_rel.offset_ps is None
                else int(herald_a_rel.offset_ps)
            ),
        )
        rel_b = CoincidenceConfig(
            window_ps=int(herald_b_rel.window_ps),
            offset_ps=(
                herald_b_delay_ps - bsm_ref_delay_ps
                if herald_b_rel.offset_ps is None
                else int(herald_b_rel.offset_ps)
            ),
        )
        _schedule_chunked_emission(timeline, source_a, int(cfg.pulses_per_delay), int(cfg.emission_chunk_pulses))
        _schedule_chunked_emission(timeline, source_b, int(cfg.pulses_per_delay), int(cfg.emission_chunk_pulses))
        timeline.run()

        bsm0, bsm1 = hom.get_click_times(flush=True, include_dark_counts=True, reset=False)
        if herald_mode == "projected":
            herald_a_readout = _read_herald_analyzer(herald_a)
            herald_b_readout = _read_herald_analyzer(herald_b)
        else:
            herald_a_readout = HeraldReadout([], [], [], {})
            herald_b_readout = HeraldReadout([], [], [], {})

        # Raman/background noise must be injected in-timeline by the fiber channels,
        # not post-processed onto detector streams. If enabled in fiber specs
        # (classical coexistence + power), those events arrive through
        # HOMInterferenceNode.receive_noise_photon during timeline.run().

        bsm_events = bsm_twofold_events(bsm0, bsm1, bsm_cfg, assume_sorted=True)
        if herald_mode == "bsm_only":
            rows.append(
                {
                    "delay_ps": int(delay_ps),
                    "bsm1_count": int(len(bsm0)),
                    "bsm2_count": int(len(bsm1)),
                    "bsm_twofold_count": int(len(bsm_events)),
                    "bsm_twofold_rate_per_pulse": float(len(bsm_events) / max(1, cfg.pulses_per_delay)),
                    "fiber_dcd_a_ps_per_nm_km": float(getattr(ch_sig_a, "DCD_ps_per_nm_km", 0.0)),
                    "fiber_dcd_b_ps_per_nm_km": float(getattr(ch_sig_b, "DCD_ps_per_nm_km", 0.0)),
                    "raman_noise_rate_total_hz": float(
                        getattr(ch_sig_a, "raman_noise_rate_Hz", 0.0)
                        + getattr(ch_sig_b, "raman_noise_rate_Hz", 0.0)
                    ),
                    "herald_mode": herald_mode,
                }
            )
            if store_streams:
                streams[int(delay_ps)] = {
                    "BSM1_ps": _sorted_array(bsm0, assume_sorted=True),
                    "BSM2_ps": _sorted_array(bsm1, assume_sorted=True),
                }
            continue

        if herald_mode == "projected":
            hom_fourfold = fourfold_count_from_bsm(
                bsm_events,
                herald_a_readout.all_times,
                herald_b_readout.all_times,
                rel_a,
                rel_b,
                assume_sorted=True,
            )
            selected_fourfold = fourfold_count_from_bsm(
                bsm_events,
                [herald_a_readout.h_times, herald_a_readout.v_times][int(cfg.herald_channel_a)],
                [herald_b_readout.h_times, herald_b_readout.v_times][int(cfg.herald_channel_b)],
                rel_a,
                rel_b,
                assume_sorted=True,
            )
            hh_fourfold = fourfold_count_from_bsm(
                bsm_events, herald_a_readout.h_times, herald_b_readout.h_times, rel_a, rel_b, assume_sorted=True
            )
            hv_fourfold = fourfold_count_from_bsm(
                bsm_events, herald_a_readout.h_times, herald_b_readout.v_times, rel_a, rel_b, assume_sorted=True
            )
            vh_fourfold = fourfold_count_from_bsm(
                bsm_events, herald_a_readout.v_times, herald_b_readout.h_times, rel_a, rel_b, assume_sorted=True
            )
            vv_fourfold = fourfold_count_from_bsm(
                bsm_events, herald_a_readout.v_times, herald_b_readout.v_times, rel_a, rel_b, assume_sorted=True
            )
        else:
            hom_fourfold = 0
            selected_fourfold = 0
            hh_fourfold = 0
            hv_fourfold = 0
            vh_fourfold = 0
            vv_fourfold = 0
        matched_fourfold = hh_fourfold + vv_fourfold
        mismatched_fourfold = hv_fourfold + vh_fourfold

        matched_projected_events = [
            e for e in hom.interference_events if float(e.get("polarization_overlap", 0.0)) > 0.5
        ]
        mismatched_projected_events = [
            e for e in hom.interference_events if float(e.get("polarization_overlap", 0.0)) <= 0.5
        ]
        matched_projected_coinc = sum(1 for e in matched_projected_events if e.get("outputs") == [0, 1])
        mismatched_projected_coinc = sum(1 for e in mismatched_projected_events if e.get("outputs") == [0, 1])
        channel_events: Dict[Tuple[int, int], List[dict]] = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
        for e in hom.interference_events:
            ch_a = herald_a_readout.pair_channel.get(str(e.get("pair_id_arm0")))
            ch_b = herald_b_readout.pair_channel.get(str(e.get("pair_id_arm1")))
            if ch_a in (0, 1) and ch_b in (0, 1):
                channel_events[(int(ch_a), int(ch_b))].append(e)

        hh_channel_events = channel_events[(0, 0)]
        hv_channel_events = channel_events[(0, 1)]
        vh_channel_events = channel_events[(1, 0)]
        vv_channel_events = channel_events[(1, 1)]
        matched_channel_events = hh_channel_events + vv_channel_events
        mismatched_channel_events = hv_channel_events + vh_channel_events

        def _mean_event_value(events: Sequence[dict], key: str) -> float:
            return float(np.mean([float(e[key]) for e in events])) if events else 0.0

        def _coinc_prob(events: Sequence[dict]) -> float:
            return float(sum(1 for e in events if e.get("outputs") == [0, 1]) / max(1, len(events)))

        def _coinc_count(events: Sequence[dict]) -> int:
            return int(sum(1 for e in events if e.get("outputs") == [0, 1]))

        rows.append(
            {
                "delay_ps": int(delay_ps),
                "bsm1_count": int(len(bsm0)),
                "bsm2_count": int(len(bsm1)),
                "heraldA_count": int(len(herald_a_readout.all_times)),
                "heraldB_count": int(len(herald_b_readout.all_times)),
                "heraldA_0_count": int(len(herald_a_readout.h_times)),
                "heraldA_1_count": int(len(herald_a_readout.v_times)),
                "heraldB_0_count": int(len(herald_b_readout.h_times)),
                "heraldB_1_count": int(len(herald_b_readout.v_times)),
                "bsm_twofold_count": int(len(bsm_events)),
                "hom_fourfold_count": int(hom_fourfold),
                "selected_herald_fourfold_count": int(selected_fourfold),
                "hh_herald_fourfold_count": int(hh_fourfold),
                "hv_herald_fourfold_count": int(hv_fourfold),
                "vh_herald_fourfold_count": int(vh_fourfold),
                "vv_herald_fourfold_count": int(vv_fourfold),
                "matched_herald_fourfold_count": int(matched_fourfold),
                "mismatched_herald_fourfold_count": int(mismatched_fourfold),
                "matched_projected_interfering_pairs": int(len(matched_projected_events)),
                "mismatched_projected_interfering_pairs": int(len(mismatched_projected_events)),
                "matched_projected_coincidence_count": int(matched_projected_coinc),
                "mismatched_projected_coincidence_count": int(mismatched_projected_coinc),
                "hh_channel_projected_pairs": int(len(hh_channel_events)),
                "hv_channel_projected_pairs": int(len(hv_channel_events)),
                "vh_channel_projected_pairs": int(len(vh_channel_events)),
                "vv_channel_projected_pairs": int(len(vv_channel_events)),
                "matched_channel_projected_pairs": int(len(matched_channel_events)),
                "mismatched_channel_projected_pairs": int(len(mismatched_channel_events)),
                "hh_channel_mean_polarization_overlap": _mean_event_value(hh_channel_events, "polarization_overlap"),
                "hv_channel_mean_polarization_overlap": _mean_event_value(hv_channel_events, "polarization_overlap"),
                "vh_channel_mean_polarization_overlap": _mean_event_value(vh_channel_events, "polarization_overlap"),
                "vv_channel_mean_polarization_overlap": _mean_event_value(vv_channel_events, "polarization_overlap"),
                "matched_channel_mean_polarization_overlap": _mean_event_value(matched_channel_events, "polarization_overlap"),
                "mismatched_channel_mean_polarization_overlap": _mean_event_value(mismatched_channel_events, "polarization_overlap"),
                "matched_channel_projected_coincidence_count": _coinc_count(matched_channel_events),
                "mismatched_channel_projected_coincidence_count": _coinc_count(mismatched_channel_events),
                "bsm_twofold_rate_per_pulse": float(len(bsm_events) / max(1, cfg.pulses_per_delay)),
                "hom_fourfold_rate_per_pulse": float(hom_fourfold / max(1, cfg.pulses_per_delay)),
                "selected_herald_fourfold_rate_per_pulse": float(selected_fourfold / max(1, cfg.pulses_per_delay)),
                "matched_herald_fourfold_rate_per_pulse": float(matched_fourfold / max(1, cfg.pulses_per_delay)),
                "mismatched_herald_fourfold_rate_per_pulse": float(mismatched_fourfold / max(1, cfg.pulses_per_delay)),
                "matched_projected_coincidence_rate_per_pulse": float(matched_projected_coinc / max(1, cfg.pulses_per_delay)),
                "mismatched_projected_coincidence_rate_per_pulse": float(mismatched_projected_coinc / max(1, cfg.pulses_per_delay)),
                "matched_projected_coincidence_prob": float(
                    matched_projected_coinc / max(1, len(matched_projected_events))
                ),
                "mismatched_projected_coincidence_prob": float(
                    mismatched_projected_coinc / max(1, len(mismatched_projected_events))
                ),
                "matched_channel_projected_coincidence_prob": _coinc_prob(matched_channel_events),
                "mismatched_channel_projected_coincidence_prob": _coinc_prob(mismatched_channel_events),
                "mean_total_overlap": float(np.mean([e["total_overlap"] for e in hom.interference_events])) if hom.interference_events else 0.0,
                "mean_temporal_overlap": float(np.mean([e["temporal_overlap"] for e in hom.interference_events])) if hom.interference_events else 0.0,
                "mean_spectral_overlap": float(np.mean([e["spectral_overlap"] for e in hom.interference_events])) if hom.interference_events else 0.0,
                "mean_polarization_overlap": float(np.mean([e["polarization_overlap"] for e in hom.interference_events])) if hom.interference_events else 0.0,
                "interfering_pairs": int(len(hom.interference_events)),
                "interfering_coincidence_count": int(
                    sum(1 for e in hom.interference_events if e.get("outputs") == [0, 1])
                ),
                "interfering_bunch_count": int(
                    sum(1 for e in hom.interference_events if e.get("outputs") in ([0, 0], [1, 1]))
                ),
                "interfering_coincidence_rate_per_pulse": float(
                    sum(1 for e in hom.interference_events if e.get("outputs") == [0, 1])
                    / max(1, cfg.pulses_per_delay)
                ),
                "interfering_coincidence_prob": float(
                    sum(1 for e in hom.interference_events if e.get("outputs") == [0, 1])
                    / max(1, len(hom.interference_events))
                ),
                "fiber_dcd_a_ps_per_nm_km": float(getattr(ch_sig_a, "DCD_ps_per_nm_km", 0.0)),
                "fiber_dcd_b_ps_per_nm_km": float(getattr(ch_sig_b, "DCD_ps_per_nm_km", 0.0)),
                "raman_noise_rate_total_hz": float(getattr(ch_sig_a, "raman_noise_rate_Hz", 0.0) + getattr(ch_sig_b, "raman_noise_rate_Hz", 0.0)),
                "heraldA_offset_ps_used": int(rel_a.offset_ps),
                "heraldB_offset_ps_used": int(rel_b.offset_ps),
                "herald_mode": herald_mode,
                "herald_basis": str(cfg.herald_basis),
                "herald_channel_a": int(cfg.herald_channel_a),
                "herald_channel_b": int(cfg.herald_channel_b),
            }
        )

        if store_streams:
            streams[int(delay_ps)] = {
                "BSM1_ps": _sorted_array(bsm0, assume_sorted=True),
                "BSM2_ps": _sorted_array(bsm1, assume_sorted=True),
                "HeraldA_ps": _sorted_array(herald_a_readout.all_times, assume_sorted=True),
                "HeraldB_ps": _sorted_array(herald_b_readout.all_times, assume_sorted=True),
                "HeraldA_0_ps": _sorted_array(herald_a_readout.h_times, assume_sorted=True),
                "HeraldA_1_ps": _sorted_array(herald_a_readout.v_times, assume_sorted=True),
                "HeraldB_0_ps": _sorted_array(herald_b_readout.h_times, assume_sorted=True),
                "HeraldB_1_ps": _sorted_array(herald_b_readout.v_times, assume_sorted=True),
            }

    df = pd.DataFrame(rows).sort_values("delay_ps").reset_index(drop=True)
    return df, streams


def run_hom_sequence_polarization_scan(
    angles_rad: Sequence[float],
    cfg: SequenceHOMConfig = SequenceHOMConfig(),
    bsm_cfg: CoincidenceConfig = CoincidenceConfig(window_ps=120, offset_ps=0),
    herald_a_rel: CoincidenceConfig = CoincidenceConfig(window_ps=300, offset_ps=0),
    herald_b_rel: CoincidenceConfig = CoincidenceConfig(window_ps=300, offset_ps=0),
    rotate_arm: int = 1,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for i, angle in enumerate(angles_rad):
        cfg_i = SequenceHOMConfig(**{**cfg.__dict__})
        if int(rotate_arm) == 0:
            cfg_i.pol_rotation_arm0_rad = float(angle)
            cfg_i.pol_rotation_arm1_rad = 0.0
        else:
            cfg_i.pol_rotation_arm0_rad = 0.0
            cfg_i.pol_rotation_arm1_rad = float(angle)
        cfg_i.seed = int(cfg.seed + 131 * i)
        df, _ = run_hom_sequence_delay_scan(
            delays_ps=[0],
            cfg=cfg_i,
            bsm_cfg=bsm_cfg,
            herald_a_rel=herald_a_rel,
            herald_b_rel=herald_b_rel,
            store_streams=False,
        )
        r = df.iloc[0].to_dict()
        r["angle_rad"] = float(angle)
        r["angle_deg"] = float(np.degrees(angle))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("angle_rad").reset_index(drop=True)
