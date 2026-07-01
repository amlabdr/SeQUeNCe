"""Full timeline simulation of polarization entanglement swapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sequence.components.fiber_quantum_channel import FiberSection, FiberSpec, fiberQuantumChannel
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.topology.optical_nodes import PolarizationAnalyzerNode, PolarizationBSMNode, SpdcSourceNode
from sequence.utils.encoding import polarization

from example.hom_validation.simulator import _schedule_chunked_emission
from .simulator import BELL_STATES, expected_swapped_state


@dataclass(frozen=True)
class FullSwappingConfig:
    duration_s: float = 0.01
    source_frequency_hz: float = 1_000_000.0
    mean_photon_num: float = 0.08
    source_bandwidth_nm: float = 0.01
    photon_statistics: str = "thermal"
    use_sparse_emission: bool = True
    emission_chunk_pulses: int = 25_000
    source_a_state: str = "psi_minus"
    source_b_state: str = "psi_minus"
    signal_wavelength_nm: float = 1550.0
    outer_wavelength_nm: float = 1550.0
    signal_length_a_m: float = 1.0
    signal_length_b_m: float = 1.0
    outer_length_a_m: float = 100.0
    outer_length_d_m: float = 100.0
    attenuation_db_per_m: float = 0.0
    detector_efficiency_bsm: float = 1.0
    detector_efficiency_outer: float = 1.0
    detector_dark_hz_bsm: float = 0.0
    detector_dark_hz_outer: float = 0.0
    detector_jitter_ps_bsm: float = 0.0
    matching_window_ps: int = 500
    bsm_coincidence_window_ps: int = 120
    fourfold_window_ps: int = 300
    extra_overlap_scale: float = 1.0
    seed: int = 11_000
    fiber_signal_a: FiberSpec = field(default_factory=FiberSpec)
    fiber_signal_b: FiberSpec = field(default_factory=FiberSpec)
    fiber_outer_a: FiberSpec = field(default_factory=FiberSpec)
    fiber_outer_d: FiberSpec = field(default_factory=FiberSpec)

    @property
    def pulses(self) -> int:
        return max(1, int(round(self.duration_s * self.source_frequency_hz)))


def _fiber(name: str, timeline: Timeline, length_m: float, attenuation: float, spec: FiberSpec):
    return fiberQuantumChannel(
        name,
        timeline,
        attenuation=float(attenuation),
        distance=float(length_m),
        sections=[FiberSection(length_m=float(length_m), spec=spec)],
    )


def _records(analyzer: PolarizationAnalyzerNode) -> list[tuple[int, int]]:
    records = analyzer.get_detection_records()
    return sorted(
        (int(record["time"]), channel)
        for channel, channel_records in enumerate(records)
        for record in channel_records
    )


def _nearest_unused(
    records: list[tuple[int, int]],
    target_ps: int,
    window_ps: int,
    used: set[int],
) -> tuple[int, int] | None:
    best = None
    for index, (time_ps, channel) in enumerate(records):
        if index in used:
            continue
        delta = abs(time_ps - target_ps)
        if delta <= window_ps and (best is None or delta < best[0]):
            best = (delta, index, channel)
    if best is None:
        return None
    used.add(best[1])
    return best[1], best[2]


def _fourfold_counts(
    bell_events: list[dict],
    alice_records: list[tuple[int, int]],
    david_records: list[tuple[int, int]],
    offset_a_ps: int,
    offset_d_ps: int,
    window_ps: int,
) -> dict[str, np.ndarray]:
    counts = {
        "psi_minus": np.zeros((2, 2), dtype=int),
        "psi_plus": np.zeros((2, 2), dtype=int),
    }
    used_a: set[int] = set()
    used_d: set[int] = set()
    for event in sorted(bell_events, key=lambda item: item["time_ps"]):
        bell_state = event["bell_state"]
        hit_a = _nearest_unused(
            alice_records, int(event["time_ps"] + offset_a_ps), window_ps, used_a
        )
        hit_d = _nearest_unused(
            david_records, int(event["time_ps"] + offset_d_ps), window_ps, used_d
        )
        if hit_a is None or hit_d is None:
            continue
        counts[bell_state][hit_a[1], hit_d[1]] += 1
    return counts


def run_basis_setting(
    basis_a: str,
    basis_d: str,
    config: FullSwappingConfig,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run actual sources, four fibers, four-detector BSM, and outer analyzers."""
    run_seed = int(config.seed if seed is None else seed)
    propagation_guard_ps = int(2e9)
    stop_ps = int(np.ceil(config.pulses * 1e12 / config.source_frequency_hz)) + propagation_guard_ps
    timeline = Timeline(stop_time=stop_ps)

    state_to_source_label = {
        "phi_plus": "phi+",
        "phi_minus": "phi-",
        "psi_plus": "psi+",
        "psi_minus": "psi-",
    }
    source_common = {
        "wavelengths": [config.signal_wavelength_nm, config.outer_wavelength_nm],
        "frequency": config.source_frequency_hz,
        "mean_photon_num": config.mean_photon_num,
        "bandwidth": config.source_bandwidth_nm,
        "encoding": polarization,
        "photon_statistics": config.photon_statistics,
        "use_sparse_emission": config.use_sparse_emission,
    }
    source_a = SpdcSourceNode(
        "source_a", timeline,
        {**source_common, "bell_state": state_to_source_label[config.source_a_state]},
    )
    source_b = SpdcSourceNode(
        "source_b", timeline,
        {**source_common, "bell_state": state_to_source_label[config.source_b_state]},
    )
    bsm = PolarizationBSMNode(
        "polarization_bsm",
        timeline,
        {
            "source_bandwidth_nm": config.source_bandwidth_nm,
            "source_bandwidth_arm0_nm": config.source_bandwidth_nm,
            "source_bandwidth_arm1_nm": config.source_bandwidth_nm,
            "center_wavelength_arm0_nm": config.signal_wavelength_nm,
            "center_wavelength_arm1_nm": config.signal_wavelength_nm,
            "matching_window_ps": config.matching_window_ps,
            "coincidence_window_ps": config.bsm_coincidence_window_ps,
            "detector_efficiency": config.detector_efficiency_bsm,
            "detector_jitter_ps": config.detector_jitter_ps_bsm,
            "dark_count_rate_hz": config.detector_dark_hz_bsm,
            "extra_overlap_scale": config.extra_overlap_scale,
            "unmatched_policy": "single",
        },
    )
    analyzer_common = {
        "mode": "hwp_qwp",
        "qwp_fidelity": 1.0,
        "hwp_fidelity": 1.0,
        "detector_efficiency": config.detector_efficiency_outer,
        "dark_count": config.detector_dark_hz_outer,
        "pbs_fidelity": 1.0,
        "mismeasure_prob": 0.0,
    }
    alice = PolarizationAnalyzerNode("alice", timeline, {**analyzer_common, "basis": basis_a})
    david = PolarizationAnalyzerNode("david", timeline, {**analyzer_common, "basis": basis_d})

    ch_b = _fiber("fiber_b", timeline, config.signal_length_a_m, config.attenuation_db_per_m, config.fiber_signal_a)
    ch_c = _fiber("fiber_c", timeline, config.signal_length_b_m, config.attenuation_db_per_m, config.fiber_signal_b)
    ch_a = _fiber("fiber_a", timeline, config.outer_length_a_m, config.attenuation_db_per_m, config.fiber_outer_a)
    ch_d = _fiber("fiber_d", timeline, config.outer_length_d_m, config.attenuation_db_per_m, config.fiber_outer_d)

    ch_b.set_ends(source_a, bsm.name)
    ch_a.set_ends(source_a, alice.name)
    ch_c.set_ends(source_b, bsm.name)
    ch_d.set_ends(source_b, david.name)
    bsm.register_input(source_a.name, 0)
    bsm.register_input(source_b.name, 1)

    for index, entity in enumerate((source_a, source_b, bsm, alice, david), start=1):
        entity.set_seed(run_seed + index)

    timeline.init()
    bsm_reference_s = max(ch_b.base_group_delay_s, ch_c.base_group_delay_s)
    if min(ch_a.base_group_delay_s, ch_d.base_group_delay_s) <= bsm_reference_s:
        raise ValueError(
            "Outer paths must include enough optical/storage delay to reach the analyzers "
            "after the BSM projection"
        )
    _schedule_chunked_emission(timeline, source_a, config.pulses, config.emission_chunk_pulses)
    _schedule_chunked_emission(timeline, source_b, config.pulses, config.emission_chunk_pulses)
    timeline.run()

    bell_events = bsm.get_bell_events(
        coincidence_window_ps=config.bsm_coincidence_window_ps,
        include_dark_counts=True,
    )
    alice_records = _records(alice)
    david_records = _records(david)
    bsm_reference_ps = bsm_reference_s * 1e12
    offset_a_ps = int(round(ch_a.base_group_delay_s * 1e12 - bsm_reference_ps))
    offset_d_ps = int(round(ch_d.base_group_delay_s * 1e12 - bsm_reference_ps))
    counts = _fourfold_counts(
        bell_events,
        alice_records,
        david_records,
        offset_a_ps,
        offset_d_ps,
        config.fourfold_window_ps,
    )

    rows = []
    for bell_state, matrix in counts.items():
        total = int(matrix.sum())
        rows.append({
            "source_a_state": config.source_a_state,
            "source_b_state": config.source_b_state,
            "bsm_state": bell_state,
            "basis_a": basis_a,
            "basis_d": basis_d,
            "seed": run_seed,
            "duration_s": config.duration_s,
            "pulses": config.pulses,
            "source_a_pairs": source_a.emission_count,
            "source_b_pairs": source_b.emission_count,
            "bsm_bell_events": sum(e["bell_state"] == bell_state for e in bell_events),
            "fourfold_count": total,
            "count_00": int(matrix[0, 0]),
            "count_01": int(matrix[0, 1]),
            "count_10": int(matrix[1, 0]),
            "count_11": int(matrix[1, 1]),
            "correlation": float((matrix[0, 0] + matrix[1, 1] - matrix[0, 1] - matrix[1, 0]) / total) if total else np.nan,
            "expectation_a": float((matrix[0, :].sum() - matrix[1, :].sum()) / total) if total else np.nan,
            "expectation_d": float((matrix[:, 0].sum() - matrix[:, 1].sum()) / total) if total else np.nan,
            "offset_a_ps": offset_a_ps,
            "offset_d_ps": offset_d_ps,
        })
    return pd.DataFrame(rows)


def run_full_tomography(config: FullSwappingConfig) -> pd.DataFrame:
    frames = []
    for i, basis_a in enumerate("XYZ"):
        for j, basis_d in enumerate("XYZ"):
            frames.append(run_basis_setting(basis_a, basis_d, config, config.seed + 100 * i + 10 * j))
    return pd.concat(frames, ignore_index=True)


def _projectors():
    states = {
        "X": (np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)),
        "Y": (np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)),
        "Z": (np.array([1, 0]), np.array([0, 1])),
    }
    return {
        (basis_a, basis_d, a, d): np.kron(
            np.outer(states[basis_a][a], states[basis_a][a].conj()),
            np.outer(states[basis_d][d], states[basis_d][d].conj()),
        )
        for basis_a in "XYZ" for basis_d in "XYZ" for a in (0, 1) for d in (0, 1)
    }


def _rho_from_parameters(parameters: np.ndarray) -> np.ndarray:
    t = np.zeros((4, 4), dtype=complex)
    t[np.diag_indices(4)] = parameters[:4]
    cursor = 4
    for row in range(1, 4):
        for column in range(row):
            t[row, column] = parameters[cursor] + 1j * parameters[cursor + 1]
            cursor += 2
    rho = t @ t.conj().T
    return rho / np.trace(rho)


def maximum_likelihood_tomography(rows: pd.DataFrame, bsm_state: str) -> tuple[np.ndarray, dict]:
    """Physical two-qubit MLE from conditional fourfold counts."""
    selected = rows[rows["bsm_state"] == bsm_state]
    if len(selected) != 9 or selected["fourfold_count"].min() <= 0:
        raise ValueError(f"Need nonzero fourfolds for all nine settings of {bsm_state}")
    projectors = _projectors()

    observations = []
    for row in selected.itertuples():
        for a in (0, 1):
            for d in (0, 1):
                observations.append((
                    int(getattr(row, f"count_{a}{d}")),
                    projectors[(row.basis_a, row.basis_d, a, d)],
                ))

    def objective(parameters):
        rho = _rho_from_parameters(parameters)
        return -sum(count * np.log(max(float(np.real(np.trace(rho @ projector))), 1e-12))
                    for count, projector in observations)

    initial = np.zeros(16)
    initial[:4] = 0.5
    result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": 3000})
    rho = _rho_from_parameters(result.x)
    expected_label, expected_ket = expected_swapped_state(
        str(selected.iloc[0]["source_a_state"]),
        str(selected.iloc[0]["source_b_state"]),
        bsm_state,
    )
    return rho, {
        "bsm_state": bsm_state,
        "expected_outer_state": expected_label,
        "fidelity": float(np.real(np.vdot(expected_ket, rho @ expected_ket))),
        "trace": float(np.real(np.trace(rho))),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(rho).min()),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }
