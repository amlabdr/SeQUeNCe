"""End-to-end polarization entanglement-swapping validation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from sequence.components.light_source import SPDCBellSource
from sequence.components.photon import Photon
from sequence.kernel.timeline import Timeline
from sequence.topology.optical_nodes import PolarizationAnalyzerNode, PolarizationBSMNode
from sequence.utils.encoding import polarization


BELL_STATES = {
    "phi_plus": np.asarray(SPDCBellSource.bell_state_map["phi+"], dtype=complex),
    "phi_minus": np.asarray(SPDCBellSource.bell_state_map["phi-"], dtype=complex),
    "psi_plus": np.asarray(SPDCBellSource.bell_state_map["psi+"], dtype=complex),
    "psi_minus": np.asarray(SPDCBellSource.bell_state_map["psi-"], dtype=complex),
}


@dataclass(frozen=True)
class SwappingValidationConfig:
    accepted_events_per_basis: int = 500
    max_attempt_factor: int = 10
    wavelength_nm: float = 1550.0
    bandwidth_nm: float = 0.0
    matching_window_ps: int = 100
    coincidence_window_ps: int = 100
    seed: int = 73_001


def expected_swapped_state(
    source_a_state: str,
    source_b_state: str,
    bsm_state: str,
) -> tuple[str, np.ndarray]:
    """Project an ideal four-qubit source state and identify the outer Bell state."""
    source_a = BELL_STATES[source_a_state]
    source_b = BELL_STATES[source_b_state]
    projector_bra = BELL_STATES[bsm_state].conj()
    state_abcd = np.kron(source_a, source_b).reshape(2, 2, 2, 2)
    projected_ad = np.einsum("bc,abcd->ad", projector_bra.reshape(2, 2), state_abcd).reshape(4)
    norm = float(np.linalg.norm(projected_ad))
    if norm <= 0:
        raise ValueError("Bell projection has zero probability")
    projected_ad /= norm

    fidelities = {
        label: float(abs(np.vdot(state, projected_ad)) ** 2)
        for label, state in BELL_STATES.items()
    }
    label = max(fidelities, key=fidelities.get)
    if fidelities[label] < 1 - 1e-10:
        raise RuntimeError("Projected outer state is not a Bell state")
    return label, projected_ad


def _new_pair(timeline: Timeline, prefix: str, bell_state: str, wavelength_nm: float):
    outer = Photon(
        f"{prefix}_outer",
        timeline,
        wavelength=wavelength_nm,
        encoding_type=polarization,
    )
    signal = Photon(
        f"{prefix}_signal",
        timeline,
        wavelength=wavelength_nm,
        encoding_type=polarization,
    )
    outer.combine_state(signal)
    outer.set_state(tuple(BELL_STATES[bell_state]))
    return outer, signal


def _build_apparatus(
    config: SwappingValidationConfig,
    basis_a: str,
    basis_d: str,
    seed: int,
):
    timeline = Timeline()
    bsm = PolarizationBSMNode(
        "polarization_bsm",
        timeline,
        {
            "source_bandwidth_nm": config.bandwidth_nm,
            "source_bandwidth_arm0_nm": config.bandwidth_nm,
            "source_bandwidth_arm1_nm": config.bandwidth_nm,
            "center_wavelength_arm0_nm": config.wavelength_nm,
            "center_wavelength_arm1_nm": config.wavelength_nm,
            "matching_window_ps": config.matching_window_ps,
            "coincidence_window_ps": config.coincidence_window_ps,
            "detector_efficiency": 1.0,
            "detector_jitter_ps": 0.0,
            "dark_count_rate_hz": 0.0,
            "pbs_fidelity": 1.0,
            "pbs_mismeasure_prob": 0.0,
            "unmatched_policy": "discard",
        },
    )
    bsm.register_input("source_a", 0)
    bsm.register_input("source_b", 1)

    analyzer_config = {
        "mode": "hwp_qwp",
        "qwp_fidelity": 1.0,
        "hwp_fidelity": 1.0,
        "detector_efficiency": 1.0,
        "dark_count": 0.0,
        "pbs_fidelity": 1.0,
        "mismeasure_prob": 0.0,
    }
    alice = PolarizationAnalyzerNode("alice", timeline, {**analyzer_config, "basis": basis_a})
    david = PolarizationAnalyzerNode("david", timeline, {**analyzer_config, "basis": basis_d})

    bsm.set_seed(seed)
    alice.set_seed(seed + 1)
    david.set_seed(seed + 2)
    bsm.init()
    alice.init()
    david.init()
    timeline.init()
    return timeline, bsm, alice, david


def run_conditional_basis_measurement(
    source_a_state: str,
    source_b_state: str,
    bsm_state: str,
    basis_a: str,
    basis_d: str,
    config: SwappingValidationConfig,
    seed: int,
) -> dict:
    """Measure one outer basis, conditioned on one recognizable BSM result."""
    timeline, bsm, alice, david = _build_apparatus(config, basis_a, basis_d, seed)
    counts = np.zeros((2, 2), dtype=int)
    accepted = 0
    attempts = 0
    max_attempts = config.accepted_events_per_basis * config.max_attempt_factor

    while accepted < config.accepted_events_per_basis and attempts < max_attempts:
        attempts += 1
        outer_a, signal_a = _new_pair(
            timeline, f"a{attempts}", source_a_state, config.wavelength_nm
        )
        outer_d, signal_c = _new_pair(
            timeline, f"d{attempts}", source_b_state, config.wavelength_nm
        )
        bsm.receive_qubit("source_a", signal_a)
        bsm.receive_qubit("source_b", signal_c)
        event = bsm.interaction_events[-1]
        timeline.time += 10_000

        if not event["accepted"] or event["bell_state"] != bsm_state:
            continue

        alice.receive_qubit("source_a", outer_a)
        david.receive_qubit("source_b", outer_d)
        result_a = alice.get_measurement_result()
        result_d = david.get_measurement_result()
        if result_a < 0 or result_d < 0:
            continue
        counts[result_a, result_d] += 1
        accepted += 1

    if accepted < config.accepted_events_per_basis:
        raise RuntimeError(
            f"Only collected {accepted}/{config.accepted_events_per_basis} "
            f"{bsm_state} events after {attempts} attempts"
        )

    same = int(counts[0, 0] + counts[1, 1])
    different = int(counts[0, 1] + counts[1, 0])
    correlation = float((same - different) / accepted)
    expectation_a = float((counts[0, :].sum() - counts[1, :].sum()) / accepted)
    expectation_d = float((counts[:, 0].sum() - counts[:, 1].sum()) / accepted)
    return {
        "source_a_state": source_a_state,
        "source_b_state": source_b_state,
        "bsm_state": bsm_state,
        "basis_a": basis_a,
        "basis_d": basis_d,
        "attempts": attempts,
        "accepted_events": accepted,
        "count_00": int(counts[0, 0]),
        "count_01": int(counts[0, 1]),
        "count_10": int(counts[1, 0]),
        "count_11": int(counts[1, 1]),
        "expectation_a": expectation_a,
        "expectation_d": expectation_d,
        "correlation": correlation,
    }


def reconstruct_density_matrix(rows: list[dict]) -> np.ndarray:
    """Reconstruct a two-qubit density matrix from nine Pauli settings."""
    pauli = {
        "I": np.eye(2, dtype=complex),
        "X": np.asarray([[0, 1], [1, 0]], dtype=complex),
        "Y": np.asarray([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.asarray([[1, 0], [0, -1]], dtype=complex),
    }
    lookup = {(row["basis_a"], row["basis_d"]): row for row in rows}
    expectation_a = {
        basis: float(np.mean([lookup[(basis, other)]["expectation_a"] for other in "XYZ"]))
        for basis in "XYZ"
    }
    expectation_d = {
        basis: float(np.mean([lookup[(other, basis)]["expectation_d"] for other in "XYZ"]))
        for basis in "XYZ"
    }

    rho = np.kron(pauli["I"], pauli["I"])
    for basis in "XYZ":
        rho += expectation_a[basis] * np.kron(pauli[basis], pauli["I"])
        rho += expectation_d[basis] * np.kron(pauli["I"], pauli[basis])
    for basis_a in "XYZ":
        for basis_d in "XYZ":
            rho += lookup[(basis_a, basis_d)]["correlation"] * np.kron(
                pauli[basis_a], pauli[basis_d]
            )
    rho /= 4
    return (rho + rho.conj().T) / 2


def run_swapping_validation(
    source_a_state: str = "psi_minus",
    source_b_state: str = "psi_minus",
    bsm_states: tuple[str, ...] = ("psi_minus", "psi_plus"),
    config: SwappingValidationConfig = SwappingValidationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return conditional basis counts and Bell-fidelity summaries."""
    rows = []
    summaries = []
    for bsm_index, bsm_state in enumerate(bsm_states):
        expected_state, _ = expected_swapped_state(
            source_a_state,
            source_b_state,
            bsm_state,
        )
        state_rows = []
        for basis_a_index, basis_a in enumerate(("X", "Y", "Z")):
            for basis_d_index, basis_d in enumerate(("X", "Y", "Z")):
                row = run_conditional_basis_measurement(
                    source_a_state,
                    source_b_state,
                    bsm_state,
                    basis_a,
                    basis_d,
                    config,
                    seed=(
                        config.seed
                        + 1000 * bsm_index
                        + 100 * basis_a_index
                        + 10 * basis_d_index
                    ),
                )
                rows.append(row)
                state_rows.append(row)

        correlations = {
            basis: next(
                row["correlation"]
                for row in state_rows
                if row["basis_a"] == basis and row["basis_d"] == basis
            )
            for basis in "XYZ"
        }
        target = BELL_STATES[expected_state]
        rho = reconstruct_density_matrix(state_rows)
        fidelity = float(np.real(np.vdot(target, rho @ target)))
        summaries.append(
            {
                "source_a_state": source_a_state,
                "source_b_state": source_b_state,
                "bsm_state": bsm_state,
                "expected_outer_state": expected_state,
                "correlation_x": correlations["X"],
                "correlation_y": correlations["Y"],
                "correlation_z": correlations["Z"],
                "estimated_fidelity": fidelity,
                "rho_real": np.real(rho).tolist(),
                "rho_imag": np.imag(rho).tolist(),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def run_all_source_combinations(
    config: SwappingValidationConfig = SwappingValidationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run tomography for all 16 source pairs and both detectable BSM states."""
    all_counts = []
    all_summaries = []
    for source_index, source_a_state in enumerate(BELL_STATES):
        for source_b_index, source_b_state in enumerate(BELL_STATES):
            run_config = replace(
                config,
                seed=config.seed + 10_000 * source_index + 1_000 * source_b_index,
            )
            counts, summary = run_swapping_validation(
                source_a_state,
                source_b_state,
                config=run_config,
            )
            all_counts.append(counts)
            all_summaries.append(summary)
    return pd.concat(all_counts, ignore_index=True), pd.concat(all_summaries, ignore_index=True)
