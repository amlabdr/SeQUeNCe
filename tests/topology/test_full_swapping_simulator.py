from pathlib import Path

import pandas as pd

from example.entanglement_swapping_validation.batch_config import load_config
from example.entanglement_swapping_validation.full_simulator import (
    FullSwappingConfig,
    maximum_likelihood_tomography,
    run_basis_setting,
)


def _ideal_rows(bsm_state):
    rows = []
    for basis_a in "XYZ":
        for basis_d in "XYZ":
            if basis_a != basis_d:
                counts = [25, 25, 25, 25]
            elif bsm_state == "psi_minus" or basis_a == "Z":
                counts = [0, 50, 50, 0]
            else:
                counts = [50, 0, 0, 50]
            rows.append({
                "source_a_state": "psi_minus",
                "source_b_state": "psi_minus",
                "bsm_state": bsm_state,
                "basis_a": basis_a,
                "basis_d": basis_d,
                "count_00": counts[0],
                "count_01": counts[1],
                "count_10": counts[2],
                "count_11": counts[3],
                "fourfold_count": sum(counts),
            })
    return rows


def test_mle_reconstructs_ideal_swapped_states():
    frame = pd.DataFrame(_ideal_rows("psi_minus") + _ideal_rows("psi_plus"))
    for bsm_state in ("psi_minus", "psi_plus"):
        rho, summary = maximum_likelihood_tomography(frame, bsm_state)
        assert summary["optimizer_success"]
        assert summary["fidelity"] > 0.999
        assert abs(summary["trace"] - 1) < 1e-12
        assert summary["minimum_eigenvalue"] > -1e-12
        assert rho.shape == (4, 4)


def test_full_physical_zz_smoke():
    config = FullSwappingConfig(
        duration_s=0.3,
        source_frequency_hz=1_000_000,
        mean_photon_num=0.02,
        source_bandwidth_nm=0.0001,
        signal_length_a_m=1,
        signal_length_b_m=1,
        outer_length_a_m=100,
        outer_length_d_m=100,
        matching_window_ps=100,
        fourfold_window_ps=100,
        seed=22,
    )
    rows = run_basis_setting("Z", "Z", config)
    assert rows["fourfold_count"].min() > 0
    assert rows["correlation"].max() < -0.8


def test_full_batch_config_loads():
    path = Path("example/entanglement_swapping_validation/configs/full_swapping_validation.ini")
    simulation, batch, slurm = load_config(path)
    assert simulation.outer_length_a_m > simulation.signal_length_a_m
    assert batch.runs == 100
    assert slurm.target == "blackbird"
