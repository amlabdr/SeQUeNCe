import numpy as np
from itertools import product

from example.entanglement_swapping_validation.simulator import (
    BELL_STATES,
    SwappingValidationConfig,
    expected_swapped_state,
    run_swapping_validation,
)
from sequence.components.bsm import BSM
from sequence.components.photon import Photon
from sequence.kernel.timeline import Timeline


def test_free_state_bell_measurement_updates_active_state():
    timeline = Timeline()
    a, b, c, d = [Photon(name, timeline) for name in "abcd"]
    a.combine_state(b)
    a.set_state(tuple(BSM._psi_minus))
    c.combine_state(d)
    c.set_state(tuple(BSM._psi_minus))
    b.combine_state(c)

    before = tuple(a.quantum_state.state)
    Photon.measure_multiple(
        (tuple(BSM._phi_plus), tuple(BSM._phi_minus), tuple(BSM._psi_plus), tuple(BSM._psi_minus)),
        [b, c],
        np.random.default_rng(3),
    )

    assert tuple(a.quantum_state.state) != before
    assert not hasattr(a.quantum_state, "entangled_photons")


def test_psi_minus_sources_have_expected_swapped_states():
    label_minus, _ = expected_swapped_state("psi_minus", "psi_minus", "psi_minus")
    label_plus, _ = expected_swapped_state("psi_minus", "psi_minus", "psi_plus")
    assert label_minus == "psi_minus"
    assert label_plus == "psi_plus"


def test_swapped_state_three_basis_fidelity():
    _, summary = run_swapping_validation(
        "psi_minus",
        "psi_minus",
        config=SwappingValidationConfig(
            accepted_events_per_basis=80,
            seed=991,
        ),
    )

    assert set(summary["bsm_state"]) == {"psi_minus", "psi_plus"}
    assert np.all(summary["estimated_fidelity"] > 0.95)


def test_all_source_bell_combinations_swap_to_predicted_state():
    for index, (source_a, source_b) in enumerate(product(BELL_STATES, repeat=2)):
        _, summary = run_swapping_validation(
            source_a,
            source_b,
            config=SwappingValidationConfig(
                accepted_events_per_basis=8,
                seed=20_000 + 100 * index,
            ),
        )
        assert np.all(summary["estimated_fidelity"] > 0.99)
