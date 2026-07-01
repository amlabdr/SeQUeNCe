import numpy as np
import pytest

from sequence.components.photon import Photon
from sequence.kernel.timeline import Timeline
from sequence.topology.optical_nodes import PolarizationBSMNode
from sequence.utils.encoding import polarization


PAIR_SPACING_PS = 10_000


def _node(**overrides):
    timeline = Timeline()
    config = {
        "source_bandwidth_nm": 0.1,
        "source_bandwidth_arm0_nm": 0.1,
        "source_bandwidth_arm1_nm": 0.1,
        "center_wavelength_arm0_nm": 1550.0,
        "center_wavelength_arm1_nm": 1550.0,
        "matching_window_ps": 100,
        "coincidence_window_ps": 100,
        "detector_efficiency": 1.0,
        "detector_jitter_ps": 0.0,
        "dark_count_rate_hz": 0.0,
        "pbs_fidelity": 1.0,
        "pbs_mismeasure_prob": 0.0,
        "unmatched_policy": "single",
    }
    config.update(overrides)
    node = PolarizationBSMNode("polarization_bsm", timeline, config)
    node.register_input("source_a", 0)
    node.register_input("source_b", 1)
    node.set_seed(1234)
    node.init()
    timeline.init()
    return timeline, node


def _photon(name, timeline, state):
    photon = Photon(name, timeline, wavelength=1550.0, encoding_type=polarization, quantum_state=state)
    return photon


def _send_pair(timeline, node, state_a, state_b):
    node.receive_qubit("source_a", _photon("a", timeline, state_a))
    node.receive_qubit("source_b", _photon("b", timeline, state_b))


def test_detector_pair_classification():
    classify = PolarizationBSMNode.classify_detector_pair
    assert classify(0, 3) == "psi_minus"
    assert classify(3, 0) == "psi_minus"
    assert classify(1, 2) == "psi_minus"
    assert classify(0, 1) == "psi_plus"
    assert classify(2, 3) == "psi_plus"
    assert classify(0, 2) is None
    assert classify(1, 3) is None
    assert classify(0, 0) is None


def test_orthogonal_inputs_produce_accepted_psi_patterns():
    timeline, node = _node()
    h_state, v_state = polarization["bases"][0]
    pair_count = 400

    for _ in range(pair_count):
        _send_pair(timeline, node, h_state, v_state)
        timeline.time += PAIR_SPACING_PS

    counts = node.count_bell_states()
    assert counts["psi_minus"] + counts["psi_plus"] == pair_count
    assert 140 < counts["psi_minus"] < 260
    assert 140 < counts["psi_plus"] < 260


def test_identical_matched_inputs_bunch_without_accepted_bell_pattern():
    timeline, node = _node()
    h_state = polarization["bases"][0][0]
    pair_count = 100

    for _ in range(pair_count):
        _send_pair(timeline, node, h_state, h_state)
        timeline.time += PAIR_SPACING_PS

    counts = node.get_detection_counts()
    assert sum(counts.values()) == pair_count
    assert node.count_bell_states() == {"psi_minus": 0, "psi_plus": 0}
    assert all(event["p_coincidence"] == pytest.approx(0.0) for event in node.interaction_events)


def test_four_named_click_streams_and_reset():
    timeline, node = _node()
    h_state, v_state = polarization["bases"][0]
    _send_pair(timeline, node, h_state, v_state)

    streams = node.get_click_times(include_dark_counts=False)
    assert tuple(streams) == PolarizationBSMNode.DETECTOR_NAMES
    assert sum(len(values) for values in streams.values()) == 2

    reset_streams = node.get_click_times(include_dark_counts=False, reset=True)
    assert reset_streams == streams
    assert node.get_click_times(include_dark_counts=False) == {
        name: [] for name in PolarizationBSMNode.DETECTOR_NAMES
    }


def test_operational_bell_events_are_derived_from_timestamps():
    _, node = _node()
    node.trigger_times = [
        [100, 1000],
        [2000],
        [2030],
        [130],
    ]

    events = node.get_bell_events(coincidence_window_ps=50, include_dark_counts=False)
    assert events == [
        {
            "time_ps": 130,
            "delta_t_ps": 30,
            "detectors": ("D3H", "D4V"),
            "bell_state": "psi_minus",
        },
        {
            "time_ps": 2030,
            "delta_t_ps": 30,
            "detectors": ("D3V", "D4H"),
            "bell_state": "psi_minus",
        },
    ]


def test_zero_detector_efficiency_suppresses_all_clicks():
    timeline, node = _node(detector_efficiency=0.0)
    h_state, v_state = polarization["bases"][0]
    for _ in range(20):
        _send_pair(timeline, node, h_state, v_state)
        timeline.time += PAIR_SPACING_PS

    assert node.get_detection_counts() == {
        name: 0 for name in PolarizationBSMNode.DETECTOR_NAMES
    }
    assert node.get_bell_events(include_dark_counts=False) == []


def test_per_detector_efficiencies_require_four_values():
    with pytest.raises(ValueError, match="four values"):
        _node(detector_efficiencies=[1.0, 1.0])


def test_zero_pbs_fidelity_suppresses_all_clicks():
    timeline, node = _node(pbs_fidelity=0.0)
    h_state, v_state = polarization["bases"][0]
    _send_pair(timeline, node, h_state, v_state)

    assert node.get_detection_counts() == {
        name: 0 for name in PolarizationBSMNode.DETECTOR_NAMES
    }


def test_unmatched_single_makes_one_click_but_no_bell_event():
    timeline, node = _node()
    h_state = polarization["bases"][0][0]
    node.receive_qubit("source_a", _photon("single", timeline, h_state))
    node.flush()

    assert sum(node.get_detection_counts().values()) == 1
    assert node.get_bell_events(include_dark_counts=False) == []


def test_noise_photons_are_routed_to_four_detector_channels():
    timeline, node = _node()
    for _ in range(200):
        node.receive_noise_photon()
        timeline.time += PAIR_SPACING_PS

    counts = node.get_detection_counts()
    assert sum(counts.values()) == 200
    assert all(count > 20 for count in counts.values())
