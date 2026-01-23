import numpy as np

from sequence.components.beam_splitter import BeamSplitter, FockBeamSplitter2, PolarizingBeamSplitter
from sequence.components.detector import FockDetector
from sequence.components.photon import Photon
from sequence.kernel.timeline import Timeline
from sequence.utils.encoding import polarization
from sequence.topology.node import Node

np.random.seed(0)
SEED = 0


def test_BeamSplitter_init():
    tl = Timeline()
    bs = BeamSplitter("bs", tl)
    bs.add_receiver(None)
    bs.add_receiver(None)
    tl.init()


class Owner:
    def __init__(self):
        self.generator = np.random.default_rng(SEED)

    def get_generator(self):
        return self.generator


class Receiver:
    def __init__(self, tl):
        self.timeline = tl
        self.log = []

    def get(self, photon=None):
        self.log.append((self.timeline.now()))

    def reset(self):
        self.log = []


def test_BeamSplitter_get():
    tl = Timeline()
    bs = BeamSplitter("bs", tl)
    own = Owner()
    bs.owner = own
    receiver0 = Receiver(tl)
    bs.add_receiver(receiver0)
    receiver1 = Receiver(tl)
    bs.add_receiver(receiver1)

    frequency = 8e7
    start_time = 0
    basis_len = 1000
    basis_list = []

    # z-basis states, measurement
    for i in range(basis_len):
        basis_list.append(0)

    bs.set_basis_list(basis_list, start_time, frequency)

    bits = []
    for i in range(basis_len):
        time = 1e12 / frequency * i
        tl.time = time
        bit = np.random.randint(2)
        bits.append(bit)
        photon = Photon(str(i), tl, quantum_state=polarization["bases"][0][bit])
        bs.get(photon)

    for i in range(basis_len):
        time = 1e12 / frequency * i
        r_i = bits[i]
        assert time in bs._receivers[r_i].log

    # x-basis states, measurement
    receiver0.log = []
    receiver1.log = []
    basis_list = []
    for i in range(basis_len):
        basis_list.append(1)

    bs.set_basis_list(basis_list, start_time, frequency)

    bits2 = []
    for i in range(basis_len):
        time = 1e12 / frequency * i
        tl.time = time
        bit = np.random.randint(2)
        bits2.append(bit)
        photon = Photon(str(i), tl, quantum_state=polarization["bases"][1][bit])
        bs.get(photon)

    for i in range(basis_len):
        time = 1e12 / frequency * i
        r_i = bits2[i]
        assert time in bs._receivers[r_i].log

    # z-basis states, x-basis measurement
    receiver0.log = []
    receiver1.log = []
    basis_list = []
    for i in range(basis_len):
        basis_list.append(1)

    bs.set_basis_list(basis_list, start_time, frequency)

    bits = []
    for i in range(basis_len):
        time = 1e12 / frequency * i
        tl.time = time
        bit = np.random.randint(2)
        bits.append(bit)
        photon = Photon(str(i), tl, quantum_state=polarization["bases"][0][bit])
        bs.get(photon)

    print(len(receiver1.log), len(receiver0.log))
    true_counter, false_counter = 0, 0
    for i in range(basis_len):
        time = 1e12 / frequency * i
        r_i = bits[i]
        if time in bs._receivers[r_i].log:
            true_counter += 1
        else:
            false_counter += 1
    assert true_counter / basis_len - 0.5 < 0.1


# def test_FockBeamSplitter_get():
#     NUM_TRIALS = 1000
#     psi_minus = [complex(0), complex(sqrt(1 / 2)), -complex(sqrt(1 / 2)), complex(0)]
#
#     tl = Timeline(formalism="density_matrix")
#     rec_0 = Receiver(tl)
#     rec_1 = Receiver(tl)
#
#     bs = FockBeamSplitter("bs", tl)
#     bs.add_receiver(rec_0)
#     bs.add_receiver(rec_1)
#
#     tl.init()
#
#     # measure unentangled
#     for _ in range(NUM_TRIALS):
#         p0 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p1 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p0.is_null = True
#         bs.get(p0)
#         bs.get(p1)
#         tl.time += 1
#
#     assert abs(len(rec_0.log) / len(rec_1.log)) - 1 < 0.1
#
#     # measure entangled, no phase
#     tl.time = 0
#     rec_0.reset()
#     rec_1.reset()
#     for _ in range(NUM_TRIALS):
#         p0 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p1 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p0.is_null = True
#         p0.entangle(p1)
#         p0.set_state(psi_minus)
#         bs.get(p0)
#         bs.get(p1)
#         tl.time += 1
#
#     assert abs(len(rec_0.log) / len(rec_1.log)) - 1 < 0.1
#
#     # measure entangled, pi/2 phase
#     tl.time = 0
#     rec_0.reset()
#     rec_1.reset()
#     circuit = Circuit(1)
#     circuit.phase(0, pi/2)
#     for _ in range(NUM_TRIALS):
#         p0 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p1 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p0.is_null = True
#         p0.entangle(p1)
#         p0.set_state(psi_minus)
#         tl.quantum_manager.run_circuit(circuit, [p0.quantum_state])
#
#         bs.get(p0)
#         bs.get(p1)
#         tl.time += 1
#
#     assert len(rec_1.log) == NUM_TRIALS
#
#     # measure entangled, 3pi/2 phase
#     tl.time = 0
#     rec_0.reset()
#     rec_1.reset()
#     circuit = Circuit(1)
#     circuit.phase(0, 3*pi/2)
#     for _ in range(NUM_TRIALS):
#         p0 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p1 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p0.is_null = True
#         p0.entangle(p1)
#         p0.set_state(psi_minus)
#         tl.quantum_manager.run_circuit(circuit, [p0.quantum_state])
#
#         bs.get(p0)
#         bs.get(p1)
#         tl.time += 1
#
#     assert len(rec_0.log) == NUM_TRIALS


def test_FockBeamSplitter2():
    """ Quantum transduction via entanglemenet swapping (node2).
        A fock beamsplitter is attached to two detectors
    """
    tl = Timeline()
    node1 = Node('Node1', tl)
    node2 = Node('Node2', tl)  # swapping node
    node3 = Node('Node3', tl)
    src_list = [node1, node3]

    detector1 = FockDetector(node2.name + '.detector1', tl, efficiency=0.5)
    detector2 = FockDetector(node2.name + '.detector2', tl, efficiency=0.5)

    fockbeamsplitter2 = FockBeamSplitter2('fock_beamsplitter2', node2, tl, efficiency=0.9, photon_counter=0, src_list=src_list)
    fockbeamsplitter2.add_outputs([detector1, detector2])

    photon = Photon('photon', tl)

    # receives 10 photons from node1 and 10 photons from node3
    photon_number = 10
    for _ in range(photon_number):
        fockbeamsplitter2.get(node1.name, photon)
    for _ in range(photon_number):
        fockbeamsplitter2.get(node2.name, photon)

    assert fockbeamsplitter2.photon_counter == photon_number * 2

def test_PolarizingBeamSplitter_init():
    """Test PolarizingBeamSplitter initialization with default parameters."""
    tl = Timeline()
    pbs = PolarizingBeamSplitter("pbs", tl)
    
    assert pbs.name == "pbs"
    assert pbs.basis_index == 0
    assert pbs.fidelity == 1.0
    assert pbs.mismeasure_prob == 0.0
    
    # Add receivers and initialize
    pbs.add_receiver(None)
    pbs.add_receiver(None)
    tl.init()


def test_PolarizingBeamSplitter_init_custom_params():
    """Test PolarizingBeamSplitter initialization with custom parameters."""
    tl = Timeline()
    pbs = PolarizingBeamSplitter("pbs", tl, basis_index=1, fidelity=0.95, mismeasure_prob=0.02)
    
    assert pbs.basis_index == 1
    assert pbs.fidelity == 0.95
    assert pbs.mismeasure_prob == 0.02


def test_PolarizingBeamSplitter_requires_two_receivers():
    """Test that PolarizingBeamSplitter requires exactly 2 receivers."""
    tl = Timeline()
    pbs = PolarizingBeamSplitter("pbs", tl)
    
    # Should fail with 1 receiver
    pbs.add_receiver(None)
    with pytest.raises(AssertionError, match="requires exactly 2 receivers"):
        tl.init()
    
    # Should succeed with 2 receivers
    tl = Timeline()
    pbs = PolarizingBeamSplitter("pbs", tl)
    pbs.add_receiver(None)
    pbs.add_receiver(None)
    tl.init()  # Should not raise


def test_PolarizingBeamSplitter_HV_basis_perfect():
    """Test PBS measurement in H/V basis with perfect fidelity and no errors."""
    tl = Timeline()
    pbs = PolarizingBeamSplitter("pbs", tl, basis_index=0, fidelity=1.0, mismeasure_prob=0.0)
    own = Owner()
    pbs.owner = own
    
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    pbs.add_receiver(receiver0)
    pbs.add_receiver(receiver1)
    
    tl.init()
    
    # Test H-polarized photons (should go to receiver 0)
    num_photons = 100
    for i in range(num_photons):
        tl.time = i
        photon = Photon(f"H_{i}", tl, quantum_state=polarization["bases"][0][0])  # |H⟩
        pbs.get(photon)
    
    assert len(receiver0.log) == num_photons
    assert len(receiver1.log) == 0
    
    # Reset and test V-polarized photons (should go to receiver 1)
    receiver0.reset()
    receiver1.reset()
    
    for i in range(num_photons):
        tl.time = i + num_photons
        photon = Photon(f"V_{i}", tl, quantum_state=polarization["bases"][0][1])  # |V⟩
        pbs.get(photon)
    
    assert len(receiver0.log) == 0
    assert len(receiver1.log) == num_photons


def test_PolarizingBeamSplitter_diagonal_basis_perfect():
    """Test PBS measurement in +/- basis with perfect fidelity."""
    tl = Timeline()
    pbs = PolarizingBeamSplitter("pbs", tl, basis_index=1, fidelity=1.0, mismeasure_prob=0.0)
    own = Owner()
    pbs.owner = own
    
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    pbs.add_receiver(receiver0)
    pbs.add_receiver(receiver1)
    
    tl.init()
    
    # Test |+⟩ photons (should go to receiver 0)
    num_photons = 100
    for i in range(num_photons):
        tl.time = i
        photon = Photon(f"plus_{i}", tl, quantum_state=polarization["bases"][1][0])  # |+⟩
        pbs.get(photon)
    
    assert len(receiver0.log) == num_photons
    assert len(receiver1.log) == 0
    
    # Reset and test |-⟩ photons (should go to receiver 1)
    receiver0.reset()
    receiver1.reset()
    
    for i in range(num_photons):
        tl.time = i + num_photons
        photon = Photon(f"minus_{i}", tl, quantum_state=polarization["bases"][1][1])  # |-⟩
        pbs.get(photon)
    
    assert len(receiver0.log) == 0
    assert len(receiver1.log) == num_photons


def test_PolarizingBeamSplitter_basis_mismatch():
    """Test PBS with wrong basis measurement (H/V photons in +/- basis)."""
    tl = Timeline()
    pbs = PolarizingBeamSplitter("pbs", tl, basis_index=1, fidelity=1.0, mismeasure_prob=0.0)
    own = Owner()
    pbs.owner = own
    
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    pbs.add_receiver(receiver0)
    pbs.add_receiver(receiver1)
    
    tl.init()
    
    # Send H-polarized photons, measure in +/- basis
    # Should get ~50/50 split
    num_photons = 1000
    for i in range(num_photons):
        tl.time = i
        photon = Photon(f"H_{i}", tl, quantum_state=polarization["bases"][0][0])  # |H⟩
        pbs.get(photon)
    
    ratio = len(receiver0.log) / num_photons
    
    # Should be approximately 0.5 with some tolerance
    assert 0.4 < ratio < 0.6, f"Expected ~0.5, got {ratio}"


def test_PolarizingBeamSplitter_fidelity():
    """Test PBS with imperfect fidelity (photon loss)."""
    tl = Timeline()
    fidelity = 0.8
    pbs = PolarizingBeamSplitter("pbs", tl, basis_index=0, fidelity=fidelity, mismeasure_prob=0.0)
    own = Owner()
    pbs.owner = own
    
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    pbs.add_receiver(receiver0)
    pbs.add_receiver(receiver1)
    
    tl.init()
    
    # Send many H-polarized photons
    num_photons = 1000
    for i in range(num_photons):
        tl.time = i
        photon = Photon(f"H_{i}", tl, quantum_state=polarization["bases"][0][0])
        pbs.get(photon)
    
    total_received = len(receiver0.log) + len(receiver1.log)
    measured_fidelity = total_received / num_photons
    
    # Should be close to specified fidelity
    assert abs(measured_fidelity - fidelity) < 0.05, \
        f"Expected fidelity ~{fidelity}, got {measured_fidelity}"
    
    # All received photons should be at receiver0 (H-polarized)
    assert len(receiver1.log) == 0


def test_PolarizingBeamSplitter_mismeasurement():
    """Test PBS with measurement errors (bit flips)."""
    tl = Timeline()
    mismeasure_prob = 0.1
    pbs = PolarizingBeamSplitter("pbs", tl, basis_index=0, fidelity=1.0, 
                                  mismeasure_prob=mismeasure_prob)
    own = Owner()
    pbs.owner = own
    
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    pbs.add_receiver(receiver0)
    pbs.add_receiver(receiver1)
    
    tl.init()
    
    # Send H-polarized photons (should mostly go to receiver0)
    num_photons = 1000
    for i in range(num_photons):
        tl.time = i
        photon = Photon(f"H_{i}", tl, quantum_state=polarization["bases"][0][0])
        pbs.get(photon)
    
    # With mismeasurement, some will go to wrong receiver
    error_rate = len(receiver1.log) / num_photons
    
    # Should be close to mismeasure_prob
    assert abs(error_rate - mismeasure_prob) < 0.03, \
        f"Expected error rate ~{mismeasure_prob}, got {error_rate}"


def test_PolarizingBeamSplitter_combined_imperfections():
    """Test PBS with both fidelity loss and measurement errors."""
    tl = Timeline()
    fidelity = 0.9
    mismeasure_prob = 0.05
    pbs = PolarizingBeamSplitter("pbs", tl, basis_index=0, fidelity=fidelity, 
                                  mismeasure_prob=mismeasure_prob)
    own = Owner()
    pbs.owner = own
    
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    pbs.add_receiver(receiver0)
    pbs.add_receiver(receiver1)
    
    tl.init()
    
    # Send H-polarized photons
    num_photons = 1000
    for i in range(num_photons):
        tl.time = i
        photon = Photon(f"H_{i}", tl, quantum_state=polarization["bases"][0][0])
        pbs.get(photon)
    
    total_received = len(receiver0.log) + len(receiver1.log)
    
    # Check fidelity (overall transmission)
    measured_fidelity = total_received / num_photons
    assert abs(measured_fidelity - fidelity) < 0.05
    
    # Check error rate among received photons
    if total_received > 0:
        error_rate = len(receiver1.log) / total_received
        assert abs(error_rate - mismeasure_prob) < 0.03


def test_PolarizingBeamSplitter_requires_polarization_encoding():
    """Test that PBS rejects non-polarization encoded photons."""
    tl = Timeline()
    pbs = PolarizingBeamSplitter("pbs", tl)
    own = Owner()
    pbs.owner = own
    
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    pbs.add_receiver(receiver0)
    pbs.add_receiver(receiver1)
    
    tl.init()
    
    # Create photon with wrong encoding (generic Photon defaults to time-bin)
    from sequence.utils.encoding import time_bin
    photon = Photon("test", tl, encoding_type=time_bin)
    
    # Should raise assertion error
    with pytest.raises(AssertionError, match="requires polarization encoding"):
        pbs.get(photon)


def test_PolarizingBeamSplitter_timing():
    """Test that PBS preserves timing information."""
    tl = Timeline()
    pbs = PolarizingBeamSplitter("pbs", tl, basis_index=0, fidelity=1.0)
    own = Owner()
    pbs.owner = own
    
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    pbs.add_receiver(receiver0)
    pbs.add_receiver(receiver1)
    
    tl.init()
    
    # Send photons at specific times
    times = [0, 100, 500, 1000, 5000]
    for t in times:
        tl.time = t
        photon = Photon(f"H_{t}", tl, quantum_state=polarization["bases"][0][0])
        pbs.get(photon)
    
    # Check all photons arrived at correct times
    assert sorted(receiver0.log) == times


def test_PolarizingBeamSplitter_statistical_distribution():
    """Test statistical distribution of measurements matches quantum predictions."""
    tl = Timeline()
    pbs = PolarizingBeamSplitter("pbs", tl, basis_index=0, fidelity=1.0, mismeasure_prob=0.0)
    own = Owner()
    pbs.owner = own
    
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    pbs.add_receiver(receiver0)
    pbs.add_receiver(receiver1)
    
    tl.init()
    
    # Test diagonal photons in H/V basis (should be 50/50)
    num_photons = 2000
    for i in range(num_photons):
        tl.time = i
        # |+⟩ state
        photon = Photon(f"plus_{i}", tl, quantum_state=polarization["bases"][1][0])
        pbs.get(photon)
    
    ratio = len(receiver0.log) / num_photons
    
    # Should be close to 0.5 (quantum prediction)
    assert 0.45 < ratio < 0.55, f"Expected ~0.5, got {ratio}"

import pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])