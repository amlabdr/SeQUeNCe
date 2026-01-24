"""Unit tests for WavePlate component.

This module tests half-wave plate and quarter-wave plate functionality
including Jones matrix correctness, polarization transformations, and
integration with SeQUeNCe's photon model.
"""

import numpy as np
import pytest
from numpy import sqrt

from sequence.components.wave_plate import WavePlate
from sequence.components.photon import Photon
from sequence.kernel.timeline import Timeline
from sequence.utils.encoding import polarization


SEED = 0


class DummyReceiver:
    """Mock receiver for testing wave plate output."""
    
    def __init__(self):
        self.photons_received = []
        self.last_photon = None
        self.last_kwargs = None
    
    def get(self, photon: Photon, **kwargs):
        self.photons_received.append(photon)
        self.last_photon = photon
        self.last_kwargs = kwargs


class DummyOwner:
    """Mock owner for random number generation."""
    
    def __init__(self, seed=SEED):
        self.generator = np.random.default_rng(seed)
    
    def get_generator(self):
        return self.generator


# ============================================================================
# Initialization and Configuration Tests
# ============================================================================

def test_WavePlate_init_default():
    """Test default initialization."""
    tl = Timeline()
    wp = WavePlate("wp", tl)
    
    assert wp.name == "wp"
    assert wp.plate_type == "HWP"
    assert wp.angle == 0.0
    assert wp.fidelity == 1.0
    assert wp.encoding_type == polarization
    assert wp.owner is None


def test_WavePlate_init_hwp():
    """Test HWP initialization with custom parameters."""
    tl = Timeline()
    wp = WavePlate("hwp", tl, plate_type="HWP", angle=np.pi/4, fidelity=0.95)
    
    assert wp.plate_type == "HWP"
    assert wp.angle == np.pi/4
    assert wp.fidelity == 0.95


def test_WavePlate_init_qwp():
    """Test QWP initialization with custom parameters."""
    tl = Timeline()
    wp = WavePlate("qwp", tl, plate_type="QWP", angle=np.pi/4)
    
    assert wp.plate_type == "QWP"
    assert wp.angle == np.pi/4


def test_WavePlate_init_invalid_plate_type():
    """Test that invalid plate type raises assertion error."""
    tl = Timeline()
    with pytest.raises(AssertionError):
        WavePlate("wp", tl, plate_type="INVALID")


def test_WavePlate_init_invalid_fidelity():
    """Test that invalid fidelity raises assertion error."""
    tl = Timeline()
    with pytest.raises(AssertionError):
        WavePlate("wp", tl, fidelity=1.5)
    
    with pytest.raises(AssertionError):
        WavePlate("wp", tl, fidelity=-0.1)


def test_WavePlate_init_must_have_receiver():
    """Test that init() requires exactly one receiver."""
    tl = Timeline()
    wp = WavePlate("wp", tl)
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    
    # Should not raise
    wp.init()
    
    # Test with no receiver
    wp2 = WavePlate("wp2", tl)
    with pytest.raises(AssertionError):
        wp2.init()


# ============================================================================
# Half-Wave Plate Tests
# ============================================================================

def test_HWP_at_0_degrees():
    """Test HWP at 0° flips V polarization."""
    tl = Timeline()
    wp = WavePlate("hwp", tl, plate_type="HWP", angle=0.0)
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    wp.init()
    
    # Test on V polarization
    photon = Photon("test", tl, quantum_state=polarization["bases"][0][1])  # |V⟩
    wp.get(photon)
    
    output_state = receiver.last_photon.quantum_state.state
    expected = np.array([0, -1])  # -|V⟩
    
    assert np.allclose(output_state, expected)


def test_HWP_at_45_degrees_H_to_V():
    """Test HWP at 45° rotates H to V (90° rotation)."""
    tl = Timeline()
    wp = WavePlate("hwp", tl, plate_type="HWP", angle=np.pi/4)
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    wp.init()
    
    # Test H polarization
    photon = Photon("test", tl, quantum_state=polarization["bases"][0][0])  # |H⟩
    wp.get(photon)
    
    output_state = receiver.last_photon.quantum_state.state
    expected = np.array([0, 1])  # |V⟩
    
    assert np.allclose(output_state, expected, atol=1e-10)


def test_HWP_at_22_5_degrees():
    """Test HWP at 22.5° rotates by 45°."""
    tl = Timeline()
    wp = WavePlate("hwp", tl, plate_type="HWP", angle=np.pi/8)  # 22.5°
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    wp.init()
    
    # Test H polarization
    photon = Photon("test", tl, quantum_state=polarization["bases"][0][0])  # |H⟩
    wp.get(photon)
    
    output_state = receiver.last_photon.quantum_state.state
    # Should rotate by 45°: (cos(45°), sin(45°))
    expected = np.array([1/sqrt(2), 1/sqrt(2)])  # |+⟩
    
    assert np.allclose(output_state, expected)


def test_HWP_double_pass():
    """Test that two HWP passes at same angle return to original."""
    tl = Timeline()
    wp1 = WavePlate("hwp1", tl, plate_type="HWP", angle=np.pi/6)  # 30°
    wp2 = WavePlate("hwp2", tl, plate_type="HWP", angle=np.pi/6)  # 30°
    receiver = DummyReceiver()
    wp1.add_receiver(wp2)
    wp2.add_receiver(receiver)
    wp1.init()
    wp2.init()
    
    # Input state
    input_state = np.array([1, 0])  # |H⟩
    photon = Photon("test", tl, quantum_state=tuple(input_state))
    wp1.get(photon)
    
    output_state = receiver.last_photon.quantum_state.state
    # Two passes of 60° rotation = 120° total, should rotate back (with phase)
    # Actually: HWP is its own inverse when applied twice at same angle
    expected = np.array([1, 0])  # Back to |H⟩
    
    assert np.allclose(output_state, expected, atol=1e-10)


# ============================================================================
# Quarter-Wave Plate Tests
# ============================================================================

def test_QWP_at_45_degrees_H_to_circular():
    """Test QWP at 45° converts H to circular polarization."""
    tl = Timeline()
    wp = WavePlate("qwp", tl, plate_type="QWP", angle=np.pi/4)
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    wp.init()
    
    # Test H polarization
    photon = Photon("test", tl, quantum_state=polarization["bases"][0][0])  # |H⟩
    wp.get(photon)
    
    output_state = receiver.last_photon.quantum_state.state
    # QWP at 45° produces: (0.5+0.5j, 0.5-0.5j) for H input
    expected = np.array([0.5+0.5j, 0.5-0.5j])
    
    assert np.allclose(output_state, expected)
    
    # Verify it's circular (equal magnitude in both components)
    assert np.allclose(np.abs(output_state[0]), np.abs(output_state[1]))


def test_QWP_at_45_degrees_V_to_circular():
    """Test QWP at 45° converts V to circular polarization."""
    tl = Timeline()
    wp = WavePlate("qwp", tl, plate_type="QWP", angle=np.pi/4)
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    wp.init()
    
    # Test V polarization
    photon = Photon("test", tl, quantum_state=polarization["bases"][0][1])  # |V⟩
    wp.get(photon)
    
    output_state = receiver.last_photon.quantum_state.state
    # QWP at 45° produces: (0.5-0.5j, 0.5+0.5j) for V input
    expected = np.array([0.5-0.5j, 0.5+0.5j])
    
    assert np.allclose(output_state, expected)
    
    # Verify it's circular (equal magnitude in both components)
    assert np.allclose(np.abs(output_state[0]), np.abs(output_state[1]))


def test_QWP_at_0_degrees():
    """Test QWP at 0° introduces phase shift only."""
    tl = Timeline()
    wp = WavePlate("qwp", tl, plate_type="QWP", angle=0.0)
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    wp.init()
    
    # Test on V polarization
    photon = Photon("test", tl, quantum_state=polarization["bases"][0][1])  # |V⟩
    wp.get(photon)
    
    output_state = receiver.last_photon.quantum_state.state
    expected = np.array([0, 1j])  # i|V⟩
    
    assert np.allclose(output_state, expected)


def test_QWP_double_pass_HWP_equivalence():
    """Test that two QWP passes at 45° equals one HWP pass."""
    tl = Timeline()
    qwp1 = WavePlate("qwp1", tl, plate_type="QWP", angle=np.pi/4)
    qwp2 = WavePlate("qwp2", tl, plate_type="QWP", angle=np.pi/4)
    receiver = DummyReceiver()
    qwp1.add_receiver(qwp2)
    qwp2.add_receiver(receiver)
    qwp1.init()
    qwp2.init()
    
    # Input state
    photon = Photon("test", tl, quantum_state=polarization["bases"][0][0])  # |H⟩
    qwp1.get(photon)
    
    output_state = receiver.last_photon.quantum_state.state
    
    # Compare with single HWP at 45°
    tl2 = Timeline()
    hwp = WavePlate("hwp", tl2, plate_type="HWP", angle=np.pi/4)
    receiver2 = DummyReceiver()
    hwp.add_receiver(receiver2)
    hwp.init()
    
    photon2 = Photon("test2", tl2, quantum_state=polarization["bases"][0][0])
    hwp.get(photon2)
    
    hwp_output = receiver2.last_photon.quantum_state.state
    
    # Should be equivalent up to global phase
    assert np.allclose(np.abs(output_state), np.abs(hwp_output))


# ============================================================================
# Angle Update Tests
# ============================================================================

def test_WavePlate_set_angle():
    """Test dynamic angle updating."""
    tl = Timeline()
    wp = WavePlate("wp", tl, plate_type="HWP", angle=0.0)
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    wp.init()
    
    # Initial angle
    photon1 = Photon("p1", tl, quantum_state=polarization["bases"][0][0])  # |H⟩
    wp.get(photon1)
    state1 = receiver.last_photon.quantum_state.state
    
    # Change angle to 45°
    wp.set_angle(np.pi/4)
    assert wp.angle == np.pi/4
    
    photon2 = Photon("p2", tl, quantum_state=polarization["bases"][0][0])  # |H⟩
    wp.get(photon2)
    state2 = receiver.last_photon.quantum_state.state
    
    # States should be different
    assert not np.allclose(state1, state2)
    # Second should be rotated to V
    assert np.allclose(state2, [0, 1], atol=1e-10)


# ============================================================================
# Fidelity Tests
# ============================================================================

def test_WavePlate_fidelity_perfect():
    """Test that fidelity=1.0 transmits all photons."""
    tl = Timeline()
    wp = WavePlate("wp", tl, plate_type="HWP", angle=0.0, fidelity=1.0)
    owner = DummyOwner()
    wp.owner = owner
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    wp.init()
    
    num_photons = 100
    for i in range(num_photons):
        photon = Photon(f"p{i}", tl, quantum_state=polarization["bases"][0][0])
        wp.get(photon)
    
    assert len(receiver.photons_received) == num_photons


def test_WavePlate_fidelity_lossy():
    """Test that fidelity<1.0 causes photon loss."""
    tl = Timeline()
    fidelity = 0.8
    wp = WavePlate("wp", tl, plate_type="HWP", angle=0.0, fidelity=fidelity)
    owner = DummyOwner(seed=42)
    wp.owner = owner
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    wp.init()
    
    num_photons = 1000
    for i in range(num_photons):
        photon = Photon(f"p{i}", tl, quantum_state=polarization["bases"][0][0])
        wp.get(photon)
    
    received = len(receiver.photons_received)
    expected = fidelity * num_photons
    
    # Should be close to expected with some statistical variation
    assert abs(received - expected) / expected < 0.1


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_WavePlate_wrong_encoding():
    """Test that wrong encoding type raises assertion error."""
    tl = Timeline()
    wp = WavePlate("wp", tl, plate_type="HWP", encoding_type=polarization)
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    wp.init()
    
    # Create photon with time_bin encoding
    from sequence.utils.encoding import time_bin
    photon = Photon("test", tl, encoding_type=time_bin)
    
    with pytest.raises(AssertionError):
        wp.get(photon)


# ============================================================================
# Jones Matrix Unitarity Tests
# ============================================================================

def test_HWP_matrix_unitary():
    """Test that HWP Jones matrix is unitary."""
    tl = Timeline()
    wp = WavePlate("hwp", tl, plate_type="HWP", angle=np.pi/6)
    
    J = wp.unitary
    J_dagger = J.conj().T
    identity = np.dot(J_dagger, J)
    
    assert np.allclose(identity, np.eye(2))


def test_QWP_matrix_unitary():
    """Test that QWP Jones matrix is unitary."""
    tl = Timeline()
    wp = WavePlate("qwp", tl, plate_type="QWP", angle=np.pi/6)
    
    J = wp.unitary
    J_dagger = J.conj().T
    identity = np.dot(J_dagger, J)
    
    assert np.allclose(identity, np.eye(2))


# ============================================================================
# Integration Tests
# ============================================================================

def test_WavePlate_chain():
    """Test chain of multiple wave plates."""
    tl = Timeline()
    
    # Create chain: HWP(22.5°) -> QWP(45°) -> HWP(22.5°)
    hwp1 = WavePlate("hwp1", tl, "HWP", angle=np.pi/8)
    qwp = WavePlate("qwp", tl, "QWP", angle=np.pi/4)
    hwp2 = WavePlate("hwp2", tl, "HWP", angle=np.pi/8)
    receiver = DummyReceiver()
    
    hwp1.add_receiver(qwp)
    qwp.add_receiver(hwp2)
    hwp2.add_receiver(receiver)
    
    hwp1.init()
    qwp.init()
    hwp2.init()
    
    # Send H polarized photon
    photon = Photon("test", tl, quantum_state=polarization["bases"][0][0])
    hwp1.get(photon)
    
    # Should produce some transformed state
    output_state = receiver.last_photon.quantum_state.state
    assert len(output_state) == 2
    
    # Verify normalization preserved
    norm = np.sqrt(np.sum(np.abs(output_state)**2))
    assert np.allclose(norm, 1.0)


def test_WavePlate_kwargs_forwarding():
    """Test that kwargs are forwarded to receiver."""
    tl = Timeline()
    wp = WavePlate("wp", tl, plate_type="HWP", angle=0.0)
    receiver = DummyReceiver()
    wp.add_receiver(receiver)
    wp.init()
    
    photon = Photon("test", tl, quantum_state=polarization["bases"][0][0])
    wp.get(photon, custom_param="test_value")
    
    assert receiver.last_kwargs is not None
    assert "custom_param" in receiver.last_kwargs
    assert receiver.last_kwargs["custom_param"] == "test_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
