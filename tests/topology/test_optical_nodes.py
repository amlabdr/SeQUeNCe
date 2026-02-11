
import numpy as np

from sequence.components.photon import Photon
from sequence.utils.encoding import polarization
from sequence.topology.optical_nodes import PolarizationAnalyzerNode
from sequence.kernel.timeline import Timeline
from sequence.topology.optical_nodes import SpdcSourceNode

# Test spacing to avoid detector dead time
PHOTON_SPACING = int(1e12)  # 1 ms


def test_PolarizationAnalyzerNode_hwp_only_rotation():
    """Test HWP-only mode rotates polarization correctly (H→V at 90°)."""
    np.random.seed(42)

    tl = Timeline()
    config = {
        'mode': 'hwp_only',
        'rotation_angle': np.pi/2,  # 90° rotation
        'detector_efficiency': 1.0,
        'dark_count': 0,
        'pbs_fidelity': 1.0,
        'hwp_fidelity': 1.0,
        'mismeasure_prob': 0.0
    }
    analyzer = PolarizationAnalyzerNode("analyzer", tl, config)
    analyzer.init()
    tl.init()

    # Send H-polarized photons
    NUM_H = 100
    for i in range(NUM_H):
        photon = Photon(f"H_{i}", tl, quantum_state=polarization["bases"][0][0])
        analyzer.receive_qubit(photon)
        tl.time += PHOTON_SPACING

    tl.run()

    counts = analyzer.get_detection_counts()
    # H rotated by 90° should become V (detector 1)
    assert counts[0] == 0, f"Expected 0 H, got {counts[0]}"
    assert counts[1] == NUM_H, f"Expected {NUM_H} V, got {counts[1]}"


def test_PolarizationAnalyzerNode_hwp_only_dynamic_angle():
    """Test dynamic rotation angle changes in HWP-only mode."""
    np.random.seed(42)

    tl = Timeline()
    config = {
        'mode': 'hwp_only',
        'rotation_angle': 0.0,
        'detector_efficiency': 1.0,
        'dark_count': 0,
        'pbs_fidelity': 1.0,
        'hwp_fidelity': 1.0,
        'mismeasure_prob': 0.0
    }
    analyzer = PolarizationAnalyzerNode("analyzer", tl, config)
    analyzer.init()
    tl.init()

    # First: no rotation (H stays H)
    photon1 = Photon("p1", tl, quantum_state=polarization["bases"][0][0])
    analyzer.receive_qubit(photon1)
    tl.time += PHOTON_SPACING

    # Change to 90° rotation (H becomes V)
    analyzer.set_rotation_angle(np.pi/2)

    photon2 = Photon("p2", tl, quantum_state=polarization["bases"][0][0])
    analyzer.receive_qubit(photon2)
    tl.time += PHOTON_SPACING

    tl.run()

    counts = analyzer.get_detection_counts()
    # Should have one in each detector
    assert counts[0] == 1, f"Expected 1 H, got {counts[0]}"
    assert counts[1] == 1, f"Expected 1 V, got {counts[1]}"


def test_PolarizationAnalyzerNode_custom_hwp_only_perfect_X_basis():
    """Test custom mode with only HWP gives perfect X basis measurement."""
    np.random.seed(42)

    tl = Timeline()
    config = {
        'mode': 'custom',
        'use_qwp': False,  # No QWP = no phase issues!
        'use_hwp': True,
        'hwp_angle': np.pi/8,  # 22.5° → 45° rotation
        'detector_efficiency': 1.0,
        'dark_count': 0,
        'pbs_fidelity': 1.0,
        'hwp_fidelity': 1.0,
        'mismeasure_prob': 0.0
    }
    analyzer = PolarizationAnalyzerNode("analyzer", tl, config)
    analyzer.init()
    tl.init()

    # Send diagonal photons |D⟩ = (|H⟩ + |V⟩)/√2
    NUM_D = 100
    diagonal_state = (1/np.sqrt(2), 1/np.sqrt(2))

    for i in range(NUM_D):
        photon = Photon(f"D_{i}", tl, quantum_state=diagonal_state)
        analyzer.receive_qubit(photon)
        tl.time += PHOTON_SPACING

    tl.run()

    counts = analyzer.get_detection_counts()

    # HWP at 22.5° rotates diagonal by 45° to H
    # Without QWP, this is PERFECT - all go to detector 0
    assert counts[0] == NUM_D, f"Expected all in detector 0, got {counts[0]}/{counts[1]}"
    assert counts[1] == 0, "No photons should reach detector 1"


def test_PolarizationAnalyzerNode_custom_no_plates():
    """Test custom mode with no wave plates (direct PBS measurement)."""
    np.random.seed(42)

    tl = Timeline()
    config = {
        'mode': 'custom',
        'use_qwp': False,
        'use_hwp': False,
        'detector_efficiency': 1.0,
        'dark_count': 0,
        'pbs_fidelity': 1.0,
        'mismeasure_prob': 0.0
    }
    analyzer = PolarizationAnalyzerNode("analyzer", tl, config)
    analyzer.init()
    tl.init()

    # Send H photons
    NUM_H = 100
    for i in range(NUM_H):
        photon = Photon(f"H_{i}", tl, quantum_state=polarization["bases"][0][0])
        analyzer.receive_qubit(photon)
        tl.time += PHOTON_SPACING

    tl.run()

    counts = analyzer.get_detection_counts()

    # Direct PBS: H → detector 0, V → detector 1
    assert counts[0] == NUM_H
    assert counts[1] == 0


def test_PolarizationAnalyzerNode_qwp_hwp_Y_basis():
    """Test Y basis (circular) measurement in qwp_hwp mode."""
    np.random.seed(0)

    tl = Timeline()
    config = {
        'mode': 'qwp_hwp',
        'basis': 'Y',
        'detector_efficiency': 1.0,
        'dark_count': 0,
        'pbs_fidelity': 1.0,
        'qwp_fidelity': 1.0,
        'hwp_fidelity': 1.0,
        'mismeasure_prob': 0.0
    }
    analyzer = PolarizationAnalyzerNode("analyzer", tl, config)
    analyzer.init()
    tl.init()

    NUM_CIRCULAR = 100
    right_circular = (1/np.sqrt(2), 1j/np.sqrt(2))  # |R⟩ = (|H⟩ + i|V⟩)/√2

    for i in range(NUM_CIRCULAR):
        photon = Photon(f"R_{i}", tl, quantum_state=right_circular)
        analyzer.receive_qubit(photon)
        tl.time += PHOTON_SPACING

    tl.run()

    counts = analyzer.get_detection_counts()
    total = counts[0] + counts[1]

    # All photons should be detected
    assert total == NUM_CIRCULAR, f"Lost photons: {total}/{NUM_CIRCULAR}"

    # Y basis should show strong preference for one detector (>60%)
    assert counts[0] > 0.6 * NUM_CIRCULAR or counts[1] > 0.6 * NUM_CIRCULAR,         f"Expected strong bias in Y basis, got {counts[0]}/{counts[1]}"


def test_PolarizationAnalyzerNode_invalid_mode():
    """Test that invalid mode raises error."""
    tl = Timeline()
    config = {'mode': 'invalid_mode'}

    with pytest.raises(ValueError, match="Unknown mode"):
        PolarizationAnalyzerNode("analyzer", tl, config)


def test_PolarizationAnalyzerNode_set_rotation_wrong_mode():
    """Test that set_rotation_angle() fails in wrong mode."""
    tl = Timeline()
    config = {'mode': 'qwp_hwp'}
    analyzer = PolarizationAnalyzerNode("analyzer", tl, config)

    with pytest.raises(ValueError, match="only valid in 'hwp_only' mode"):
        analyzer.set_rotation_angle(np.pi/4)


def test_PolarizationAnalyzerNode_measurement_result_api():
    """Test get_measurement_result() API returns correct binary outcome."""
    np.random.seed(42)

    tl = Timeline()
    config = {
        'mode': 'hwp_only',
        'rotation_angle': 0.0,
        'detector_efficiency': 1.0,
        'dark_count': 0,
        'pbs_fidelity': 1.0,
        'mismeasure_prob': 0.0
    }
    analyzer = PolarizationAnalyzerNode("analyzer", tl, config)
    analyzer.init()
    tl.init()

    # No detections yet - should return -1
    result = analyzer.get_measurement_result()
    assert result == -1, "Should return -1 when no photons detected"

    # Send H photon (should go to detector 0 → result 0)
    photon = Photon("H", tl, quantum_state=polarization["bases"][0][0])
    analyzer.receive_qubit(photon)
    tl.run()

    result = analyzer.get_measurement_result()
    assert result == 0, f"H photon should give result 0, got {result}"


def test_SpdcSourceNode_basic_emission():
    """Test that SpdcSourceNode emits photon pairs through quantum channels."""
    np.random.seed(42)
    
    tl = Timeline()
    
    # Create source node
    config = {
        'frequency': 1e8,
        'mean_photon_num': 0.1,
        'bell_state': 'psi-'
    }
    source = SpdcSourceNode("source", tl, config)
    
    # Create mock receivers to track photons
    class MockReceiver:
        def __init__(self, name):
            self.name = name
            self.received = []
        
        def receive_qubit(self, photon):
            self.received.append((tl.now(), photon))
    
    receiver0 = MockReceiver("recv0")
    receiver1 = MockReceiver("recv1")
    
    # Mock quantum channels that implement send_qubit's expected interface
    qc0 = receiver0  # Direct receiver
    qc1 = receiver1
    source.qchannels = [qc0, qc1]
    
    tl.init()
    source.emit(num_pulses=100)
    tl.run()
    
    # Both receivers should get same number of photons
    assert len(receiver0.received) == len(receiver1.received), \
        f"Unequal photon counts: {len(receiver0.received)} vs {len(receiver1.received)}"
    
    # Should have emitted some pairs (mean 0.1 × 100 ≈ 10)
    assert len(receiver0.received) > 0, "No photons emitted"
    assert len(receiver0.received) < 50, "Too many photons"
    
    # Check emission counter
    assert source.emission_count == len(receiver0.received), \
        f"Emission count mismatch: {source.emission_count} vs {len(receiver0.received)}"


def test_SpdcSourceNode_bell_state_configuration():
    """Test dynamic Bell state reconfiguration."""
    np.random.seed(0)
    
    tl = Timeline()
    
    config = {
        'frequency': 1e8,
        'mean_photon_num': 1.0,
        'bell_state': 'phi+'
    }
    source = SpdcSourceNode("source", tl, config)
    
    # Check initial state
    assert source.spdc.bell_state_label == 'phi+'
    assert source.spdc.bell_state == source.spdc.bell_state_map['phi+']
    
    # Change to psi-
    source.set_bell_state('psi-')
    assert source.spdc.bell_state_label == 'psi-'
    assert source.spdc.bell_state == source.spdc.bell_state_map['psi-']
    
    # Test all valid states
    for state in ['phi+', 'phi-', 'psi+', 'psi-']:
        source.set_bell_state(state)
        assert source.spdc.bell_state_label == state
    
    # Test invalid state
    with pytest.raises(ValueError, match="Invalid Bell state"):
        source.set_bell_state('invalid')


def test_SpdcSourceNode_frequency_and_brightness_control():
    """Test dynamic frequency and mean photon number configuration."""
    np.random.seed(123)
    
    tl = Timeline()
    
    config = {
        'frequency': 1e8,
        'mean_photon_num': 0.1,
        'bell_state': 'psi+'
    }
    source = SpdcSourceNode("source", tl, config)
    
    # Test frequency change
    source.set_frequency(2e8)
    assert source.spdc.frequency == 2e8
    
    # Test invalid frequency
    with pytest.raises(ValueError, match="must be positive"):
        source.set_frequency(-1e8)
    
    with pytest.raises(ValueError, match="must be positive"):
        source.set_frequency(0)
    
    # Test mean photon number change
    source.set_mean_photon_num(0.5)
    assert source.spdc.mean_photon_num == 0.5
    
    # Test invalid mean
    with pytest.raises(ValueError, match="must be non-negative"):
        source.set_mean_photon_num(-0.1)
    
    # Edge case: zero is valid (no emission)
    source.set_mean_photon_num(0.0)
    assert source.spdc.mean_photon_num == 0.0


def test_SpdcSourceNode_emission_counting():
    """Test emission counting without timestamp storage."""
    np.random.seed(999)
    
    tl = Timeline()
    
    config = {
        'frequency': 1e9,
        'mean_photon_num': 0.5,
        'bell_state': 'phi+'
    }
    source = SpdcSourceNode("source", tl, config)
    
    # Create mock receivers
    class MockReceiver:
        def __init__(self):
            self.photons = []
        
        def receive_qubit(self, photon):
            self.photons.append(photon)
    
    recv0 = MockReceiver()
    recv1 = MockReceiver()
    source.qchannels = [recv0, recv1]
    
    tl.init()
    
    # Initial state
    assert source.emission_count == 0
    
    # First emission batch
    source.emit(num_pulses=100)
    tl.run()
    
    first_count = source.emission_count
    
    assert first_count > 0, "No photons emitted"
    assert len(recv0.photons) == first_count
    assert len(recv1.photons) == first_count
    
    # Second emission batch (should accumulate)
    source.emit(num_pulses=100)
    tl.run()
    
    second_count = source.emission_count
    assert second_count > first_count, "Count should accumulate"
    assert second_count == len(recv0.photons)


def test_SpdcSourceNode_default_config():
    """Test that source works with minimal/default configuration."""
    # Test with None config
    tl1 = Timeline()
    source1 = SpdcSourceNode("source1", tl1, config=None)
    
    # Should have reasonable defaults
    assert source1.spdc.frequency == 8e7
    assert source1.spdc.mean_photon_num == 0.1
    assert source1.spdc.bell_state_label == 'psi+'
    assert source1.spdc.wavelengths == [1550, 1550]
    
    # Test with empty config dict
    tl2 = Timeline()
    source2 = SpdcSourceNode("source2", tl2, config={})
    assert source2.spdc.frequency == 8e7
    
    # Test partial override
    tl3 = Timeline()
    source3 = SpdcSourceNode("source3", tl3, config={'frequency': 1e9})
    assert source3.spdc.frequency == 1e9
    assert source3.spdc.mean_photon_num == 0.1  # Still default


def test_SpdcSourceNode_component_ownership():
    """Test that SPDC source and ports have correct ownership."""
    tl = Timeline()
    
    config = {'bell_state': 'phi-'}
    source = SpdcSourceNode("test_source", tl, config)
    
    # Check SPDC ownership
    assert source.spdc.owner == source, "SPDC should be owned by source node"
    
    # Check ports ownership
    for port in source.ports.values():
        assert port.owner == source, "Ports should be owned by source node"
    
    # Check component registration
    assert source.spdc in source.components
    assert source.ports[0] in source.components
    assert source.ports[1] in source.components
    
    # Check first component
    assert source.first_component_name == source.spdc.name


def test_SpdcSourceNode_zero_emission():
    """Test behavior when mean_photon_num is set to zero."""
    np.random.seed(42)
    
    tl = Timeline()
    
    config = {
        'frequency': 1e8,
        'mean_photon_num': 0.0,  # No photons
        'bell_state': 'psi+'
    }
    source = SpdcSourceNode("source", tl, config)
    
    class MockReceiver:
        def __init__(self):
            self.photons = []
        
        def receive_qubit(self, photon):
            self.photons.append(photon)
    
    recv0 = MockReceiver()
    recv1 = MockReceiver()
    source.qchannels = [recv0, recv1]
    
    tl.init()
    source.emit(num_pulses=100)
    tl.run()
    
    # Should emit no photons
    assert source.emission_count == 0
    assert len(recv0.photons) == 0
    assert len(recv1.photons) == 0


import pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])