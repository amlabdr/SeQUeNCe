from numpy import random
from sequence.components.light_source import LightSource, SPDCBellSource
from sequence.kernel.timeline import Timeline
from sequence.utils.encoding import polarization
from math import sqrt

random.seed(0)


class Receiver:
    def __init__(self, timeline):
        self.timeline = timeline
        self.log = []
        self.name = ""

    def get(self, photon):
        self.log.append((self.timeline.now(), photon))


def test_light_source():
    tl = Timeline()
    FREQ, MEAN = 1e8, 0.1
    ls = LightSource("ls", tl, frequency=FREQ, mean_photon_num=MEAN)
    receiver = Receiver(tl)
    ls.add_receiver(receiver)

    state_list = []
    STATE_LEN = 1000
    for _ in range(STATE_LEN):
        basis = random.randint(2)
        bit = random.randint(2)
        state_list.append(polarization["bases"][basis][bit])

    tl.init()
    ls.emit(state_list)
    tl.run()

    assert (len(receiver.log) / STATE_LEN) - MEAN < 0.1
    for time, qubit in receiver.log:
        index = int(qubit.name)
        assert state_list[index] == qubit.quantum_state.state
        assert time == index * (1e12 / FREQ)


def test_SPDCBellSource_basic_emission():
    """Test that SPDCBellSource emits photon pairs to two receivers."""
    random.seed(42)
    
    tl = Timeline()
    source = SPDCBellSource("spdc", tl, frequency=1e8, mean_photon_num=0.1, 
                           photon_statistics="poisson", bell_state="psi-")
    
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    source.add_receiver(receiver0)
    source.add_receiver(receiver1)
    
    tl.init()
    source.emit(num_pulses=100)
    tl.run()
    
    # Both receivers should get same number of photons
    assert len(receiver0.log) == len(receiver1.log)
    
    # Should have some photons (Poisson with mean 0.1 × 100 ≈ 10)
    assert len(receiver0.log) > 0
    assert len(receiver0.log) < 50
    
    # Photon pairs should arrive at same time
    for i in range(len(receiver0.log)):
        time0, photon0 = receiver0.log[i]
        time1, photon1 = receiver1.log[i]
        assert time0 == time1


def test_SPDCBellSource_bell_states():
    """Test that different Bell states are correctly set."""
    random.seed(0)
    
    bell_states = ["phi+", "phi-", "psi+", "psi-"]
    expected_states = {
        "phi+": (1/sqrt(2), 0, 0, 1/sqrt(2)),
        "phi-": (1/sqrt(2), 0, 0, -1/sqrt(2)),
        "psi+": (0, 1/sqrt(2), 1/sqrt(2), 0),
        "psi-": (0, 1/sqrt(2), -1/sqrt(2), 0)
    }
    
    for bell_state in bell_states:
        tl = Timeline()
        source = SPDCBellSource("spdc", tl, frequency=1e8, mean_photon_num=1.0,
                               photon_statistics="poisson", bell_state=bell_state)
        
        receiver0 = Receiver(tl)
        receiver1 = Receiver(tl)
        source.add_receiver(receiver0)
        source.add_receiver(receiver1)
        
        tl.init()
        source.emit(num_pulses=10)
        tl.run()
        
        assert len(receiver0.log) > 0
        assert source.bell_state_label == bell_state
        assert source.bell_state == expected_states[bell_state]


def test_SPDCBellSource_wavelength_correlation():
    """Test that signal and idler wavelengths satisfy energy conservation."""
    random.seed(123)
    
    tl = Timeline()
    wavelengths = [1540, 1560]  # 20nm range
    source = SPDCBellSource("spdc", tl, wavelengths=wavelengths, 
                           frequency=1e8, mean_photon_num=1.0,
                           photon_statistics="poisson", bell_state="psi+")
    
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    source.add_receiver(receiver0)
    source.add_receiver(receiver1)
    
    tl.init()
    source.emit(num_pulses=50)
    tl.run()
    
    lam0 = 0.5 * (wavelengths[0] + wavelengths[1])
    
    for i in range(len(receiver0.log)):
        _, photon0 = receiver0.log[i]
        _, photon1 = receiver1.log[i]
        
        lambda_signal = photon0.wavelength
        lambda_idler = photon1.wavelength
        
        # Check energy conservation:
        product = lambda_signal * lambda_idler
        expected_product = lam0 * lam0
        
        relative_error = abs(product - expected_product) / expected_product
        assert relative_error < 0.01


def test_SPDCBellSource_photon_statistics():
    """Test thermal vs Poisson photon statistics."""
    random.seed(999)
    
    NUM_PULSES = 1000
    MEAN_PHOTON_NUM = 0.5
    
    # Poisson
    tl_poisson = Timeline()
    source_poisson = SPDCBellSource("spdc_p", tl_poisson, frequency=1e9, 
                                   mean_photon_num=MEAN_PHOTON_NUM,
                                   photon_statistics="poisson", bell_state="phi+")
    receiver_p0 = Receiver(tl_poisson)
    receiver_p1 = Receiver(tl_poisson)
    source_poisson.add_receiver(receiver_p0)
    source_poisson.add_receiver(receiver_p1)
    
    tl_poisson.init()
    source_poisson.emit(num_pulses=NUM_PULSES)
    tl_poisson.run()
    
    # Thermal
    tl_thermal = Timeline()
    source_thermal = SPDCBellSource("spdc_t", tl_thermal, frequency=1e9,
                                   mean_photon_num=MEAN_PHOTON_NUM,
                                   photon_statistics="thermal", bell_state="phi+")
    receiver_t0 = Receiver(tl_thermal)
    receiver_t1 = Receiver(tl_thermal)
    source_thermal.add_receiver(receiver_t0)
    source_thermal.add_receiver(receiver_t1)
    
    tl_thermal.init()
    source_thermal.emit(num_pulses=NUM_PULSES)
    tl_thermal.run()
    
    # Both should emit photons
    assert len(receiver_p0.log) > 0
    assert len(receiver_t0.log) > 0
    
    # Check counts are reasonable
    expected = MEAN_PHOTON_NUM * NUM_PULSES
    assert 0.5 * expected < len(receiver_p0.log) < 1.5 * expected
    assert 0.5 * expected < len(receiver_t0.log) < 1.5 * expected

import pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])