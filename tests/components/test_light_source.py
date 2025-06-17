from numpy import random
from sequence.components.light_source import LightSource, SPDCBellSource
from sequence.kernel.timeline import Timeline
from sequence.utils.encoding import polarization
from math import isclose

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

def test_spdc_bell_source():
    FREQ = 1e8
    MEAN = 0.1
    BELL_STATE = "psi+"
    STATE_VEC = SPDCBellSource.bell_state_map[BELL_STATE]
    TOL = 1e-5

    # Setup timeline and source
    tl = Timeline()
    source = SPDCBellSource("bell_src", tl, frequency=FREQ, mean_photon_num=MEAN, bell_state=BELL_STATE)
    
    # Attach receivers
    receiver0 = Receiver(tl)
    receiver1 = Receiver(tl)
    receiver0.name = "recv0"
    receiver1.name = "recv1"
    source.add_receiver(receiver0)
    source.add_receiver(receiver1)

    # Emit N pulses and run
    N = 1000
    tl.init()
    source.emit(N)
    tl.run()

    # Collect received photons
    assert len(receiver0.log) == len(receiver1.log)
    received = len(receiver0.log)
    expected = MEAN * N

    # Check that emission rate is statistically plausible
    assert abs(received - expected) / expected < 0.3, "Unexpected number of emitted pairs"

    # Check that all states match the Bell state
    for (t0, photon0), (t1, photon1) in zip(receiver0.log, receiver1.log):
        assert t0 == t1  # Coincident
        state = photon0.quantum_state.state
        assert isclose(abs(state[0]), abs(STATE_VEC[0]), rel_tol=TOL)
        assert isclose(abs(state[1]), abs(STATE_VEC[1]), rel_tol=TOL)
        assert isclose(abs(state[2]), abs(STATE_VEC[2]), rel_tol=TOL)
        assert isclose(abs(state[3]), abs(STATE_VEC[3]), rel_tol=TOL)

    print("Test passed: SPDCBellSource emits correct Bell states with expected frequency.")

