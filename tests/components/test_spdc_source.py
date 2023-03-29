from numpy import random
from sequence.components.light_source import SPDCSource
from sequence.kernel.timeline import Timeline
from sequence.utils.encoding import polarization

random.seed(0)


class Receiver:
    def __init__(self, timeline):
        self.timeline = timeline
        self.log = []

    def get(self, photon):
        self.log.append((self.timeline.now(), photon))


def test_spdc_light_source():
    tl = Timeline()
    FREQ, MEAN = 1e8, 1
    spdc = SPDCSource("spdc", tl, frequency=FREQ, mean_photon_num=MEAN,random_frequency=True)
    receiver1 = Receiver(tl)
    receiver2 = Receiver(tl)
    spdc.add_receiver(receiver1)
    spdc.add_receiver(receiver2)

    state_list = []
    STATE_LEN = 1000
    for _ in range(STATE_LEN):
        basis = random.randint(2)
        bit = random.randint(2)
        state_list.append(polarization["bases"][1][0])

    tl.init()
    spdc.emit(state_list)
    tl.run()

    #assert (len(receiver.log) / STATE_LEN) - MEAN < 0.1
    for time, qubit in receiver1.log:
        #index = int(qubit.name)
        print("state: ", qubit.quantum_state.state)
        print("arrived in time: ", time)
        #assert state_list[index] == qubit.quantum_state.state
        #assert time == index * (1e12 / FREQ)


test_spdc_light_source()