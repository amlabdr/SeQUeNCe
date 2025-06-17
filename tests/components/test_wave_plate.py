from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sequence.kernel.timeline import Timeline

import numpy as np
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node
from sequence.utils.encoding import polarization
from sequence.components.optical_channel import QuantumChannel
from sequence.components.wave_plate import WavePlate
from sequence.components.spdc_source_node import SpdcSourceNode
from sequence.components.photon import Photon
from numpy import random


class MiddleNode(Node):
    def __init__(self, name, timeline, wp):
        super().__init__(name, timeline)
        self.wp = wp
        self.wp.owner = self
        self.last_qubit_state = None

    def receive_qubit(self, src, qubit):
        self.wp.get(qubit)
        self.last_qubit_state = qubit.quantum_state.state

    def get(self, photon: "Photon", **kwargs):
        for index, dst in enumerate(self.qchannels):
            self.send_qubit(dst, photon)


class Receiver(Node):
    def __init__(self, name, timeline):
        super().__init__(name, timeline)
        self.log = []

    def receive_qubit(self, src, qubit):
        self.log.append((self.timeline.now(), src, qubit))


def test_waveplate_with_spdc():
    tl = Timeline()
    LS_FREQ = 8e7
    MEAN = 0.1
    ANGLE = np.pi / 4  # 45°

    # Setup source node using your custom SourceNode class
    source = SpdcSourceNode("source", tl, config={
        "frequency": LS_FREQ,
        "mean_photon_num": MEAN,
        "encoding": polarization,
        "bell_state": "psi+"
    })

    # Setup wave plate and receiver
    wp = WavePlate("wp", tl, plate_type="HWP", angle=ANGLE)
    receiver = Receiver("receiver", tl)
    middle = MiddleNode("mid", tl, wp)
    wp.add_receiver(middle)

    source.set_seed(0)
    middle.set_seed(1)
    receiver.set_seed(2)

    # Create quantum channels from source to middle and middle to receiver
    qc0 = QuantumChannel("qc0", tl, distance=1e5, attenuation=0)
    qc1 = QuantumChannel("qc1", tl, distance=1e5, attenuation=0)
    qc0.set_ends(source, middle.name)
    qc1.set_ends(middle, receiver.name)

    # Start simulation
    tl.init()
    source.spdc.emit(num_pulses=1000)
    tl.run()

    # Test: photon 0 should have been flipped from |H⟩ = [1, 0] → |V⟩ = [0, 1]
    expected_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    assert len(receiver.log) > 0
    for _, _, qubit in receiver.log:
        print(f"Received qubit state: {qubit.quantum_state.state}")
        actual = qubit.quantum_state.state
        np.testing.assert_almost_equal(np.abs(actual), np.abs(expected_state), decimal=5)


