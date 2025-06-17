from sequence.components.spdc_source_node import SpdcSourceNode
from numpy import random
from sequence.kernel.timeline import Timeline
from sequence.utils.encoding import polarization

def test_spdc_configuration_loaded_correctly():

    config = {
        'wavelengths': [1550, 1550],
        'frequency': '8e6',
        'mean_photon_num': '0.1',
        'phase_error': '0.0',
        'bandwidth': '0',
        'bell_state': 'phi+'
    }

    timeline = Timeline()
    source_node = SpdcSourceNode("src", timeline, config)

    spdc = source_node.spdc
    assert spdc.wavelengths == [1550, 1550]
    assert spdc.frequency == 8e6
    assert spdc.mean_photon_num == 0.1
    assert spdc.bell_state_label == "phi+"

# SPDC emits and routes photons to both source_port receivers
def test_spdc_emits_and_routes_photons():

    # Track received photons per port
    class ReceiverSpy:
        def __init__(self, name=None):
            self.received = []
        def get(self, photon, **kwargs):
            self.received.append(photon)

    config = {
        'wavelengths': [1550, 1550],
        'frequency': '1e6',  # Low for test
        'mean_photon_num': '1',
        'phase_error': '0.0',
        'bandwidth': '0',
        'bell_state': 'psi-'
    }

    timeline = Timeline()
    source_node = SpdcSourceNode("src", timeline, config)

    # Replace ports with spies
    port_spy_0 = ReceiverSpy('0')
    port_spy_1 = ReceiverSpy('1')
    source_node.ports[0]._receivers[0] = port_spy_0
    source_node.ports[1]._receivers[0] = port_spy_1

    source_node.spdc.init()
    source_node.spdc.emit(num_pulses=5)
    timeline.run()

    assert len(port_spy_0.received) > 0
    assert len(port_spy_1.received) > 0

def test_source_node_get_routes_photon_correctly():    
    class FakeChannel:
        def __init__(self):
            self.sent = []
        def send_qubit(self, dst, photon):
            self.sent.append((dst, photon))

    timeline = Timeline()
    config = {
        'wavelengths': [1550, 1550],
        'frequency': '1e6',  # Low for test
        'mean_photon_num': '1',
        'phase_error': '0.0',
        'bandwidth': '0',
        'bell_state': 'psi+'
    }

    source_node = SpdcSourceNode("src", timeline, config)

    # Setup fake qchannels
    fake_channel_0 = FakeChannel()
    fake_channel_1 = FakeChannel()
    source_node.qchannels = {"det0": fake_channel_0, "det1": fake_channel_1}

    class Photon:
        def __init__(self, name): self.name = name

    photon_0 = Photon("0")
    photon_1 = Photon("1")

    source_node.send_qubit = lambda dst, photon: fake_channel_0.sent.append((dst, photon)) if photon.name == "0" else fake_channel_1.sent.append((dst, photon))

    source_node.get(photon_0)
    source_node.get(photon_1)

    assert fake_channel_0.sent[0][1].name == "0"
    assert fake_channel_1.sent[0][1].name == "1"
