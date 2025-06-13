from sequence.components.spdc_source_node import SourceNode
from numpy import random
from sequence.components.light_source import LightSource
from sequence.kernel.timeline import Timeline
from sequence.utils.encoding import polarization

def test_spdc_configuration_loaded_correctly():

    config = {
        'wavelengths': [1550, 1550],
        'frequency': '8e6',
        'mean_photon_num': '0.1',
        'phase_error': '0.0',
        'bandwidth': '0',
        
    }

    timeline = Timeline()
    source_node = SourceNode("src", timeline, config)

    spdc = source_node.spdc
    assert spdc.wavelengths == [1550, 1550]
    assert spdc.frequency == 8e6
    assert spdc.mean_photon_num == 0.1

# SPDC emits and routes photons to both source_port receivers
def test_spdc_emits_and_routes_photons():

    # Track received photons per port
    class ReceiverSpy:
        def __init__(self, name=None):
            self.name = name
            self.received = []
        def get(self, photon, **kwargs):
            #print(f"Received photon: {photon} from {self.name}" )
            self.received.append(photon)

    config = {
        'wavelengths': [1550, 1550],
        'frequency': '1e6',  # Low for test
        'mean_photon_num': '1',
        'phase_error': '0.0',
        'bandwidth': '0'
    }

    timeline = Timeline()
    source_node = SourceNode("src", timeline, config)

    # Replace ports with spies
    port_spy_0 = ReceiverSpy('0')
    port_spy_1 = ReceiverSpy('1')
    source_node.ports[0]._receivers[0] = port_spy_0
    source_node.ports[1]._receivers[0] = port_spy_1

    source_node.spdc.init()
    state_list = []
    STATE_LEN = 5
    for _ in range(STATE_LEN):
        basis = random.randint(2)
        bit = random.randint(2)
        state_list.append(polarization["bases"][basis][bit])
    source_node.spdc.emit(state_list)  # Emit 5 state-pairs

    timeline.run()

    #print(f"Port 0 received {len(port_spy_0.received)} photons")
    #print(f"Port 1 received {len(port_spy_1.received)} photons")
    

    # We expect that both ports received something
    assert len(port_spy_0.received) > 0
    assert len(port_spy_1.received) > 0

def test_source_node_get_routes_photon_correctly():    
    class FakeChannel:
        def __init__(self):
            self.sent = []
        def send_qubit(self, dst, photon):
            #print(f"Sending photon: {photon} to {dst}")
            self.sent.append((dst, photon))

    timeline = Timeline()
    config = {
        'wavelengths': [1550, 1550],
        'frequency': '1e6',  # Low for test
        'mean_photon_num': '1',
        'phase_error': '0.0',
        'bandwidth': '0'
    }

    source_node = SourceNode("src", timeline, config)

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

    #print("Fake channel 0 sent:", fake_channel_0.sent)
    #print("Fake channel 1 sent:", fake_channel_1.sent)
    assert fake_channel_0.sent[0][1].name == "0"
    assert fake_channel_1.sent[0][1].name == "1"




test_spdc_configuration_loaded_correctly()
test_spdc_emits_and_routes_photons()
test_source_node_get_routes_photon_correctly()
