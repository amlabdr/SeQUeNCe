from sequence.topology.node import Node
from sequence.components.light_source import SPDCBellSource
from sequence.kernel.entity import Entity
from sequence.utils.encoding import polarization
import os, json

class SourcePort(Entity):
    def __init__(self, name, timeline, owner:Node):
        super().__init__(name, timeline)
        self.owner = owner
        self.add_receiver(owner)

    def init(self):
        pass

    def get(self, photon, **kwargs):
        photon.name = self.name
        self._receivers[0].get(photon)

class SpdcSourceNode(Node):
    """
    Node that emits entangled photon pairs using an SPDCBellSource.
    """

    def __init__(self, name, timeline, config):
        super().__init__(name, timeline)
        self.name = name
        self.emission_count = 0
        self.timestamps = []

        # Default values for SPDC configuration
        default_config = {
            'wavelengths': [1550, 1550],
            'frequency': 8e7,
            'mean_photon_num': 0.1,
            'phase_error': 0.0,
            'bandwidth': 0,
            'encoding': polarization,
            'bell_state': 'psi+'
        }

        # Merge with user config
        merged_config = {**default_config, **(config or {})}

        # Create the Bell-state SPDC source
        self.spdc = SPDCBellSource(
            name=self.name + "_SPDC",
            timeline=self.timeline,
            wavelengths=merged_config['wavelengths'],
            frequency=float(merged_config['frequency']),
            mean_photon_num=float(merged_config['mean_photon_num']),
            phase_error=float(merged_config['phase_error']),
            bandwidth=float(merged_config['bandwidth']),
            encoding_type=merged_config['encoding'],
            bell_state=merged_config['bell_state']
        )

        # Create and connect output ports
        self.ports = {}
        for i in range(2):
            self.ports[i] = SourcePort(str(i), self.timeline, self)
            self.spdc.add_receiver(self.ports[i])

        self.first_component_name = self.spdc.name

    def get(self, photon, **kwargs):
        if photon.name == "0":  # Only log photon 0 (assume it's consistent)
            self.emission_count += 1
            self.timestamps.append(self.timeline.now())

        for index, dst in enumerate(self.qchannels):
            if photon.name == str(index):
                self.send_qubit(dst, photon)
                break
    
    def export_timestamps(self, directory="logs"):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name}_timestamps.json")
        with open(path, "w") as f:
            json.dump(self.timestamps, f, indent=2)