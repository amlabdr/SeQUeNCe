from sequence.topology.node import Node
from sequence.components.light_source import SPDCSource
from sequence.kernel.entity import Entity
from sequence.utils.encoding import polarization
import os, json

class source_port(Entity):
    def __init__(self, name, timeline, owner:Node):
        super().__init__(name, timeline)
        self.owner = owner
        self.add_receiver(owner)

    def init(self):
        pass
    def get(self, photon, **kwargs):
        photon.name = self.name
        self._receivers[0].get(photon)


class SourceNode(Node):
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
            'encoding': polarization
        }

        # Merge provided config with defaults
        if config is not None:
            merged_config = {**default_config, **config}
        else:
            merged_config = default_config

        # Initialize SPDCSource
        self.spdc = SPDCSource(
            name=self.name + "_SPDC",
            timeline=self.timeline,
            wavelengths=merged_config['wavelengths'],
            frequency=float(merged_config['frequency']),
            mean_photon_num=float(merged_config['mean_photon_num']),
            phase_error=float(merged_config['phase_error']),
            bandwidth=float(merged_config['bandwidth']),
            encoding_type=merged_config['encoding']
        )

        self.ports = {}
        for i in range (2):
            self.ports[i] = source_port(str(i), self.timeline, self)
            self.spdc.add_receiver(self.ports[i])
        self.first_component_name = self.spdc.name

    def get(self, photon, **kwargs):
        index = 0
        if photon.name == "0":  # Log only for the first photon in the pair
            self.emission_count += 1
            self.timestamps.append(self.timeline.now())
            
        for dst in self.qchannels:
            if int(photon.name) == index:
                self.send_qubit(dst, photon)
                break
            index +=1
    
    def export_timestamps(self, directory="logs"):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name}_timestamps.json")
        with open(path, "w") as f:
            json.dump(self.timestamps, f, indent=2)