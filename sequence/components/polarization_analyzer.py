from sequence.topology.node import Node
from sequence.components.detector import FixedBasisPolarizationDetector
from sequence.components.wave_plate import WavePlate
from sequence.components.optical_channel import QuantumChannel

class PolarizationAnalyzer(Node):
    """
    Component that models a polarization analyzer: 
    A half-wave plate followed by a polarization-sensitive detector.
    
    Attributes:
        wp (WavePlate): half-wave plate used to rotate polarization basis.
        detector (QSDetectorPolarization): polarization detector with two outputs.
    """

    def __init__(self, name: str, timeline, angle: float, which = None):
        super().__init__(name, timeline)

        # Half-Wave Plate
        self.wp = WavePlate(f"{name}_HWP", timeline, "HWP", angle=angle)
        self.which = which  # Optional parameter to specify which qubit to act on signal or idler

        # Polarization-sensitive QSDetector
        self.detector = FixedBasisPolarizationDetector(f"{name}_detector", timeline, basis_index = 0)

        self.wp.add_receiver(self.detector)
        self.add_component(self.wp)
        self.add_component(self.detector)
        self.add_component(self)
        self.set_first_component(self.name)
        
        
    def init(self):
        self.wp.init()
        self.detector.init()

    def get(self, photon, **kwargs):
        """Receive photon and pass it through HWP then polarization detector."""
        self.wp.get(photon, which = self.which)

    def get_detection_counts(self):
        """Returns the number of detections on each detector."""
        return self.detector.get_photon_times()