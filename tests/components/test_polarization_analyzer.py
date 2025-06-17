import numpy as np
import matplotlib.pyplot as plt
from sequence.kernel.timeline import Timeline
from sequence.components.photon import Photon
from sequence.utils.encoding import polarization
from sequence.components.polarization_analyzer import PolarizationAnalyzer  # Replace with actual import
from numpy import sqrt
from sequence.components.spdc_source_node import SpdcSourceNode
from sequence.topology.node import Node
from sequence.components.optical_channel import QuantumChannel
def test_polarization_analyzer(angle_rad, num_pulses=10000):
    tl = Timeline()
    # Setup source
    source = SpdcSourceNode("source", tl, config={
        "frequency": 1e6,
        "mean_photon_num": 0.1,
        "bell_state": "psi+"
    })

    analyzer = PolarizationAnalyzer("analyzer", tl, angle=angle_rad)

    # Connect source to analyzer and sink
    qc_signal = QuantumChannel("qc_signal", tl, distance=1e5, attenuation=0)
    qc_signal.set_ends(source, analyzer.name)

    # Run
    tl.init()
    source.spdc.emit(num_pulses=num_pulses)
    tl.run()

    return analyzer.get_detection_counts()

res = test_polarization_analyzer(45)

