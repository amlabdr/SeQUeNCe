"""Models for simulation of wave plate.

This module introduces the HalfWavePlates class.
The wave plate modifies the phase of incoming photons, but does not add additional delay or losses.
"""
from numpy import pi, cos, sin, multiply
import numpy as np
from cmath import exp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .photon import Photon

from ..kernel.entity import Entity
from .circuit import Circuit


class HalfWavePlate(Entity):
    """Class implementing a simple half wave plate.

    Attributes:
        name (str): name of the wave plate instance.
        timeline (Timeline): simulation timeline.
        angle with fast axis at angle
        fidelity (float): fraction of qubits not lost on the reflective surface
    """

    def __init__(self, name, timeline, angle = 0, fidelity=1):
        """Constructor for wave plate class.

        Args:
            name (str): name of the wave plate.
            timeline (Timeline): simulation timeline.
            angle with fast axis at angle: (default 0)
            phase (float): phase to apply to incoming photons (default 0.0).
        """

        super().__init__(name, timeline)
        self.fidelity = fidelity
        self.angle = angle

    def init(self):
        """Implementation of Entity interface (see base class)."""

        assert len(self._receivers) == 1, "BeamSplitter should only be attached to 1 output."


    def set_angle(self, angle: float):
        """Method to change the angle with fast axis 
        Args:
            angle (float): new phase to use.
        """
        self.angle = angle


    def get(self, photon: "Photon", **kwargs):
        """Method to receive a photon for measurement.

        Args:
            photon (Photon): photon to measure (must have polarization encoding)

        """

        assert photon.encoding_type["name"] == "polarization", "Beamsplitter should only be used with polarization."
        rng = self.get_generator()
        state = photon.quantum_state.state
        if rng.random() < self.fidelity:
            if rng.random() < self.phase_error:
                theta = self.angle
                c = exp(-1j * pi / 2)
                hwp_matrix = np.multiply(exp(-1j * pi / 2), np.array([[cos(theta)**2 - sin(theta)**2 , 2*cos(theta)*sin(theta)],
                                                            [2*cos(theta)*sin(theta),   cos(theta)**2 - sin(theta)**2]]))
                state = np.dot(hwp_matrix, state)
                photon.set_state(state)