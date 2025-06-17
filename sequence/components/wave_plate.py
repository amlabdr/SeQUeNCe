import numpy as np
from numpy import dot
from typing import TYPE_CHECKING
from .photon import Photon
from ..kernel.entity import Entity
from ..utils.encoding import polarization

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline


class WavePlate(Entity):
    def __init__(self, name: str, timeline: "Timeline", plate_type="HWP", angle=0.0, encoding_type=polarization):
        super().__init__(name, timeline)
        assert plate_type in ["HWP", "QWP"], "Invalid wave plate type"
        self.plate_type = plate_type
        self.angle = angle
        self.encoding_type = encoding_type

    def init(self):
        assert len(self._receivers) == 1
        pass

    def get(self, photon: Photon, **kwargs):
        assert photon.encoding_type["name"] == self.encoding_type["name"]
        full_state = photon.quantum_state.state
        jones = self._get_jones_matrix()
        which = kwargs.get("which")  

        # Only handle 2D or 4D polarization vectors
        if len(full_state) == 2:
            # Single-photon polarization (e.g., before entanglement)
            new_state = tuple(dot(jones, full_state))
            photon.set_state(new_state.tolist())
        elif len(full_state) == 4:
            if which == "signal":
                op = np.kron(jones, np.identity(2))  # act on qubit 0
            elif which == "idler":
                op = np.kron(np.identity(2), jones)  # act on qubit 1
            else:
                raise ValueError("For entangled states, specify which='signal' or 'idler'")
            
            new_state = dot(op, full_state)
            photon.set_state(tuple(new_state))

        else:
            raise ValueError("Unexpected photon state dimension")

        self._receivers[0].get(photon)

    def set_angle(self, theta: float):
        self.angle = theta

    def _get_jones_matrix(self):
        theta = self.angle
        c = np.cos(2 * theta)
        s = np.sin(2 * theta)

        if self.plate_type == "HWP":
            return np.array([[c, s], [s, -c]])
        else:  # QWP
            return np.array([[1j * c + s, (1j - 1) * s],
                             [(1j - 1) * s, -1j * c + s]])
