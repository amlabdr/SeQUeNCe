"""Models for optical wave plate components.

This module provides wave plate models (half-wave and quarter-wave plates)
for manipulating photon polarization states using Jones calculus.
"""

import numpy as np
from numpy import dot
from typing import TYPE_CHECKING, Literal
from sequence.components.photon import Photon
from sequence.kernel.entity import Entity
from sequence.utils.encoding import polarization

if TYPE_CHECKING:
    from sequence.kernel.timeline import Timeline


class WavePlate(Entity):
    """Optical wave plate for polarization manipulation.
    
    Models half-wave plates (HWP) and quarter-wave plates (QWP) using
    Jones matrix formalism.
    
    Wave plate effects:
    - **HWP**: Rotating the plate by angle θ rotates linear polarization by 2θ.
               Used for polarization rotation without changing polarization type.
    - **QWP**: Converts between linear and circular/elliptical polarization.
               Rotation angle changes ellipticity and handedness, not just rotation.
    
    Attributes:
        name (str): component name.
        timeline (Timeline): simulation timeline.
        plate_type (str): "HWP" for half-wave or "QWP" for quarter-wave.
        angle (float): wave plate rotation angle in radians.
        fidelity (float): transmission probability (accounts for insertion losses).
        encoding_type (dict): encoding scheme (default: polarization).
        unitary (np.ndarray): 2x2 Jones matrix for single photon.
        unitary_signal (np.ndarray): 4x4 matrix acting on first qubit of entangled pair.
        unitary_idler (np.ndarray): 4x4 matrix acting on second qubit of entangled pair.
    """
    
    def __init__(
        self,
        name: str,
        timeline: "Timeline",
        plate_type: Literal["HWP", "QWP"] = "HWP",
        angle: float = 0.0,
        fidelity: float = 1.0,
        encoding_type: dict = polarization
    ):
        """Constructor for wave plate.
        
        Args:
            name (str): component name.
            timeline (Timeline): simulation timeline.
            plate_type (str): "HWP" or "QWP" (default "HWP").
            angle (float): wave plate rotation angle in radians (default 0.0).
            fidelity (float): transmission probability 0-1 (default 1.0).
            encoding_type (dict): encoding scheme (default polarization).
        """
        Entity.__init__(self, name, timeline)
        
        assert plate_type in ["HWP", "QWP"], f"Invalid plate_type: {plate_type}"
        assert 0 <= fidelity <= 1, f"Fidelity must be between 0 and 1, got {fidelity}"
        
        self.plate_type = plate_type
        self.angle = angle
        self.fidelity = fidelity
        self.encoding_type = encoding_type
        self.owner = None  # Will be set by parent component
        
        # Compute Jones matrices once
        self.unitary = self._get_jones_matrix()
        self.unitary_signal = np.kron(self.unitary, np.identity(2))
        self.unitary_idler = np.kron(np.identity(2), self.unitary)

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""
        assert len(self._receivers) == 1, "WavePlate must have exactly 1 receiver"

    def get(self, photon: Photon, **kwargs) -> None:
        """Method to receive and transform a photon.
        
        Applies wave plate Jones matrix transformation to photon polarization state.
        
        Args:
            photon (Photon): photon to transform (must have polarization encoding).
            
        Side Effects:
            May discard photon if fidelity check fails.
            Modifies photon quantum state.
            Forwards photon to attached receiver.
        """
        assert photon.encoding_type["name"] == self.encoding_type["name"], \
            f"Photon encoding {photon.encoding_type['name']} doesn't match wave plate {self.encoding_type['name']}"
        
        # Apply fidelity (photon loss due to insertion loss)
        if self.fidelity < 1.0:
            if self.owner and hasattr(self.owner, 'get_generator'):
                if self.owner.get_generator().random() > self.fidelity:
                    # Photon lost
                    return
        
        full_state = photon.quantum_state.state
        
        # Handle different state dimensions
        if len(full_state) == 2:
            # Single-photon polarization state (2D)
            new_state = dot(self.unitary, full_state)
            new_state_tuple = tuple(complex(x) for x in new_state)
            photon.set_state(new_state_tuple)
            
        elif len(full_state) == 4:
            # Entangled two-photon state (4D)            
            if photon.name == "signal":
                op = self.unitary_signal  # Act on first qubit (signal)
            elif photon.name == "idler":
                op = self.unitary_idler   # Act on second qubit (idler)
            else:
                raise ValueError(f"Photon name must be 'signal' or 'idler', got {photon.name}")
            
            new_state = dot(op, full_state)
            photon.set_state(tuple(new_state))
            
        else:
            raise ValueError(f"Unexpected photon state dimension: {len(full_state)}")
        
        # Forward to next component
        self._receivers[0].get(photon, **kwargs)

    def set_angle(self, theta: float) -> None:
        """Update wave plate rotation angle.
        
        Args:
            theta (float): new wave plate angle in radians.
            
        Side Effects:
            Recomputes all Jones matrices.
        """
        self.angle = theta
        self.unitary = self._get_jones_matrix()
        self.unitary_signal = np.kron(self.unitary, np.identity(2))
        self.unitary_idler = np.kron(np.identity(2), self.unitary)

    def _get_jones_matrix(self) -> np.ndarray:
        """Compute Jones matrix for current plate type and angle.
        
        Returns:
            np.ndarray: 2x2 Jones matrix.
        """
        theta = self.angle
        
        if self.plate_type == "HWP":
            # Half-wave plate: rotates polarization by 2θ
            c = np.cos(2 * theta)
            s = np.sin(2 * theta)
            return np.array([[c, s], [s, -c]], dtype=complex)
            
        elif self.plate_type == "QWP":
            # Quarter-wave plate: introduces π/2 phase shift
            c = np.cos(theta)
            s = np.sin(theta)
            return np.array([
                [c**2 + 1j * s**2, (1 - 1j) * c * s],
                [(1 - 1j) * c * s, s**2 + 1j * c**2]
            ], dtype=complex)
        
        else:
            raise ValueError(f"Unknown plate_type: {self.plate_type}")
