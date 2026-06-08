"""Models for simulation of a polarization beam splitter.

This module defines the class BeamSplitter, which is used for simulating polarization beam splitters. 
The beam splitter receives photons with polarization encoding and forwards photons to one of two 
attached receivers (which can be any entity).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from ..topology.node import Node

from numpy import trace
import numpy as np

from .photon import Photon
from ..kernel.quantum_utils import povm_0
from ..utils.encoding import polarization
from ..kernel.entity import Entity
import random



class BeamSplitter(Entity):
    """Class modeling a polarization beamsplitter.

    Simulates operation of a polarization beam splitter (PBS).
    The BeamSplitter class can be configured to measure polarization in different bases at different times.
    
    Attributes:
        name (str): label for beamsplitter instance.
        timeline (Timeline): timeline for simulation.
        fidelity (float): probability of transmitting a received photon.
        start_time (int): start time (in ps) of photon interaction.
        frequency (float): frequency with which to switch measurement bases.
        basis_list (list[int]): 0/1 indices of measurement bases over time.
    """

    def __init__(self, name: str, timeline: "Timeline", fidelity=1):
        """Constructor for the beamsplitter class.

        Args:
            name (str): name of the beamsplitter instance.
            timeline (Timeline): simulation timeline.
            fidelity (float): probability of transmitting a received photon (default 1).
        """

        Entity.__init__(self, name, timeline)  # Splitter is part of the QSDetector, and does not have its own name
        self.fidelity = fidelity
        # for BB84
        self.start_time = 0
        self.frequency = 0
        self.basis_list = []

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""

        assert len(self._receivers) == 2, "BeamSplitter should only be attached to 2 outputs."

    def get(self, photon, **kwargs) -> None:
        """Method to receive a photon for measurement.

        Args:
            photon (Photon): photon to measure (must have polarization encoding)

        Side Effects:
            May call get method of one receiver.
        """

        assert photon.encoding_type["name"] == "polarization", "Beamsplitter should only be used with polarization."

        if self.get_generator().random() < self.fidelity:
            index = int((self.timeline.now() - self.start_time) * self.frequency * 1e-12)

            if 0 > index or index >= len(self.basis_list):
                return

            res = Photon.measure(polarization["bases"][self.basis_list[index]],
                                 photon, self.get_generator())
            self._receivers[res].get(photon)

    def set_basis_list(self, basis_list: list[int], start_time: int, frequency: float) -> None:
        """Sets the basis_list, start_time, and frequency attributes."""

        self.basis_list = basis_list
        self.start_time = start_time
        self.frequency = frequency


class HOMBeamSplitter(Entity):
    """50:50 beam splitter model for Hong-Ou-Mandel interference.

    This component provides event-level sampling for one-photon-per-input HOM events.
    It does not require both photons to have a shared source pulse id. Instead, users
    should provide arrival times from network events and pair photons externally.

    The model uses:
    - relative delay between two input arms,
    - temporal wavepacket width derived from source bandwidth,
    - mode overlap / distinguishability factor.

    For one photon in each input arm:
    - distinguishable limit -> coincidence probability 0.5,
    - ideal indistinguishable at zero delay -> full bunching, zero coincidence.
    """

    def __init__(self, name: str, timeline):
        Entity.__init__(self, name, timeline)

    def init(self) -> None:
        pass

    @staticmethod
    def temporal_overlap(delta_t_ps: float, lambda_nm: float, bandwidth_nm: float) -> float:
        """Gaussian temporal mode overlap in [0, 1].

        Temporal width is derived from source spectral width:
        """
        c_nm_per_ps = 299792.458
        if bandwidth_nm is None or bandwidth_nm <= 0:
            # Monochromatic limit -> very long coherence time.
            return 1.0
        if lambda_nm is None or lambda_nm <= 0:
            return 0.0
        temporal_width_ps = (lambda_nm ** 2) / (c_nm_per_ps * bandwidth_nm)
        if temporal_width_ps <= 0:
            return 0.0
        return float(np.exp(-(delta_t_ps / temporal_width_ps) ** 2))

    @staticmethod
    def _sigma_omega_from_bandwidth(lambda_nm: float, bandwidth_nm: float) -> float:
        """Approximate spectral width sigma in angular-frequency units (rad/ps).

        Assumes `bandwidth_nm` is FWHM in wavelength domain and converts via
        sigma_lambda = FWHM / 2.355 and delta_omega ~= (2*pi*c/lambda^2) * delta_lambda.
        """
        if lambda_nm is None or lambda_nm <= 0 or bandwidth_nm is None or bandwidth_nm <= 0:
            return 0.0
        c_nm_per_ps = 299792.458
        sigma_lambda_nm = float(bandwidth_nm) / 2.355
        return float((2.0 * np.pi * c_nm_per_ps / (float(lambda_nm) ** 2)) * sigma_lambda_nm)

    @staticmethod
    def _single_photon_density(photon: Photon) -> np.ndarray | None:
        """Extract single-photon polarization density matrix from photon state."""
        try:
            state = np.asarray(photon.quantum_state.state, dtype=complex).flatten()
        except Exception:
            return None

        if state.size == 2:
            norm = np.linalg.norm(state)
            if norm <= 0:
                return None
            ket = state / norm
            return np.outer(ket, np.conj(ket))

        if state.size == 4:
            amp = state.reshape(2, 2)
            # In this codebase, signal is first subsystem, idler second.
            name = str(getattr(photon, "name", "")).lower()
            if name == "signal":
                rho = amp @ np.conj(amp.T)   # trace out idler
            elif name == "idler":
                rho = amp.T @ np.conj(amp)   # trace out signal
            else:
                rho = amp @ np.conj(amp.T)
            tr = float(np.real(np.trace(rho)))
            if tr <= 0:
                return None
            return rho / tr

        return None

    @classmethod
    def polarization_overlap_from_states(cls, photon_arm0: Photon | None, photon_arm1: Photon | None) -> float:
        """State-based polarization overlap in [0,1].

        Uses Tr(rho_A rho_B), which reduces to |<p_A|p_B>|^2 for pure states.
        """
        if photon_arm0 is None or photon_arm1 is None:
            return 1.0
        rho0 = cls._single_photon_density(photon_arm0)
        rho1 = cls._single_photon_density(photon_arm1)
        if rho0 is None or rho1 is None:
            return 1.0
        overlap = float(np.real(np.trace(rho0 @ rho1)))
        return float(np.clip(overlap, 0.0, 1.0))

    def two_photon_outcome(
        self,
        arrival_arm0_ps: int,
        arrival_arm1_ps: int,
        photon_arm0: Photon | None = None,
        photon_arm1: Photon | None = None,
        scan_delay_ps: int = 0,
        lambda_nm: float = 1550.0,
        bandwidth_nm: float = 0.0,
        lambda_arm0_nm: float | None = None,
        lambda_arm1_nm: float | None = None,
        bandwidth_arm0_nm: float | None = None,
        bandwidth_arm1_nm: float | None = None,
        extra_overlap_scale: float = 1.0,
    ) -> dict:
        """Sample one HOM event with one photon in each input arm.

        Args:
            arrival_arm0_ps (int): photon arrival time at input arm 0 (ps).
            arrival_arm1_ps (int): photon arrival time at input arm 1 before scan delay (ps).
            scan_delay_ps (int): additional delay on arm 1 in ps.
            lambda_nm (float): central wavelength used for temporal-width conversion.
            bandwidth_nm (float): effective source bandwidth in nm.
            extra_overlap_scale (float): optional extra overlap reduction in [0, 1].

        Returns:
            dict: event fields including sampled output mode occupations.
                `outputs` is one of [0,0], [1,1], [0,1].
        """
        eff_t1 = int(arrival_arm1_ps + scan_delay_ps)
        delta_t_ps = float(arrival_arm0_ps - eff_t1)
        lam0 = float(lambda_nm if lambda_arm0_nm is None else lambda_arm0_nm)
        lam1 = float(lambda_nm if lambda_arm1_nm is None else lambda_arm1_nm)
        bw0 = float(bandwidth_nm if bandwidth_arm0_nm is None else bandwidth_arm0_nm)
        bw1 = float(bandwidth_nm if bandwidth_arm1_nm is None else bandwidth_arm1_nm)

        sigma_a = self._sigma_omega_from_bandwidth(lam0, bw0)
        sigma_b = self._sigma_omega_from_bandwidth(lam1, bw1)
        sigma_sum = sigma_a * sigma_a + sigma_b * sigma_b

        if sigma_sum <= 0:
            overlap_t = 1.0
            overlap_w = 1.0
        else:
            # Eq: O_t = exp[- (sigma_a^2 sigma_b^2)/(sigma_a^2+sigma_b^2) * (Delta t)^2 ]
            temporal_coeff = (sigma_a * sigma_a * sigma_b * sigma_b) / sigma_sum
            overlap_t = float(np.exp(-temporal_coeff * (delta_t_ps ** 2)))

            # Eq: O_w = (2 sigma_a sigma_b)/(sigma_a^2+sigma_b^2) * exp[-(Delta w)^2/(sigma_a^2+sigma_b^2)]
            c_nm_per_ps = 299792.458
            delta_w = float(2.0 * np.pi * c_nm_per_ps * (1.0 / lam0 - 1.0 / lam1))
            pref = (2.0 * sigma_a * sigma_b) / sigma_sum
            overlap_w = float(pref * np.exp(-(delta_w * delta_w) / sigma_sum))
            overlap_w = float(np.clip(overlap_w, 0.0, 1.0))

        overlap_pol = self.polarization_overlap_from_states(photon_arm0, photon_arm1)
        total_scale = (
            float(np.clip(extra_overlap_scale, 0.0, 1.0))
            * overlap_w
            * overlap_pol
        )
        overlap = overlap_t * total_scale
        overlap = float(np.clip(overlap, 0.0, 1.0))

        # For a symmetric 50:50 BS with partial indistinguishability.
        p_coinc = 0.5 * (1.0 - overlap)
        p_bunch = 1.0 - p_coinc
        p_out0 = 0.5 * p_bunch
        p_out1 = 0.5 * p_bunch

        draw = self.get_generator().random()
        if draw < p_out0:
            outputs = [0, 0]
        elif draw < (p_out0 + p_out1):
            outputs = [1, 1]
        else:
            outputs = [0, 1]

        return {
            "arrival_arm0_ps": int(arrival_arm0_ps),
            "arrival_arm1_ps": int(arrival_arm1_ps),
            "effective_arrival_arm1_ps": eff_t1,
            "delta_t_ps": delta_t_ps,
            "temporal_overlap": overlap_t,
            "spectral_overlap": overlap_w,
            "polarization_overlap": overlap_pol,
            "extra_overlap_scale": float(np.clip(extra_overlap_scale, 0.0, 1.0)),
            "sigma_a_rad_per_ps": sigma_a,
            "sigma_b_rad_per_ps": sigma_b,
            "total_overlap": overlap,
            "p_coincidence": p_coinc,
            "p_bunching": p_bunch,
            "outputs": outputs,
        }

    def single_photon_output(self) -> int:
        """Sample output detector (0/1) for a single photon input event."""
        return int(self.get_generator().choice([0, 1]))


class FockBeamSplitter(Entity):
    """WIP"""

    def __init__(self, name, timeline, fidelity=1):
        super().__init__(name, timeline)
        self.fidelity = fidelity
        self.most_recent_time = -1

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""

        assert len(self._receivers) == 2, "BeamSplitter should only be attached to 2 outputs."

    def get(self, photon, **kwargs) -> None:
        assert photon.encoding_type["name"] == "absorptive"

        if not photon.is_null:
            state = self.timeline.quantum_manager.get(photon.quantum_state)

            if len(state.keys) == 2:  # entangled; calculate probability of measurement
                prob_0 = trace(state.state @ povm_0).real
                if prob_0 > 1:
                    prob_0 = 1
                elif prob_0 < 0:
                    prob_0 = 0

            else:  # unentangled; send to a random output
                if self.timeline.now() == self.most_recent_time:  # if already measured right now, return (HOM effect)
                    return
                prob_0 = 0.5

            detector_num = self.get_generator().choice([0, 1], p=[prob_0, 1-prob_0])
            self.most_recent_time = self.timeline.now()
            self._receivers[detector_num].get()


class FockBeamSplitter2(Entity):
    """Class modeling a Fock beam splitter. The '2' for avoiding naming conflicts.

    A Fock beam splitter can send a single photon randomly in one of its ports. 
    See https://arxiv.org/abs/2411.11377, Simulation of Quantum Transduction Strategies for Quantum Networks

    Attributes:
        name (str): the name
        owner (Node): the owner
        timeline (Timeline): the timeline
        efficiency (float): the efficiency of the beamsplitter
        photon_counter (int): counter for counting photons
        src_list (str): a list of photon source names
    """
    def __init__(self, name: str, owner: "Node", timeline: "Timeline", efficiency: float, photon_counter: int, src_list: list[str]):

        Entity.__init__(self, name, timeline)
        self.owner = owner
        self.timeline = timeline
        self.efficiency = efficiency
        self.photon_counter = photon_counter
        self.src_list = src_list
        self.swapping_protocol = None
        self.update_counter = 0


    def init(self):
        assert len(self._receivers) == 2


    def get(self, source: str, photon: Photon) -> None:
        """Receive photon from two end nodes"""

        self.photon_counter += 1
            
        print(f"Photon counter BEAM SPLITTER: {self.photon_counter} at time {self.timeline.now()}")

        selected_receiver = random.choice(self._receivers)

        if self.photon_counter == 1:
            
            selected_receiver.get(photon)
            selected_receiver.get_2(photon) 

        elif self.photon_counter == 2:
            
            self._receivers[0].photon_counter = 0
            self._receivers[1].photon_counter = 0
            self._receivers[0].photon_counter2 = 0
            self._receivers[1].photon_counter2 = 0
            selected_receiver.getx2(photon)
            selected_receiver.get_2x2(photon)


    def add_outputs(self, outputs: list):
        """Add outputs, i.e., receivers
        
        Args:
            outputs (list): a list of entities, i.e., detectors
        """
        for i in outputs:
            self.add_receiver(i)


class PolarizingBeamSplitter(Entity):
    """Class modeling a polarizing beam splitter with fixed orientation.

    Simulates a polarization beam splitter (PBS) that always measures in the same basis.
    Unlike the time-varying BeamSplitter used for BB84 protocols, this PBS has a fixed
    measurement orientation and includes error modeling for realistic simulations.
    
    Attributes:
        name (str): label for beam splitter instance.
        timeline (Timeline): timeline for simulation.
        basis_index (int): index of measurement basis (0 for H/V, 1 for +/-).
        fidelity (float): probability of successfully transmitting a received photon.
        mismeasure_prob (float): probability of measurement error (bit flip).
    """

    def __init__(self, name: str, timeline: "Timeline", basis_index: int = 0, 
                 fidelity: float = 1.0, mismeasure_prob: float = 0.0):
        """Constructor for the polarizing beam splitter class.

        Args:
            name (str): name of the beam splitter instance.
            timeline (Timeline): simulation timeline.
            basis_index (int): polarization basis index (0 for H/V, 1 for +/-, default 0).
            fidelity (float): probability of transmitting a received photon (default 1.0).
            mismeasure_prob (float): probability of bit flip error in measurement (default 0.0).
        """
        Entity.__init__(self, name, timeline)
        self.fidelity = fidelity
        self.basis_index = basis_index
        self.mismeasure_prob = mismeasure_prob

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""
        assert len(self._receivers) == 2, \
            "PolarizingBeamSplitter requires exactly 2 receivers."

    def get(self, photon: Photon, **kwargs) -> None:
        """Method to receive a photon for measurement.

        Measures the photon in the fixed basis and routes it to the appropriate receiver.
        Includes optional measurement errors via the mismeasure_prob parameter.

        Args:
            photon (Photon): photon to measure (must have polarization encoding).

        Side Effects:
            May call get method of one receiver based on measurement result.
        """
        assert photon.encoding_type["name"] == "polarization", \
            "PolarizingBeamSplitter requires polarization encoding."

        if self.get_generator().random() < self.fidelity:
            # Measure in fixed basis
            basis = polarization["bases"][self.basis_index]
            result = Photon.measure(basis, photon, self.get_generator())

            # Apply measurement error (bit flip)
            if self.get_generator().random() < self.mismeasure_prob:
                result = 1 - result

            photon.pbs_measurement_result = int(result)
            
            self._receivers[result].get(photon)
