"""Models for simulation of photon emission devices.

This module defines the LightSource class to supply individual photons and the SPDCSource class to supply pre-entangled photons.
These classes should be connected to one or two entities, respectively, that are capable of receiving photons.
"""

from numpy import multiply, sqrt, zeros, kron, outer

from .photon import Photon
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils.encoding import polarization, fock
from ..utils import log
import numpy as np


class LightSource(Entity):
    """Model for a laser light source.

    The LightSource component acts as a simple low intensity laser, providing photon clusters at a set frequency.

    Attributes:
        name (str): label for beamsplitter instance
        timeline (Timeline): timeline for simulation
        frequency (float): frequency (in Hz) of photon creation.
        wavelength (float): wavelength (in nm) of emitted photons.
        linewidth (float): st. dev. in photon wavelength (in nm).
        mean_photon_num (float): mean number of photons emitted each period.
        encoding_type (dict[str, Any]): encoding scheme of emitted photons (as defined in the encoding module).
        phase_error (float): phase error applied to qubits.
        photon_counter (int): counter for number of photons emitted.
    """

    def __init__(self, name, timeline, frequency=8e7, wavelength=1550, bandwidth=0, mean_photon_num=0.1,
                 encoding_type=polarization, phase_error=0):
        """Constructor for the LightSource class.

        Arguments:
            name (str): name of the light source instance.
            timeline (Timeline): simulation timeline.
            frequency (float): frequency (in Hz) of photon creation (default 8e7).
            wavelength (float): wavelength (in nm) of emitted photons (default 1550).
            bandwidth (float): st. dev. in photon wavelength (default 0).
            mean_photon_num (float): mean number of photons emitted each period (default 0.1).
            encoding_type (dict): encoding scheme of emitted photons (as defined in the encoding module) (default polarization).
            phase_error (float): phase error applied to qubits (default 0).
        """

        Entity.__init__(self, name, timeline)
        self.frequency = frequency  # measured in Hz
        self.wavelength = wavelength  # measured in nm
        self.linewidth = bandwidth  # st. dev. in photon wavelength (nm)
        self.mean_photon_num = mean_photon_num
        self.encoding_type = encoding_type
        self.phase_error = phase_error
        self.photon_counter = 0

    def init(self):
        """Implementation of Entity interface (see base class)."""

        pass

    # for general use
    def emit(self, state_list) -> None:
        """Method to emit photons.

        Will emit photons for a length of time determined by the `state_list` parameter.
        The number of photons emitted per period is calculated as a poisson random variable.

        Arguments:
            state_list (list[list[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
        """

        log.logger.info(f"{self.name} emitting {len(state_list)} photons")

        time = self.timeline.now()
        period = int(round(1e12 / self.frequency))

        for i, state in enumerate(state_list):
            num_photons = self.get_generator().poisson(self.mean_photon_num)

            if self.get_generator().random() < self.phase_error:
                state = multiply([1, -1], state)

            for _ in range(num_photons):
                wavelength = self.linewidth * self.get_generator().standard_normal() + self.wavelength
                new_photon = Photon(str(i), self.timeline,
                                    wavelength=wavelength,
                                    location=self.owner,
                                    encoding_type=self.encoding_type,
                                    quantum_state=state)
                process = Process(self._receivers[0], "get", [new_photon])
                event = Event(time, process)
                self.timeline.schedule(event)
                self.photon_counter += 1

            time += period


class SPDCBellSource(LightSource):
    """
    A light source that emits entangled photon pairs in Bell states using
    spontaneous parametric down-conversion (SPDC).

    This component models a pulsed laser with an SPDC crystal that generates
    entangled photon pairs at a specified frequency. Each photon pair is emitted
    in one of the four maximally-entangled Bell states (|Φ±⟩, |Ψ±⟩).

    The source implements realistic SPDC physics:
    - Energy-conserving wavelength correlation (lambda_signal * lambda_idler = lambda_pump²)
    - Gaussian wavelength distribution around central value
    - Configurable photon statistics (thermal or Poisson)

    Attributes:
        bell_state_label (str): Label of the Bell state ("phi+", "phi-", "psi+", "psi-").
        bell_state (tuple): 4D state vector of the selected Bell state in
            computational basis |HH⟩, |HV⟩, |VH⟩, |VV⟩.
        wavelengths (list[float]): Two-element list [lambda_min, lambda_max] defining
            wavelength range in nm for energy-conserving sampling.
        photon_statistics (str): Distribution type ("thermal" or "poisson").
    """

    bell_state_map = {
        "phi+": (1 / sqrt(2), 0, 0, 1 / sqrt(2)),
        "phi-": (1 / sqrt(2), 0, 0, -1 / sqrt(2)),
        "psi+": (0, 1 / sqrt(2), 1 / sqrt(2), 0),
        "psi-": (0, 1 / sqrt(2), -1 / sqrt(2), 0)
    }

    def __init__(self, name, timeline, wavelengths=None, frequency=8e7, mean_photon_num=0.1,
                 encoding_type=polarization, phase_error=0, bandwidth=0, photon_statistics="thermal", bell_state="psi+"):
        """
        Constructor for SPDCBellSource.

        Args:
            name (str): Name of the source instance.
            timeline (Timeline): Simulation timeline.
            wavelengths (list[float], optional): Two-element list [lambda_min, lambda_max] 
                defining wavelength sampling range in nm. If None, defaults to [1550, 1550].
            frequency (float): Pulse repetition frequency in Hz (default 80 MHz).
            mean_photon_num (float): Mean number of photon pairs per pulse (default 0.1).
            encoding_type (dict): Photon encoding scheme (default polarization encoding).
            phase_error (float): Phase flip probability (currently unused, default 0).
            bandwidth (float): Wavelength range for Gaussian sampling in nm (default 0).
                The standard deviation is bandwidth/3.
            photon_statistics (str): Photon pair distribution type (default "thermal").
                Options: "thermal" (Bose-Einstein) or "poisson" (coherent).
            bell_state (str): Bell state to emit (default "psi-").
                Options: "phi+" (|Φ⁺⟩), "phi-" (|Φ⁻⟩), "psi+" (|Ψ⁺⟩), "psi-" (|Ψ⁻⟩).

        Raises:
            AssertionError: If not connected to exactly 2 receivers (checked in init()).
        """
        super().__init__(name, timeline, frequency, 0, bandwidth, mean_photon_num, encoding_type, phase_error)
        self.wavelengths = wavelengths
        self.photon_statistics = photon_statistics
        if self.wavelengths is None or len(self.wavelengths) != 2:
            self.set_wavelength()
        self.bell_state_label = bell_state
        self.bell_state = self.bell_state_map[bell_state]

    def init(self):
        assert len(self._receivers) == 2, "SPDCBellSource source must connect to 2 receivers."

    def sample_photon_pairs(self):
        """Sample number of photon pairs from configured distribution.

        Uses either thermal (Bose-Einstein) or Poisson statistics based on
        the photon_statistics attribute.

        Returns:
            int: Number of photon pairs to emit (>= 0).

        Raises:
            ValueError: If photon_statistics is not "thermal" or "poisson".
        """
        if self.photon_statistics == "thermal":
            # Thermal (Bose-Einstein) distribution
            p = 1 / (1 + self.mean_photon_num)
            return self.get_generator().geometric(p) - 1
        
        elif self.photon_statistics == "poisson":
            # Poisson distribution (coherent state)
            return self.get_generator().poisson(self.mean_photon_num)
        
        else:
            raise ValueError(f"Unknown photon_statistics: {self.photon_statistics}")
    
    def emit(self, num_pulses=1):
        """
        Emit entangled photon pairs in the specified Bell state.

        Each pulse emits a random number of photon pairs sampled from the
        configured distribution (thermal or Poisson). Photons are sent to 
        the two connected receivers with appropriate time delays.

        Wavelength correlation: For each pair, signal and idler wavelengths
        are sampled to satisfy energy conservation (lambda_signal * lambda_idler = lambda_pump^2).

        Timing: Pulses are spaced by 1/frequency. Photon pairs within a pulse
        are sent at the same time.

        Args:
            num_pulses (int): Number of emission pulses to generate (default 1).

        Side Effects:
            - Schedules photon arrival events at connected receivers
            - Increments self.photon_counter for each emitted pair
            - Advances internal time by num_pulses * (1/frequency)
        """
        time = self.timeline.now()
        period = int(round(1e12 / self.frequency))

        lam_min, lam_max = self.wavelengths
        lam0 = 0.5 * (lam_min + lam_max)
        delta_max = 0.5 * (lam_max - lam_min)
        sigma = delta_max / 3

        for _ in range(num_pulses):
            num_pairs = self.sample_photon_pairs()
            for _ in range(num_pairs):
                delta = sigma * self.get_generator().standard_normal()
                delta = np.clip(delta, -delta_max, delta_max)

                lambda_signal = lam0 + delta
                lambda_idler  = lam0 * lam0 / lambda_signal

                new_photon0 = Photon("signal", self.timeline,
                                     wavelength=lambda_signal,
                                     location=self,
                                     encoding_type=self.encoding_type)
                new_photon1 = Photon("idler", self.timeline,
                                     wavelength=lambda_idler,
                                     location=self,
                                     encoding_type=self.encoding_type)

                new_photon0.combine_state(new_photon1)
                new_photon0.set_state(self.bell_state)
                self.send_photons(time, [new_photon0, new_photon1])
                self.photon_counter += 1
            time += period
    
    def send_photons(self, time, photons: list["Photon"]):
        """
        Dispatch photon pair to the connected receivers.

        Args:
            time (float): Emission time in ps.
            photons (list): List of two Photon objects.
        """
        log.logger.debug("SPDC source {} sending photons to {} at time {}".format(
            self.name, self._receivers, time
        ))
        assert len(photons) == 2
        for dst, photon in zip(self._receivers, photons):
            process = Process(dst, "get", [photon])
            event = Event(int(round(time)), process)
            self.timeline.schedule(event)

    def set_wavelength(self, wavelength1=1550, wavelength2=1550):
        """Method to set the wavelengths of photons emitted in two output modes."""
        self.wavelengths = [wavelength1, wavelength2]


class SPDCSource(LightSource):
    """Model for a laser light source for entangled photons, via spontaneous parametric down-conversion (SPDC).

    The SPDCLightSource component acts as a simple low intensity laser with an SPDC lens.
    It provides entangled photon clusters at a set frequency.

    Attributes:
        name (str): label for beamsplitter instance
        timeline (Timeline): timeline for simulation
        frequency (float): frequency (in Hz) of photon creation.
        wavelengths (list[float]): wavelengths (in nm) of emitted entangled photons.
            If a list is given, it should contain two elements (corresponding to two modes).
        linewidth (float): st. dev. in photon wavelength (in nm) (currently unused).
        mean_photon_num (float): mean number of photons emitted each period.
        encoding_type (dict): encoding scheme of emitted photons (as defined in the encoding module).
        phase_error (float): phase error applied to qubits.
    """

    def __init__(self, name, timeline, wavelengths=None, frequency=8e7, mean_photon_num=0.1,
                 encoding_type=fock, phase_error=0, bandwidth=0):
        super().__init__(name, timeline, frequency, 0, bandwidth, mean_photon_num, encoding_type, phase_error)
        self.wavelengths = wavelengths
        if self.wavelengths is None or len(self.wavelengths) != 2:
            self.set_wavelength()

    def init(self):
        assert len(self._receivers) == 2, "SPDC source must connect to 2 receivers."

    def _generate_tmsv_state(self):
        """Method to generate two-mode squeezed vacuum state of two output photonic modes

        Returns:
            array: generated state.
        """

        mean_num = self.mean_photon_num
        truncation = self.timeline.quantum_manager.truncation

        # create state component amplitudes list
        amp_list = [(sqrt(mean_num / (mean_num + 1)) ** m) / sqrt(mean_num + 1) for m in range(truncation)]
        amp_square_list = [amp ** 2 for amp in amp_list]
        amp_list.append(sqrt(1 - sum(amp_square_list)))

        # create two-mode state vector
        state_vec = zeros((truncation+1) ** 2)

        for i in range(truncation+1):
            amp = amp_list[i]
            basis = zeros(truncation+1)
            basis[i] = 1
            basis = kron(basis,basis)
            state_vec += amp * basis

        return state_vec

    def emit(self, state_list):
        """Method to emit photons.

        Will emit photons for a length of time determined by the `state_list` parameter.
        The number of photons emitted per period is calculated as a poisson random variable.

        Arguments:
            state_list (list[list[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
                This is ignored for absorptive and Fock encoding types.
                For these encoding types only the length of list matters and elements can be arbitrary.
        """

        log.logger.info(f"SPDC sourcee {self.name} emitting {len(state_list)} photons")

        time = self.timeline.now()

        if self.encoding_type["name"] == "fock":
            # Use Fock encoding.
            # The two generated photons should be entangled and should have keys pointing to same Fock state.
            for _ in state_list:
                # generate two new photons
                new_photon0 = Photon("", self.timeline,
                                     wavelength=self.wavelengths[0],
                                     location=self,
                                     encoding_type=self.encoding_type,
                                     use_qm=True)
                new_photon1 = Photon("", self.timeline,
                                     wavelength=self.wavelengths[1],
                                     location=self,
                                     encoding_type=self.encoding_type,
                                     use_qm=True)

                # set shared state to squeezed state
                state = self._generate_tmsv_state()
                keys = [new_photon0.quantum_state, new_photon1.quantum_state]
                self.timeline.quantum_manager.set(keys, state)

                self.send_photons(time, [new_photon0, new_photon1])
                self.photon_counter += 1
                time += 1e12 / self.frequency

        elif self.encoding_type["name"] == "absorptive":
            for _ in state_list:
                num_photon_pairs = self.get_generator().poisson(self.mean_photon_num)

                for _ in range(num_photon_pairs):
                    new_photon0 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[0],
                                         location=self,
                                         encoding_type=self.encoding_type,
                                         use_qm=True)
                    new_photon1 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[1],
                                         location=self,
                                         encoding_type=self.encoding_type,
                                         use_qm=True)

                    new_photon0.combine_state(new_photon1)
                    new_photon0.set_state((complex(0), complex(0), complex(0), complex(1)))
                    self.send_photons(time, [new_photon0, new_photon1])
                    self.photon_counter += 1

                if num_photon_pairs == 0:
                    # send two null photons for purposes of entanglement
                    new_photon0 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[0],
                                         location=self,
                                         encoding_type=self.encoding_type,
                                         use_qm=True)
                    new_photon1 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[1],
                                         location=self,
                                         encoding_type=self.encoding_type,
                                         use_qm=True)

                    new_photon0.is_null = True
                    new_photon1.is_null = True
                    new_photon0.combine_state(new_photon1)
                    new_photon0.set_state((complex(1), complex(0), complex(0), complex(0)))
                    self.send_photons(time, [new_photon0, new_photon1])

                time += 1e12 / self.frequency

        else:
            for state in state_list:
                num_photon_pairs = self.get_generator().poisson(
                self.mean_photon_num)

                if self.get_generator().random() < self.phase_error:
                    state = multiply([1, -1], state)

                for _ in range(num_photon_pairs):
                    new_photon0 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[0],
                                         location=self,
                                         encoding_type=self.encoding_type)
                    new_photon1 = Photon("", self.timeline,
                                         wavelength=self.wavelengths[1],
                                         location=self,
                                         encoding_type=self.encoding_type)

                    new_photon0.combine_state(new_photon1)
                    new_photon0.set_state((state[0], complex(0), complex(0), state[1]))
                    self.send_photons(time, [new_photon0, new_photon1])
                    self.photon_counter += 1

                time += 1e12 / self.frequency

    def send_photons(self, time, photons: list["Photon"]):
        log.logger.debug("SPDC source {} sending photons to {} at time {}".format(
            self.name, self._receivers, time
        ))

        assert len(photons) == 2
        for dst, photon in zip(self._receivers, photons):
            process = Process(dst, "get", [photon])
            event = Event(int(round(time)), process)
            self.timeline.schedule(event)

    def set_wavelength(self, wavelength1=1550, wavelength2=1550):
        """Method to set the wavelengths of photons emitted in two output modes."""
        self.wavelengths = [wavelength1, wavelength2]