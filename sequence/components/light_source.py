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
                 encoding_type=polarization, phase_error=0, photon_statistics="thermal"):
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
        self.photon_statistics = photon_statistics  # "thermal" or "poisson"

    def init(self):
        """Implementation of Entity interface (see base class)."""

        pass

    def sample_photon_pairs(self):
        if self.photon_statistics == "thermal":
            p = 1 / (1 + self.mean_photon_num)
            return self.get_generator().geometric(p) - 1
        elif self.photon_statistics == "poisson":
            return self.get_generator().poisson(self.mean_photon_num)
        else:
            raise ValueError(f"Unknown photon_statistics mode: {self.photon_statistics}")

    # for general use
    def emit(self, state_list) -> None:
        """Method to emit photons.

        Will emit photons for a length of time determined by the `state_list` parameter.
        The number of photons emitted per period is calculated as a poisson random variable.

        Arguments:
            state_list (list[list[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
        """

        log.logger.info("{} emitting {} photons".format(self.name, len(state_list)))

        time = self.timeline.now()
        period = int(round(1e12 / self.frequency))

        for i, state in enumerate(state_list):
            num_photons = self.sample_photon_pairs()

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

        log.logger.info("SPDC sourcee {} emitting {} photons".format(self.name, len(state_list)))

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

class SPDCBellSource(LightSource):
    """
    A light source that emits entangled photon pairs in Bell states using
    spontaneous parametric down-conversion (SPDC).

    This component models a low-intensity laser with an SPDC crystal that
    produces entangled photon pairs at a specified frequency and wavelength.
    The output state of each photon pair is one of the four Bell states.

    Attributes:
        bell_state_label (str): The label of the Bell state to emit ("phi+", "phi-", "psi+", "psi-").
        bell_state (tuple): The corresponding 4D state vector of the selected Bell state.
        wavelengths (list): List of two wavelengths (nm) for the photon pair.
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
            name (str): Name of the source.
            timeline (Timeline): The simulation timeline.
            wavelengths (list): List of two photon wavelengths in nm (optional).
            frequency (float): Emission frequency in Hz.
            mean_photon_num (float): Average number of photon pairs emitted per cycle.
            encoding_type (dict): Encoding scheme (default is polarization).
            phase_error (float): Phase error probability (currently unused).
            bandwidth (float): Spectral bandwidth (currently unused).
            photon_statistics (str): Distribution for pair generation ("thermal" or "poisson").
            bell_state (str): Desired Bell state to emit ("phi+", "phi-", "psi+", "psi-").
        """
        super().__init__(name, timeline, frequency, 0, bandwidth, mean_photon_num, encoding_type, phase_error, photon_statistics)
        self.wavelengths = wavelengths
        if self.wavelengths is None or len(self.wavelengths) != 2:
            self.set_wavelength()
        self.bell_state_label = bell_state
        self.bell_state = self.bell_state_map[bell_state]

    def init(self):
        assert len(self._receivers) == 2, "SPDCBellSource source must connect to 2 receivers."

    def emit(self, num_pulses=1):
        """
        Emit entangled photon pairs in the specified Bell state.

        Each pulse may emit zero or more photon pairs, depending on the
        specified photon number distribution (thermal or Poisson).

        Args:
            num_pulses (int): Number of emission pulses (default is 1).
        """
        time = self.timeline.now()
        period = int(round(1e12 / self.frequency))
        for _ in range(num_pulses):
            num_pairs = self.sample_photon_pairs()
            for _ in range(num_pairs):
                new_photon0 = Photon("", self.timeline,
                                     wavelength=self.wavelengths[0],
                                     location=self,
                                     encoding_type=self.encoding_type)
                new_photon1 = Photon("", self.timeline,
                                     wavelength=self.wavelengths[1],
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

        



