"""Models for simulation of photon emission devices.

This module defines the LightSource class to supply individual photons and the SPDCSource class to supply pre-entangled photons.
These classes should be connected to one or two entities, respectively, that are capable of receiving photons.
"""

from typing import List

from numpy import multiply

from .photon import Photon
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils.encoding import polarization
from ..utils import log


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
        encoding_type (Dict[str, Any]): encoding scheme of emitted photons (as defined in the encoding module).
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
            encoding_type (Dict): encoding scheme of emitted photons (as defined in the encoding module) (default polarization).
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
            state_list (List[List[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
        """

        log.logger.info("{} emitting {} photons".format(self.name, len(state_list)))

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


class SPDCSource(LightSource):
    """Model for a laser light source for entangled photons (via SPDC).

    The SPDCLightSource component acts as a simple low intensity laser with an SPDC lens.
    It provides entangled photon clusters at a set frequency.

    Attributes:
        name (str): label for beamsplitter instance
        timeline (Timeline): timeline for simulation
        frequency (float): frequency (in Hz) of photon creation.
        wavelengths (List[float]): wavelengths (in nm) of emitted entangled photons.
        linewidth (float): st. dev. in photon wavelength (in nm) (currently unused).
        mean_photon_num (float): mean number of photons emitted each period.
        encoding_type (Dict): encoding scheme of emitted photons (as defined in the encoding module).
        phase_error (float): phase error applied to qubits.
    """

    def __init__(self, name, timeline, wavelengths=None, frequency=8e7, bandwidth=0, mean_photon_num=0.1,
                 encoding_type=polarization, phase_error=0):
        super().__init__(name, timeline, frequency, 0, bandwidth, mean_photon_num, encoding_type, phase_error)
        if wavelengths is None:
            wavelengths = [1550, 1550]
        self.wavelengths = wavelengths

    def init(self):
        assert len(self._receivers) == 2, "SPDC source must connect to 2 receivers."

    def emit(self, state_list):
        """Method to emit photons.

        Will emit photons for a length of time determined by the `state_list` parameter.
        The number of photons emitted per period is calculated as a poisson random variable.

        Arguments:
            state_list (List[List[complex]]): list of complex coefficient arrays to send as photon-encoded qubits.
                This is ignored for the absorptive encoding type.
        """

        log.logger.info("SPDC sourcee {} emitting {} photons".format(self.name, len(state_list)))

        time = self.timeline.now()

        if self.encoding_type["name"] == "absorptive":
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

                    new_photon0.entangle(new_photon1)
                    new_photon0.set_state((complex(0), complex(0), complex(0), complex(1)))
                    self.send_photons(time, [new_photon0, new_photon1])
                    self.photon_counter += 1

                if num_photon_pairs is 0:
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
                    new_photon0.entangle(new_photon1)
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

                    new_photon0.entangle(new_photon1)
                    new_photon0.set_state((state[0], complex(0), complex(0), state[1]))
                    self.send_photons(time, [new_photon0, new_photon1])
                    self.photon_counter += 1

                time += 1e12 / self.frequency

    def send_photons(self, time, photons: List["Photon"]):
        log.logger.debug("SPDC source {} sending photons to {} at time {}".format(
            self.name, self._receivers, time
        ))

        assert len(photons) == 2
        for dst, photon in zip(self._receivers, photons):
            process = Process(dst, "get", [photon])
            event = Event(int(round(time)), process)
            self.timeline.schedule(event)
