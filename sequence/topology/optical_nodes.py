"""Polarization analyzer node for quantum state measurements.

This module provides a configurable compound node for polarization analysis
with optional QWP, HWP, and detector components.
"""

from ..topology.node import Node
from ..components.detector import QSDetectorPolarizationStatic
from ..components.wave_plate import WavePlate
from ..components.light_source import SPDCBellSource
from ..components.beam_splitter import HOMBeamSplitter
from ..utils.encoding import polarization
from ..kernel.entity import Entity
from ..components.photon import Photon
from typing import Optional, Dict, Any
import numpy as np
from collections import deque


class SourcePort(Entity):
    def __init__(self, name, timeline, owner:Node):
        super().__init__(name, timeline)
        self.owner = owner
        self.add_receiver(owner)

    def init(self):
        pass

    def get(self, photon, **kwargs):
        self._receivers[0].get(photon)


class SpdcSourceNode(Node):
    """
    Node that emits entangled photon pairs using an SPDCBellSource.
    """

    def __init__(self, name, timeline, config):
        super().__init__(name, timeline)
        self.name = name
        self.emission_count = 0
        self.direct_receivers = {}

        # Default values for SPDC configuration
        default_config = {
            'wavelengths': [1550, 1550],
            'frequency': 8e7,
            'mean_photon_num': 0.1,
            'phase_error': 0.0,
            'bandwidth': 0,
            'encoding': polarization,
            'photon_statistics': 'thermal',
            'bell_state': 'psi-',
            'use_sparse_emission': False,
        }

        # Merge with user config
        merged_config = {**default_config, **(config or {})}
        
        # Create the Bell-state SPDC source
        self.spdc = SPDCBellSource(
            name=self.name + "_SPDC",
            timeline=self.timeline,
            wavelengths=merged_config['wavelengths'],
            frequency=float(merged_config['frequency']),
            mean_photon_num=float(merged_config['mean_photon_num']),
            phase_error=float(merged_config['phase_error']),
            bandwidth=float(merged_config['bandwidth']),
            encoding_type=merged_config['encoding'],
            photon_statistics=merged_config['photon_statistics'],
            bell_state=merged_config['bell_state'],
            use_sparse_emission=bool(merged_config['use_sparse_emission']),

        )

        # Create and connect output ports
        self.ports = {}
        for i in range(2):
            port_name = f"{self.name}.port{i}"
            self.ports[i] = SourcePort(port_name, self.timeline, self)
            self.spdc.add_receiver(self.ports[i])

        self.first_component_name = self.spdc.name
        self.spdc.owner = self

    def emit(self, num_pulses: int):
        """
        Emit entangled photon pairs.
        """
        self.spdc.emit(num_pulses=num_pulses)


    def get(self, photon, **kwargs):
        #Count only for signal no need to duplicate for idler
        if hasattr(photon, 'name') and photon.name == "signal":
            self.emission_count += 1

        # Route to appropriate quantum channel based on port name
        try:
            photon_type = getattr(photon, 'name', None)
            if photon_type == "signal":
                port_index = 0
            elif photon_type == "idler":
                port_index = 1
            else:
                raise ValueError(f"Unknown photon type: {photon_type}")
            if port_index in self.direct_receivers:
                self.direct_receivers[port_index].receive_qubit(self.name, photon)
                return
            for index, dst in enumerate(self.qchannels):
                if str(port_index) == str(index):
                    self.send_qubit(dst, photon)
                    break
        except (AttributeError, IndexError):
            pass
    
    # ========================================================================
    # Configuration API
    # ========================================================================

    def set_direct_receiver(self, port_index: int, receiver: Node) -> None:
        """Route one source output directly to a receiver without a channel."""
        if port_index not in (0, 1):
            raise ValueError("SPDC source port_index must be 0 or 1")
        self.direct_receivers[int(port_index)] = receiver

    def set_bell_state(self, bell_state: str) -> None:
        """Change the Bell state emitted by the source.
        
        Allows dynamic reconfiguration of the entangled state without
        recreating the source. Changes take effect on next emit() call.
        
        Args:
            bell_state (str): Bell state identifier - one of:
                - "phi+" : |Φ⁺⟩ = (|HH⟩ + |VV⟩) / √2
                - "phi-" : |Φ⁻⟩ = (|HH⟩ - |VV⟩) / √2
                - "psi+" : |Ψ⁺⟩ = (|HV⟩ + |VH⟩) / √2
                - "psi-" : |Ψ⁻⟩ = (|HV⟩ - |VH⟩) / √2
        
        Raises:
            ValueError: If bell_state is not one of the valid options.
        
        Example:
            >>> source.set_bell_state("phi+")
            >>> source.emit(100)  # Emits |Φ⁺⟩ pairs
        """
        if bell_state not in self.spdc.bell_state_map:
            valid = list(self.spdc.bell_state_map.keys())
            raise ValueError(
                f"Invalid Bell state: '{bell_state}'. "
                f"Must be one of {valid}"
            )
        
        self.spdc.bell_state_label = bell_state
        self.spdc.bell_state = self.spdc.bell_state_map[bell_state]


    def set_frequency(self, frequency: float) -> None:
        """Set pulse repetition frequency.
        
        Controls the rate at which the source attempts to emit photon pairs.
        Higher frequency = faster emission but shorter time between pulses.
        
        Args:
            frequency (float): Pulse repetition rate in Hz.
                Typical values: 1e6 to 1e9 Hz (1 MHz to 1 GHz)
        
        Raises:
            ValueError: If frequency <= 0.
        
        Example:
            >>> source.set_frequency(1e8)  # 100 MHz pulse rate
        """
        if frequency <= 0:
            raise ValueError(f"Frequency must be positive, got {frequency}")
        
        self.spdc.frequency = float(frequency)


    def set_mean_photon_num(self, mean: float) -> None:
        """Set mean number of photon pairs per pulse.
        
        Controls the brightness/intensity of the source. The actual number
        of pairs per pulse follows either Poisson or thermal statistics
        (depending on photon_statistics setting).
        
        Args:
            mean (float): Average number of photon pairs emitted per pulse.
                Typical values: 0.01 to 1.0
                - Low (0.01-0.1): Heralded single-photon regime
                - Medium (0.1-0.5): Moderate brightness
                - High (0.5-1.0): High brightness but more multi-pair events
        
        Raises:
            ValueError: If mean < 0.
        
        Example:
            >>> source.set_mean_photon_num(0.01)  # ~1% chance per pulse
        """
        if mean < 0:
            raise ValueError(f"mean_photon_num must be non-negative, got {mean}")
        
        self.spdc.mean_photon_num = float(mean)

class PolarizationAnalyzerNode(Node):
    """Configurable node for polarization state analysis.
    
    Supports three architecture modes:
    - 'hwp_only': Photon → HWP → PBS (for linear polarization rotation)
    - 'hwp_qwp': Photon → HWP → QWP → PBS (for full tomography)
    - 'custom': Manual angle control
    
    Attributes:
        mode (str): Architecture mode ('hwp_only', 'hwp_qwp', 'custom').
        qwp (WavePlate | None): Quarter-wave plate (if mode uses it).
        hwp (WavePlate | None): Half-wave plate (if mode uses it).
        detector (QSDetectorPolarizationStatic): Two-output PBS detector.
    """

    def __init__(
        self, 
        name: str, 
        timeline, 
        config: Optional[Dict[str, Any]] = None
    ):
        """Constructor for polarization analyzer node.
        
        Args:
            name (str): node name.
            timeline (Timeline): simulation timeline.
            config (dict): configuration dictionary with optional keys:
                - 'mode' (str): 'hwp_only', 'hwp_qwp', or 'custom' (default 'hwp_qwp').
                - 'rotation_angle' (float): For 'hwp_only' mode, rotation angle in radians.
                - 'basis' (str): For 'hwp_qwp' mode, Pauli basis 'Z', 'X', or 'Y'.
                - 'qwp_angle' (float): For 'custom' mode, QWP angle in radians.
                - 'hwp_angle' (float): For 'custom' mode, HWP angle in radians.
                - 'qwp_fidelity' (float): QWP transmission (default 1.0).
                - 'hwp_fidelity' (float): HWP transmission (default 1.0).
                - 'detector_efficiency' (float): SPD efficiency (default 1.0).
                - 'dark_count' (float): SPD dark count rate in Hz (default 0).
                - 'pbs_fidelity' (float): PBS transmission (default 1.0).
                - 'mismeasure_prob' (float): PBS measurement error (default 0.0).
        """
        Node.__init__(self, name, timeline)
        
        # Parse configuration
        config = config or {}
        mode = config.get('mode', 'hwp_qwp')
        qwp_fidelity = config.get('qwp_fidelity', 1.0)
        hwp_fidelity = config.get('hwp_fidelity', 1.0)
        detector_efficiency = config.get('detector_efficiency', 1.0)
        dark_count = config.get('dark_count', 0)
        pbs_fidelity = config.get('pbs_fidelity', 1.0)
        mismeasure_prob = config.get('mismeasure_prob', 0.0)
        
        self.mode = mode
        self.qwp = None
        self.hwp = None
        
        # Create detector (always present)
        self.detector = QSDetectorPolarizationStatic(
            f"{name}.detector", 
            timeline,
            basis_index=0,
            PBS_fidelity=pbs_fidelity,
            mismeasure_prob=mismeasure_prob,
            detector_efficiency=detector_efficiency,
            dark_count=dark_count
        )

        self.add_component(self.detector)
        
        # Build component chain based on mode
        if mode == 'hwp_only':
            # Only HWP for linear polarization rotation
            rotation_angle = config.get('rotation_angle', 0.0)
            
            self.hwp = WavePlate(
                f"{name}.HWP",
                timeline,
                plate_type="HWP",
                angle=rotation_angle / 2.0,  # HWP rotates by 2θ
                fidelity=hwp_fidelity
            )
            self.add_component(self.hwp)
            # Wire: HWP → Detector
            self.hwp.add_receiver(self.detector)
            self.hwp.owner = self
            
            self.set_first_component(self.hwp.name)

        elif mode == 'hwp_qwp':
            # HWP + QWP for full tomography
            self.hwp = WavePlate(
                f"{name}.HWP",
                timeline,
                plate_type="HWP",
                angle=0.0,
                fidelity=hwp_fidelity
            )
            
            self.qwp = WavePlate(
                f"{name}.QWP",
                timeline,
                plate_type="QWP",
                angle=0.0,
                fidelity=qwp_fidelity
            )
            self.add_component(self.qwp)
            self.add_component(self.hwp)
            # Wire: HWP → QWP → Detector
            self.hwp.add_receiver(self.qwp)
            self.qwp.add_receiver(self.detector)
            self.qwp.owner = self
            self.hwp.owner = self
            
            self.set_first_component(self.hwp.name)
            
            # Set basis if specified
            basis = config.get('basis')
            if basis:
                self.set_basis(basis)
                
        elif mode == 'custom':
            # Custom angle control
            qwp_angle = config.get('qwp_angle', 0.0)
            hwp_angle = config.get('hwp_angle', 0.0)
            use_qwp = config.get('use_qwp', True)
            use_hwp = config.get('use_hwp', True)
            
            first_component = None
            last_component = None
            
            # Build chain dynamically
            if use_qwp:
                self.qwp = WavePlate(
                    f"{name}.QWP",
                    timeline,
                    plate_type="QWP",
                    angle=qwp_angle,
                    fidelity=qwp_fidelity
                )
                self.qwp.owner = self
                first_component = self.qwp
                last_component = self.qwp
                self.add_component(self.qwp)
            
            if use_hwp:
                self.hwp = WavePlate(
                    f"{name}.HWP",
                    timeline,
                    plate_type="HWP",
                    angle=hwp_angle,
                    fidelity=hwp_fidelity
                )
                self.hwp.owner = self
                
                if last_component:
                    last_component.add_receiver(self.hwp)
                else:
                    first_component = self.hwp
                    
                last_component = self.hwp
                self.add_component(self.hwp)
            
            # Connect to detector
            if last_component:
                last_component.add_receiver(self.detector)
            else:
                first_component = self.detector
            
            self.set_first_component(first_component.name)
            
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'hwp_only', 'hwp_qwp', or 'custom'.")

        self.classical_noise_count = 0
    
    def init(self) -> None:
        """Initialize all components."""
        super().init()
        if self.qwp:
            self.qwp.init()
        if self.hwp:
            self.hwp.init()
        self.detector.init()
    
    # ========================================================================
    # Configuration API
    # ========================================================================
    
    def set_rotation_angle(self, theta_rad: float) -> None:
        """Set polarization rotation angle (for hwp_only mode).
        
        Rotates linear polarization by angle theta.
        HWP physical angle set to theta/2 (since HWP rotates by 2θ).
        
        Args:
            theta_rad (float): desired rotation angle in radians.
            
        Raises:
            ValueError: if mode is not 'hwp_only'.
        """
        if self.mode != 'hwp_only':
            raise ValueError("set_rotation_angle() only valid in 'hwp_only' mode")
        
        self.set_hwp_angle(theta_rad / 2.0)
    
    def set_qwp_angle(self, theta_rad: float) -> None:
        """Set QWP physical angle.
        
        Args:
            theta_rad (float): angle in radians.
            
        Raises:
            ValueError: if QWP not present in this mode.
        """
        if self.qwp is None:
            raise ValueError("QWP not present in current mode")
        self.qwp.set_angle(theta_rad)
    
    def set_hwp_angle(self, theta_rad: float) -> None:
        """Set HWP physical angle.
        
        Args:
            theta_rad (float): angle in radians.
            
        Raises:
            ValueError: if HWP not present in this mode.
        """
        if self.hwp is None:
            raise ValueError("HWP not present in current mode")
        self.hwp.set_angle(theta_rad)
    
    def set_basis(self, basis: str) -> None:
        """Configure analyzer to measure in a Pauli basis (hwp_qwp mode only).
        
        Basis configurations:
        - 'Z' (computational): Measures H/V polarization
          Settings: QWP=0°, HWP=0°
          
        - 'X' (diagonal): Measures D/A polarization (±45°)
          Settings: QWP=0°, HWP=22.5°
          
        - 'Y' (circular): Measures R/L polarization
          Settings: QWP=45°, HWP=0°
        
        Args:
            basis (str): 'Z', 'X', or 'Y'.
            
        Raises:
            ValueError: if mode is not 'hwp_qwp' or basis invalid.
        """
        if self.mode != 'hwp_qwp':
            raise ValueError("set_basis() only valid in 'hwp_qwp' mode")
        
        basis = basis.upper()
        
        if basis == "Z":
            self.set_qwp_angle(0.0)
            self.set_hwp_angle(0.0)
            
        elif basis == "X":
            self.set_qwp_angle(0.0)
            self.set_hwp_angle(np.pi/8)  # 22.5°
            
        elif basis == "Y":
            self.set_qwp_angle(np.pi/4)  # 45°
            self.set_hwp_angle(0.0)
            
        else:
            raise ValueError(f"Unknown basis '{basis}'. Use 'Z', 'X', or 'Y'.")
    
    # ========================================================================
    # Measurement Results API
    # ========================================================================
    
    def get_photon_times(self) -> list:
        """Get detection times from both detector outputs.
        
        Returns:
            list[list[int]]: [times_detector0, times_detector1]
            
        Side Effects:
            Clears internal detector buffer.
        """
        return self.detector.get_photon_times()

    def get_detection_records(self) -> list:
        """Get per-channel detection records from the analyzer detector."""
        return self.detector.get_detection_records()
    
    def get_detection_counts(self) -> tuple:
        """Get number of detections on each output.
        
        Returns:
            tuple[int, int]: (count_detector0, count_detector1)
            
        Side Effects:
            Clears internal detector buffer.
        """
        times = self.detector.get_photon_times()
        return (len(times[0]), len(times[1]))
    
    def get_measurement_result(self) -> int:
        """Get binary measurement result (0 or 1) from latest detection.
        
        Returns:
            int: 0 if last photon detected on detector 0, 1 if detector 1.
                 Returns -1 if no photons detected.
        """
        times = self.detector.trigger_times
        
        last_0 = times[0][-1] if times[0] else -1
        last_1 = times[1][-1] if times[1] else -1
        
        if last_0 == -1 and last_1 == -1:
            return -1
        elif last_0 > last_1:
            return 0
        else:
            return 1

    def receive_noise_photon(self) -> None:
        """Record a Raman noise detection event on a random detector.
    
        Raman-scattered noise photons are unpolarized and arrive with random
        polarization states. Instead of simulating the full photon propagation,
        this method directly triggers a detection event on a randomly selected
        detector (0 or 1) with 50% probability each.
        
        Physical reasoning:
        - Raman noise is unpolarized (random polarization)
        - PBS splits unpolarized light 50/50 between outputs
        - Direct detector triggering is computationally efficient
        
        Called by fiber channel when Raman noise photon arrives.
        """
        # Randomly select detector (50/50 probability for unpolarized light)
        detector_index = 0 if self.get_generator().random() < 0.5 else 1
        
        # Record detection at current time
        self.detector.detectors[detector_index].get()

        self.classical_noise_count += 1

class HOMInputPort(Entity):
    """Input port for the HOM beam splitter node."""

    def __init__(self, name, timeline, owner: Node, port_index: int):
        super().__init__(name, timeline)
        self.owner = owner
        self.port_index = port_index

    def init(self):
        pass

    def get(self, photon, **kwargs):
        self.owner.receive_photon(photon, self.port_index)


class PhotonSinkNode(Node):
    """Dummy sink node used to absorb unused photons (e.g., source signal arms)."""

    def __init__(self, name, timeline):
        super().__init__(name, timeline)

    def init(self):
        super().init()
    
    def receive_qubit(self, src, qubit):
        # Absorb and ignore incoming photons
        pass

    def get(self, photon, **kwargs):
        # Intentionally discard photon
        pass


class HeraldDetectorNode(Node):
    """Single-detector node that timestamps photon arrivals with simple SPD behavior."""

    def __init__(self, name: str, timeline, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, timeline)
        config = config or {}

        self.detector_efficiency = float(config.get("detector_efficiency", 1.0))
        self.detector_jitter_ps = float(config.get("detector_jitter_ps", 0.0))
        self.dark_count_rate_hz = float(
            config.get("dark_count_rate_hz", config.get("dark_count", 0.0))
        )

        self.trigger_times = []
        self._run_start_ps = int(self.timeline.now())
        self._last_event_ps = int(self.timeline.now())
        self._dark_counts_injected = False

    def init(self):
        super().init()

    def receive_qubit(self, src, qubit):
        self._record_detection(int(self.timeline.now()))

    def get(self, photon, **kwargs):
        self._record_detection(int(self.timeline.now()))

    def _sample_jitter_ps(self) -> int:
        if self.detector_jitter_ps <= 0:
            return 0
        return int(np.rint(self.get_generator().normal(0.0, self.detector_jitter_ps)))

    def _record_detection(self, nominal_click_ps: int) -> None:
        self._dark_counts_injected = False
        self._last_event_ps = max(self._last_event_ps, nominal_click_ps)

        eta = float(np.clip(self.detector_efficiency, 0.0, 1.0))
        if self.get_generator().random() > eta:
            return

        jitter = self._sample_jitter_ps()
        click_ps = int(max(0, nominal_click_ps + jitter))
        self.trigger_times.append(click_ps)
        self._last_event_ps = max(self._last_event_ps, click_ps)

    def _inject_dark_counts(self) -> None:
        if self._dark_counts_injected or self.dark_count_rate_hz <= 0:
            self._dark_counts_injected = True
            return

        end_ps = max(self._last_event_ps, int(self.timeline.now()))
        duration_ps = max(0, end_ps - self._run_start_ps)
        if duration_ps <= 0:
            self._dark_counts_injected = True
            return

        expected = self.dark_count_rate_hz * (duration_ps * 1e-12)
        count = int(self.get_generator().poisson(expected))
        if count > 0:
            draws = self.get_generator().random(count)
            for draw in draws:
                t_ps = int(self._run_start_ps + draw * duration_ps)
                self.trigger_times.append(t_ps)

        self._dark_counts_injected = True

    def get_click_times(self, include_dark_counts: bool = True, reset: bool = False) -> list[int]:
        if include_dark_counts:
            self._inject_dark_counts()

        out = sorted(self.trigger_times)
        if reset:
            self.reset_logs()
        return out

    def get_photon_times(self) -> list[int]:
        return self.get_click_times(include_dark_counts=True, reset=True)

    def reset_logs(self) -> None:
        self.trigger_times = []
        self._dark_counts_injected = False
        now_ps = int(self.timeline.now())
        self._run_start_ps = now_ps
        self._last_event_ps = now_ps


class HOMInterferenceNode(Node):
    """Central timing-based Hong-Ou-Mandel interference node.

    Two remote sources deliver photons through standard quantum channels via
    `receive_qubit(src, qubit)`. Input-arm identity is inferred from `src`.

    Matching and interference are timing-based:
    - Per-arm arrival queues are paired by effective arrival-time compatibility.
    - A configurable scan delay is applied to one arm.
    - Paired photons are processed by a dedicated `HOMBeamSplitter`.
    - Unmatched photons follow an explicit single-photon policy.

    Detector model:
    - efficiency,
    - Gaussian timing jitter,
    - optional dark counts,
    - coincidence extraction with configurable window.
    """

    def __init__(self, name: str, timeline, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, timeline)
        config = config or {}

        self.scan_delay_ps = int(config.get("scan_delay_ps", config.get("delay_ps", 0)))
        self.delayed_arm = int(config.get("delayed_arm", 1))
        self.source_bandwidth_nm = float(config.get("source_bandwidth_nm", 0.0))
        self.source_bandwidth_arm0_nm = float(
            config.get("source_bandwidth_arm0_nm", self.source_bandwidth_nm)
        )
        self.source_bandwidth_arm1_nm = float(
            config.get("source_bandwidth_arm1_nm", self.source_bandwidth_nm)
        )
        self.center_wavelength_arm0_nm = config.get("center_wavelength_arm0_nm", None)
        self.center_wavelength_arm1_nm = config.get("center_wavelength_arm1_nm", None)
        self.coincidence_window_ps = int(config.get("coincidence_window_ps", 1000))
        self.detector_efficiency = float(config.get("detector_efficiency", 0.9))
        self.detector_jitter_ps = float(config.get("detector_jitter_ps", 0.0))
        self.matching_window_ps = int(config.get("matching_window_ps", config.get("match_window_ps", 2000)))
        self.spectral_overlap_sigma_nm = config.get("spectral_overlap_sigma_nm", None)

        # Single catch-all overlap reduction factor.
        # Legacy keys map here for backward compatibility.
        self.extra_overlap_scale = float(
            config.get("extra_overlap_scale", config.get("mode_overlap", config.get("visibility", 1.0)))
        )
        self.dark_count_rate_hz = float(config.get("dark_count_rate_hz", config.get("dark_count", 0.0)))

        # Pre-BS per-arm loss probabilities.
        losses = config.get("arm_loss", [0.0, 0.0])
        if isinstance(losses, dict):
            self.arm_loss = {
                0: float(losses.get(0, losses.get("0", 0.0))),
                1: float(losses.get(1, losses.get("1", 0.0))),
            }
        else:
            self.arm_loss = {
                0: float(losses[0]) if len(losses) > 0 else 0.0,
                1: float(losses[1]) if len(losses) > 1 else 0.0,
            }

        # Policy for photons that cannot be paired in time:
        # "single" routes as distinguishable single-photon BS events,
        # "discard" drops them.
        self.unmatched_policy = str(config.get("unmatched_policy", "single")).lower()

        # Source-name -> arm index (0/1).
        self.input_map: Dict[str, int] = {}
        self.pending = {0: deque(), 1: deque()}

        # Click logs (detector 0 / detector 1), in ps.
        self.trigger_times = [[], []]
        self.interference_events = []

        self._run_start_ps = int(self.timeline.now())
        self._last_event_ps = int(self.timeline.now())
        self._dark_counts_injected = False

        self.hom_splitter = HOMBeamSplitter(f"{name}.hom_bs", timeline)
        self.add_component(self.hom_splitter)

    def init(self):
        super().init()
        self.hom_splitter.init()

    # ------------------------------------------------------------------
    # Configuration API
    # ------------------------------------------------------------------

    def register_input(self, src_name: str, port_index: int) -> None:
        if port_index not in (0, 1):
            raise ValueError(f"port_index must be 0 or 1, got {port_index}")
        self.input_map[src_name] = int(port_index)

    def set_scan_delay_ps(self, scan_delay_ps: int) -> None:
        self.scan_delay_ps = int(scan_delay_ps)

    def set_delay_ps(self, delay_ps: int) -> None:
        # Compatibility alias.
        self.set_scan_delay_ps(delay_ps)

    def set_source_bandwidth_nm(self, source_bandwidth_nm: float) -> None:
        self.source_bandwidth_nm = float(source_bandwidth_nm)

    def set_mode_overlap(self, mode_overlap: float) -> None:
        # Compatibility alias: mode_overlap maps to extra overlap scale.
        self.extra_overlap_scale = float(mode_overlap)

    def set_visibility(self, visibility: float) -> None:
        # Legacy knob retained as optional additional overlap scale.
        self.extra_overlap_scale = float(visibility)

    def set_matching_window_ps(self, matching_window_ps: int) -> None:
        self.matching_window_ps = int(matching_window_ps)

    def set_coincidence_window_ps(self, coincidence_window_ps: int) -> None:
        self.coincidence_window_ps = int(coincidence_window_ps)

    # ------------------------------------------------------------------
    # Photon handling
    # ------------------------------------------------------------------

    def receive_qubit(self, src, qubit) -> None:
        """Receive a photon from a quantum channel and enqueue by source arm."""
        if src not in self.input_map:
            return

        arm = self.input_map[src]
        arrival_ps = int(self.timeline.now())
        self._last_event_ps = max(self._last_event_ps, arrival_ps)
        self._dark_counts_injected = False

        if self.get_generator().random() < self.arm_loss.get(arm, 0.0):
            return

        self.pending[arm].append({
            "arrival_ps": arrival_ps,
            "source": src,
            "photon": qubit,
        })
        self._process_pending()

    def _effective_arrival(self, arm: int, arrival_ps: int) -> int:
        if arm == self.delayed_arm:
            return int(arrival_ps + self.scan_delay_ps)
        return int(arrival_ps)

    def _process_pending(self) -> None:
        while self.pending[0] and self.pending[1]:
            p0 = self.pending[0][0]
            p1 = self.pending[1][0]
            eff0 = self._effective_arrival(0, p0["arrival_ps"])
            eff1 = self._effective_arrival(1, p1["arrival_ps"])
            dt = int(eff0 - eff1)

            if abs(dt) <= self.matching_window_ps:
                self.pending[0].popleft()
                self.pending[1].popleft()
                self._process_interfering_pair(p0, p1, eff0, eff1)
                continue

            if eff0 < eff1:
                self.pending[0].popleft()
                self._handle_unmatched_single(0, p0, eff0)
            else:
                self.pending[1].popleft()
                self._handle_unmatched_single(1, p1, eff1)

    def _process_interfering_pair(self, p0: dict, p1: dict, eff0_ps: int, eff1_ps: int) -> None:
        lambda0 = float(getattr(p0["photon"], "wavelength", 1550.0))
        lambda1 = float(getattr(p1["photon"], "wavelength", 1550.0))
        center0 = float(lambda0 if self.center_wavelength_arm0_nm is None else self.center_wavelength_arm0_nm)
        center1 = float(lambda1 if self.center_wavelength_arm1_nm is None else self.center_wavelength_arm1_nm)
        lambda_nm = 0.5 * (center0 + center1)

        if self.delayed_arm == 1:
            result = self.hom_splitter.two_photon_outcome(
                arrival_arm0_ps=p0["arrival_ps"],
                arrival_arm1_ps=p1["arrival_ps"],
                photon_arm0=p0["photon"],
                photon_arm1=p1["photon"],
                scan_delay_ps=self.scan_delay_ps,
                lambda_nm=lambda_nm,
                bandwidth_nm=self.source_bandwidth_nm,
                lambda_arm0_nm=center0,
                lambda_arm1_nm=center1,
                bandwidth_arm0_nm=self.source_bandwidth_arm0_nm,
                bandwidth_arm1_nm=self.source_bandwidth_arm1_nm,
                extra_overlap_scale=self.extra_overlap_scale,
            )
        else:
            result = self.hom_splitter.two_photon_outcome(
                arrival_arm0_ps=p1["arrival_ps"],
                arrival_arm1_ps=p0["arrival_ps"],
                photon_arm0=p1["photon"],
                photon_arm1=p0["photon"],
                scan_delay_ps=self.scan_delay_ps,
                lambda_nm=lambda_nm,
                bandwidth_nm=self.source_bandwidth_nm,
                lambda_arm0_nm=center1,
                lambda_arm1_nm=center0,
                bandwidth_arm0_nm=self.source_bandwidth_arm0_nm,
                bandwidth_arm1_nm=self.source_bandwidth_arm1_nm,
                extra_overlap_scale=self.extra_overlap_scale,
            )

        outputs = result["outputs"]
        n0 = int(outputs.count(0))
        n1 = int(outputs.count(1))
        nominal_click_ps = int(max(eff0_ps, eff1_ps))
        self._apply_detector_response(n0, n1, nominal_click_ps)

        result["effective_arrival_arm0_ps"] = int(eff0_ps)
        result["effective_arrival_arm1_ps"] = int(eff1_ps)
        result["matching_delta_t_ps"] = int(eff0_ps - eff1_ps)
        result["src_arm0"] = p0["source"]
        result["src_arm1"] = p1["source"]
        result["pair_id_arm0"] = getattr(p0["photon"], "pair_id", None)
        result["pair_id_arm1"] = getattr(p1["photon"], "pair_id", None)
        self.interference_events.append(result)


    def _handle_unmatched_single(self, arm: int, photon_entry: dict, effective_arrival_ps: int) -> None:
        if self.unmatched_policy == "discard":
            return

        detector_index = self.hom_splitter.single_photon_output()
        n0 = 1 if detector_index == 0 else 0
        n1 = 1 if detector_index == 1 else 0
        self._apply_detector_response(n0, n1, int(effective_arrival_ps))

    def _sample_jitter_ps(self) -> int:
        if self.detector_jitter_ps <= 0:
            return 0
        return int(np.rint(self.get_generator().normal(0.0, self.detector_jitter_ps)))

    def _apply_detector_response(self, photons_det0: int, photons_det1: int, nominal_click_ps: int) -> None:
        # Threshold detector response: P(click|n photons)=1-(1-eta)^n.
        eta = float(np.clip(self.detector_efficiency, 0.0, 1.0))
        for detector_index, n_photons in enumerate([photons_det0, photons_det1]):
            if n_photons <= 0:
                continue

            p_click = 1.0 - (1.0 - eta) ** int(n_photons)
            if self.get_generator().random() <= p_click:
                jitter = self._sample_jitter_ps()
                click_ps = int(max(0, nominal_click_ps + jitter))
                self.trigger_times[detector_index].append(click_ps)
                self._last_event_ps = max(self._last_event_ps, click_ps)

    def receive_noise_photon(self) -> None:
        """Handle Raman/background noise photon arrival at the BSM node.

        The fiber channel schedules this callback on the receiver when
        classical coexistence noise is enabled. We model unpolarized noise
        as a 50/50 random click on one of the two BSM detectors.
        """
        det_idx = 0 if self.get_generator().random() < 0.5 else 1
        n0 = 1 if det_idx == 0 else 0
        n1 = 1 if det_idx == 1 else 0
        self._apply_detector_response(n0, n1, int(self.timeline.now()))

    def _inject_dark_counts(self) -> None:
        if self._dark_counts_injected or self.dark_count_rate_hz <= 0:
            self._dark_counts_injected = True
            return

        end_ps = max(self._last_event_ps, int(self.timeline.now()))
        duration_ps = max(0, end_ps - self._run_start_ps)
        if duration_ps <= 0:
            self._dark_counts_injected = True
            return

        expected = self.dark_count_rate_hz * (duration_ps * 1e-12)
        for detector_index in (0, 1):
            count = int(self.get_generator().poisson(expected))
            if count <= 0:
                continue
            draws = self.get_generator().random(count)
            for draw in draws:
                t_ps = int(self._run_start_ps + draw * duration_ps)
                self.trigger_times[detector_index].append(t_ps)

        self._dark_counts_injected = True

    # ------------------------------------------------------------------
    # Data access + analysis helpers
    # ------------------------------------------------------------------

    def flush(self) -> None:
        while self.pending[0]:
            p0 = self.pending[0].popleft()
            eff0 = self._effective_arrival(0, p0["arrival_ps"])
            self._handle_unmatched_single(0, p0, eff0)
        while self.pending[1]:
            p1 = self.pending[1].popleft()
            eff1 = self._effective_arrival(1, p1["arrival_ps"])
            self._handle_unmatched_single(1, p1, eff1)

    def get_click_times(self, flush: bool = True, include_dark_counts: bool = True, reset: bool = False) -> list:
        if flush:
            self.flush()
        if include_dark_counts:
            self._inject_dark_counts()

        out = [sorted(self.trigger_times[0]), sorted(self.trigger_times[1])]
        if reset:
            self.reset_logs(clear_pending=True)
        return out

    def get_photon_times(self) -> list:
        # Compatibility behavior with previous HOM node: returns and clears logs.
        return self.get_click_times(flush=True, include_dark_counts=True, reset=True)

    def get_detection_counts(self) -> tuple:
        times = self.get_photon_times()
        return (len(times[0]), len(times[1]))

    def count_coincidences(self, coincidence_window_ps: Optional[int] = None) -> int:
        window = int(self.coincidence_window_ps if coincidence_window_ps is None else coincidence_window_ps)
        t0, t1 = self.get_click_times(flush=True, include_dark_counts=True, reset=False)
        i = 0
        j = 0
        count = 0
        while i < len(t0) and j < len(t1):
            dt = int(t0[i] - t1[j])
            if abs(dt) <= window:
                count += 1
                i += 1
                j += 1
            elif dt < -window:
                i += 1
            else:
                j += 1
        return count

    def click_time_differences(self, max_abs_dt_ps: Optional[int] = None) -> list:
        t0, t1 = self.get_click_times(flush=True, include_dark_counts=True, reset=False)
        deltas = []
        j0 = 0
        for t in t0:
            while j0 < len(t1) and t1[j0] < t - self.coincidence_window_ps:
                j0 += 1
            j = j0
            while j < len(t1) and t1[j] <= t + self.coincidence_window_ps:
                dt = int(t - t1[j])
                if max_abs_dt_ps is None or abs(dt) <= int(max_abs_dt_ps):
                    deltas.append(dt)
                j += 1
        return deltas

    def reset_logs(self, clear_pending: bool = True) -> None:
        self.trigger_times = [[], []]
        self.interference_events = []
        self._dark_counts_injected = False
        now_ps = int(self.timeline.now())
        self._run_start_ps = now_ps
        self._last_event_ps = now_ps
        if clear_pending:
            self.pending[0].clear()
            self.pending[1].clear()


class PolarizationBSMNode(HOMInterferenceNode):
    """Polarization-resolved linear-optics Bell-state measurement node.

    The node reuses :class:`HOMBeamSplitter` for the non-polarizing 50:50
    beam splitter, then measures each spatial output in the H/V basis. Its
    detector order is ``D3H, D3V, D4H, D4V``.

    With threshold detectors, the accepted passive-linear-optics patterns are:

    - psi-minus: D3H-D4V or D3V-D4H
    - psi-plus: D3H-D3V or D4H-D4V

    The node does not claim to distinguish phi-plus from phi-minus. For
    temporally and spectrally indistinguishable matched photons, it performs
    one joint Bell-basis projection and derives the detector pattern from that
    same outcome. This projects any remote photons entangled with the measured
    inputs, as required for entanglement swapping.
    """

    DETECTOR_NAMES = ("D3H", "D3V", "D4H", "D4V")
    PSI_MINUS_PATTERNS = frozenset({frozenset((0, 3)), frozenset((1, 2))})
    PSI_PLUS_PATTERNS = frozenset({frozenset((0, 1)), frozenset((2, 3))})
    BELL_BASIS = (
        (1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)),
        (1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)),
        (0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0),
        (0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0),
    )
    BELL_LABELS = ("phi_plus", "phi_minus", "psi_plus", "psi_minus")

    def __init__(self, name: str, timeline, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(name, timeline, config)

        efficiencies = config.get("detector_efficiencies", self.detector_efficiency)
        if isinstance(efficiencies, dict):
            self.detector_efficiencies = [
                float(efficiencies.get(detector, self.detector_efficiency))
                for detector in self.DETECTOR_NAMES
            ]
        elif isinstance(efficiencies, (list, tuple, np.ndarray)):
            if len(efficiencies) != 4:
                raise ValueError("detector_efficiencies must contain four values")
            self.detector_efficiencies = [float(value) for value in efficiencies]
        else:
            self.detector_efficiencies = [float(efficiencies)] * 4

        self.pbs_fidelity = float(config.get("pbs_fidelity", 1.0))
        self.pbs_mismeasure_prob = float(config.get("pbs_mismeasure_prob", 0.0))
        self.trigger_times = [[], [], [], []]
        self.interaction_events = []

    @classmethod
    def classify_detector_pair(cls, detector_a: int, detector_b: int) -> Optional[str]:
        """Return the identifiable Bell state for a two-detector pattern."""
        if detector_a == detector_b:
            return None
        pattern = frozenset((int(detector_a), int(detector_b)))
        if pattern in cls.PSI_MINUS_PATTERNS:
            return "psi_minus"
        if pattern in cls.PSI_PLUS_PATTERNS:
            return "psi_plus"
        return None

    def _measure_hv(self, photon: Photon) -> Optional[int]:
        if self.get_generator().random() > float(np.clip(self.pbs_fidelity, 0.0, 1.0)):
            return None
        result = int(Photon.measure(polarization["bases"][0], photon, self.get_generator()))
        if self.get_generator().random() < float(np.clip(self.pbs_mismeasure_prob, 0.0, 1.0)):
            result = 1 - result
        return result

    def _apply_channel_response(self, detector_index: int, photon_count: int, nominal_click_ps: int) -> bool:
        if photon_count <= 0:
            return False
        eta = float(np.clip(self.detector_efficiencies[detector_index], 0.0, 1.0))
        p_click = 1.0 - (1.0 - eta) ** int(photon_count)
        if self.get_generator().random() > p_click:
            return False
        click_ps = int(max(0, nominal_click_ps + self._sample_jitter_ps()))
        self.trigger_times[detector_index].append(click_ps)
        self._last_event_ps = max(self._last_event_ps, click_ps)
        return True

    def _apply_pbs_and_detector_response(
        self,
        ideal_detector_indices: list[int],
        nominal_click_ps: int,
    ) -> list[int]:
        """Apply PBS loss/error and threshold-detector response to ideal channels."""
        channel_occupancy = [0, 0, 0, 0]
        for detector_index in ideal_detector_indices:
            if self.get_generator().random() > float(np.clip(self.pbs_fidelity, 0.0, 1.0)):
                continue
            observed_index = int(detector_index)
            if self.get_generator().random() < float(np.clip(self.pbs_mismeasure_prob, 0.0, 1.0)):
                observed_index ^= 1
            channel_occupancy[observed_index] += 1

        return [
            detector_index
            for detector_index, photon_count in enumerate(channel_occupancy)
            if self._apply_channel_response(detector_index, photon_count, nominal_click_ps)
        ]

    def _project_bell_pair(self, photon0: Photon, photon1: Photon) -> int:
        """Jointly project two inputs while preserving collapse of remote partners."""
        if photon0.use_qm or photon1.use_qm:
            raise NotImplementedError(
                "PolarizationBSMNode Bell projection currently requires local FreeQuantumState photons"
            )
        if photon1.quantum_state not in photon0.quantum_state.entangled_states:
            photon0.combine_state(photon1)
        return int(
            Photon.measure_multiple(
                self.BELL_BASIS,
                [photon0, photon1],
                self.get_generator(),
            )
        )

    def _bell_outcome_channels(self, bell_result: int) -> tuple[list[int], list[int]]:
        """Return ideal detector channels and spatial outputs for one Bell result."""
        if bell_result == 2:  # psi-plus: opposite polarization, same BS output
            channels = [0, 1] if self.get_generator().random() < 0.5 else [2, 3]
        elif bell_result == 3:  # psi-minus: opposite polarization, opposite BS outputs
            channels = [0, 3] if self.get_generator().random() < 0.5 else [1, 2]
        else:  # phi outcomes bunch into one unresolved threshold-detector channel
            detector_index = int(self.get_generator().choice(4))
            channels = [detector_index, detector_index]
        return channels, [channel // 2 for channel in channels]

    def _route_photons(
        self,
        photons: list[Photon],
        spatial_outputs: list[int],
        nominal_click_ps: int,
    ) -> list[int]:
        channel_occupancy = [0, 0, 0, 0]
        for photon, spatial_output in zip(photons, spatial_outputs):
            polarization_output = self._measure_hv(photon)
            if polarization_output is None:
                continue
            detector_index = 2 * int(spatial_output) + int(polarization_output)
            channel_occupancy[detector_index] += 1

        clicked = []
        for detector_index, photon_count in enumerate(channel_occupancy):
            if self._apply_channel_response(detector_index, photon_count, nominal_click_ps):
                clicked.append(detector_index)
        return clicked

    def _process_interfering_pair(self, p0: dict, p1: dict, eff0_ps: int, eff1_ps: int) -> None:
        lambda0 = float(getattr(p0["photon"], "wavelength", 1550.0))
        lambda1 = float(getattr(p1["photon"], "wavelength", 1550.0))
        center0 = float(lambda0 if self.center_wavelength_arm0_nm is None else self.center_wavelength_arm0_nm)
        center1 = float(lambda1 if self.center_wavelength_arm1_nm is None else self.center_wavelength_arm1_nm)
        lambda_nm = 0.5 * (center0 + center1)

        if self.delayed_arm == 1:
            result = self.hom_splitter.two_photon_outcome(
                arrival_arm0_ps=p0["arrival_ps"],
                arrival_arm1_ps=p1["arrival_ps"],
                photon_arm0=p0["photon"],
                photon_arm1=p1["photon"],
                scan_delay_ps=self.scan_delay_ps,
                lambda_nm=lambda_nm,
                bandwidth_nm=self.source_bandwidth_nm,
                lambda_arm0_nm=center0,
                lambda_arm1_nm=center1,
                bandwidth_arm0_nm=self.source_bandwidth_arm0_nm,
                bandwidth_arm1_nm=self.source_bandwidth_arm1_nm,
                extra_overlap_scale=self.extra_overlap_scale,
            )
        else:
            result = self.hom_splitter.two_photon_outcome(
                arrival_arm0_ps=p1["arrival_ps"],
                arrival_arm1_ps=p0["arrival_ps"],
                photon_arm0=p1["photon"],
                photon_arm1=p0["photon"],
                scan_delay_ps=self.scan_delay_ps,
                lambda_nm=lambda_nm,
                bandwidth_nm=self.source_bandwidth_nm,
                lambda_arm0_nm=center1,
                lambda_arm1_nm=center0,
                bandwidth_arm0_nm=self.source_bandwidth_arm1_nm,
                bandwidth_arm1_nm=self.source_bandwidth_arm0_nm,
                extra_overlap_scale=self.extra_overlap_scale,
            )

        nominal_click_ps = int(max(eff0_ps, eff1_ps))
        mode_overlap = float(np.clip(
            result["temporal_overlap"]
            * result["spectral_overlap"]
            * result["extra_overlap_scale"],
            0.0,
            1.0,
        ))
        coherent_projection = self.get_generator().random() < mode_overlap
        bell_projection = None

        if coherent_projection:
            bell_result = self._project_bell_pair(p0["photon"], p1["photon"])
            bell_projection = self.BELL_LABELS[bell_result]
            ideal_channels, outputs = self._bell_outcome_channels(bell_result)
            clicked = self._apply_pbs_and_detector_response(ideal_channels, nominal_click_ps)
        else:
            outputs = [int(value) for value in result["outputs"]]
            if outputs == [0, 1] and self.get_generator().random() < 0.5:
                outputs.reverse()
            clicked = self._route_photons(
                [p0["photon"], p1["photon"]],
                outputs,
                nominal_click_ps,
            )
        bell_state = self.classify_detector_pair(*clicked) if len(clicked) == 2 else None

        result.update({
            "effective_arrival_arm0_ps": int(eff0_ps),
            "effective_arrival_arm1_ps": int(eff1_ps),
            "matching_delta_t_ps": int(eff0_ps - eff1_ps),
            "src_arm0": p0["source"],
            "src_arm1": p1["source"],
            "pair_id_arm0": getattr(p0["photon"], "pair_id", None),
            "pair_id_arm1": getattr(p1["photon"], "pair_id", None),
            "detector_indices": clicked,
            "detector_names": [self.DETECTOR_NAMES[index] for index in clicked],
            "bell_state": bell_state,
            "bell_projection": bell_projection,
            "mode_overlap": mode_overlap,
            "coherent_projection": coherent_projection,
            "accepted": bell_state is not None,
        })
        self.interference_events.append(result)
        self.interaction_events.append(result)

    def _handle_unmatched_single(self, arm: int, photon_entry: dict, effective_arrival_ps: int) -> None:
        if self.unmatched_policy == "discard":
            return
        spatial_output = self.hom_splitter.single_photon_output()
        self._route_photons(
            [photon_entry["photon"]],
            [spatial_output],
            int(effective_arrival_ps),
        )

    def receive_noise_photon(self) -> None:
        """Route one unpolarized noise photon to a random BSM detector."""
        detector_index = int(self.get_generator().choice(4))
        self._apply_channel_response(detector_index, 1, int(self.timeline.now()))

    def _inject_dark_counts(self) -> None:
        if self._dark_counts_injected or self.dark_count_rate_hz <= 0:
            self._dark_counts_injected = True
            return

        end_ps = max(self._last_event_ps, int(self.timeline.now()))
        duration_ps = max(0, end_ps - self._run_start_ps)
        expected = self.dark_count_rate_hz * (duration_ps * 1e-12)
        for detector_index in range(4):
            count = int(self.get_generator().poisson(expected))
            if count > 0:
                draws = self.get_generator().random(count)
                self.trigger_times[detector_index].extend(
                    int(self._run_start_ps + draw * duration_ps) for draw in draws
                )
        self._dark_counts_injected = True

    def get_click_times(
        self,
        flush: bool = True,
        include_dark_counts: bool = True,
        reset: bool = False,
    ) -> dict[str, list[int]]:
        """Return four polarization-resolved timestamp streams."""
        if flush:
            self.flush()
        if include_dark_counts:
            self._inject_dark_counts()
        out = {
            name: sorted(self.trigger_times[index])
            for index, name in enumerate(self.DETECTOR_NAMES)
        }
        if reset:
            self.reset_logs(clear_pending=True)
        return out

    def get_photon_times(self) -> dict[str, list[int]]:
        return self.get_click_times(flush=True, include_dark_counts=True, reset=True)

    def get_detection_counts(self) -> dict[str, int]:
        times = self.get_click_times(flush=True, include_dark_counts=True, reset=False)
        return {name: len(values) for name, values in times.items()}

    def drain_click_times(self, include_dark_counts: bool = True) -> dict[str, list[int]]:
        """Return and clear completed telemetry without clearing pending photons."""
        streams = self.get_click_times(
            flush=False,
            include_dark_counts=include_dark_counts,
            reset=False,
        )
        self.trigger_times = [[], [], [], []]
        self.interference_events = []
        self.interaction_events = []
        self._dark_counts_injected = False
        now_ps = int(self.timeline.now())
        self._run_start_ps = now_ps
        self._last_event_ps = now_ps
        return streams

    def get_bell_events(
        self,
        coincidence_window_ps: Optional[int] = None,
        include_dark_counts: bool = True,
    ) -> list[dict]:
        """Greedily classify accepted Bell patterns from detector timestamps."""
        window = int(self.coincidence_window_ps if coincidence_window_ps is None else coincidence_window_ps)
        streams = self.get_click_times(
            flush=True,
            include_dark_counts=include_dark_counts,
            reset=False,
        )
        clicks = sorted(
            (int(time_ps), detector_index)
            for detector_index, name in enumerate(self.DETECTOR_NAMES)
            for time_ps in streams[name]
        )
        used = set()
        events = []
        for i, (time_i, detector_i) in enumerate(clicks):
            if i in used:
                continue
            best = None
            for j in range(i + 1, len(clicks)):
                if j in used:
                    continue
                time_j, detector_j = clicks[j]
                if time_j - time_i > window:
                    break
                bell_state = self.classify_detector_pair(detector_i, detector_j)
                if bell_state is None:
                    continue
                delta = abs(time_j - time_i)
                if best is None or delta < best[0]:
                    best = (delta, j, time_j, detector_j, bell_state)
            if best is None:
                continue
            _, j, time_j, detector_j, bell_state = best
            used.update((i, j))
            events.append({
                "time_ps": int(max(time_i, time_j)),
                "delta_t_ps": int(time_j - time_i),
                "detectors": (
                    self.DETECTOR_NAMES[detector_i],
                    self.DETECTOR_NAMES[detector_j],
                ),
                "bell_state": bell_state,
            })
        return events

    def count_bell_states(self, coincidence_window_ps: Optional[int] = None) -> dict[str, int]:
        counts = {"psi_minus": 0, "psi_plus": 0}
        for event in self.get_bell_events(coincidence_window_ps=coincidence_window_ps):
            counts[event["bell_state"]] += 1
        return counts

    def reset_logs(self, clear_pending: bool = True) -> None:
        self.trigger_times = [[], [], [], []]
        self.interference_events = []
        self.interaction_events = []
        self._dark_counts_injected = False
        now_ps = int(self.timeline.now())
        self._run_start_ps = now_ps
        self._last_event_ps = now_ps
        if clear_pending:
            self.pending[0].clear()
            self.pending[1].clear()


class HOMBeamSplitterNode(HOMInterferenceNode):
    """Compatibility wrapper for older notebooks using HOMBeamSplitterNode name."""
