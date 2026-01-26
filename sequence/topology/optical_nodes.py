"""Polarization analyzer node for quantum state measurements.

This module provides a configurable compound node for polarization analysis
with optional QWP, HWP, and detector components.
"""

from ..topology.node import Node
from ..components.detector import QSDetectorPolarizationStatic
from ..components.wave_plate import WavePlate
from ..components.light_source import SPDCBellSource
from ..utils.encoding import polarization
from ..kernel.entity import Entity
from typing import Optional, Dict, Any
import numpy as np


class SourcePort(Entity):
    def __init__(self, name, timeline, owner:Node):
        super().__init__(name, timeline)
        self.owner = owner
        self.port_index = int(name)
        self.add_receiver(owner)

    def init(self):
        pass

    def get(self, photon, **kwargs):
        photon.name = self.name
        photon.port_index = self.port_index
        self._receivers[0].get(photon)


class SpdcSourceNode(Node):
    """
    Node that emits entangled photon pairs using an SPDCBellSource.
    """

    def __init__(self, name, timeline, config):
        super().__init__(name, timeline)
        self.name = name
        self.emission_count = 0

        # Default values for SPDC configuration
        default_config = {
            'wavelengths': [1550, 1550],
            'frequency': 8e7,
            'mean_photon_num': 0.1,
            'phase_error': 0.0,
            'bandwidth': 0,
            'encoding': polarization,
            'bell_state': 'psi+'
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
            bell_state=merged_config['bell_state']
        )

        # Create and connect output ports
        self.ports = {}
        for i in range(2):
            self.ports[i] = SourcePort(str(i), self.timeline, self)
            self.spdc.add_receiver(self.ports[i])

        self.first_component_name = self.spdc.name
        self.spdc.owner = self
        self.components = [self.spdc] + list(self.ports.values())

    def emit(self, num_pulses: int):
        """
        Emit entangled photon pairs.
        """
        self.spdc.emit(num_pulses=num_pulses)


    def get(self, photon, **kwargs):
        if hasattr(photon, 'port_index') and photon.port_index == 0:
            self.emission_count += 1

        # Route to appropriate quantum channel based on port name
        try:
            port_index = getattr(photon, 'port_index', None)
            if port_index is not None and port_index < len(self.qchannels):
                self.qchannels[port_index].get(photon)
        except (AttributeError, IndexError):
            pass
    
    # ========================================================================
    # Configuration API
    # ========================================================================

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
    - 'qwp_hwp': Photon → QWP → HWP → PBS (for full tomography)
    - 'custom': Manual angle control
    
    Attributes:
        mode (str): Architecture mode ('hwp_only', 'qwp_hwp', 'custom').
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
                - 'mode' (str): 'hwp_only', 'qwp_hwp', or 'custom' (default 'qwp_hwp').
                - 'rotation_angle' (float): For 'hwp_only' mode, rotation angle in radians.
                - 'basis' (str): For 'qwp_hwp' mode, Pauli basis 'Z', 'X', or 'Y'.
                - 'qwp_angle' (float): For 'custom' mode, QWP angle in radians.
                - 'hwp_angle' (float): For 'custom' mode, HWP angle in radians.
                - 'qwp_fidelity' (float): QWP transmission (default 1.0).
                - 'hwp_fidelity' (float): HWP transmission (default 1.0).
                - 'detector_efficiency' (float): SPD efficiency (default 0.9).
                - 'dark_count' (float): SPD dark count rate in Hz (default 0).
                - 'pbs_fidelity' (float): PBS transmission (default 1.0).
                - 'mismeasure_prob' (float): PBS measurement error (default 0.0).
        """
        Node.__init__(self, name, timeline)
        
        # Parse configuration
        config = config or {}
        mode = config.get('mode', 'qwp_hwp')
        qwp_fidelity = config.get('qwp_fidelity', 1.0)
        hwp_fidelity = config.get('hwp_fidelity', 1.0)
        detector_efficiency = config.get('detector_efficiency', 0.9)
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
            
            # Wire: HWP → Detector
            self.hwp.add_receiver(self.detector)
            self.hwp.owner = self
            
            self.components = [self.hwp, self.detector]
            self.set_first_component(self.hwp.name)
            
        elif mode == 'qwp_hwp':
            # QWP + HWP for full tomography
            self.qwp = WavePlate(
                f"{name}.QWP",
                timeline,
                plate_type="QWP",
                angle=0.0,
                fidelity=qwp_fidelity
            )
            
            self.hwp = WavePlate(
                f"{name}.HWP",
                timeline,
                plate_type="HWP",
                angle=0.0,
                fidelity=hwp_fidelity
            )
            
            # Wire: QWP → HWP → Detector
            self.qwp.add_receiver(self.hwp)
            self.hwp.add_receiver(self.detector)
            self.qwp.owner = self
            self.hwp.owner = self
            
            self.components = [self.qwp, self.hwp, self.detector]
            self.set_first_component(self.qwp.name)
            
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
            
            # Connect to detector
            if last_component:
                last_component.add_receiver(self.detector)
            else:
                first_component = self.detector
            
            # Register components
            self.components = [c for c in [self.qwp, self.hwp, self.detector] if c]
            self.set_first_component(first_component.name)
            
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'hwp_only', 'qwp_hwp', or 'custom'.")
    
    def init(self) -> None:
        """Initialize all components."""
        super().init()
        if self.qwp:
            self.qwp.init()
        if self.hwp:
            self.hwp.init()
        self.detector.init()
    
    def get(self, photon, **kwargs):
        """Receive photon and pass to first component.
        
        Args:
            photon (Photon): incoming photon.
            **kwargs: additional arguments passed through chain.
        """
        # Get first component (depends on mode)
        first = self.components[0]
        first.get(photon, **kwargs)
    
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
        
        self.hwp.set_angle(theta_rad / 2.0)
    
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
        """Configure analyzer to measure in a Pauli basis (qwp_hwp mode only).
        
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
            ValueError: if mode is not 'qwp_hwp' or basis invalid.
        """
        if self.mode != 'qwp_hwp':
            raise ValueError("set_basis() only valid in 'qwp_hwp' mode")
        
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
