"""
Realistic fiber channel for SeQUeNCe.

Implementation of physical effects described in:
Banner, Rolston, Britton, "BIFROST: A first-principles model of polarization mode dispersion in optical fiber," arXiv:2510.01212, 2025.

This code is written from the paper's equations/sections (Jones matrices, birefringence mechanisms, dispersion, DGD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np

from ..components.optical_channel import QuantumChannel
from ..constants import SPEED_OF_LIGHT_M_PER_S, SPEED_OF_LIGHT, PLANCK_CONSTANT

from sequence.kernel.process import Process
from sequence.kernel.event import Event
from ..components.photon import Photon

if TYPE_CHECKING:
    from ..topology.node import Node
    from ..kernel.timeline import Timeline

PI = np.pi


# -----------------------------
# Sellmeier with Temperature Dependence
# -----------------------------
@dataclass(frozen=True)
class SellmeierModel:
    """
    Temperature-dependent Sellmeier model.
    """
    B: Tuple[float, ...]           # resonance strengths (unitless)
    C_um: Tuple[float, ...]        # resonance wavelengths in micrometers (μm)
    
    # Temperature dependence (optional)
    B_temp_coeffs: Optional[np.ndarray] = None  # shape (N_resonances, 5) for T^0..T^4
    C_temp_coeffs: Optional[np.ndarray] = None  # shape (N_resonances, 5) for T^0..T^4
    dn_dT: float = 0.0                          # thermo-optic coefficient (1/K)
    T_ref_C: float = 20.0                       # reference temperature (°C)

    def n(self, wavelength_m: float, temperature_C: float = 20.0) -> float:
        """
        Calculate refractive index at given wavelength and temperature.
        
        Args:
            wavelength_m: Wavelength in meters
            temperature_C: Temperature in Celsius
            
        Returns:
            Refractive index
        """
        lam_um = wavelength_m * 1e6
        lam2 = lam_um * lam_um
        
        # If temperature-dependent coefficients exist
        if self.B_temp_coeffs is not None and self.C_temp_coeffs is not None:
            # Temperature powers:
            T_K = temperature_C + 273.15
            T_powers = np.array([1.0, T_K, T_K**2, T_K**3, T_K**4])
            
            n2 = 1.0
            for i, (_, _) in enumerate(zip(self.B, self.C_um)):
                B_T = np.dot(self.B_temp_coeffs[i], T_powers)
                C_T = np.dot(self.C_temp_coeffs[i], T_powers)
                
                C_T2 = C_T * C_T
                n2 += B_T * lam2 / (lam2 - C_T2)
            
            return float(np.sqrt(max(n2, 1.0)))
        
        # Otherwise use standard Sellmeier at reference temp
        n2 = 1.0
        for Bi, Ci_um in zip(self.B, self.C_um):
            Ci2 = Ci_um * Ci_um
            n2 += Bi * lam2 / (lam2 - Ci2)
        
        n_ref = np.sqrt(max(n2, 1.0))
        
        # Apply thermo-optic correction if dn_dT is provided
        if self.dn_dT != 0.0:
            n_T = n_ref + self.dn_dT * (temperature_C - self.T_ref_C)
            return float(n_T)
        
        return float(n_ref)

SILICA_B_TEMP = np.array([
    # Resonance 1: B coefficients [T^0, T^1, T^2, T^3, T^4]
    [1.10127, -4.94251e-5, 5.27414e-7, -1.59700e-9, 1.75949e-12],
    # Resonance 2
    [1.78752e-5, 4.76391e-5, -4.49019e-7, 1.44546e-9, -1.57223e-12],
    # Resonance 3
    [7.93552e-1, -1.27815e-3, 1.84595e-5, -9.20275e-8, 1.48829e-10],
])

SILICA_C_TEMP = np.array([
    # Resonance 1: C coefficients [T^0, T^1, T^2, T^3, T^4]
    [-8.906e-2, 9.0873e-6, -6.53638e-8, 7.77072e-11, 6.84605e-14],
    # Resonance 2
    [2.97562e-1, -8.59578e-4, 6.59069e-6, -1.09482e-8, 7.85145e-13],
    # Resonance 3
    [9.34454, -70.9788e-3, 1.01968e-4, -5.07660e-7, 8.21348e-10],
])

SILICA_SELLMEIER = SellmeierModel(
    B=(1.10127, 1.78752e-5, 7.93552e-1),  # T^0 terms (for reference)
    C_um=(-8.906e-2, 2.97562e-1, 9.34454),  # T^0 terms (for reference)
    B_temp_coeffs=SILICA_B_TEMP,
    C_temp_coeffs=SILICA_C_TEMP,
    T_ref_C=20.0,  # Temperature is in Celsius in 
)

# Germania - uses thermo-optic correction
GERMANIA_SELLMEIER = SellmeierModel(
    B=(0.80686642, 0.71815848, 0.85416831),
    C_um=(0.068972606, 0.15396605, 11.841931),
    dn_dT=1.31e-5,  # Calculated from quartic formula in calcNGe
    T_ref_C=24.0,
)

class RamanScatteringConstants:
    '''
    Background-generation constants for Raman scattering.

    From: Burenkov et al., "Synchronization and coexistence in quantum networks,"
          Optics Express 31, 11431 (2023), Table 2.

    Units: 10^-23 m^-1 Hz^-1
    '''

    # Beta constants: (beta_FS, beta_BS) for each wavelength pair
    BETA_TABLE = {
        # Classical → Quantum : (Forward Scattering, Back Scattering)
        (1270, 1550): (0.058e-23, 0.061e-23),  # Best option (lowest noise)
        (1310, 1550): (0.421e-23, 0.449e-23),  # 7x more noise
        (1330, 1550): (0.699e-23, 0.745e-23),  # 12x more noise
        (1490, 1550): (3.69e-23,  3.75e-23),   # 64x more noise (avoid!)
    }

    @staticmethod
    def get_beta(classical_nm: float, quantum_nm: float) -> tuple[float, float]:
        '''
        Get scattering constants for wavelength pair.

        Args:
            classical_nm: Classical signal wavelength (nm)
            quantum_nm: Quantum channel wavelength (nm)

        Returns:
            (beta_FS, beta_BS): Forward and backward scattering constants [m^-1 Hz^-1]

        Raises:
            ValueError: If wavelength pair not in table
        '''
        # Round to nearest 10 nm to match table entries
        key = (int(round(classical_nm/10)*10), int(round(quantum_nm/10)*10))

        if key not in RamanScatteringConstants.BETA_TABLE:
            available = list(RamanScatteringConstants.BETA_TABLE.keys())
            raise ValueError(
                f"No Raman scattering data for wavelength pair {key}. "
                f"Available pairs: {available}. "
                f"Use 1270 nm classical for lowest noise."
            )

        return RamanScatteringConstants.BETA_TABLE[key]


DEFAULT_RAMAN = RamanScatteringConstants()




# -----------------------------
# Material mixing (Eq. 23 from paper)
# -----------------------------
@dataclass(frozen=True)
class GlassMixture:
    """
    Linear additive model for properties vs molar fraction m (Eq. 23).
    Now supports temperature-dependent refractive indices.
    """
    m_germania: float  # 0..1

    silica: SellmeierModel = SILICA_SELLMEIER
    germania: SellmeierModel = GERMANIA_SELLMEIER

    def n(self, wavelength_m: float, temperature_C: float = 20.0) -> float:
        """
        Get refractive index at wavelength and temperature.
        
        Uses linear mixing: n = (1-m)*n_silica + m*n_germania
        """
        m = float(np.clip(self.m_germania, 0.0, 1.0))
        if m <= 0:
            return self.silica.n(wavelength_m, temperature_C)
        
        ns = self.silica.n(wavelength_m, temperature_C)
        ng = self.germania.n(wavelength_m, temperature_C)
        return (1.0 - m) * ns + m * ng


# -----------------------------
# Fiber parameters
# -----------------------------
@dataclass
class FiberSpec:
    # geometry
    core_radius_m: float = 4.1e-6
    cladding_radius_m: float = 62.5e-6

    # composition (molar fractions)
    core_m_germania: float = 0.036
    clad_m_germania: float = 0.0

    # operating conditions
    wavelength_m: float = 1550e-9
    temperature_C: float = 20.0
    reference_temp_C: float = 20.0

    # ellipticity (a/b where a > b); if 1.0 => circular
    core_ellipticity: float = 1.0

    # bending & tension
    bend_radius_m: float = 0.0        # R in Eq. (9); 0 -> straight
    axial_tension_N: float = 0.0      # F in Eq. (10)

    # twist
    twist_rate_rad_per_m: float = 0.0 # tau  in Eq. (11)

    # DGD numerical derivative step (paper suggests ~0.1 nm)
    d_lambda_m: float = 0.1e-9

    # Classical coexistence (OPTIONAL - default disabled)
    classical_coexist_enabled: bool = False
    classical_wavelength_nm: float = 1270.0
    classical_power_mW: float = 1.0
    quantum_wavelength_nm: float = wavelength_m * 1e9
    quantum_bandwidth_Hz: float = 100e9


@dataclass
class FiberSection:
    """
    Represents a segment of fiber with its own spec and length.
    Enables multi-section fiber modeling.
    """
    length_m: float
    spec: FiberSpec
    
    def __post_init__(self):
        if self.length_m <= 0:
            raise ValueError(f"Section length must be positive, got {self.length_m}")

# -----------------------------
# Physics helpers
# -----------------------------
def v_parameter(r0_m: float, wavelength_m: float, nco: float, ncl: float) -> float:
    """Eq. (3): V = r0*k0*sqrt(nco^2 - ncl^2)"""
    k0 = 2 * PI / wavelength_m
    val = r0_m * k0 * np.sqrt(max(nco * nco - ncl * ncl, 0.0))
    return float(val)


def beta_fundamental_approx(r0_m: float, wavelength_m: float, nco: float, ncl: float) -> float:
    """
    Eq. (2) from paper: approximation for propagation constant beta of fundamental mode.
    """
    k0 = 2 * PI / wavelength_m
    V = v_parameter(r0_m, wavelength_m, nco, ncl)
    if V <= 0:
        return float(nco * k0)

    term = ((1.0 + np.sqrt(2.0)) * V) / (1.0 + (4.0 + V**4)**0.25)
    beta_squared = (nco * k0)**2 - (term / r0_m)**2
    beta_squared = max(beta_squared, 0.0)
    return float(np.sqrt(beta_squared))


def jones_linear(delta_beta: float, length_m: float, beta_avg: float) -> np.ndarray:
    """Jones matrix with both propagation phase and birefringence"""
    # Average phase both modes accumulate
    phase_avg = beta_avg * length_m
    # Differential phase from birefringence
    phase_diff = 0.5 * delta_beta * length_m
    
    return np.array([
        [np.exp(1j * (phase_avg + phase_diff)), 0.0],
        [0.0, np.exp(1j * (phase_avg - phase_diff))]
    ], dtype=complex)


def jones_circular(delta_beta: float, length_m: float) -> np.ndarray:
    """Eq. (16): rotation matrix"""
    a = 0.5 * delta_beta * length_m
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], dtype=complex)

def get_fiber_attenuation_per_m(wavelength_nm: float) -> float:
    '''
    Get fiber attenuation coefficient at any wavelength.
    
    Uses SMF-28 measured data from Burenkov et al., Table 1,
    with linear interpolation for intermediate wavelengths.
    
    Args:
        wavelength_nm: Wavelength in nanometers (valid: 1200-1700 nm)
        
    Returns:
        Attenuation constant alpha in m^-1 (for exponential decay)
    '''
    # Measured SMF-28 data points (Table 1)
    lam_data = np.array([1270, 1310, 1330, 1490, 1550, 1625])
    alpha_dB_km = np.array([0.34, 0.32, 0.28, 0.19, 0.17, 0.18])
    
    # Interpolate to get dB/km at requested wavelength
    alpha_dB_per_km = float(np.interp(wavelength_nm, lam_data, alpha_dB_km))
    
    # Convert dB/km to m^-1:
    # alpha[m^-1] = alpha[dB/km] × ln(10) / (10 × 1000)
    alpha_per_m = alpha_dB_per_km * np.log(10) / 10000.0
    
    return alpha_per_m



# -----------------------------
# Material constants (Table I from paper)
# -----------------------------
@dataclass(frozen=True)
class MaterialConstants:
    """Material properties from BIFROST paper Table I"""
    alpha_silica: float = 5.4e-7
    alpha_germania: float = 1.0e-5
    nu_silica: float = 0.170
    nu_germania: float = 0.212
    p11_silica: float = 0.121
    p12_silica: float = 0.270
    p11_germania: float = 0.130
    p12_germania: float = 0.288
    young_silica_GPa: float = 74.0
    young_germania_GPa: float = 45.5
    softening_silica_C: float = 1100.0
    softening_germania_C: float = 300.0


DEFAULT_CONST = MaterialConstants()


def mix_linear(m: float, ys: float, yg: float) -> float:
    """Linear mixing rule for material properties"""
    m = float(np.clip(m, 0.0, 1.0))
    return (1.0 - m) * ys + m * yg


# -----------------------------
# Birefringence mechanisms (Equations 6, 8-11 from paper)
# -----------------------------

def delta_beta_core_ellipticity(spec: FiberSpec, nco: float, ncl: float) -> float:
    """
    Calculate birefringence (Δbeta) due to core ellipticity.
    
    Based on equation (6) from the BIFROST paper:
    
    """
    if spec.core_ellipticity <= 1.0:
        return 0.0
    
    epsilon = spec.core_ellipticity
    e2 = 1.0 - 1.0 / (epsilon * epsilon)
    
    r0 = spec.core_radius_m
    V = v_parameter(r0, spec.wavelength_m, nco, ncl)
    if V <= 1.0:
        return 0.0

    Delta = 0.5 * (1.0 - (ncl**2 / nco**2))
    
    logV = np.log(V)

    numerator = 4.0 * (logV**3)
    denominator = (V**3) * (1.0 + logV)
    
    # Note the negative sign: The definition already includes the orientation
    result = (e2 * ((2.0 * Delta)**(3/2)) / r0) * (numerator / denominator)
    
    return float(result)


def delta_beta_asym_thermal(spec: FiberSpec, nco: float, ncl: float, const: MaterialConstants) -> float:
    """
    Eq. (8) from BIFROST paper: Δbeta due to asymmetric thermal stress.
    """
    if spec.core_ellipticity <= 1.0:
        return 0.0

    # Material properties (mixing rules)
    alpha_co = mix_linear(spec.core_m_germania, const.alpha_silica, const.alpha_germania)
    alpha_cl = mix_linear(spec.clad_m_germania, const.alpha_silica, const.alpha_germania)
    nu_p = mix_linear(spec.core_m_germania, const.nu_silica, const.nu_germania)
    p11 = mix_linear(spec.core_m_germania, const.p11_silica, const.p11_germania)
    p12 = mix_linear(spec.core_m_germania, const.p12_silica, const.p12_germania)
    Ts_C = mix_linear(spec.core_m_germania, const.softening_silica_C, const.softening_germania_C)

    epsilon = spec.core_ellipticity
    r0 = spec.core_radius_m
    
    k0 = 2.0 * PI / spec.wavelength_m
    V = v_parameter(r0, spec.wavelength_m, nco, ncl)
    beta = beta_fundamental_approx(r0, spec.wavelength_m, nco, ncl)

    inside = max((nco * k0)**2 - beta**2, 0.0)
    u_squared = (r0**2) * inside
    factor_mode = 1.0 - u_squared / (V * V) if V > 0 else 0.0

    ellipt_factor = (epsilon - 1.0) / (epsilon + 1.0)

    T = spec.temperature_C
    thermal_term = (alpha_cl - alpha_co) * abs(Ts_C - T) / (1.0 - nu_p**2)
    
    result = k0 * factor_mode * 0.5 * (nco**3) * (p11 - p12) * thermal_term * ellipt_factor

    return float(result)

def delta_beta_bending(spec: FiberSpec, nco: float, const: MaterialConstants) -> float:
    """
    Eq. (9) and (10) from BIFROST paper: bending birefringence.
    
    """
    if spec.bend_radius_m <= 0.0:
        return 0.0

    p11 = mix_linear(spec.core_m_germania, const.p11_silica, const.p11_germania)
    p12 = mix_linear(spec.core_m_germania, const.p12_silica, const.p12_germania)
    nu_p = mix_linear(spec.core_m_germania, const.nu_silica, const.nu_germania)
    E_Pa = 1e9 * mix_linear(spec.core_m_germania, const.young_silica_GPa, const.young_germania_GPa)

    k0 = 2.0 * PI / spec.wavelength_m
    R = spec.bend_radius_m
    rcl = spec.cladding_radius_m

    d_beta = k0 * (nco**3 / 4.0) * (p11 - p12) * (1.0 + nu_p) * (rcl**2 / R**2)

    if spec.axial_tension_N > 0.0:
        F = spec.axial_tension_N
        tension_term = ((2.0 - 3.0 * nu_p) / (1.0 - nu_p)) * (rcl / R) * (F / (PI * rcl**2 * E_Pa))
        d_beta += k0 * (nco**3 / 2.0) * (p11 - p12) * (1.0 + nu_p) * tension_term

    return float(d_beta)

def delta_beta_twist(spec: FiberSpec, nco: float, const: MaterialConstants) -> float:
    """
    Birefringence due to twisting.
    """
    if spec.twist_rate_rad_per_m == 0.0:
        return 0.0
    
    p11 = mix_linear(spec.core_m_germania, const.p11_silica, const.p11_germania)
    p12 = mix_linear(spec.core_m_germania, const.p12_silica, const.p12_germania)
    
    return float(-((nco**2 / 2.0) * (p11 - p12)) * spec.twist_rate_rad_per_m)



# -----------------------------
# Channel implementation
# -----------------------------
class fiberQuantumChannel(QuantumChannel):
    """
    SeQUeNCe QuantumChannel with physics-based polarization + dispersion + DGD.

    Based on BIFROST paper:
    - Jones matrix from birefringence mechanisms (Eqs. 6, 8-11, 15-16)
    - DGD from Jones derivative method (Eq. 24)
    - Chromatic dispersion from Eq. (5)
    """

    def __init__(
        self,
        name: str,
        timeline: "Timeline",
        attenuation: float,
        distance: float,
        polarization_fidelity: float = 1.0,
        light_speed: float = SPEED_OF_LIGHT,
        frequency: float = 8e7,
        spec: Optional[FiberSpec] = None,  # For single uniform fiber
        sections: Optional[list[FiberSection]] = None,  # For multi-section fiber
        const: MaterialConstants = DEFAULT_CONST,
    ):
        super().__init__(name, timeline, attenuation, distance,
                         polarization_fidelity, light_speed, frequency)
        
        # Validate inputs
        if spec is not None and sections is not None:
            raise ValueError("Cannot specify both 'spec' and 'sections'. Choose one.")
        
        if sections is not None:
            # Multi-section mode
            self.sections = sections
            self.spec = None  # No single spec
            # Validate total length matches
            total_section_length = sum(s.length_m for s in sections)
            if abs(total_section_length - distance) > 1e-6:
                raise ValueError(
                    f"Sum of section lengths ({total_section_length:.3f} m) "
                    f"must equal total distance ({distance:.3f} m)"
                )
        else:
            # Single uniform fiber mode
            self.spec = spec or FiberSpec()
            self.sections = [FiberSection(distance, self.spec)]
        
        self.const = const

        self.J_total: Optional[np.ndarray] = None
        self.DCD_ps_per_nm_km: float = 0.0
        self.tau_dgd_s: float = 0.0
        self.base_group_delay_s: float = 0.0

        # Raman scattering noise parameters
        self.raman_noise_rate_Hz: float = 0.0
        self.noise_enabled: bool = False
        self.raman_const: RamanScatteringConstants = DEFAULT_RAMAN


        self.cd_delays_ps = []  # Track all CD delays
        self.wavelengths_nm = []  # Track all wavelengths
        self.track_delays = False  # Enable/disable tracking

    def init(self) -> None:
        super().init()
        self._compute_link_model()

        if self._is_classical_coexist_enabled():
            self.raman_noise_rate_FS_Hz, self.raman_noise_rate_BS_Hz = self._compute_raman_noise_rate()

            self.raman_noise_rate_Hz = (
                self.raman_noise_rate_FS_Hz + self.raman_noise_rate_BS_Hz
            )
            self.noise_enabled = True
            self._schedule_next_noise_photon()

    def enable_delay_tracking(self):
        """Enable tracking of CD delays for diagnostics."""
        self.track_delays = True
        self.cd_delays_ps = []
        self.wavelengths_nm = []
    
    def disable_delay_tracking(self):
        """Disable tracking to save memory."""
        self.track_delays = False
    
    def get_cd_delay_statistics(self) -> dict:
        """Get statistics about tracked CD delays."""
        if not self.cd_delays_ps:
            return {"count": 0}
        
        delays = np.array(self.cd_delays_ps)
        wavelengths = np.array(self.wavelengths_nm)
        
        return {
            "count": len(delays),
            "delays_ps": delays,
            "wavelengths_nm": wavelengths,
            "mean_delay_ps": np.mean(delays),
            "std_delay_ps": np.std(delays),
            "min_delay_ps": np.min(delays),
            "max_delay_ps": np.max(delays),
        }
    
    def clear_delay_tracking(self):
        """Clear tracked delays."""
        self.cd_delays_ps = []
        self.wavelengths_nm = []

    def set_ends(self, sender: "Node", receiver_name: str) -> None:
        """
        Override to start noise photon generation after sender is connected.
        """
        super().set_ends(sender, receiver_name)
        
        # NOW we can schedule noise photons (sender is connected)
        if self.noise_enabled and self.raman_noise_rate_Hz > 0:
            self._schedule_next_noise_photon()
            
    def _compute_link_model(self) -> None:
        """Compute Jones matrix, DGD, and chromatic dispersion """

        # Build composite Jones matrix by multiplying section matrices
        J_composite = np.eye(2, dtype=complex)
    
        for section in self.sections:
            s = section.spec
            L = section.length_m
            
            # Calculate refractive indices
            nco = GlassMixture(s.core_m_germania).n(s.wavelength_m, s.temperature_C)
            ncl = GlassMixture(s.clad_m_germania).n(s.wavelength_m, s.temperature_C)
            
            # Birefringence for this section
            d_beta_lin = (
                delta_beta_core_ellipticity(s, nco, ncl)
                + delta_beta_asym_thermal(s, nco, ncl, self.const)
                + delta_beta_bending(s, nco, self.const)
            )
            d_beta_circ = delta_beta_twist(s, nco, self.const)
            
            # Build Jones matrix for this section
            J_section = self._build_link_jones(
                length_m=L,
                delta_beta_linear=d_beta_lin,
                delta_beta_circular=d_beta_circ,
                spec=s
            )
            
            J_composite = J_section @ J_composite
        
        self.J_total = J_composite
        
        self.DCD_ps_per_nm_km = self._compute_total_chromatic_dispersion()
        self.tau_dgd_s = self._compute_dgd_seconds()
        self.base_group_delay_s = self._compute_total_group_delay()
    
    def _compute_total_chromatic_dispersion(self) -> float:
        """
        Calculate TOTAL chromatic dispersion accounting for ALL sections.
        
        This is physically correct: total dispersion is the weighted average
        of each section's contribution.
        """
        total_delay_per_nm = 0.0  # ps/nm
        total_length_km = 0.0
        
        for section in self.sections:
            # Calculate DCD for THIS section
            DCD_section = self._compute_chromatic_dispersion_numerical(section.spec)
            
            # Length contribution in km
            L_km = section.length_m / 1000.0
            
            # Accumulate: DCD × L (units: ps/nm)
            total_delay_per_nm += DCD_section * L_km
            total_length_km += L_km
        
        # Effective DCD = total_delay / total_length
        # This gives the "average" DCD that characterizes the whole link
        if total_length_km > 0:
            return float(total_delay_per_nm / total_length_km)
        return 0.0
    
    def _compute_total_group_delay(self) -> float:
        """
        Calculate TOTAL group delay from ALL sections.
        
        Each section contributes: tau _g,i = n_g,i × L_i / c
        Total: tau _g = sum(tau _g,i)
        """
        total_delay_s = 0.0
        
        for section in self.sections:
            s = section.spec
            lam = s.wavelength_m
            dlam = s.d_lambda_m
            
            # Calculate n_eff at three wavelengths for this section
            def calc_neff(wavelength):
                nco = GlassMixture(s.core_m_germania).n(wavelength, s.temperature_C)
                ncl = GlassMixture(s.clad_m_germania).n(wavelength, s.temperature_C)
                beta = beta_fundamental_approx(s.core_radius_m, wavelength, nco, ncl)
                return beta / (2.0 * PI / wavelength)
            
            neff_minus = calc_neff(lam - dlam)
            neff_center = calc_neff(lam)
            neff_plus = calc_neff(lam + dlam)
            
            # Group index for this section
            dn_dlam = (neff_plus - neff_minus) / (2.0 * dlam)
            n_group = neff_center - lam * dn_dlam
            
            # Thermal expansion of this section's length
            alpha_core = mix_linear(s.core_m_germania, 
                                    self.const.alpha_silica, 
                                    self.const.alpha_germania)
            L_thermal = section.length_m * (1.0 + alpha_core * (s.temperature_C - s.reference_temp_C))
            
            # Add this section's group delay
            total_delay_s += n_group * L_thermal / SPEED_OF_LIGHT_M_PER_S
        
        return float(total_delay_s)

    def _chromatic_delay_picoseconds(self, qubit: "Photon") -> float:
        """
        Chromatic dispersion delay accounting for ALL sections.
        This is already captured in self.DCD_ps_per_nm_km which is the
        effective weighted average.
        """
        if self.DCD_ps_per_nm_km == 0.0 or not hasattr(qubit, "wavelength"):
            return 0.0
        lambda_q_nm = float(qubit.wavelength)
        lambda_ref_nm = self.sections[0].spec.wavelength_m * 1e9
        dlam_nm = lambda_q_nm - lambda_ref_nm
        L_km = self.distance / 1000.0
        
        # This now correctly uses the effective DCD from all sections
        delay_ps = self.DCD_ps_per_nm_km * L_km * dlam_nm

        if self.track_delays:
            self.cd_delays_ps.append(delay_ps)
            self.wavelengths_nm.append(lambda_q_nm)
        
        return delay_ps
    
    def _compute_chromatic_dispersion_numerical(self, spec: FiberSpec) -> float:
        s = spec
        lam_center = s.wavelength_m
        dlam = s.d_lambda_m
        
        def calc_neff(wavelength):
            nco = GlassMixture(s.core_m_germania).n(wavelength, s.temperature_C)
            ncl = GlassMixture(s.clad_m_germania).n(wavelength, s.temperature_C)
            beta = beta_fundamental_approx(s.core_radius_m, wavelength, nco, ncl)
            return beta / (2.0 * PI / wavelength)
        
        neff_A = calc_neff(lam_center - dlam)
        neff_B = calc_neff(lam_center)
        neff_C = calc_neff(lam_center + dlam)
        
        d2neff_dlam2 = (neff_C - 2.0 * neff_B + neff_A) / (dlam ** 2)
        
        DCD = -(lam_center / SPEED_OF_LIGHT_M_PER_S) * d2neff_dlam2 * 1e12 * 1e-9 * 1e3

        
        return float(DCD)


    def _build_link_jones(self, length_m: float, delta_beta_linear: float, delta_beta_circular: float, spec: FiberSpec) -> np.ndarray:
        """
        Compose Jones matrix from birefringence elements .
        CRITICAL: Must use thermally-adjusted length .
        """
        s = spec
        
        # Calculate thermal expansion coefficient for core 
        alpha_core = mix_linear(
            s.core_m_germania, 
            self.const.alpha_silica, 
            self.const.alpha_germania
        )
        
        # Thermally adjusted length 
        # Lt = L0 * (1 + alpha * (T - Tref))
        # Assuming reference temp is 20 C
        length_thermal = length_m * (1.0 + alpha_core * (s.temperature_C - s.reference_temp_C))
        
        # Calculate average propagation constant beta at center wavelength
        nco = GlassMixture(s.core_m_germania).n(s.wavelength_m, s.temperature_C)
        ncl = GlassMixture(s.clad_m_germania).n(s.wavelength_m, s.temperature_C)
        beta_avg = beta_fundamental_approx(s.core_radius_m, s.wavelength_m, nco, ncl)

        # Build Jones matrices with THERMAL length 
        J_lin = jones_linear(delta_beta_linear, length_thermal, beta_avg)
        if delta_beta_circular != 0:
            J_circ = jones_circular(delta_beta_circular, length_thermal)
            return J_circ @ J_lin
        return J_lin


    def _compute_dgd_seconds(self) -> float:
        """
        DGD calculation for multi-section fiber.
        Computes derivative of total Jones matrix.
        """
        if self.J_total is None:
            return 0.0
        
        # Use first section's spec for wavelength stepping
        s = self.sections[0].spec
        lam = s.wavelength_m
        dlam = s.d_lambda_m
        
        d_omega_per_step = 2.0 * PI * SPEED_OF_LIGHT_M_PER_S / (lam * lam) * dlam
        
        J_minus = self._build_composite_jones_at_wavelength(lam - dlam)
        
        J_plus = self._build_composite_jones_at_wavelength(lam + dlam)
        
        try:
            J_base_inv = np.linalg.inv(self.J_total)
            
            M_back = np.matmul(J_minus, J_base_inv)
            evals_back = np.linalg.eigvals(M_back)
            if len(evals_back) < 2:
                return 0.0
            dgd_back = abs(np.angle(evals_back[0] / evals_back[1])) / d_omega_per_step
            
            M_forw = np.matmul(J_plus, J_base_inv)
            evals_forw = np.linalg.eigvals(M_forw)
            if len(evals_forw) < 2:
                return 0.0
            dgd_forw = abs(np.angle(evals_forw[0] / evals_forw[1])) / d_omega_per_step
            
            dgd = (dgd_back + dgd_forw) / 2.0
            return float(dgd)
            
        except np.linalg.LinAlgError:
            return 0.0

    def _build_composite_jones_at_wavelength(self, wavelength_m: float) -> np.ndarray:
        """Helper: Build composite Jones matrix at specified wavelength."""
        J_composite = np.eye(2, dtype=complex)
        
        for section in self.sections:
            s = section.spec
            L = section.length_m
            
            # Create temporary spec with different wavelength
            spec_temp = FiberSpec(
                core_radius_m=s.core_radius_m,
                cladding_radius_m=s.cladding_radius_m,
                core_m_germania=s.core_m_germania,
                clad_m_germania=s.clad_m_germania,
                wavelength_m=wavelength_m,
                temperature_C=s.temperature_C,
                reference_temp_C=s.reference_temp_C,
                core_ellipticity=s.core_ellipticity,
                bend_radius_m=s.bend_radius_m,
                axial_tension_N=s.axial_tension_N,
                twist_rate_rad_per_m=s.twist_rate_rad_per_m,
            )
            
            nco = GlassMixture(spec_temp.core_m_germania).n(wavelength_m, spec_temp.temperature_C)
            ncl = GlassMixture(spec_temp.clad_m_germania).n(wavelength_m, spec_temp.temperature_C)
            
            d_beta_lin = (
                delta_beta_core_ellipticity(spec_temp, nco, ncl)
                + delta_beta_asym_thermal(spec_temp, nco, ncl, self.const)
                + delta_beta_bending(spec_temp, nco, self.const)
            )
            d_beta_circ = delta_beta_twist(spec_temp, nco, self.const)
            
            J_section = self._build_link_jones(L, d_beta_lin, d_beta_circ, spec_temp)
            J_composite = J_section @ J_composite
        
        return J_composite

    def transmit(self, qubit: "Photon", source: "Node") -> None:
        """
        Apply Jones transform and schedule arrival with PMD + CD delays .
        """
        if (
            qubit.encoding_type["name"] == "polarization"
            and not qubit.is_null
            and self.sender.get_generator().random() > self.loss
            and self.J_total is not None
        ):
            self._apply_jones(qubit, self.J_total)

            extra_delay = 0.0
            pmd_delay = self._sample_pmd_delay_picoseconds(qubit)
            cd_delay = self._chromatic_delay_picoseconds(qubit)

            extra_delay += pmd_delay
            extra_delay += cd_delay
            
            """if hasattr(qubit, 'wavelength') and self.timeline.now() < 1e8:  # Only first ~100 photons
                print(f"[{self.name}] Photon: λ={qubit.wavelength:.3f} nm, "
                    f"PMD={pmd_delay*1e12:.1f} ps, "
                    f"CD={cd_delay*1e12:.1f} ps, "
                    f"Total extra={extra_s*1e12:.1f} ps")"""

            if not extra_delay:
                print("extra delay is : ", extra_delay)
            
            base_ps = int(round(self.base_group_delay_s * 1e12))
            future_time = self.timeline.now() + base_ps + extra_delay
            future_time = max(self.timeline.now(), future_time)

            process = Process(self.receiver, "receive_qubit", [source.name, qubit])
            self.timeline.schedule(Event(future_time, process))
            
            return

    def _apply_jones(self, qubit: "Photon", J: np.ndarray) -> None:
        """Apply Jones matrix to qubit state """
        state = np.array(qubit.quantum_state.state, dtype=complex)

        if state.size == 2:
            qubit.set_state(tuple(J @ state))
            return

        if state.size == 4:
            if qubit.name == "signal":
                op = np.kron(J, np.eye(2))
            elif qubit.name == "idler":
                op = np.kron(np.eye(2), J)
            else:
                return
            qubit.set_state(tuple(op @ state))
    

    def _sample_pmd_delay_picoseconds(self, qubit: "Photon") -> float:
        """Sample PMD delay: ±tau _DGD/2 based on H-polarization probability """
        if self.tau_dgd_s <= 0.0:
            return 0.0

        state = np.array(qubit.quantum_state.state, dtype=complex)
        if state.size == 2:
            pH = float(abs(state[0])**2)
        elif state.size == 4:
            if qubit.name == "0":
                pH = float(abs(state[0])**2 + abs(state[1])**2)
            elif qubit.name == "1":
                pH = float(abs(state[0])**2 + abs(state[2])**2)
            else:
                return 0.0
        else:
            return 0.0

        r = self.sender.get_generator().random()
        return (-0.5 if r < pH else 0.5) * self.tau_dgd_s * 1e12

    # =========================================================================
    # CLASSICAL COEXISTENCE NOISE METHODS
    # =========================================================================

    def _is_classical_coexist_enabled(self) -> bool:
        """Check if any section has classical coexistence enabled."""
        return any(s.spec.classical_coexist_enabled for s in self.sections)

    def _compute_raman_noise_rate(self) -> tuple[float, float]:
        '''
        Calculate total Raman scattering noise rate from all sections.

        Based on Burenkov et al., Opt. Express 31, 11431 (2023):
        - Equation 2: Backward scattering (BS)
        - Equation 3: Forward scattering (FS)

        Returns:
            -> tuple[float, float]:
        '''
        total_FS = 0.0
        total_BS = 0.0

        for section in self.sections:
            s = section.spec

            if not s.classical_coexist_enabled:
                continue

            # Section length
            L = section.length_m

            # Convert classical power to photons/s
            h = PLANCK_CONSTANT
            c = SPEED_OF_LIGHT_M_PER_S
            lambda_s = s.classical_wavelength_nm * 1e-9

            E_photon = h * c / lambda_s
            P_watts = s.classical_power_mW * 1e-3
            P_in = P_watts / E_photon

            # Get attenuation constants using GLOBAL function (not self.method!)
            alpha_s = get_fiber_attenuation_per_m(s.classical_wavelength_nm)
            alpha_n = get_fiber_attenuation_per_m(s.quantum_wavelength_nm)

            # Get scattering constants
            try:
                beta_FS, beta_BS = self.raman_const.get_beta(
                    s.classical_wavelength_nm,
                    s.quantum_wavelength_nm
                )
            except ValueError as e:
                print(f"Warning: {e}. Skipping noise for section {section}")
                continue

            # Channel bandwidth
            Delta_nu = s.quantum_bandwidth_Hz

            # Equation 3: Forward Scattering
            if abs(alpha_s - alpha_n) > 1e-12:
                numerator_FS = np.exp(-alpha_n * L) - np.exp(-alpha_s * L)
                P_FS = (numerator_FS / (alpha_s - alpha_n)) * beta_FS * Delta_nu * P_in
            else:
                P_FS = L * np.exp(-alpha_n * L) * beta_FS * Delta_nu * P_in

            # Equation 2: Backward Scattering
            numerator_BS = 1.0 - np.exp(-(alpha_s + alpha_n) * L)
            P_BS = (numerator_BS / (alpha_s + alpha_n)) * beta_BS * Delta_nu * P_in

            total_FS += P_FS
            total_BS += P_BS

        return float(total_FS), float(total_BS)


    def _schedule_next_noise_photon(self) -> None:
        """
        Schedule the next Raman noise photon arrival.

        Uses Poisson process: inter-arrival times follow exponential distribution.
        Creates an actual noise photon and transmits it through the channel.
        """
        if (
            not self.noise_enabled
            or self.raman_noise_rate_Hz <= 0
            or self.sender is None 
        ):
            return
        # Sample inter-arrival time from exponential distribution
        rate = self.raman_noise_rate_Hz  # [photons/s]
        u = self.sender.get_generator().random()
        wait_time_s = -np.log(max(u, 1e-10)) / rate
        wait_time_ps = int(round(wait_time_s * 1e12))

        # Schedule creation and transmission of noise photon
        generation_time = self.timeline.now() + wait_time_ps

        # Create process to generate and send noise photon
        process = Process(self, "_generate_and_transmit_noise_photon", [])
        event = Event(generation_time, process)
        self.timeline.schedule(event)

    def _generate_and_transmit_noise_photon(self) -> None:
        """
        Generate and record a Raman noise photon detection.
        
        Instead of creating and transmitting a full photon object, this method
        directly triggers a detection event at the receiver. This is computationally
        efficient and physically accurate for unpolarized Raman noise.
        
        Physical model:
        - Raman scattering produces unpolarized light
        - Unpolarized photons have 50/50 probability at PBS
        - Photons still experience fiber loss (accounted for in noise rate calculation)
        """
        
        # Schedule noise detection at receiver
        arrival_time = self.timeline.now()
        
        # Call receiver's noise handling method
        process = Process(self.receiver, "receive_noise_photon", [])
        self.timeline.schedule(Event(arrival_time, process))
        
        # Schedule next noise photon (maintain Poisson process)
        self._schedule_next_noise_photon()
        

    def get_noise_info(self) -> dict:
        """
        Get diagnostic information about Raman noise configuration.

        Returns:
            Dictionary with noise parameters and expected rates
        """
        if not self.noise_enabled:
            return {"enabled": False}

        # Collect info from first enabled section (for display)
        enabled_sections = [s for s in self.sections if s.spec.classical_coexist_enabled]

        if not enabled_sections:
            return {"enabled": False}

        s = enabled_sections[0].spec

        return {
            "enabled": True,
            "total_noise_rate_Hz": self.raman_noise_rate_Hz,
            "total_noise_rate_per_sec": self.raman_noise_rate_Hz,
            "classical_wavelength_nm": s.classical_wavelength_nm,
            "classical_power_mW": s.classical_power_mW,
            "quantum_wavelength_nm": s.quantum_wavelength_nm,
            "quantum_bandwidth_GHz": s.quantum_bandwidth_Hz / 1e9,
            "fiber_length_km": self.distance / 1000.0,
            "expected_accidentals_per_ms": self.raman_noise_rate_Hz / 1000.0,
        }

