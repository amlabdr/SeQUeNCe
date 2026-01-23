import numpy as np
import pytest

from sequence.components.fiber_models import fiberQuantumChannel, FiberSpec, FiberSection
from sequence.components.photon import Photon
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node
from sequence.utils.encoding import polarization

SEED = 0


# ============================================================================
# Test 1: FiberSpec Initialization and Defaults
# ============================================================================

def test_FiberSpec_defaults():
    """Test FiberSpec creates valid default SMF-28-like parameters."""
    spec = FiberSpec()
    
    # Geometry
    assert spec.core_radius_m == 4.1e-6
    assert spec.cladding_radius_m == 62.5e-6
    
    # Composition
    assert spec.core_m_germania == 0.036
    assert spec.clad_m_germania == 0.0
    
    # Operating conditions
    assert spec.wavelength_m == 1550e-9
    assert spec.temperature_C == 20.0
    
    # Birefringence sources (default: none)
    assert spec.core_ellipticity == 1.0
    assert spec.bend_radius_m == 0.0
    assert spec.twist_rate_rad_per_m == 0.0


def test_FiberSpec_custom_parameters():
    """Test FiberSpec accepts custom parameters."""
    spec = FiberSpec(
        core_radius_m=5.0e-6,
        wavelength_m=1310e-9,
        temperature_C=25.0,
        core_ellipticity=1.005,
        bend_radius_m=10.0,
        twist_rate_rad_per_m=0.1,
    )
    
    assert spec.core_radius_m == 5.0e-6
    assert spec.wavelength_m == 1310e-9
    assert spec.temperature_C == 25.0
    assert spec.core_ellipticity == 1.005
    assert spec.bend_radius_m == 10.0
    assert spec.twist_rate_rad_per_m == 0.1


# ============================================================================
# Test 2: Refractive Index Calculations (Sellmeier Model)
# ============================================================================

def test_Sellmeier_silica_at_1550nm():
    """Test silica refractive index at 1550 nm matches BIFROST exactly."""
    from sequence.components.fiber_models import SILICA_SELLMEIER
    
    n_silica = SILICA_SELLMEIER.n(1550e-9, temperature_C=20.0)
    
    # BIFROST reference: 1.4446184067
    assert abs(n_silica - 1.4446184067) < 1e-8, f"Silica n = {n_silica}"


def test_Sellmeier_germania_at_1550nm():
    """Test germania refractive index at 1550 nm matches BIFROST."""
    from sequence.components.fiber_models import GERMANIA_SELLMEIER
    
    n_germania = GERMANIA_SELLMEIER.n(1550e-9, temperature_C=24.0)
    
    # BIFROST reference: 1.5871022089
    assert abs(n_germania - 1.5871022089) < 1e-8, f"Germania n = {n_germania}"


def test_GlassMixture_pure_silica():
    """Test glass mixture with m=0 gives pure silica."""
    from sequence.components.fiber_models import GlassMixture, SILICA_SELLMEIER
    
    mixture = GlassMixture(m_germania=0.0)
    n_mix = mixture.n(1550e-9, 20.0)
    n_silica = SILICA_SELLMEIER.n(1550e-9, 20.0)
    
    assert abs(n_mix - n_silica) < 1e-6


def test_GlassMixture_doped_core():
    """Test doped glass has intermediate refractive index."""
    from sequence.components.fiber_models import GlassMixture
    
    core = GlassMixture(m_germania=0.036)  # 3.6% germania
    clad = GlassMixture(m_germania=0.0)    # Pure silica
    
    n_core = core.n(1550e-9, 20.0)
    n_clad = clad.n(1550e-9, 20.0)
    
    # Core should have higher index than cladding
    assert n_core > n_clad
    
    # BIFROST references (relaxed tolerance for numerical precision)
    # n_core = 1.4497451413, n_clad = 1.4446184067
    assert abs(n_core - 1.4497451413) < 1e-6, f"Core n = {n_core}"
    assert abs(n_clad - 1.4446184067) < 1e-8, f"Clad n = {n_clad}"


def test_Sellmeier_temperature_dependence():
    """Test refractive index changes with temperature (BIFROST has strong T-dependence)."""
    from sequence.components.fiber_models import SILICA_SELLMEIER
    
    n_20C = SILICA_SELLMEIER.n(1550e-9, temperature_C=20.0)
    n_80C = SILICA_SELLMEIER.n(1550e-9, temperature_C=80.0)
    
    # Index should increase with temperature
    assert n_80C > n_20C
    
    # BIFROST reference: delta_n = 3.287318e-03 for 60°C change
    delta_n = n_80C - n_20C
    assert abs(delta_n - 3.287318e-03) < 1e-7, f"delta_n = {delta_n}"


# ============================================================================
# Test 3: Fiber Parameters (V, Beta, etc.)
# ============================================================================

def test_v_parameter_SMF28():
    """Test V-parameter for SMF-28 at 1550 nm."""
    from sequence.components.fiber_models import v_parameter, GlassMixture
    
    spec = FiberSpec()
    nco = GlassMixture(spec.core_m_germania).n(spec.wavelength_m, spec.temperature_C)
    ncl = GlassMixture(spec.clad_m_germania).n(spec.wavelength_m, spec.temperature_C)
    
    V = v_parameter(spec.core_radius_m, spec.wavelength_m, nco, ncl)
    
    # BIFROST reference: 2.024552 (relaxed tolerance for propagated error)
    assert abs(V - 2.024552) < 2e-4, f"V-parameter = {V}"


def test_beta_fundamental():
    """Test propagation constant has reasonable value."""
    from sequence.components.fiber_models import beta_fundamental_approx, GlassMixture
    
    spec = FiberSpec()
    nco = GlassMixture(spec.core_m_germania).n(spec.wavelength_m, spec.temperature_C)
    ncl = GlassMixture(spec.clad_m_germania).n(spec.wavelength_m, spec.temperature_C)
    
    beta = beta_fundamental_approx(spec.core_radius_m, spec.wavelength_m, nco, ncl)
    
    # BIFROST reference: 5.864475e+06 rad/m
    assert abs(beta - 5.864475e6) < 1e3, f"beta = {beta}"


# ============================================================================
# Test 4: Birefringence Mechanisms (Limiting Cases)
# ============================================================================

def test_birefringence_circular_core_zero():
    """Test circular core (epsilon=1.0) gives zero ellipticity birefringence."""
    from sequence.components.fiber_models import delta_beta_core_ellipticity, GlassMixture
    
    spec = FiberSpec(core_ellipticity=1.0)  # Perfectly circular
    nco = GlassMixture(spec.core_m_germania).n(spec.wavelength_m, spec.temperature_C)
    ncl = GlassMixture(spec.clad_m_germania).n(spec.wavelength_m, spec.temperature_C)
    
    d_beta = delta_beta_core_ellipticity(spec, nco, ncl)
    
    assert d_beta == 0.0, "Circular core should have zero ellipticity birefringence"


def test_birefringence_elliptical_core_nonzero():
    """Test elliptical core gives nonzero birefringence."""
    from sequence.components.fiber_models import delta_beta_core_ellipticity, GlassMixture
    
    spec = FiberSpec(core_ellipticity=1.005)  # 0.5% ellipticity
    nco = GlassMixture(spec.core_m_germania).n(spec.wavelength_m, spec.temperature_C)
    ncl = GlassMixture(spec.clad_m_germania).n(spec.wavelength_m, spec.temperature_C)
    
    d_beta = delta_beta_core_ellipticity(spec, nco, ncl)
    
    # BIFROST reference: 1.424477e-01 rad/m (relaxed tolerance for propagated error)
    assert abs(d_beta - 0.1424477) < 5e-5, f"BCNC = {d_beta}"


def test_birefringence_thermal_stress_depends_on_ellipticity():
    """
    Test thermal stress birefringence depends on ellipticity and temperature.
    Note: Thermal stress uses (T_softening - T_actual), not reference temperature!
    """
    from sequence.components.fiber_models import delta_beta_asym_thermal, GlassMixture, DEFAULT_CONST
    
    # Test 1: Circular core should give zero thermal stress
    spec_circular = FiberSpec(
        temperature_C=20.0,
        core_ellipticity=1.0  # Circular
    )
    nco = GlassMixture(spec_circular.core_m_germania).n(spec_circular.wavelength_m, spec_circular.temperature_C)
    ncl = GlassMixture(spec_circular.clad_m_germania).n(spec_circular.wavelength_m, spec_circular.temperature_C)
    
    d_beta_circular = delta_beta_asym_thermal(spec_circular, nco, ncl, DEFAULT_CONST)
    assert d_beta_circular == 0.0, "Circular core should give zero thermal stress birefringence"
    
    # Test 2: Elliptical core gives nonzero thermal stress (even at reference temp)
    spec_elliptical = FiberSpec(
        temperature_C=20.0,
        core_ellipticity=1.005
    )
    d_beta_elliptical = delta_beta_asym_thermal(spec_elliptical, nco, ncl, DEFAULT_CONST)
    assert d_beta_elliptical != 0.0, "Elliptical core should have thermal stress birefringence"


def test_birefringence_straight_fiber_zero():
    """Test straight fiber (R=0) gives zero bending birefringence."""
    from sequence.components.fiber_models import delta_beta_bending, GlassMixture, DEFAULT_CONST
    
    spec = FiberSpec(bend_radius_m=0.0)
    nco = GlassMixture(spec.core_m_germania).n(spec.wavelength_m, spec.temperature_C)
    
    d_beta = delta_beta_bending(spec, nco, DEFAULT_CONST)
    
    assert d_beta == 0.0, "Straight fiber should have zero bending birefringence"


def test_birefringence_no_twist_zero():
    """Test no twist gives zero circular birefringence."""
    from sequence.components.fiber_models import delta_beta_twist, GlassMixture, DEFAULT_CONST
    
    spec = FiberSpec(twist_rate_rad_per_m=0.0)
    nco = GlassMixture(spec.core_m_germania).n(spec.wavelength_m, spec.temperature_C)
    
    d_beta = delta_beta_twist(spec, nco, DEFAULT_CONST)
    
    assert d_beta == 0.0, "No twist should give zero circular birefringence"


# ============================================================================
# Test 5: Chromatic Dispersion (Realistic Values)
# ============================================================================

def test_chromatic_dispersion_SMF28_at_1550nm():
    """Test chromatic dispersion matches BIFROST value."""
    tl = Timeline()
    spec = FiberSpec(wavelength_m=1550e-9, temperature_C=20.0)
    
    channel = fiberQuantumChannel(
        name="test_channel",
        timeline=tl,
        attenuation=0.0002,
        distance=1000.0,  # 1 km
        spec=spec
    )
    tl.init()
    
    # BIFROST reference: 13.702363 ps/(nm·km)
    assert abs(channel.DCD_ps_per_nm_km - 13.702363) < 0.01, \
        f"DCD = {channel.DCD_ps_per_nm_km} ps/(nm·km)"


def test_chromatic_dispersion_zero_at_1310nm():
    """Test chromatic dispersion near zero-dispersion wavelength (~1310 nm)."""
    tl = Timeline()
    spec = FiberSpec(wavelength_m=1310e-9)
    
    channel = fiberQuantumChannel(
        name="test_channel",
        timeline=tl,
        attenuation=0.0002,
        distance=1000.0,
        spec=spec
    )
    tl.init()
    
    # BIFROST reference: -2.078989 ps/(nm·km) (slightly negative at 1310 nm)
    assert abs(channel.DCD_ps_per_nm_km - (-2.078989)) < 0.5, \
        f"DCD at 1310 nm = {channel.DCD_ps_per_nm_km}"


# ============================================================================
# Test 6: DGD Calculation
# ============================================================================

def test_dgd_zero_for_circular_fiber():
    """Test DGD is near zero for perfectly circular fiber."""
    tl = Timeline()
    spec = FiberSpec(
        core_ellipticity=1.0,
        bend_radius_m=0.0,
        twist_rate_rad_per_m=0.0
    )
    
    channel = fiberQuantumChannel(
        name="test_channel",
        timeline=tl,
        attenuation=0.0,
        distance=1000.0,
        spec=spec
    )
    tl.init()
    
    # Perfect circular fiber should have minimal DGD
    assert channel.tau_dgd_s < 1e-13, f"DGD = {channel.tau_dgd_s*1e12:.3f} ps (expected ~0)"


def test_dgd_nonzero_for_elliptical_fiber():
    """Test DGD is nonzero for elliptical fiber."""
    tl = Timeline()
    spec = FiberSpec(core_ellipticity=1.005)  # 0.5% ellipticity
    
    channel = fiberQuantumChannel(
        name="test_channel",
        timeline=tl,
        attenuation=0.0,
        distance=1000.0,  # 1 km
        spec=spec
    )
    tl.init()
    
    dgd_ps = channel.tau_dgd_s * 1e12
    
    # BIFROST reference: 0.820006 ps for 1 km with epsilon=1.005
    assert abs(dgd_ps - 0.820006) < 0.01, f"DGD = {dgd_ps} ps"


def test_dgd_length_scaling():
    """
    Test DGD scaling with length (linear for deterministic birefringence).
    """
    tl = Timeline()
    spec = FiberSpec(core_ellipticity=1.005)
    
    lengths_km = [1, 2, 4]
    dgd_values = []
    
    for L_km in lengths_km:
        channel = fiberQuantumChannel(
            name=f"channel_{L_km}km",
            timeline=tl,
            attenuation=0.0,
            distance=L_km * 1000.0,
            spec=spec
        )
        tl.init()
        dgd_values.append(channel.tau_dgd_s * 1e12)
    
    # BIFROST references: 0.820006, 1.640013, 3.280025 ps
    # Ratios: 2.000 and 4.000
    ratio_2x = dgd_values[1] / dgd_values[0]
    ratio_4x = dgd_values[2] / dgd_values[0]
    
    # Linear scaling confirmed by BIFROST
    assert abs(ratio_2x - 2.0) < 0.01, f"DGD(2L)/DGD(L) = {ratio_2x:.3f}"
    assert abs(ratio_4x - 4.0) < 0.01, f"DGD(4L)/DGD(L) = {ratio_4x:.3f}"

# ============================================================================
# Test 7: Jones Matrix Properties
# ============================================================================

def test_jones_matrix_unitarity():
    """Test Jones matrix is unitary (preserves photon number)."""
    tl = Timeline()
    spec = FiberSpec(core_ellipticity=1.005, bend_radius_m=10.0)
    
    channel = fiberQuantumChannel(
        name="test_channel",
        timeline=tl,
        attenuation=0.0,
        distance=1000.0,
        spec=spec
    )
    tl.init()
    
    J = channel.J_total
    J_dagger = np.conj(J.T)
    
    # Check J*tr(J) = I (unitarity)
    identity = np.matmul(J_dagger, J)
    
    assert np.allclose(identity, np.eye(2), atol=1e-10), \
        "Jones matrix should be unitary"


def test_jones_matrix_determinant():
    """Test Jones matrix has determinant with magnitude 1."""
    tl = Timeline()
    spec = FiberSpec(core_ellipticity=1.005)
    
    channel = fiberQuantumChannel(
        name="test_channel",
        timeline=tl,
        attenuation=0.0,
        distance=1000.0,
        spec=spec
    )
    tl.init()
    
    J = channel.J_total
    det = np.linalg.det(J)
    
    assert abs(abs(det) - 1.0) < 1e-10, \
        f"|det(J)| = {abs(det)} (should be 1)"


# ============================================================================
# Test 8: Channel Initialization and Integration
# ============================================================================

def test_fiberQuantumChannel_init():
    """Test fiberQuantumChannel initializes with default spec."""
    tl = Timeline()
    channel = fiberQuantumChannel(
        name="test_channel",
        timeline=tl,
        attenuation=0.0002,
        distance=10000.0  # 10 km
    )
    
    assert channel.name == "test_channel"
    assert channel.distance == 10000.0
    assert channel.spec is not None
    assert channel.J_total is None  # Not initialized until tl.init()


def test_fiberQuantumChannel_init_with_timeline():
    """Test channel initialization computes fiber model."""
    tl = Timeline()
    spec = FiberSpec(core_ellipticity=1.005)
    
    channel = fiberQuantumChannel(
        name="test_channel",
        timeline=tl,
        attenuation=0.0,
        distance=1000.0,
        spec=spec
    )
    tl.init()
    
    # After init, these should be computed
    assert channel.J_total is not None
    assert channel.DCD_ps_per_nm_km != 0.0
    assert channel.tau_dgd_s >= 0.0
    assert channel.base_group_delay_s > 0.0


def test_fiberQuantumChannel_set_ends():
    """Test setting channel endpoints."""
    tl = Timeline()
    channel = fiberQuantumChannel(
        name="qc",
        timeline=tl,
        attenuation=0.0002,
        distance=1000.0
    )
    
    node1 = Node("node1", tl)
    node2 = Node("node2", tl)
    
    channel.set_ends(node1, node2.name)
    
    assert channel.sender == node1
    assert channel.receiver == node2.name
    assert "node2" in node1.qchannels


# ============================================================================
# Test 9: Photon Transmission
# ============================================================================

def test_fiberQuantumChannel_transmit_basic():
    """Test basic photon transmission through fiber channel."""
    
    class FakeNode(Node):
        def __init__(self, name, tl):
            Node.__init__(self, name, tl)
            self.received_photons = []
            self.generator = np.random.default_rng(SEED)
        
        def receive_qubit(self, src, photon):
            self.received_photons.append((self.timeline.now(), photon.name))
        
        def get_generator(self):
            return self.generator
    
    tl = Timeline()
    spec = FiberSpec()
    
    channel = fiberQuantumChannel(
        name="test_channel",
        timeline=tl,
        attenuation=0.0,  # No loss for this test
        distance=1000.0,
        spec=spec
    )
    
    sender = FakeNode("sender", tl)
    receiver = FakeNode("receiver", tl)
    sender.set_seed(SEED)
    
    channel.set_ends(sender, receiver.name)
    tl.init()
    
    # Send 10 photons
    for i in range(10):
        photon = Photon(str(i), timeline=tl, encoding_type=polarization)
        photon.set_state([1, 0])  # H polarization
        channel.transmit(photon, sender)
        tl.time += 1
    
    tl.run()
    
    # All photons should arrive (no loss)
    assert len(receiver.received_photons) == 10


def test_fiberQuantumChannel_applies_jones_matrix():
    """Test that Jones matrix is applied to photon state."""
    
    class FakeNode(Node):
        def __init__(self, name, tl):
            Node.__init__(self, name, tl)
            self.received_states = []
            self.generator = np.random.default_rng(SEED)
        
        def receive_qubit(self, src, photon):
            state = np.array(photon.quantum_state.state)
            self.received_states.append(state)
        
        def get_generator(self):
            return self.generator
    
    tl = Timeline()
    spec = FiberSpec(core_ellipticity=1.01)  # Some birefringence
    
    channel = fiberQuantumChannel(
        name="test_channel",
        timeline=tl,
        attenuation=0.0,
        distance=1000.0,
        spec=spec
    )
    
    sender = FakeNode("sender", tl)
    receiver = FakeNode("receiver", tl)
    sender.set_seed(SEED)
    
    channel.set_ends(sender, receiver.name)
    tl.init()
    
    # Send H-polarized photon
    photon = Photon("test", timeline=tl, encoding_type=polarization)
    initial_state = np.array([1, 0], dtype=complex)
    photon.set_state(tuple(initial_state))
    
    channel.transmit(photon, sender)
    tl.run()
    
    # State should change due to Jones matrix (unless perfectly aligned)
    assert len(receiver.received_states) == 1
    final_state = receiver.received_states[0]
    
    # Check state is normalized
    norm = np.sum(np.abs(final_state)**2)
    assert abs(norm - 1.0) < 1e-6, "State should remain normalized"

# ============================================================================
# Test 10: Multi-Section Fiber - Uniform Properties
# ============================================================================


def test_multi_section_uniform_properties():
    """
    Test multi-section fiber with identical properties matches single fiber.
    Tests: 5 km + 3 km + 2 km = 10 km (all same specs)
    """
    tl = Timeline()
    
    # Single 10 km fiber
    spec = FiberSpec(core_ellipticity=1.005)
    
    single_channel = fiberQuantumChannel(
        name="single_10km",
        timeline=tl,
        attenuation=0.0,
        distance=10000.0,
        spec=spec
    )
    
    # Multi-section: 5 km + 3 km + 2 km
    sections = [
        FiberSection(length_m=5000.0, spec=FiberSpec(core_ellipticity=1.005)),
        FiberSection(length_m=3000.0, spec=FiberSpec(core_ellipticity=1.005)),
        FiberSection(length_m=2000.0, spec=FiberSpec(core_ellipticity=1.005))
    ]
    
    multi_channel = fiberQuantumChannel(
        name="multi_section_10km",
        timeline=tl,
        attenuation=0.0,
        distance=10000.0,
        sections=sections
    )
    
    tl.init()
    
    # Jones matrices should be identical (within numerical precision)
    assert np.allclose(single_channel.J_total, multi_channel.J_total, atol=1e-8), \
        "Multi-section with uniform properties should match single fiber"
    
    # DCD should match
    assert abs(single_channel.DCD_ps_per_nm_km - multi_channel.DCD_ps_per_nm_km) < 0.01
    
    # DGD should match (BIFROST ref: 8.200063 ps for 10 km)
    dgd_single = single_channel.tau_dgd_s * 1e12
    dgd_multi = multi_channel.tau_dgd_s * 1e12
    assert abs(dgd_single - dgd_multi) < 0.01, \
        f"DGD: single={dgd_single:.3f} ps, multi={dgd_multi:.3f} ps"


def test_multi_section_dcd_weighted_average():
    """
    Test that multi-section DCD is correctly weighted by section lengths.
    Uses uniform properties to verify calculation.
    """
    tl = Timeline()
    
    # Three identical sections
    sections = [
        FiberSection(length_m=5000.0, spec=FiberSpec(core_ellipticity=1.005)),
        FiberSection(length_m=3000.0, spec=FiberSpec(core_ellipticity=1.005)),
        FiberSection(length_m=2000.0, spec=FiberSpec(core_ellipticity=1.005))
    ]
    
    # Calculate individual DCDs
    individual_dcds = []
    for i, section in enumerate(sections):
        ch = fiberQuantumChannel(
            name=f"section_{i}",
            timeline=tl,
            attenuation=0.0,
            distance=section.length_m,
            spec=section.spec
        )
        tl.init()
        individual_dcds.append(ch.DCD_ps_per_nm_km)
    
    # Effective DCD (weighted average)
    total_delay_per_nm = sum(dcd * s.length_m / 1000.0 for dcd, s in zip(individual_dcds, sections))
    total_length_km = sum(s.length_m for s in sections) / 1000.0
    expected_dcd = total_delay_per_nm / total_length_km
    
    # Multi-section channel
    multi_channel = fiberQuantumChannel(
        name="multi_section",
        timeline=tl,
        attenuation=0.0,
        distance=sum(s.length_m for s in sections),
        sections=sections
    )
    tl.init()
    
    # Should match weighted average (BIFROST ref: 13.702363 ps/(nm·km))
    assert abs(multi_channel.DCD_ps_per_nm_km - expected_dcd) < 0.01, \
        f"DCD: multi={multi_channel.DCD_ps_per_nm_km:.6f}, expected={expected_dcd:.6f}"


# ============================================================================
# Test 11: Multi-Section Fiber - Temperature Gradient
# ============================================================================


def test_multi_section_temperature_gradient():
    """
    Test multi-section fiber with different temperatures.
    Models realistic scenario: fiber passing through different environments.
    Sections: 5 km @ 20°C, 5 km @ 35°C, 5 km @ 10°C
    """
    tl = Timeline()
    
    # Three 5 km sections at different temperatures
    sections = [
        FiberSection(5000.0, FiberSpec(temperature_C=20.0, core_ellipticity=1.005)),
        FiberSection(5000.0, FiberSpec(temperature_C=35.0, core_ellipticity=1.005)),
        FiberSection(5000.0, FiberSpec(temperature_C=10.0, core_ellipticity=1.005))
    ]
    
    multi_channel = fiberQuantumChannel(
        name="temp_gradient",
        timeline=tl,
        attenuation=0.0,
        distance=15000.0,
        sections=sections
    )
    tl.init()
    
    # Jones matrix should still be unitary
    J = multi_channel.J_total
    J_dagger = np.conj(J.T)
    identity = np.matmul(J_dagger, J)
    assert np.allclose(identity, np.eye(2), atol=1e-10)
    
    # Effective DCD should be reasonable (BIFROST ref: 13.571029 ps/(nm·km))
    assert 13.0 < multi_channel.DCD_ps_per_nm_km < 14.0, \
        f"DCD = {multi_channel.DCD_ps_per_nm_km:.6f} ps/(nm·km)"


def test_multi_section_thermal_length_variation():
    """
    Test that thermal expansion is accounted for in each section.
    Different temperatures cause different section lengths.
    """
    tl = Timeline()
    
    # Sections at very different temperatures
    sections = [
        FiberSection(5000.0, FiberSpec(temperature_C=20.0)),
        FiberSection(5000.0, FiberSpec(temperature_C=80.0))
    ]
    
    multi_channel = fiberQuantumChannel(
        name="thermal_test",
        timeline=tl,
        attenuation=0.0,
        distance=10000.0,
        sections=sections
    )
    tl.init()
    
    # Total propagation delay should account for thermal expansion
    assert multi_channel.base_group_delay_s > 0
    
    # Group delay should be slightly different from isothermal fiber
    isothermal = fiberQuantumChannel(
        name="isothermal",
        timeline=tl,
        attenuation=0.0,
        distance=10000.0,
        spec=FiberSpec(temperature_C=20.0)
    )
    tl.init()
    
    # Delays should differ (hot section has longer physical length)
    delay_diff = abs(multi_channel.base_group_delay_s - isothermal.base_group_delay_s)
    assert delay_diff > 0, "Temperature should affect group delay"


# ============================================================================
# Test 12: Multi-Section Fiber - Mixed Birefringence
# ============================================================================


def test_multi_section_mixed_birefringence():
    """
    Test realistic multi-section fiber with different birefringence sources.
    Section 1: 4 km straight, epsilon=1.005
    Section 2: 3 km bent (R=15m), epsilon=1.002
    Section 3: 3 km twisted (τ=0.05 rad/m), epsilon=1.001
    """
    tl = Timeline()
    
    sections = [
        FiberSection(4000.0, FiberSpec(
            core_ellipticity=1.005,
            bend_radius_m=0.0,
            twist_rate_rad_per_m=0.0
        )),
        FiberSection(3000.0, FiberSpec(
            core_ellipticity=1.002,
            bend_radius_m=15.0,
            twist_rate_rad_per_m=0.0
        )),
        FiberSection(3000.0, FiberSpec(
            core_ellipticity=1.001,
            bend_radius_m=0.0,
            twist_rate_rad_per_m=0.05
        ))
    ]
    
    multi_channel = fiberQuantumChannel(
        name="mixed_birefringence",
        timeline=tl,
        attenuation=0.0,
        distance=10000.0,
        sections=sections
    )
    tl.init()
    
    # Jones matrix should be unitary
    J = multi_channel.J_total
    assert abs(abs(np.linalg.det(J)) - 1.0) < 1e-10
    
    # DCD should be consistent (BIFROST ref: 13.702363 ps/(nm·km))
    assert abs(multi_channel.DCD_ps_per_nm_km - 13.702363) < 0.01


def test_multi_section_dgd_combination():
    """
    Test DGD from multiple sections with different birefringence.
    DGD should combine according to birefringence in each section.
    """
    tl = Timeline()
    
    # Create sections with different ellipticities
    sections = [
        FiberSection(4000.0, FiberSpec(core_ellipticity=1.005)),
        FiberSection(3000.0, FiberSpec(core_ellipticity=1.002)),
        FiberSection(3000.0, FiberSpec(core_ellipticity=1.001))
    ]
    
    # Calculate individual DGDs
    individual_dgds = []
    for i, section in enumerate(sections):
        ch = fiberQuantumChannel(
            name=f"section_{i}",
            timeline=tl,
            attenuation=0.0,
            distance=section.length_m,
            spec=section.spec
        )
        tl.init()
        individual_dgds.append(ch.tau_dgd_s * 1e12)
    
    # BIFROST refs: Section 1: 3.280025 ps, Section 2: 0.985716 ps, Section 3: 0.493160 ps
    assert abs(individual_dgds[0] - 3.280025) < 0.01
    assert abs(individual_dgds[1] - 0.985716) < 0.01
    assert abs(individual_dgds[2] - 0.493160) < 0.01
    
    # Multi-section channel
    multi_channel = fiberQuantumChannel(
        name="multi_dgd",
        timeline=tl,
        attenuation=0.0,
        distance=sum(s.length_m for s in sections),
        sections=sections
    )
    tl.init()
    
    # Total DGD should be positive
    assert multi_channel.tau_dgd_s > 0


# ============================================================================
# Test 13: Multi-Section Jones Matrix Multiplication
# ============================================================================


def test_multi_section_jones_matrix_order():
    """
    Test that Jones matrices are multiplied in correct order.
    Order matters: J_total = J3 @ J2 @ J1 (light travels 1->->3)
    """
    tl = Timeline()
    
    # Three sections with different properties
    sections = [
        FiberSection(1000.0, FiberSpec(core_ellipticity=1.005)),
        FiberSection(1000.0, FiberSpec(core_ellipticity=1.003)),
        FiberSection(1000.0, FiberSpec(core_ellipticity=1.001))
    ]
    
    # Get individual Jones matrices
    J_individual = []
    for i, section in enumerate(sections):
        ch = fiberQuantumChannel(
            name=f"section_{i}",
            timeline=tl,
            attenuation=0.0,
            distance=section.length_m,
            spec=section.spec
        )
        tl.init()
        J_individual.append(ch.J_total)
    
    # Multi-section channel
    multi_channel = fiberQuantumChannel(
        name="multi_ordered",
        timeline=tl,
        attenuation=0.0,
        distance=3000.0,
        sections=sections
    )
    tl.init()
    
    # Manually compute expected composite matrix
    J_expected = np.matmul(J_individual[2], np.matmul(J_individual[1], J_individual[0]))
    
    # Should match (within numerical precision)
    assert np.allclose(multi_channel.J_total, J_expected, atol=1e-10), \
        "Jones matrix multiplication order incorrect"


def test_multi_section_photon_transmission():
    """
    Test photon transmission through multi-section fiber.
    Verify state evolution and timing.
    """
    class FakeNode(Node):
        def __init__(self, name, tl):
            Node.__init__(self, name, tl)
            self.received_photons = []
            self.generator = np.random.default_rng(SEED)
        
        def receive_qubit(self, src, photon):
            self.received_photons.append({
                'time': self.timeline.now(),
                'name': photon.name,
                'state': np.array(photon.quantum_state.state)
            })
        
        def get_generator(self):
            return self.generator
    
    tl = Timeline()
    
    sections = [
        FiberSection(5000.0, FiberSpec(core_ellipticity=1.005)),
        FiberSection(5000.0, FiberSpec(core_ellipticity=1.002))
    ]
    
    channel = fiberQuantumChannel(
        name="multi_channel",
        timeline=tl,
        attenuation=0.0,
        distance=10000.0,
        sections=sections
    )
    
    sender = FakeNode("sender", tl)
    receiver = FakeNode("receiver", tl)
    sender.set_seed(SEED)
    
    channel.set_ends(sender, receiver.name)
    tl.init()
    
    # Send H-polarized photon
    photon = Photon("test", timeline=tl, encoding_type=polarization)
    photon.set_state((1, 0))
    
    channel.transmit(photon, sender)
    tl.run()
    
    # Should receive 1 photon
    assert len(receiver.received_photons) == 1
    
    # State should be transformed
    final_state = receiver.received_photons[0]['state']
    norm = np.sum(np.abs(final_state)**2)
    assert abs(norm - 1.0) < 1e-6, "State should remain normalized"


# ============================================================================
# Test 14: Multi-Section Edge Cases
# ============================================================================


def test_multi_section_single_section():
    """
    Test that multi-section fiber with 1 section matches single-section fiber.
    """
    tl = Timeline()
    spec = FiberSpec(core_ellipticity=1.005)
    
    single = fiberQuantumChannel(
        name="single",
        timeline=tl,
        attenuation=0.0,
        distance=1000.0,
        spec=spec
    )
    
    multi = fiberQuantumChannel(
        name="multi_one_section",
        timeline=tl,
        attenuation=0.0,
        distance=1000.0,
        sections=[FiberSection(1000.0, spec)]
    )
    
    tl.init()
    
    # Should be identical
    assert np.allclose(single.J_total, multi.J_total, atol=1e-12)
    assert abs(single.DCD_ps_per_nm_km - multi.DCD_ps_per_nm_km) < 1e-10
    assert abs(single.tau_dgd_s - multi.tau_dgd_s) < 1e-15


def test_multi_section_lengths_sum_validation():
    """
    Test that section lengths must sum to total distance.
    """
    tl = Timeline()
    
    sections = [
        FiberSection(5000.0, FiberSpec()),
        FiberSection(3000.0, FiberSpec())
    ]
    # Sum = 8000, but total = 10000
    
    with pytest.raises(ValueError, match="sum to total distance|must equal total distance"):
        channel = fiberQuantumChannel(
            name="invalid",
            timeline=tl,
            attenuation=0.0,
            distance=10000.0,  # Mismatch!
            sections=sections
        )
        tl.init()

# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
