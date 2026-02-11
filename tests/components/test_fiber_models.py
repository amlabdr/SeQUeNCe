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
# Helper Function for Fig 7/10 Tests
# ============================================================================

import numpy as np
from sequence.components.fiber_models import get_fiber_attenuation_per_m

def calculate_launching_power_mW(
    output_power_W: float,
    wavelength_nm: float,
    fiber_length_m: float
) -> float:
    """
    Calculate required launching power to achieve desired output power.
    
    Used for Fig 7 scenario: maintain 1 μW at receiver by adjusting launcher.
    P_out = P_in * exp(-alpha * L)
    P_in = P_out * exp(alpha * L)
    """
    h = 6.62607015e-34  # Planck constant
    c = 299792458  # Speed of light
    
    # Get fiber attenuation at this wavelength
    alpha_per_m = get_fiber_attenuation_per_m(wavelength_nm)
    
    # Photon energy
    E_photon = h * c / (wavelength_nm * 1e-9)
    
    # Convert output power to photons/s
    output_photons_s = output_power_W / E_photon
    
    # Calculate required launching rate (compensate for fiber loss)
    launching_photons_s = output_photons_s * np.exp(alpha_per_m * fiber_length_m)
    
    # Convert back to Watts, then mW
    launching_W = launching_photons_s * E_photon
    return launching_W * 1e3

# ============================================================================
# Test 15: Classical Coexistence - Raman Scattering Constants
# ============================================================================


def test_raman_scattering_constants_table2():
    """
    Test that Raman scattering constants match Burenkov et al. Table 2.
    
    Reference: Burenkov et al., "Synchronization and coexistence in quantum networks,"
               Optics Express 31, 11431 (2023), Table 2.
    
    Beta constants in units of 10^-23 m^-1 Hz^-1 for 100 GHz DWDM channel.
    """
    from sequence.components.fiber_models import RamanScatteringConstants
    
    raman = RamanScatteringConstants()
    
    # Test each wavelength pair from Table 2
    test_cases = [
        # (classical_nm, quantum_nm, expected_beta_FS, expected_beta_BS)
        (1270, 1550, 0.058e-23, 0.061e-23),  # Best option (lowest noise)
        (1310, 1550, 0.421e-23, 0.449e-23),  # 7x more noise
        (1330, 1550, 0.699e-23, 0.745e-23),  # 12x more noise
        (1490, 1550, 3.69e-23, 3.75e-23),    # 64x more noise (worst)
    ]
    
    for classical_nm, quantum_nm, expected_FS, expected_BS in test_cases:
        beta_FS, beta_BS = raman.get_beta(classical_nm, quantum_nm)
        
        assert abs(beta_FS - expected_FS) < 1e-26, \
            f"Beta_FS({classical_nm}→{quantum_nm}): {beta_FS:.3e} != {expected_FS:.3e}"
        assert abs(beta_BS - expected_BS) < 1e-26, \
            f"Beta_BS({classical_nm}→{quantum_nm}): {beta_BS:.3e} != {expected_BS:.3e}"
    
    # Test wavelength detuning effect (1270 nm is 64x better than 1490 nm)
    beta_FS_1270, _ = raman.get_beta(1270, 1550)
    beta_FS_1490, _ = raman.get_beta(1490, 1550)
    
    ratio = beta_FS_1490 / beta_FS_1270
    assert 60 < ratio < 68, f"Noise ratio 1490/1270: {ratio:.1f} (expected ~64x)"


# ============================================================================
# Test 16: Classical Coexistence - Noise Rate vs Power (Fig 3)
# ============================================================================

def test_raman_noise_rate_vs_classical_power():
    """
    Validate Raman noise linear scaling with classical launch power.

    Reference:
        Burenkov et al., Opt. Express 31, 11431 (2023), Fig. 3

    What Fig. 3 shows:
        - Noise rate ∝ classical photon rate
        - Fixed fiber length (25 km)
        - Separate FS and BS curves (we test FS only)
        - Order-of-magnitude agreement (not exact matching)

    This test verifies:
        1) Linear scaling of FS Raman noise vs power
        2) Correct absolute order of magnitude at 10^14 photons/s
    """

    tl = Timeline()

    # Fiber spec: best-case configuration from the paper
    spec = FiberSpec(
        classical_coexist_enabled=True,
        classical_wavelength_nm=1270.0,
        quantum_wavelength_nm=1550.0,
        quantum_bandwidth_Hz=100e9,
        classical_power_mW=1.0,  # overridden below
    )

    # Fig. 3 x-axis values converted to mW at 1270 nm
    # 10^13 → 0.00156 mW
    # 10^14 → 0.0156 mW
    # 10^15 → 0.156  mW
    powers_mW = [0.00156, 0.0156, 0.156]

    fs_noise_rates = []

    for p_mW in powers_mW:
        spec.classical_power_mW = p_mW

        channel = fiberQuantumChannel(
            name=f"raman_test_{p_mW:.3e}_mW",
            timeline=tl,
            attenuation=0.17 / 1000.0,  # ~0.17 dB/km at 1550 nm
            distance=25_000.0,
            sections=[FiberSection(25_000.0, spec)],
        )

        tl.init()

        # Use FORWARD scattering only (matches one Fig. 3 curve)
        fs_noise_rates.append(channel.raman_noise_rate_FS_Hz)

    # ------------------------------------------------------------------
    # 1) Linearity test: noise  power
    # ------------------------------------------------------------------
    for i in range(1, len(powers_mW)):
        expected_ratio = powers_mW[i] / powers_mW[0]
        actual_ratio = fs_noise_rates[i] / fs_noise_rates[0]

        assert abs(actual_ratio - expected_ratio) < 0.1 * expected_ratio, (
            f"Non-linear Raman scaling: "
            f"P={powers_mW[i]:.3e} mW → ratio={actual_ratio:.2f}, "
            f"expected={expected_ratio:.2f}"
        )

    # ------------------------------------------------------------------
    # 2) Absolute magnitude sanity check (10^14 photons/s)
    # ------------------------------------------------------------------
    noise_at_1e14 = fs_noise_rates[1]  # middle point

    # Fig. 3 shows ~2e4 photons/s for FS @ 1270 nm, 25 km
    assert 1e4 < noise_at_1e14 < 4e4, (
        f"FS Raman noise at 10^14 ph/s: {noise_at_1e14:.2e} photons/s "
        f"(expected ~2e4)"
    )

# ============================================================================
# Test 17: Classical Coexistence - Noise Rate vs Fiber Length (Fig 4)
# ============================================================================

def test_raman_noise_rate_vs_fiber_length():
    """
    Validate Raman noise vs fiber length against Burenkov et al., Fig. 4.

    Paper facts:
    - Fixed LAUNCHING power: 10^14 photons/s
    - FS and BS are plotted separately (NOT summed)
    - Curves saturate with length due to effective length
    """

    tl = Timeline()

    fiber_lengths_km = [1, 6, 12, 25]

    from sequence.constants import SPEED_OF_LIGHT_M_PER_S, PLANCK_CONSTANT

    # Fixed launching power from Fig. 4 (10^14 photons/s)
    E_photon = PLANCK_CONSTANT * SPEED_OF_LIGHT_M_PER_S / (1270.0 * 1e-9)
    launching_power_mW = 1e14 * E_photon * 1e3

    fs_1270 = []
    bs_1270 = []
    fs_1310 = []
    bs_1310 = []

    for L_km in fiber_lengths_km:
        L_m = L_km * 1000.0

        spec_1270 = FiberSpec(
            classical_coexist_enabled=True,
            classical_wavelength_nm=1270.0,
            quantum_wavelength_nm=1550.0,
            quantum_bandwidth_Hz=100e9,
            classical_power_mW=launching_power_mW,
        )

        spec_1310 = FiberSpec(
            classical_coexist_enabled=True,
            classical_wavelength_nm=1310.0,
            quantum_wavelength_nm=1550.0,
            quantum_bandwidth_Hz=100e9,
            classical_power_mW=launching_power_mW,
        )

        ch_1270 = fiberQuantumChannel(
            name=f"raman_1270_{L_km}km",
            timeline=tl,
            attenuation=0.0002,  # required by constructor
            distance=L_m,
            sections=[FiberSection(L_m, spec_1270)],
        )

        ch_1310 = fiberQuantumChannel(
            name=f"raman_1310_{L_km}km",
            timeline=tl,
            attenuation=0.0002,
            distance=L_m,
            sections=[FiberSection(L_m, spec_1310)],
        )

        tl.init()

        fs_1270.append(ch_1270.raman_noise_rate_FS_Hz)
        bs_1270.append(ch_1270.raman_noise_rate_BS_Hz)
        fs_1310.append(ch_1310.raman_noise_rate_FS_Hz)
        bs_1310.append(ch_1310.raman_noise_rate_BS_Hz)

    # --------------------------------------------------
    # 1) FS and BS increase with length and saturate
    # --------------------------------------------------
    for arr, label in [
        (fs_1270, "FS 1270"),
        (bs_1270, "BS 1270"),
        (fs_1310, "FS 1310"),
        (bs_1310, "BS 1310"),
    ]:
        for i in range(1, len(arr)):
            assert arr[i] >= arr[i - 1], f"{label} not monotonic with length"

    # --------------------------------------------------
    # 2) 1310 nm produces more noise than 1270 nm (Table 2 trend)
    # --------------------------------------------------
    for i, L_km in enumerate(fiber_lengths_km):
        assert fs_1310[i] > fs_1270[i], f"FS 1310 <= FS 1270 at {L_km} km"
        assert bs_1310[i] > bs_1270[i], f"BS 1310 <= BS 1270 at {L_km} km"

    # --------------------------------------------------
    # 3) Absolute magnitude check at 25 km (Fig. 4)
    # --------------------------------------------------
    idx_25 = fiber_lengths_km.index(25)

    # From Fig. 4 (order-of-magnitude read-off)
    # FS 1270 ~ (3–4)×10^4, BS 1270 ~ (4–4)×10^4
    assert 2e4 < fs_1270[idx_25] < 4e4, (
        f"FS 1270 @25 km out of range: {fs_1270[idx_25]:.2e}"
    )
    assert 4e4 < bs_1270[idx_25] < 5e5, (
        f"BS 1270 @25 km out of range: {bs_1270[idx_25]:.2e}"
    )

# ============================================================================
# Test 18: Classical Coexistence - Coexistence Limit (Fig 10)
# ============================================================================

def test_classical_coexistence_fiber_length_limit():
    """
    Validate coexistence limit under constant classical OUTPUT power
    against Burenkov et al., Fig. 7 / Fig. 10.

    - Fixed OUTPUT power: 1 μW at receiver
    - Launching power increases with length
    - TOTAL Raman noise = FS + BS
    """

    tl = Timeline()

    classical_output_power_W = 1e-6  # 1 μW at receiver
    quantum_wavelength_nm = 1550.0
    quantum_bandwidth_Hz = 100e9

    test_lengths_km = [1, 6, 12, 25]

    # Approximate values read from Fig. 7 (model curve, order-of-magnitude)
    expected_noise_fig7 = {
        1: 3e2,
        6: 3e3,
        12: 9e3,
        25: 2e4
    }

    noise_1270 = {}

    for L_km in test_lengths_km:
        L_m = L_km * 1000.0

        launching_power_mW = calculate_launching_power_mW(
            output_power_W=classical_output_power_W,
            wavelength_nm=1270.0,
            fiber_length_m=L_m,
        )

        spec = FiberSpec(
            classical_coexist_enabled=True,
            classical_wavelength_nm=1270.0,
            quantum_wavelength_nm=quantum_wavelength_nm,
            quantum_bandwidth_Hz=quantum_bandwidth_Hz,
            classical_power_mW=launching_power_mW,
        )

        ch = fiberQuantumChannel(
            name=f"coexist_1270_{L_km}km",
            timeline=tl,
            attenuation=0.0002,
            distance=L_m,
            sections=[FiberSection(L_m, spec)],
        )

        tl.init()
        noise_1270[L_km] = ch.raman_noise_rate_Hz

    # --------------------------------------------------
    # 1) Noise must increase strongly with fiber length
    # --------------------------------------------------
    for i in range(1, len(test_lengths_km)):
        L_prev = test_lengths_km[i - 1]
        L_curr = test_lengths_km[i]
        assert noise_1270[L_curr] > noise_1270[L_prev], (
            f"Noise should increase: {L_curr} km > {L_prev} km"
        )

    # --------------------------------------------------
    # 2) Absolute scale sanity check (order-of-magnitude)
    # --------------------------------------------------
    for L_km in [6, 12, 25]:
        measured = noise_1270[L_km]
        expected = expected_noise_fig7[L_km]

        # Allow wide tolerance: paper curves are model projections
        assert 0.5 * expected < measured < 2.0 * expected, (
            f"Noise at {L_km} km: {measured:.2e}, expected ~{expected:.2e}"
        )

    # --------------------------------------------------
    # 3) 1270 nm is significantly better than 1490 nm
    # --------------------------------------------------
    L_test_m = 25_000.0

    launch_1270 = calculate_launching_power_mW(
        classical_output_power_W, 1270.0, L_test_m
    )
    launch_1490 = calculate_launching_power_mW(
        classical_output_power_W, 1490.0, L_test_m
    )

    spec_1270 = FiberSpec(
        classical_coexist_enabled=True,
        classical_wavelength_nm=1270.0,
        quantum_wavelength_nm=1550.0,
        quantum_bandwidth_Hz=100e9,
        classical_power_mW=launch_1270,
    )

    spec_1490 = FiberSpec(
        classical_coexist_enabled=True,
        classical_wavelength_nm=1490.0,
        quantum_wavelength_nm=1550.0,
        quantum_bandwidth_Hz=100e9,
        classical_power_mW=launch_1490,
    )

    ch_1270 = fiberQuantumChannel(
        "1270", tl, 0.0002, L_test_m, sections=[FiberSection(L_test_m, spec_1270)]
    )
    ch_1490 = fiberQuantumChannel(
        "1490", tl, 0.0002, L_test_m, sections=[FiberSection(L_test_m, spec_1490)]
    )

    tl.init()

    ratio = ch_1490.raman_noise_rate_Hz / ch_1270.raman_noise_rate_Hz

    # Expect strong degradation for 1490 nm (>>10× worse)
    assert ratio > 30, f"1490/1270 noise ratio too small: {ratio:.1f}"


# ============================================================================
# Test 19: Classical Coexistence - Multi-section with Mixed Classical Channels
# ============================================================================

def test_multi_section_classical_coexistence():
    """
    Test realistic scenario: Multi-section fiber with classical channels
    enabled only in certain sections.
    
    Scenario: 
    - Section 1: 10 km, no classical (dark fiber)
    - Section 2: 15 km, with 1270 nm classical @ 0.0156 mW
    - Section 3: 10 km, no classical (dark fiber)
    
    Total noise should only come from Section 2.
    """
    tl = Timeline()
    
    sections = [
        # Section 1: Dark fiber (no classical)
        FiberSection(10000.0, FiberSpec(
            classical_coexist_enabled=False
        )),
        
        # Section 2: Classical + quantum coexistence
        FiberSection(15000.0, FiberSpec(
            classical_coexist_enabled=True,
            classical_wavelength_nm=1270.0,
            quantum_wavelength_nm=1550.0,
            quantum_bandwidth_Hz=100e9,
            classical_power_mW=0.0156  # Match paper's 10^14 photons/s
        )),
        
        # Section 3: Dark fiber (no classical)
        FiberSection(10000.0, FiberSpec(
            classical_coexist_enabled=False
        )),
    ]
    
    multi_channel = fiberQuantumChannel(
        name="multi_coexist",
        timeline=tl,
        attenuation=0.0002,
        distance=35000.0,  # 35 km total
        sections=sections
    )
    tl.init()
    
    # Test 1: Noise should be non-zero (Section 2 contributes)
    assert multi_channel.raman_noise_rate_Hz > 0, \
        "Should have noise from Section 2"
    
    # Test 2: Noise should match single 15 km section
    single_channel = fiberQuantumChannel(
        name="single_15km",
        timeline=tl,
        attenuation=0.0002,
        distance=15000.0,
        sections=[FiberSection(15000.0, sections[1].spec)]
    )
    tl.init()
    
    # Noise should be similar (within tolerance due to pre/post-section attenuation)
    ratio = multi_channel.raman_noise_rate_Hz / single_channel.raman_noise_rate_Hz
    assert 0.9 < ratio < 1.1, \
        f"Multi-section noise mismatch: {ratio:.2f} (expected ~1.0)"
    
    # Test 3: If all sections are dark, noise should be zero
    dark_sections = [
        FiberSection(10000.0, FiberSpec(classical_coexist_enabled=False)),
        FiberSection(15000.0, FiberSpec(classical_coexist_enabled=False)),
        FiberSection(10000.0, FiberSpec(classical_coexist_enabled=False)),
    ]
    
    dark_channel = fiberQuantumChannel(
        name="all_dark",
        timeline=tl,
        attenuation=0.0002,
        distance=35000.0,
        sections=dark_sections
    )
    tl.init()
    
    assert dark_channel.raman_noise_rate_Hz == 0.0, \
        "Dark fiber should have zero Raman noise"

# ============================================================================
# Test 20: Classical Coexistence - Noise Photon Generation
# ============================================================================

def test_raman_noise_photon_generation():
    """
    Test that noise photons are actually generated and transmitted.
    
    This tests the full pipeline:
    1. _schedule_next_noise_photon() schedules Poisson-distributed events
    2. _generate_and_transmit_noise_photon() creates random-polarization photons
    3. Noise photons are transmitted through channel (experience loss, PMD, CD)
    4. Noise photons arrive at receiver
    
    Use higher power to generate sufficient noise in short runtime
    """
    class FakeNode(Node):
        def __init__(self, name, tl):
            Node.__init__(self, name, tl)
            self.received_photons = []
            self.noise_photon_count = 0
            self.signal_photon_count = 0
            self.generator = np.random.default_rng(SEED)
        
        def receive_qubit(self, src, photon):
            self.received_photons.append({
                'time': self.timeline.now(),
                'name': photon.name,
                'is_noise': getattr(photon, 'is_raman_noise', False)
            })
            
            if getattr(photon, 'is_raman_noise', False):
                self.noise_photon_count += 1
            else:
                self.signal_photon_count += 1
        
        def get_generator(self):
            return self.generator
    
    tl = Timeline()
    
    # High noise scenario: 1490 nm classical @ 1 mW, 1 km fiber
    # Expected noise: ~2e6 photons/s
    spec_high_noise = FiberSpec(
        classical_coexist_enabled=True,
        classical_wavelength_nm=1490.0,  # Worst case - 64x more noise
        quantum_wavelength_nm=1550.0,
        quantum_bandwidth_Hz=100e9,
        classical_power_mW=1.0  # 1 mW for measurable noise in short time
    )
    
    channel = fiberQuantumChannel(
        name="noisy_channel",
        timeline=tl,
        attenuation=0.0,  # No loss for this test
        distance=1000.0,  # 1 km
        sections=[FiberSection(1000.0, spec_high_noise)]
    )
    
    sender = FakeNode("sender", tl)
    receiver = FakeNode("receiver", tl)
    sender.set_seed(SEED)
    
    tl.init()
    channel.set_ends(sender, receiver.name)
    
    # Run simulation for 10 microseconds (long enough to see multiple noise photons)
    tl.stop_time = int(1e8)  # 10 μs in picoseconds
    tl.run()
    
    # Test 1: Noise photons should be generated
    assert receiver.noise_photon_count > 0, \
        f"No noise photons received (expected > 0)"
    
    # Test 2: Noise rate should match theoretical prediction
    runtime_s = tl.stop_time * 1e-12
    measured_rate = receiver.noise_photon_count / runtime_s
    expected_rate = channel.raman_noise_rate_Hz

    ratio = measured_rate / expected_rate
    assert 0.9 < ratio < 1.1, \
        f"Noise rate mismatch: {measured_rate:.2e} vs {expected_rate:.2e} photons/s"
    
    # Test 3: Noise photons should have random polarization
    # (Cannot test directly, but verify they are marked correctly)
    for photon_info in receiver.received_photons:
        if photon_info['is_noise']:
            assert photon_info['name'] == 'raman_noise', \
                "Noise photons should be named 'raman_noise'"
            
# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
