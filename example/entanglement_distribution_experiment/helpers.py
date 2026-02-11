import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning

# Pauli matrices
I2 = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
PAULIS = {"X": X, "Y": Y, "Z": Z}

def sliding_windows_coincidences(config, timestamps1, timestamps2):
    """
    Calculate coincidences between two timestamp arrays using a sliding window around the peak.

    Args:
        config (dict): Parameters for the coincidence detection:
            - peak0 (float): time offset to add to timestamps2 (in ps), default 0
            - range_ns (float): total histogram range in ns, default 10
            - time_bin (float): bin width in ns, default 0.5
            - coincidence_window_width (float): width of coincidence window around peak (in ps), default 1000
            - time_integration_ms (float): time period in ms for normalization, default 1000
        timestamps1 (array-like): Timestamps for channel 1 (in ps)
        timestamps2 (array-like): Timestamps for channel 2 (in ps)

    Returns:
        dict with histogram, peak info, coincidence rates, and window info
    """

    # Extract config with defaults
    peak0 = config.get("peak0", 0)
    range_ns = config.get("range_ns", 10)
    time_bin = config.get("time_bin", 0.5)
    coincidence_width = config.get("coincidence_window_width", 1000)  # in ps
    integration_time_s = config.get("time_integration_ms", 1000) / 1000

    # Prepare parameters
    range_ps = range_ns * 1e3
    half_range_ps = range_ps / 2
    bins_n = int(range_ns / time_bin)

    # Shift timestamps2
    timestamps2 = timestamps2 + peak0

    # Sliding window coincidence detection
    coincidences = []
    idx2 = 0
    for t1 in timestamps1:
        while idx2 < len(timestamps2) and timestamps2[idx2] < t1 - half_range_ps:
            idx2 += 1

        start_idx = idx2
        while idx2 < len(timestamps2) and timestamps2[idx2] <= t1 + half_range_ps:
            dtime = t1 - timestamps2[idx2]
            coincidences.append(dtime)
            idx2 += 1
        idx2 = start_idx

    coincidences = np.array(coincidences)

    # Histogram
    histo_vals, bin_edges = np.histogram(coincidences, bins=bins_n, range=(-half_range_ps, half_range_ps))

    # Peak value
    peak_index = np.argmax(histo_vals)
    peak = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2

    low_win = peak - coincidence_width / 2
    high_win = peak + coincidence_width / 2
    coinc_mask = (bin_edges[:-1] >= low_win) & (bin_edges[:-1] < high_win)
    coincidences_count = np.sum(histo_vals[coinc_mask])
    coincidence_rate = coincidences_count / integration_time_s

    # Accidentals: average counts from side regions (exclude central window)
    side_mask = ~coinc_mask
    accidental_counts = np.mean(histo_vals[side_mask]) * np.sum(coinc_mask)

    # CAR
    CAR = coincidences_count / accidental_counts if accidental_counts > 0 else np.inf

    return {
        "histo_vals": histo_vals,
        "bin_edges": bin_edges,
        "peak": peak,
        "window_low": low_win,
        "window_high": high_win,
        "coincidences_rate": coincidence_rate,
        "coincidences_count": coincidences_count,
        "channel1_rate": len(timestamps1) / integration_time_s,
        "channel2_rate": len(timestamps2) / integration_time_s,
        "accidentals": accidental_counts,
        "CAR": CAR
    }

def cos2_model(theta_deg, A, theta0, B):
    theta_rad = np.radians(theta_deg)
    return A * (np.cos(theta_rad - np.radians(theta0))**2) + B


def compute_visibility_from_fit(stats_list, angles):
    """
    Fit cos^2 model to coincidence data and compute visibility.
    Formula: V = (Cmax - Cmin) / (Cmax + Cmin).
    """
    x = np.degrees(angles)
    y_means = np.array([m for m, _ in stats_list])
    y_stds  = np.array([s for _, s in stats_list])
    
    # Check for completely flat data
    if np.ptp(y_means) < 1e-9:
        return 0.0, 0.0  # No modulation = zero visibility
    
    # Initial guess
    y_max, y_min = y_means.max(), y_means.min()
    A0 = (y_max - y_min) / 2
    B0 = (y_max + y_min) / 2  # Midpoint between max and min
    theta0_guess = x[np.argmax(y_means)]
    p0 = [A0, theta0_guess, B0]
    
    # Handle errors - use Poisson if stds are too small
    eps = np.where(y_stds > 0, y_stds, np.sqrt(np.maximum(y_means, 1)))
    eps = np.maximum(eps, 1e-10)  # Prevent zero weights
    
    try:
        popt, pcov = curve_fit(
            cos2_model, x, y_means, p0=p0,
            sigma=eps,
            absolute_sigma=True,
            maxfev=10000
        )
        
        A, theta0, B = popt
        
        # Check if covariance is valid
        if not np.all(np.isfinite(pcov)):
            raise RuntimeError("Covariance matrix contains inf/nan")
        
        # Compute Cmax and Cmin from fitted model
        Cmax = A + B
        Cmin = B
        
        # Check for physical validity
        if Cmax + Cmin <= 0:
            # Fallback to direct calculation
            raise RuntimeError("Unphysical fit parameters")
        
        # Calculate visibility
        V = abs((Cmax - Cmin) / (Cmax + Cmin))
        
        # Clamp to [0, 1]
        V = np.clip(V, 0.0, 1.0)
        
        # Error propagation
        # V = |A| / (A + 2B), so dV/dA = 2B/(A+2B)^2, dV/dB = -2A/(A+2B)^2
        denom = A + 2*B
        if abs(denom) < 1e-10:
            sigma_V = 0.0
        else:
            dV_dA = 2 * B / (denom**2)
            dV_dB = -2 * A / (denom**2)
            J = np.array([dV_dA, 0.0, dV_dB])
            
            sigma_V_squared = J @ pcov @ J.T
            
            if sigma_V_squared < 0 or not np.isfinite(sigma_V_squared):
                sigma_V = 0.0
            else:
                sigma_V = np.sqrt(sigma_V_squared)
        
        return V, sigma_V
        
    except Exception as e:
        # Fit failed - use direct min/max from data
        Cmax = y_max
        Cmin = y_min
        
        if Cmax + Cmin <= 0:
            return 0.0, 0.0
        
        V = (Cmax - Cmin) / (Cmax + Cmin)
        V = np.clip(V, 0.0, 1.0)
        
        # Simple error estimate
        idx_max = np.argmax(y_means)
        idx_min = np.argmin(y_means)
        sigma_max = y_stds[idx_max] if y_stds[idx_max] > 0 else np.sqrt(Cmax)
        sigma_min = y_stds[idx_min] if y_stds[idx_min] > 0 else np.sqrt(Cmin)
        
        # Error propagation for V = (Cmax - Cmin)/(Cmax + Cmin)
        dV_dCmax = 2 * Cmin / ((Cmax + Cmin)**2)
        dV_dCmin = -2 * Cmax / ((Cmax + Cmin)**2)
        sigma_V = np.sqrt((dV_dCmax * sigma_max)**2 + (dV_dCmin * sigma_min)**2)
        
        return V, sigma_V


def compute_visibility(stats_list):
    """
    Compute visibility from a list of (mean, std) coincidence values.

    Args:
        stats_list: list of (mean, std) tuples

    Returns:
        visibility (float)
    """
    means = [mean for mean, _ in stats_list]
    if not means or (max(means) + min(means)) == 0:
        return 0
    return (max(means) - min(means)) / (max(means) + min(means))

def compute_visibility_with_error(coinc_data):
    """
    Args:
        coinc_data: list of (mean, std) tuples, one per angle
    Returns:
        (visibility, visibility_error)
    """
    # Extract means and stds
    means = [c[0] for c in coinc_data]
    stds = [c[1] for c in coinc_data]

    c_max = max(means)
    c_min = min(means)
    i_max = means.index(c_max)
    i_min = means.index(c_min)
    
    sigma_max = stds[i_max]
    sigma_min = stds[i_min]

    V = (c_max - c_min) / (c_max + c_min)
    denom_squared = (c_max + c_min)**2
    sigma_V = (2 / denom_squared) * ((c_min * sigma_max)**2 + (c_max * sigma_min)**2)**0.5

    return V, sigma_V

def get_mean_std(stats_list):
    means = np.array([mean for mean, _ in stats_list])
    stds = np.array([std for _, std in stats_list])
    return means, stds

def nearest_angle_idx(angles, target_deg):
    angles_deg = np.degrees(angles)
    return int(np.argmin(np.abs(angles_deg - target_deg)))

def compute_E_with_error(results, angle_A_deg, angle_B_deg, all_angles_rad):
    key_A = f"{int(round(angle_A_deg))}°" 
    idx = nearest_angle_idx(all_angles_rad, angle_B_deg)

    # Extract means and stds
    hh_mean, hh_std = get_mean_std(results[key_A]["hh"])
    vv_mean, vv_std = get_mean_std(results[key_A]["vv"])
    hv_mean, hv_std = get_mean_std(results[key_A]["hv"])
    vh_mean, vh_std = get_mean_std(results[key_A]["vh"])

    hh = hh_mean[idx]
    vv = vv_mean[idx]
    hv = hv_mean[idx]
    vh = vh_mean[idx]

    hh_err = hh_std[idx]
    vv_err = vv_std[idx]
    hv_err = hv_std[idx]
    vh_err = vh_std[idx]

    numerator = hh + vv - hv - vh
    denominator = hh + vv + hv + vh
    print(f"Alice {angle_A_deg}°, Bob {angle_B_deg}°: hh={hh}±{hh_err}, vv={vv}±{vv_err}, hv={hv}±{hv_err}, vh={vh}±{vh_err}")

    if denominator == 0:
        return 0, 0

    E = numerator / denominator

    # Error propagation
    dE_dhh = ((1 - E) / denominator)
    dE_dvv = ((1 - E) / denominator)
    dE_dhv = ((-1 - E) / denominator)
    dE_dvh = ((-1 - E) / denominator)

    sigma_E = np.sqrt(
        (dE_dhh * hh_err)**2 +
        (dE_dvv * vv_err)**2 +
        (dE_dhv * hv_err)**2 +
        (dE_dvh * vh_err)**2
    )

    return E, sigma_E


def compute_CHSH_S(results, all_angles_rad):
    a, a_prime = 0, 45
    b, b_prime = 22.5, 67.5

    E_ab, err_ab = compute_E_with_error(results, a, b, all_angles_rad)
    E_abp, err_abp = compute_E_with_error(results, a, b_prime, all_angles_rad)
    E_apb, err_apb = compute_E_with_error(results, a_prime, b, all_angles_rad)
    E_apbp, err_apbp = compute_E_with_error(results, a_prime, b_prime, all_angles_rad)
    print(f"E({a},{b}) = {E_ab} ± {err_ab}")
    print(f"E({a},{b_prime}) = {E_abp} ± {err_abp}")
    print(f"E({a_prime},{b}) = {E_apb} ± {err_apb}")
    print(f"E({a_prime},{b_prime}) = {E_apbp} ± {err_apbp}")
    S = abs(E_ab - E_abp) + abs(E_apb + E_apbp)

    # Error propagation (assuming errors are uncorrelated)
    err_S = np.sqrt(err_ab**2 + err_abp**2 + err_apb**2 + err_apbp**2)

    return S, err_S

def horodecki_S(rho):
    # Step 1: Build correlation tensor T_ij = Tr(rho (σ_i ⊗ σ_j))
    T = np.zeros((3, 3), dtype=float)
    pauli_list = [PAULIS["X"], PAULIS["Y"], PAULIS["Z"]]
    
    for i, si in enumerate(pauli_list):
        for j, sj in enumerate(pauli_list):
            op = np.kron(si, sj)
            T[i, j] = np.real(np.trace(rho @ op))

    # Step 2: Form U = T^T T
    U = T.T @ T

    # Step 3: Eigenvalues of U
    eigvals = np.linalg.eigvals(U)
    eigvals = np.sort(np.real(eigvals))[::-1]  # sort descending

    # Step 4: Horodecki formula
    S_max = 2 * np.sqrt(eigvals[0] + eigvals[1])

    return S_max, T, U, eigvals


def reconstruct_rho_from_paulis(T2, singlesA=None, singlesB=None):
    # T2[(i,j)] = <σ_i ⊗ σ_j> with i,j in {X,Y,Z}
    if singlesA is None: singlesA = {"X":0.0, "Y":0.0, "Z":0.0}
    if singlesB is None: singlesB = {"X":0.0, "Y":0.0, "Z":0.0}

    term = np.kron(I2, I2).astype(complex)
    for i in "XYZ":
        term += singlesA[i]*np.kron(PAULIS[i], I2)
    for j in "XYZ":
        term += singlesB[j]*np.kron(I2, PAULIS[j])
    for i in "XYZ":
        for j in "XYZ":
            term += T2[(i,j)]*np.kron(PAULIS[i], PAULIS[j])

    rho = 0.25*term
    rho = 0.5*(rho + rho.conj().T)  # enforce Hermitian
    # project to PSD (clip negatives)
    evals, vecs = np.linalg.eigh(rho)
    evals = np.clip(evals, 0, None)
    if evals.sum() > 0:
        rho = (vecs @ np.diag(evals) @ vecs.conj().T) / np.trace(np.diag(evals))
    else:
        rho = np.eye(4, dtype=complex)/4
    return rho

def get_bell_state(label="phi+"):
    """
    Return a 4x1 column vector for the chosen Bell state.
    label: "phi+", "phi-", "psi+", "psi-"
    """
    phi = np.zeros((4,1), dtype=complex)

    if label.lower() == "phi+":
        phi[0,0] = 1/np.sqrt(2)   # |HH>
        phi[3,0] = 1/np.sqrt(2)   # |VV>
    elif label.lower() == "phi-":
        phi[0,0] = 1/np.sqrt(2)
        phi[3,0] = -1/np.sqrt(2)
    elif label.lower() == "psi+":
        phi[1,0] = 1/np.sqrt(2)   # |HV>
        phi[2,0] = 1/np.sqrt(2)   # |VH>
    elif label.lower() == "psi-":
        phi[1,0] = 1/np.sqrt(2)
        phi[2,0] = -1/np.sqrt(2)
    else:
        raise ValueError("Unknown Bell state. Use 'phi+', 'phi-', 'psi+', 'psi-'.")

    return phi

def fidelity(rho, target):
    """
    Fidelity between density matrix rho and pure state target (vector).
    """
    target = target.reshape(-1,1)   # ensure column vector
    return float(np.real((target.conj().T @ rho @ target)[0,0]))

def counts_to_probs(hh, hv, vh, vv):
    total = hh + hv + vh + vv
    if total <= 0:
        return {"pp":0, "pm":0, "mp":0, "mm":0}
    return {"pp": hh/total, "pm": hv/total, "mp": vh/total, "mm": vv/total}

def correlator_from_probs(P):
    # E = P++ + P-- − P+- − P-+
    return (P["pp"] + P["mm"]) - (P["pm"] + P["mp"])


def plot_density_matrix_bars(rho, part="both", title_prefix="Density matrix ρ", figsize=(12,5)):
    """
    Plot 3D bar charts of a 4x4 density matrix.
    Args:
        rho: 4x4 complex numpy array (Hermitian, trace 1 ideally)
        part: "real", "imag", "abs", or "both" (real & imag side-by-side)
        title_prefix: chart title prefix
        figsize: figure size when part="both"
    """
    rho = np.asarray(rho, dtype=complex)
    assert rho.shape == (4,4), "rho must be 4x4"

    labels = ["HH", "HV", "VH", "VV"]
    X, Y = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    xpos, ypos = X.flatten(), Y.flatten()
    zpos = np.zeros_like(xpos, dtype=float)
    dx = dy = 0.6 * np.ones_like(zpos)

    def _bar3d(ax, Zvals, title):
        Z = Zvals.flatten().real  # ensure real heights for bars
        ax.bar3d(xpos, ypos, zpos, dx, dy, Z, shade=True)
        ax.set_xticks(np.arange(4) + 0.3)
        ax.set_yticks(np.arange(4) + 0.3)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_zlim(-0.5, 0.5)
        ax.set_title(title)

    if part.lower() == "real":
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection="3d")
        _bar3d(ax, np.real(rho), f"{title_prefix} — Real")
        plt.show()
    elif part.lower() == "imag":
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection="3d")
        _bar3d(ax, np.imag(rho), f"{title_prefix} — Imag")
        plt.show()
    elif part.lower() == "abs":
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection="3d")
        _bar3d(ax, np.abs(rho), f"{title_prefix} — |ρ|")
        plt.show()
    else:  # "both"
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(121, projection="3d")
        _bar3d(ax1, np.real(rho), f"{title_prefix} Real")
        ax2 = fig.add_subplot(122, projection="3d")
        _bar3d(ax2, np.imag(rho), f"{title_prefix} Imag")
        plt.show()