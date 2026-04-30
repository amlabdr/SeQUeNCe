"""Canonical queueing-model utilities for two-link swapping queues.

This module is the maintained implementation used by the queueing-model
evaluation workflow. Supporting scripts and notebooks should import this file
directly instead of keeping local copies of the same equations.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

DECOHERE_WHEN_SWAP = 1

def normalize_probs(probs):
    total = sum(probs)
    if total > 0:
        return [p / total for p in probs]
    else:
        return [0 for p in probs]

def channel2Pauli(depol, depha):
    v0 = 1-depol
    w0 = 1-depha
    pI = (1 + v0*(1+2*w0))/4
    pX = (1 - v0)/4
    pY = (1 - v0)/4
    pZ = (1 + v0*(1-2*w0))/4
    return pI, normalize_probs((pX, pY, pZ))

def simple_rate(lam1, T1, W1, lam2, T2, W2, Tswp=0):
    """
    Calculates the swapping rate for two queues with
    Poisson arrivals, unit capacity, deadlines W, and reset delays T.
    Uses a simplified approximation based on the expected match time and success probabilities.

    Parameters:
    lam1, lam2 : float : Arrival rates for Queue 1 and Queue 2
    W1, W2     : float : Item lifetimes (max wait time) for Queue 1 and Queue 2
    T1, T2     : float : Reset delays for Queue 1 and Queue 2
    Tswp       : float : Additional swapping time (default=0)

    Returns:
    float : Expected rate.
    """
    a1 = 1-np.exp(-lam2*W1)
    a2 = 1-np.exp(-lam1*W2)
    Tpair = max(T1, T2) + Tswp
    E12 = lam1* a1
    E21 = lam2* a2

    R = (E12+E21)/(1   + E12*(1/lam2 + Tpair) + 1*lam1*(1-a1)*T1
                       + E21*(1/lam1 + Tpair) + 1*lam2*(1-a2)*T2 )
    return R

def maxage_of_swap(lambdas, L_list, T_coh, t_prep=None):
    lam1, lam2 = lambdas
    tau_1 = L_list[0]/2e5 # signaling latency
    tau_2 = L_list[1]/2e5 # signaling latency
    tau = max(tau_1, tau_2)# max signaling latency
    if t_prep is None:
        t_prep = [4.2*tau_1, 4.2*tau_2]
    T_1prep, T_2prep = t_prep

    W1 = max(1e-10, T_coh - tau - T_1prep) # time to wait
    W2 = max(1e-10, T_coh - tau - T_2prep) # time to wait
    W1 = max(W1, T_2prep-T_1prep)
    W2 = max(W2, T_1prep-T_2prep)

    P12 = 1-np.exp(-lam2*W1)
    P21 = 1-np.exp(-lam1*W2)

    P1match2 = lam1*P12/(lam1+lam2)
    P2match1 = lam2*P21/(lam1+lam2)

    E12 = T_1prep + 1/lam2 - W1*(1-P12)/P12
    E21 = T_2prep + 1/lam1 - W2*(1-P21)/P21

    age = tau + (P1match2*E12 + P2match1*E21)/(P1match2 + P2match1)

    return age

def solve_expected_match_time(lam1, lam2, W1, W2, T1, T2):
    """
    Calculates the expected time until the first match for two queues with
    Poisson arrivals, unit capacity, deadlines W, and reset delays T.

    Parameters:
    lam1, lam2 : float : Arrival rates for Queue 1 and Queue 2
    W1, W2     : float : Item lifetimes (max wait time) for Queue 1 and Queue 2
    T1, T2     : float : Reset delays for Queue 1 and Queue 2
    Tswp       : float : Additional swapping time (default=0)

    Returns:
    float : Expected time until the first match from t=0 (simultaneous reset).
    """

    # Basic constraints check to ensure validity of the derived integral limits
    if W1 < T2 or W2 < T1:
        print("Warning: The derivation assumes W_i >= T_j (Lifetime >= Other's Delay).")
        print("Results may be inaccurate if items expire before the other queue can wake up.")

    Lambda = lam1 + lam2

    # Probabilities of expiration (alpha terms)
    # alpha1: Prob Q1 expires before Q2 arrives (given Q1 is waiting) -> e^(-lam2 * W1)
    # But in the formula alpha is defined as probability 'other' fails to arrive
    alpha1 = np.exp(-lam2 * W1)
    alpha2 = np.exp(-lam1 * W2)

    # --- Step 1: Define Coefficients for the Linear System ---
    # System: A * [V_wait, V1, V2]^T = B
    # V_wait: Both waiting
    # V1: Q1 resetting (delay T1), Q2 waiting
    # V2: Q2 resetting (delay T2), Q1 waiting

    # --- Equation 1: V_wait ---
    # V_wait = K_w + P_w1 * V1 + P_w2 * V2
    # Rearranged: 1*V_wait - P_w1*V1 - P_w2*V2 = K_w

    P_w1 = (lam1 / Lambda) * alpha1
    P_w2 = (lam2 / Lambda) * alpha2

    # K_wait calculation
    # Term 1: Expected time to next arrival (1/Lambda)
    # Term 2: Contribution if Q1 arrives first
    # Term 3: Contribution if Q2 arrives first
    term_q1 = (1 - alpha1) / lam2
    term_q2 = (1 - alpha2) / lam1
    K_wait = (1/Lambda) + (lam1/Lambda)*term_q1 + (lam2/Lambda)*term_q2

    # --- Helper to calculate Reset State Coefficients ---
    def get_reset_coeffs(T_delay, lam_resetting, lam_active, W_active):
        """
        Returns (K, P_to_wait, P_to_other) for a state where:
        - 'resetting' queue is in delay T_delay
        - 'active' queue is waiting
        """
        # P_to_wait: Active queue does NOT arrive during delay
        P_to_wait = np.exp(-lam_active * T_delay)

        # P_to_other: Active queue arrives, but expires before resetting queue matches
        # Formula derived: (lam_active / Lambda) * e^(-lam_resetting * (W_active - T_delay)) * (1 - np.exp(-Lambda * T_delay))
        # Note: W_active must be >= T_delay for this form
        exp_factor = np.exp(-lam_resetting * (W_active - T_delay))
        P_to_other = (lam_active / Lambda) * exp_factor * (1 - np.exp(-Lambda * T_delay))

        # K (Constant time contribution)
        # K = T_delay + (1/lam_resetting)*(1 - P_to_wait) - (1/lam_resetting)*P_to_other
        K = T_delay + (1/lam_resetting)*(1 - P_to_wait) - (1/lam_resetting)*P_to_other

        return K, P_to_wait, P_to_other

    # --- Equation 2: V1 (Q1 is resetting, Q2 is active) ---
    # V1 = K1 + P1_w * V_wait + P1_2 * V2
    # Rearranged: -P1_w * V_wait + 1*V1 - P1_2 * V2 = K1
    K1, P1_w, P1_2 = get_reset_coeffs(T1, lam1, lam2, W2)

    # --- Equation 3: V2 (Q2 is resetting, Q1 is active) ---
    # V2 = K2 + P2_w * V_wait + P2_1 * V1
    # Rearranged: -P2_w * V_wait - P2_1 * V1 + 1*V2 = K2
    K2, P2_w, P2_1 = get_reset_coeffs(T2, lam2, lam1, W1)

    # --- Step 2: Solve Linear System ---
    A = np.array([
        [1.0,   -P_w1,  -P_w2],  # V_wait eq
        [-P1_w,  1.0,   -P1_2],  # V1 eq
        [-P2_w, -P2_1,   1.0 ]   # V2 eq
    ])

    B = np.array([K_wait, K1, K2])

    # Solve for [V_wait, V1, V2]
    V_solution = np.linalg.solve(A, B)
    V_wait_sol, V1_sol, V2_sol = V_solution

    # --- Step 3: Handle Initial Conditions (t=0) ---
    # Case A: T1 == T2
    if np.isclose(T1, T2):
        # Both wake up exactly at T1. System enters V_wait immediately.
        return T1 + V_wait_sol

    # Case B: T1 < T2 (Q1 wakes first)
    elif T1 < T2:
        # At time T1, Q1 wakes up. Q2 is still resetting.
        # Q2 has remaining delay: delta = T2 - T1.
        # System state: Q1 Active, Q2 Resetting (delta remaining).
        # This maps to the logic of V2, but with delay=delta.
        delta = T2 - T1
        K_init, P_init_w, P_init_1 = get_reset_coeffs(delta, lam2, lam1, W1)

        # The future values (V_wait_sol, V1_sol) are already known steady states
        V_at_T1 = K_init + P_init_w * V_wait_sol + P_init_1 * V1_sol
        return T1 + V_at_T1

    # Case C: T2 < T1 (Q2 wakes first)
    else: # T2 < T1
        # At time T2, Q2 wakes up. Q1 has remaining delay: delta = T1 - T2.
        # System state: Q2 Active, Q1 Resetting (delta remaining).
        # This maps to the logic of V1, but with delay=delta.
        delta = T1 - T2
        K_init, P_init_w, P_init_2 = get_reset_coeffs(delta, lam1, lam2, W2)

        V_at_T2 = K_init + P_init_w * V_wait_sol + P_init_2 * V2_sol
        return T2 + V_at_T2


def wait2swap(lambdas, Ws, gammas):
    lam1, lam2 = lambdas
    W1, W2 = Ws
    gam1, gam2 = gammas

    a1 = 1-np.exp(-lam2*W1)
    a2 = 1-np.exp(-lam1*W2)

    # h = lambda a: a + (1-a)*np.log(1-a)  # entropy-like function for the expected time calculation
    h = lambda a: a + (1-a)*np.log(1-a) if a < 1 else 1 # modified to avoid log(0) when a=1, which can happen when W is large and lam is high. In that case, we can treat h(a)=1 since it's the limit as a->1.

    wait_time = (gam1*h(a1)*lam1/lam2 + gam2*h(a2)*lam2/lam1)/(a1*lam1 + a2*lam2)

    return wait_time

def swap_2Q(Q1, Q2, Tswp=0, method='simple', capacity='unit'):
    """
    Calculates the swapping rate for two queues with
    Poisson arrivals, unit capacity, deadlines W, and reset delays T.

    Parameters:
    Q1, Q2 : tuple : (Q, λ, tau, W, T, f) for Queue 1 and Queue 2
    Tswp    : float : Additional swapping time (default=0)

    Returns:
    Expected rate, expected wait time until swap.
    """
    m1, lam1_list, _, W1, T1, f1 = Q1 # Renamed lam1 to lam1_list to reflect its type
    m2, lam2_list, _, W2, T2, f2 = Q2 # Renamed lam2 to lam2_list to reflect its type

    mem = [m1, m2]
    if mem == [1,1] and capacity in ['single', 'unit', '1']: # use simple formula for single memory case
        # Extract scalar rates for single memory case
        scalar_lam1 = lam1_list[0] if isinstance(lam1_list, list) else lam1_list
        scalar_lam2 = lam2_list[0] if isinstance(lam2_list, list) else lam2_list

        if method == 'simple':
            R= simple_rate(scalar_lam1, T1, W1, scalar_lam2, T2, W2, Tswp=Tswp)
        else: # solve_expected_match_time
            R= 1/(Tswp + solve_expected_match_time(scalar_lam1, scalar_lam2, W1, W2, T1, T2))

        wait_time = wait2swap((scalar_lam1, scalar_lam2), (W1, W2), (sum(f1['decohere_rates']), sum(f2['decohere_rates'])))

        return R, wait_time
    else: # for multiple memories, use CTMC model
        # print('\n------ CTMC ------')
        # set up rate vectors:
        # Original lam1 and lam2 here are already lists (lam1_list, lam2_list)
        lambda1 = lam1_list if isinstance(lam1_list, list) else [lam1_list]*m1 # Ensure it's a list for CTMC
        lambda2 = lam2_list if isinstance(lam2_list, list) else [lam2_list]*m2 # Ensure it's a list for CTMC

        if method == 'simple':
            res = solve_simplified_case(mem[0], mem[1], lambda1, lambda2, 1/W1, 1/W2)
        else: # use more accurate CTMC model for multiple memories, which requires solving the generator matrix.
            res = solve_general_case(mem[0], mem[1], lambda1, lambda2, 1/W1, 1/W2, Tswp, T1, T2)

        R = res['match_rate']
        wait_time = sum(f1['decohere_rates']) * res['E_tau1'] + sum(f2['decohere_rates']) * res['E_tau2']

        return R, wait_time


# CTMC solvers

def solve_simplified_case(K1, K2, lam1, lam2, theta1, theta2):
    """
    Solves the simplified case (T_i -> 0).
    Returns dictionary with metrics.
    """
    min_n = -K2
    max_n = K1
    size = max_n - min_n + 1

    p_unnorm = np.zeros(size)
    idx_0 = -min_n
    p_unnorm[idx_0] = 1.0

    # Forward P_n calc
    for n in range(1, max_n + 1):
        idx = n + idx_0
        rate_up = lam1[n-1]
        rate_down = lam2[0] + n * theta1
        p_unnorm[idx] = p_unnorm[idx-1] * (rate_up / rate_down)

    for n in range(-1, min_n - 1, -1):
        idx = n + idx_0
        items_q2 = abs(n + 1)
        rate_down = lam2[items_q2]
        rate_up = lam1[0] + abs(n) * theta2
        p_unnorm[idx] = p_unnorm[idx+1] * (rate_down / rate_up)

    p = p_unnorm / np.sum(p_unnorm)

    match_rate = 0.0
    L_q1 = 0.0
    L_q2 = 0.0
    Lam1_eff = 0.0
    Lam2_eff = 0.0

    for n in range(min_n, max_n + 1):
        idx = n + idx_0
        prob = p[idx]

        if n < 0: match_rate += prob * lam1[0]
        elif n > 0: match_rate += prob * lam2[0]

        # Queue Lengths
        if n > 0: L_q1 += n * prob
        if n < 0: L_q2 += abs(n) * prob

        # Effective Arrival Rates
        # Q1 enters if n < K1.
        if n < K1:
            q1_items = n if n > 0 else 0
            Lam1_eff += prob * lam1[q1_items]

        # Q2 enters if |n| < K2 -> n > -K2
        if n > -K2:
            q2_items = abs(n) if n < 0 else 0
            Lam2_eff += prob * lam2[q2_items]

    # T1 = T2 = 0 for simplified case
    E_tau1 = (L_q1 / Lam1_eff) if Lam1_eff > 0 else 0
    E_tau2 = (L_q2 / Lam2_eff) if Lam2_eff > 0 else 0

    return {
        'match_rate': match_rate,
        'E_tau1': E_tau1,
        'E_tau2': E_tau2
    }


def solve_general_case(K1, K2, lam1, lam2, theta1, theta2, Tp, T1, T2):
    """
    Solves the general case using a Continuous Time Markov Chain (CTMC).
    Returns a dictionary containing rates and age metrics.
    Includes a shared processing phase Tp prior to individual delays.
    """
    mup = 1.0 / Tp if Tp > 0 else 1e99
    mu1 = 1.0 / T1 if T1 > 0 else 1e99
    mu2 = 1.0 / T2 if T2 > 0 else 1e99

    # 1. Generate State Space
    # State = (n, mp, m1, m2)
    states = []
    state_to_idx = {}

    for n in range(-K2, K1 + 1):
        for mp in range(min(K1, K2) + 1):
            max_m1 = K1 - n - mp if n > 0 else K1 - mp
            max_m2 = K2 - abs(n) - mp if n < 0 else K2 - mp

            if max_m1 < 0 or max_m2 < 0:
                continue

            for m1 in range(max_m1 + 1):
                for m2 in range(max_m2 + 1):
                    s = (n, mp, m1, m2)
                    state_to_idx[s] = len(states)
                    states.append(s)

    num_states = len(states)

    # 2. Build Generator Matrix Q
    data = []
    row_ind = []
    col_ind = []

    def add_rate(i, j, rate):
        if rate > 0:
            data.append(rate)
            row_ind.append(i)
            col_ind.append(j)

    diag_rates = np.zeros(num_states)

    for i, (n, mp, m1, m2) in enumerate(states):
        # Q1 Arrivals
        #
        # lam1[i] is the aggregate arrival rate when i Q1 memories are already
        # unavailable.  Unavailable memories include waiting items plus memories
        # locked in shared swap processing (mp) and endpoint reset/result delay
        # (m1).  Using lam1[0] while mp/m1 > 0 overestimates multiplexed rates.
        if n >= 0:
            unavailable_q1 = n + mp + m1
            if unavailable_q1 < K1:
                rate = lam1[unavailable_q1]
                if rate > 0 and (n + 1) + mp + m1 <= K1:
                    next_s = (n + 1, mp, m1, m2)
                    if next_s in state_to_idx:
                        add_rate(i, state_to_idx[next_s], rate)
                        diag_rates[i] += rate
        else: # n < 0
            unavailable_q1 = mp + m1
            rate = lam1[unavailable_q1] if unavailable_q1 < K1 else 0
            if rate > 0 and unavailable_q1 + 1 <= K1:
                next_s = (n + 1, mp + 1, m1, m2)
                if next_s in state_to_idx:
                    add_rate(i, state_to_idx[next_s], rate)
                    diag_rates[i] += rate

        # Q2 Arrivals; same convention as Q1.
        if n <= 0:
            items_q2 = abs(n)
            unavailable_q2 = items_q2 + mp + m2
            if unavailable_q2 < K2:
                rate = lam2[unavailable_q2]
                if rate > 0 and items_q2 + 1 + mp + m2 <= K2:
                    next_s = (n - 1, mp, m1, m2)
                    if next_s in state_to_idx:
                        add_rate(i, state_to_idx[next_s], rate)
                        diag_rates[i] += rate
        else: # n > 0
            unavailable_q2 = mp + m2
            rate = lam2[unavailable_q2] if unavailable_q2 < K2 else 0
            if rate > 0 and unavailable_q2 + 1 <= K2:
                next_s = (n - 1, mp + 1, m1, m2)
                if next_s in state_to_idx:
                    add_rate(i, state_to_idx[next_s], rate)
                    diag_rates[i] += rate

        # Shared Processing Exits
        if mp > 0:
            rate = mp * mup
            next_s = (n, mp - 1, m1 + 1, m2 + 1)
            if next_s in state_to_idx:
                add_rate(i, state_to_idx[next_s], rate)
                diag_rates[i] += rate

        # Reneging (Bypasses shared phase)
        if n > 0:
            rate = n * theta1
            next_s = (n - 1, mp, m1 + 1, m2)
            if next_s in state_to_idx:
                add_rate(i, state_to_idx[next_s], rate)
                diag_rates[i] += rate
        if n < 0:
            rate = abs(n) * theta2
            next_s = (n + 1, mp, m1, m2 + 1)
            if next_s in state_to_idx:
                add_rate(i, state_to_idx[next_s], rate)
                diag_rates[i] += rate

        # Individual Exits
        if m1 > 0:
            rate = m1 * mu1
            next_s = (n, mp, m1 - 1, m2)
            if next_s in state_to_idx:
                add_rate(i, state_to_idx[next_s], rate)
                diag_rates[i] += rate
        if m2 > 0:
            rate = m2 * mu2
            next_s = (n, mp, m1, m2 - 1)
            if next_s in state_to_idx:
                add_rate(i, state_to_idx[next_s], rate)
                diag_rates[i] += rate

    for i in range(num_states):
        data.append(-diag_rates[i])
        row_ind.append(i)
        col_ind.append(i)

    # 3. Solve Linear System
    Q = csr_matrix((data, (row_ind, col_ind)), shape=(num_states, num_states))
    A = Q.transpose().tolil()
    A[-1, :] = 1.0
    A = A.tocsr()
    b = np.zeros(num_states)
    b[-1] = 1.0

    try:
        pi = spsolve(A, b)
    except Exception as e:
        print("Matrix Solver Error:", e)
        return None

    # 4. Calculate Metrics
    match_rate = 0.0
    L_q1 = 0.0
    L_q2 = 0.0
    L_mp = 0.0
    L_d1 = 0.0
    L_d2 = 0.0
    Lam1_eff = 0.0
    Lam2_eff = 0.0

    for i, (n, mp, m1, m2) in enumerate(states):
        prob = pi[i]

        # Match Rates
        if n < 0 and mp + m1 < K1: match_rate += prob * lam1[mp + m1]
        if n > 0 and mp + m2 < K2: match_rate += prob * lam2[mp + m2]

        # Queue Lengths (waiting only)
        if n > 0: L_q1 += n * prob
        if n < 0: L_q2 += abs(n) * prob

        # Post-matching lengths
        L_mp += mp * prob
        L_d1 += m1 * prob
        L_d2 += m2 * prob

        # Effective Arrival Rates
        if n > 0:
            unavailable_q1 = n + mp + m1
            if unavailable_q1 < K1: Lam1_eff += prob * lam1[unavailable_q1]
        else:
            unavailable_q1 = mp + m1
            if unavailable_q1 < K1: Lam1_eff += prob * lam1[unavailable_q1]

        if n < 0:
            unavailable_q2 = abs(n) + mp + m2
            if unavailable_q2 < K2: Lam2_eff += prob * lam2[unavailable_q2]
        else:
            unavailable_q2 = mp + m2
            if unavailable_q2 < K2: Lam2_eff += prob * lam2[unavailable_q2]

    # Use exact Little's Law covering all phases in the system
    # NOTE: Ignore the L_mp and L_di terms as they are counted in A0 term in paper
    E_tau1 = (L_q1 + 0*L_mp + 0*L_d1) / Lam1_eff if Lam1_eff > 0 else 0
    E_tau2 = (L_q2 + 0*L_mp + 0*L_d2) / Lam2_eff if Lam2_eff > 0 else 0

    return {
        'match_rate': match_rate,
        'E_tau1': E_tau1,
        'E_tau2': E_tau2
    }
