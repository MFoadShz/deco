"""
=====================================================================
 Decoherence in a Two-Layer Spin-Chain Environment  --  ANALYTIC BUILD
=====================================================================

This Streamlit application reproduces every feature of the original
"brute-force" notebook (case selection, frequency modes, comparison
panel, log view, verification tab, etc.), but the decoherence factors
are computed from the *closed-form analytic* formulas derived in the
accompanying LaTeX document (see Sections "Case 1", "Case 2", "Case 3"
and "Case 4 - Combined formulation").

The numerical SVD / partial-trace route is still kept in this single
file -- but ONLY as a verification tool inside the "Analytic
verification" tab.  The plotted curves use the analytic expressions
exclusively.

Key analytic objects (per branch i, with c_i = cos(omega_i t),
s_i = sin(omega_i t)/omega_i, omega_i = sqrt(g_i^2 + J_i^2)) :

    gamma_i(t)            scalar coherence factor (S only)
    nu_i^{(L1)}(t)        trace norm of 2x2 block (S + L1)
    nu_i^{(L2)}(t)        trace norm of 2x2 block (S + L2)
    nu_i^{(full)} = 1     fully observed chain

The full decoherence factor is the product of the per-branch factors,
exactly as in eq. (Dtotal-full) of the LaTeX document.

The frequency generator now separates three mathematically different
irrational regimes:

    1. well-approximable incommensurate ratios, placed near an integer
       ladder so finite-time near-revivals are visible;
    2. a metallic quadratic-irrational family kept as a useful
       badly-approximable-inspired baseline;
    3. a research-grade algebraic power-basis construction whose
       frequency-ratio vector is a standard simultaneously badly
       approximable vector, with a finite-window Diophantine diagnostic
       reported inside the app.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from numpy.linalg import svd
from matplotlib.lines import Line2D

try:
    import sympy as sp
    _HAS_SYMPY = True
except Exception:
    sp = None
    _HAS_SYMPY = False


# =====================================================================
# Streamlit page
# =====================================================================

st.set_page_config(
    page_title="Decoherence in a Two-Layer Spin Chain (Analytic)",
    layout="wide",
)


# =====================================================================
# Basic Pauli matrices  (used ONLY by the verification routine)
# =====================================================================

I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
I4 = np.eye(4, dtype=complex)


def rho0_matrix(ax, ay, az):
    return 0.5 * (I2 + ax * X + ay * Y + az * Z)


def H_branch(g, J, m):
    return m * g * np.kron(Z, I2) + J * np.kron(X, Z)


def U_branch(g, J, m, t):
    H = H_branch(g, J, m)
    omega = np.sqrt(g**2 + J**2)
    if omega == 0:
        return I4.copy()
    c = np.cos(omega * t)
    s = np.sin(omega * t) / omega
    return c * I4 - 1j * s * H


def trace_over_L2(M):
    M4 = M.reshape(2, 2, 2, 2)
    return np.einsum("abcb->ac", M4)


def trace_over_L1(M):
    M4 = M.reshape(2, 2, 2, 2)
    return np.einsum("abad->bd", M4)


def trace_norm(M):
    return float(np.sum(svd(M, compute_uv=False)))


def branch_factors_numeric(g, J, ax, ay, az, t):
    """Reference 'brute force' implementation used only for verification."""
    r0 = rho0_matrix(ax, ay, az)
    R0 = np.kron(r0, r0)
    Up = U_branch(g, J, +1, t)
    Um = U_branch(g, J, -1, t)
    G  = Up @ R0 @ Um.conj().T
    gamma_abs = abs(np.trace(G))
    nu_L1 = trace_norm(trace_over_L2(G))
    nu_L2 = trace_norm(trace_over_L1(G))
    nu_full = trace_norm(G)
    return gamma_abs, nu_L1, nu_L2, nu_full


# =====================================================================
# ANALYTIC closed-form per-branch factors
# (from Sections 2-5 of the LaTeX document)
# =====================================================================

def _cs(omega, t):
    """Return (cos, sin/omega) handling the omega -> 0 limit."""
    if omega == 0:
        return np.ones_like(t), np.zeros_like(t)
    return np.cos(omega * t), np.sin(omega * t) / omega


def gamma_abs2_analytic(g, J, ax, ay, az, t):
    """
    |gamma_i(t)|^2  -- scalar coherence factor when the whole branch
    is traced out.  Eq. (Case 3).

        gamma_i(t) = 1 - 2 g^2 s^2  -  2 i g s ( c a_z + J s a_y a_z )
    """
    omega = np.sqrt(g**2 + J**2)
    c, s  = _cs(omega, t)
    Re = 1.0 - 2.0 * g**2 * s**2
    Im = -2.0 * g * s * (c * az + J * s * ay * az)
    return Re * Re + Im * Im


def nu_L1_sq_analytic(g, J, ax, ay, az, t):
    """
    [ nu_i^{(L1)}(t) ]^2  =  S_i + 2*sqrt(X_i^2 + Y_i^2)   (Section 2.2)
    """
    omega = np.sqrt(g**2 + J**2)
    c, s  = _cs(omega, t)

    s2 = s * s
    s4 = s2 * s2
    az2 = az * az
    az4 = az2 * az2
    ay2 = ay * ay
    a2  = ax * ax + ay * ay + az2          # |a|^2
    J2  = J * J
    J4  = J2 * J2
    g2  = g * g

    X = (
        0.25 * (1.0 - a2)
        + s2 * J2 * (ay2 - ay2 * az2 + az2 - az4)
        + s4 * (
            J4 * (ay2 * az2 - ay2 + az4 - az2)
            + g2 * J2 * (az4 - 1.0)
        )
    )
    Y = 2.0 * g * J2 * c * s2 * s * az * (az2 - 1.0)
    S = (
        0.5 * (1.0 + a2)
        + 2.0 * s2 * J2 * (ay2 * az2 - ay2 + az4 - az2)
        + 2.0 * s4 * (
            J4 * (-ay2 * az2 + ay2 - az4 + az2)
            + g2 * J2 * (-az4 + 2.0 * az2 - 1.0)
        )
    )
    return S + 2.0 * np.sqrt(X * X + Y * Y)


def nu_L2_sq_analytic(g, J, ax, ay, az, t):
    """
    [ nu_i^{(L2)}(t) ]^2  =  S_j + 2*sqrt(X_j^2 + Y_j^2)   (Section 3)
    """
    omega = np.sqrt(g**2 + J**2)
    c, s  = _cs(omega, t)

    c2 = c * c
    s2 = s * s
    s3 = s2 * s
    g2 = g * g
    J2 = J * J
    az2 = az * az
    ax2 = ax * ax
    ay2 = ay * ay
    axy2 = ax2 + ay2

    one_m_2gs = 1.0 - 2.0 * g2 * s2
    Delta = c2 - (g2 + J2) * s2

    S = (
        0.5 * (1.0 + az2) * one_m_2gs**2
        + 2.0 * g2 * s2 * (1.0 + az2) * (c2 * az2 + J2 * s2 * ay2)
        + 8.0 * g2 * J * c * s3 * ay * az2
        + 0.5 * axy2 * Delta**2
        + 2.0 * c2 * s2 * axy2 * (J2 * ax2 + g2 * az2)
    )
    X = (
        0.25 * (1.0 - az2) * one_m_2gs**2
        - g2 * s2 * (1.0 - az2) * (c2 * az2 - J2 * s2 * ay2)
        - 0.25 * axy2 * Delta**2
        - c2 * s2 * axy2 * (J2 * ax2 - g2 * az2)
    )
    Y = -g * c * s * az * (
        (1.0 - az2) * one_m_2gs - axy2 * Delta
    )
    return S + 2.0 * np.sqrt(X * X + Y * Y)


# =====================================================================
# Vectorised helpers: take an array of time points
# =====================================================================

def gamma_abs2_vec(g, J, ax, ay, az, t_grid):
    return gamma_abs2_analytic(g, J, ax, ay, az, t_grid)


def nu_L1_sq_vec(g, J, ax, ay, az, t_grid):
    return nu_L1_sq_analytic(g, J, ax, ay, az, t_grid)


def nu_L2_sq_vec(g, J, ax, ay, az, t_grid):
    return nu_L2_sq_analytic(g, J, ax, ay, az, t_grid)


# =====================================================================
# Stable log-sum product of positive arrays
# =====================================================================

def log_sum_clip(arr, floor=1e-300):
    arr = np.clip(arr, floor, None)
    return np.log(arr)


# =====================================================================
# Frequency-mode generation
# =====================================================================

FREQUENCY_MODE_TEXT = {
    "identical": {
        "title": "Identical branches",
        "desc": (
            "All branches have exactly the same frequency scale. "
            "This is the cleanest finite-environment situation."
        ),
        "expect": (
            "Strong coherence collapse away from t = 0 and a sharp "
            "exact revival when all branches rephase together."
        ),
    },
    "integer_commensurate": {
        "title": "Integer commensurate frequencies",
        "desc": (
            "Branch frequencies are integer multiples of one base "
            "frequency.  Not identical, but they share a common period."
        ),
        "expect": "Exact revival, richer pre-revival oscillation pattern.",
    },
    "rational_commensurate": {
        "title": "Rational commensurate frequencies",
        "desc": (
            "Frequency ratios are rationals like 1, 3/2, 2, 5/2.  A "
            "common period still exists but it may appear later."
        ),
        "expect": "Exact revival, generally later than the integer case.",
    },
    "incommensurate": {
        "title": "Well-approximable incommensurate frequencies",
        "desc": (
            "Ratios are irrational but deliberately close to an "
            "integer-commensurate ladder: n + eps*sqrt(p_n), with eps "
            "chosen rationally.  This keeps the ratios non-commensurate "
            "while making rational near-rephasings visible at small N."
        ),
        "expect": (
            "No exact revival.  Finite-time near-revivals are relatively "
            "visible because the ratios were placed close to a simple "
            "rational ladder."
        ),
    },
    "badly_approximable": {
        "title": "Algebraic simultaneously badly-approximable frequencies",
        "desc": (
            "Research-grade construction.  A real algebraic number beta "
            "of degree N is selected from an irreducible trinomial "
            "x^N - a x - 1.  The frequency ratios are proportional to "
            "1, beta, beta^2, ..., beta^(N-1).  The vector "
            "(beta, beta^2, ..., beta^(N-1)) is the standard algebraic "
            "power-basis example of a simultaneously badly approximable "
            "vector."
        ),
        "expect": (
            "No exact revival and systematically suppressed finite-time "
            "near-rephasing compared with the well-approximable "
            "incommensurate family.  The app reports a finite-window "
            "Diophantine score c_Q as a numerical diagnostic."
        ),
    },
    "metallic_ba_inspired": {
        "title": "Metallic badly-approximable-inspired frequencies",
        "desc": (
            "Legacy baseline retained for comparison.  Ratios use "
            "metallic quadratic irrationals m_n=(n+sqrt(n^2+4))/2, "
            "normalised to the same mean scale.  Each raw m_n is a "
            "quadratic irrational with bounded periodic continued "
            "fraction, but this does not certify that the full frequency "
            "vector is simultaneously badly approximable."
        ),
        "expect": (
            "Usually weaker near-revivals than the well-approximable "
            "incommensurate family, but it should be described as a "
            "BA-inspired baseline rather than a rigorous vector-level "
            "construction."
        ),
    },
    "weak_disorder": {
        "title": "Weak disorder",
        "desc": (
            "Branches are almost identical but with a small random "
            "perturbation.  Same frequency scale, broken synchrony."
        ),
        "expect": (
            "The clean revival of the identical case becomes broadened, "
            "weakened, or destroyed."
        ),
    },
}


def _first_primes(n):
    """Return the first n primes; small helper for frequency families."""
    primes = []
    x = 2
    while len(primes) < n:
        is_prime = True
        for p in range(2, int(np.sqrt(x)) + 1):
            if x % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(x)
        x += 1
    return primes


def _normalise_to_unit_mean(values):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return []
    mean = float(np.mean(values))
    if mean == 0:
        return list(values)
    return list(values / mean)


def _params_from_ratios(g0, J0, ratios):
    """Convert dimensionless frequency ratios into branch couplings."""
    ratios = np.array(ratios, dtype=float)
    return [(g0 * float(r), J0 * float(r)) for r in ratios]


def _frequency_info(mode, ratios, construction, claim, **metadata):
    ratios = np.array(ratios, dtype=float)
    return {
        "mode": mode,
        "ratios": ratios,
        "construction": construction,
        "claim": claim,
        **metadata,
    }


def _well_approximable_incommensurate_ratios(N, eps_num=35, eps_den=1000):
    """
    Irrational but intentionally well-approximable ratios.

    Construction:
        r_n = n + eps*sqrt(p_n),  n = 1,...,N,
        eps = eps_num / eps_den.

    The integer ladder gives a nearby common period, while the sqrt(p_n)
    perturbation makes the ratios genuinely irrational at the mathematical
    construction level.  This is useful as a controlled contrast with the
    badly-approximable mode: both are non-commensurate, but this family admits
    much better finite-time near-rephasings.
    """
    if N <= 0:
        return []
    primes = _first_primes(N)
    ladder = np.arange(1, N + 1, dtype=float)
    eps = float(eps_num) / float(eps_den)
    raw = ladder + eps * np.sqrt(np.array(primes, dtype=float))
    return _normalise_to_unit_mean(raw)


def _metallic_badly_approximable_ratios(N):
    """
    Metallic quadratic irrationals, normalised to unit mean.

    m_n = (n + sqrt(n^2 + 4))/2 has periodic continued fraction
    [n; n, n, n, ...].  Each m_n is a quadratic irrational and is individually
    badly approximable.  This is retained as a useful BA-inspired baseline, but
    it is not a rigorous certificate that the whole N-frequency vector is
    simultaneously badly approximable.
    """
    if N <= 0:
        return []
    n = np.arange(1, N + 1, dtype=float)
    raw = 0.5 * (n + np.sqrt(n * n + 4.0))
    return _normalise_to_unit_mean(raw)


def _trinomial_coefficients(degree, a):
    """Coefficients of x^degree - a*x - 1 in descending order."""
    if degree < 2:
        return [1.0, -1.0]
    return [1.0] + [0.0] * (degree - 2) + [-float(a), -1.0]


def _real_root_larger_than_one_of_trinomial(degree, a):
    """Numerically find the real root beta > 1 of x^degree - a*x - 1."""
    coeffs = _trinomial_coefficients(degree, a)
    roots = np.roots(coeffs)
    real_roots = [float(z.real) for z in roots
                  if abs(z.imag) < 1e-8 and z.real > 1.0]
    if not real_roots:
        raise ValueError(
            f"Could not find a real root > 1 for x^{degree} - {a} x - 1."
        )
    return max(real_roots)


def _is_irreducible_trinomial_over_Q(degree, a):
    """
    Return True/False if SymPy is available; otherwise return None.

    Irreducibility over Q is the important algebraic certificate: if the
    polynomial has exact degree N and beta is one of its roots, then beta has
    algebraic degree N.  The power basis (beta, ..., beta^(N-1)) is then the
    standard simultaneous badly-approximable vector.
    """
    if not _HAS_SYMPY:
        return None
    x = sp.Symbol("x")
    poly = sp.Poly(x**degree - int(a) * x - 1, x, domain=sp.QQ)
    return bool(poly.is_irreducible)


def _select_irreducible_trinomial_parameter(degree, seed=123,
                                             a_min=1, a_max=80):
    """
    Pick an integer a such that x^degree - a*x - 1 is irreducible over Q.

    The search is deterministic for a given seed.  With SymPy installed the
    irreducibility check is exact.  Without SymPy, the code falls back to a=1
    and labels the certificate as unavailable rather than pretending to prove
    irreducibility.
    """
    if degree < 2:
        return 1, None

    if not _HAS_SYMPY:
        return 1, None

    rng = np.random.default_rng(seed)
    candidates = np.arange(int(a_min), int(a_max) + 1, dtype=int)
    rng.shuffle(candidates)

    for a in candidates:
        if _is_irreducible_trinomial_over_Q(degree, int(a)):
            return int(a), True

    # Deterministic fallback search outside the original window.
    for a in range(int(a_max) + 1, int(a_max) + 500):
        if _is_irreducible_trinomial_over_Q(degree, int(a)):
            return int(a), True

    raise RuntimeError(
        "No irreducible trinomial x^N - a*x - 1 was found in the search range."
    )


def _algebraic_simultaneously_badly_ratios(N, seed=123):
    """
    Research-grade simultaneous badly-approximable frequency family.

    For N branches choose beta as the real root > 1 of an irreducible
    polynomial

        p(x) = x^N - a x - 1.

    Then beta has algebraic degree N.  The frequency-ratio vector relevant for
    simultaneous rephasing is

        theta = (omega_2/omega_1, ..., omega_N/omega_1)
              = (beta, beta^2, ..., beta^(N-1)).

    Since 1, beta, ..., beta^(N-1) are linearly independent over Q, the power
    basis gives the standard algebraic example of a simultaneously badly
    approximable vector.  The app additionally computes the finite-window
    diagnostic c_Q for the actually plotted time window.
    """
    if N <= 0:
        return [], {}
    if N == 1:
        return [1.0], {
            "degree": 1,
            "a": None,
            "beta": None,
            "polynomial": "trivial one-frequency case",
            "irreducible_over_Q": True,
            "certificate": "trivial",
        }

    degree = int(N)
    a, irreducible = _select_irreducible_trinomial_parameter(
        degree, seed=seed
    )
    beta = _real_root_larger_than_one_of_trinomial(degree, a)
    raw = np.array([beta**k for k in range(N)], dtype=float)
    ratios = np.array(_normalise_to_unit_mean(raw), dtype=float)

    certificate = (
        "exact SymPy irreducibility certificate over Q"
        if irreducible is True else
        "SymPy unavailable: algebraic construction used, irreducibility not certified"
    )

    metadata = {
        "degree": degree,
        "a": int(a),
        "beta": float(beta),
        "polynomial": f"x^{degree} - {int(a)} x - 1",
        "irreducible_over_Q": irreducible,
        "certificate": certificate,
    }
    return list(ratios), metadata


def _distance_to_nearest_integer(x):
    return np.abs(x - np.rint(x))


def simultaneous_badness_score_from_ratios(ratios, Qmax):
    """
    Finite-window simultaneous Diophantine diagnostic.

    For theta = (omega_2/omega_1, ..., omega_N/omega_1), compute

        c_Q = min_{1 <= q <= Qmax} q^(1/d) max_j || q theta_j ||,
        d = len(theta).

    A larger c_Q means that, up to denominator Qmax, the frequency vector is
    harder to rephase simultaneously.  This is not a proof of the infinite-q
    badly-approximable constant, but it is the physically relevant finite-time
    test for the plotted window.
    """
    ratios = np.array(ratios, dtype=float)
    if len(ratios) <= 1:
        return {
            "c_Q": np.nan,
            "q_best": None,
            "max_distance_at_q_best": np.nan,
            "dimension": 0,
            "Qmax": int(Qmax),
        }

    if ratios[0] == 0:
        raise ValueError("The first frequency ratio is zero; cannot form theta.")

    theta = ratios[1:] / ratios[0]
    d = len(theta)
    Qmax = max(1, int(Qmax))

    q_values = np.arange(1, Qmax + 1, dtype=float)
    distances = _distance_to_nearest_integer(q_values[:, None] * theta[None, :])
    max_distances = np.max(distances, axis=1)
    scaled = (q_values ** (1.0 / d)) * max_distances
    idx = int(np.argmin(scaled))

    return {
        "c_Q": float(scaled[idx]),
        "q_best": int(q_values[idx]),
        "max_distance_at_q_best": float(max_distances[idx]),
        "dimension": int(d),
        "Qmax": int(Qmax),
    }


def rephasing_candidate_from_ratios(ratios, Qmax, exact_tol=1e-12):
    """
    Report the earliest exact commensurate rephasing, if it exists within Qmax,
    and the best finite-window near-rephasing candidate.

    q is defined in units of the first branch frequency: t = q*pi/omega_1.
    The quantity time_factor_pi_over_omega0 converts this to the app's common
    unit t/(pi/omega0), where omega0 = sqrt(g0^2 + J0^2).
    """
    ratios = np.array(ratios, dtype=float)
    if len(ratios) <= 1:
        return {
            "exact_q": 1,
            "exact_time_factor_pi_over_omega0": (1.0 / ratios[0]) if len(ratios) else None,
            "best_q": 1,
            "best_time_factor_pi_over_omega0": (1.0 / ratios[0]) if len(ratios) else None,
            "best_max_phase_distance": 0.0,
            "Qmax": int(max(1, Qmax)),
        }

    if ratios[0] == 0:
        raise ValueError("The first frequency ratio is zero; cannot form theta.")

    theta = ratios[1:] / ratios[0]
    d = len(theta)
    Qmax = max(1, int(Qmax))
    q_values = np.arange(1, Qmax + 1, dtype=float)
    distances = _distance_to_nearest_integer(q_values[:, None] * theta[None, :])
    max_distances = np.max(distances, axis=1)
    scaled = (q_values ** (1.0 / d)) * max_distances

    exact_indices = np.where(max_distances <= exact_tol)[0]
    exact_q = int(q_values[int(exact_indices[0])]) if len(exact_indices) else None
    best_idx = int(np.argmin(scaled))
    best_q = int(q_values[best_idx])

    exact_L = (exact_q / ratios[0]) if exact_q is not None else None
    best_L = best_q / ratios[0]

    return {
        "exact_q": exact_q,
        "exact_time_factor_pi_over_omega0": None if exact_L is None else float(exact_L),
        "best_q": best_q,
        "best_time_factor_pi_over_omega0": float(best_L),
        "best_max_phase_distance": float(max_distances[best_idx]),
        "best_scaled_distance": float(scaled[best_idx]),
        "Qmax": int(Qmax),
    }


def suggested_Qmax_from_time_window(t_grid, params, hard_cap=25000):
    """Choose a denominator cutoff matched to the plotted time window."""
    if len(t_grid) == 0 or len(params) == 0:
        return 1
    omega1 = float(np.sqrt(params[0][0]**2 + params[0][1]**2))
    if omega1 <= 0:
        return 1
    t_extent = float(max(abs(np.min(t_grid)), abs(np.max(t_grid))))
    q_guess = int(np.ceil(max(1.0, t_extent * omega1 / np.pi)))
    return int(max(25, min(hard_cap, q_guess)))


def suggested_Qmax_strictly_inside_time_window(t_grid, params, hard_cap=25000):
    """Largest q whose candidate time t=q*pi/omega_1 lies inside the plot."""
    if len(t_grid) == 0 or len(params) == 0:
        return 1
    omega1 = float(np.sqrt(params[0][0]**2 + params[0][1]**2))
    if omega1 <= 0:
        return 1
    t_extent = float(max(abs(np.min(t_grid)), abs(np.max(t_grid))))
    q_inside = int(np.floor(max(1.0, t_extent * omega1 / np.pi)))
    return int(max(1, min(hard_cap, q_inside)))


def build_params_with_info(N, g0, J0, mode, disorder=0.05, seed=123):
    """
    Build a list of (g_i, J_i) per branch plus frequency metadata.

    The plotted curves only need the couplings, but the metadata is important
    for research use: it states what was actually constructed and what can be
    legitimately claimed about the frequency ratios.
    """
    if N <= 0:
        return [], _frequency_info(
            mode, [], "empty", "No branches were generated."
        )

    if mode == "identical":
        ratios = [1.0] * N
        info = _frequency_info(
            mode, ratios,
            "r_i = 1 for all i",
            "Exact common period; fully commensurate."
        )

    elif mode == "integer_commensurate":
        base = [1.0, 2.0, 3.0, 4.0]
        raw = [base[i % len(base)] for i in range(N)]
        ratios = _normalise_to_unit_mean(raw)
        info = _frequency_info(
            mode, ratios,
            "integer ladder repeated and normalised to unit mean",
            "Exact common period; fully commensurate."
        )

    elif mode == "rational_commensurate":
        base = [1.0, 1.5, 2.0, 2.5, 3.0]
        raw = [base[i % len(base)] for i in range(N)]
        ratios = _normalise_to_unit_mean(raw)
        info = _frequency_info(
            mode, ratios,
            "rational ladder repeated and normalised to unit mean",
            "Exact common period; fully commensurate."
        )

    elif mode == "incommensurate":
        ratios = _well_approximable_incommensurate_ratios(N, eps_num=35)
        info = _frequency_info(
            mode, ratios,
            "r_n = n + (35/1000)*sqrt(p_n), then normalised",
            "Irrational and non-commensurate, deliberately well approximable."
        )

    elif mode == "badly_approximable":
        ratios, metadata = _algebraic_simultaneously_badly_ratios(
            N, seed=seed
        )
        info = _frequency_info(
            mode, ratios,
            "algebraic power basis 1, beta, ..., beta^(N-1), normalised",
            "Vector-level simultaneous badly-approximable construction.",
            **metadata,
        )

    elif mode == "metallic_ba_inspired":
        ratios = _metallic_badly_approximable_ratios(N)
        info = _frequency_info(
            mode, ratios,
            "m_n=(n+sqrt(n^2+4))/2, normalised",
            "Each raw entry is individually badly approximable; vector-level "
            "simultaneous BA is not certified."
        )

    elif mode == "weak_disorder":
        rng = np.random.default_rng(seed)
        ratios = list(1.0 + disorder * rng.normal(size=N))
        ratios = [max(0.05, r) for r in ratios]
        ratios = _normalise_to_unit_mean(ratios)
        info = _frequency_info(
            mode, ratios,
            f"r_i = 1 + normal disorder, sigma={disorder:g}, then clipped",
            "Generic random disorder; no exact common period expected."
        )

    else:
        ratios = [1.0] * N
        info = _frequency_info(
            mode, ratios,
            "fallback identical ratios",
            "Unknown mode, used identical frequencies."
        )

    return _params_from_ratios(g0, J0, ratios), info


def build_params(N, g0, J0, mode, disorder=0.05, seed=123):
    """Backward-compatible wrapper: return only branch couplings."""
    params, _ = build_params_with_info(
        N, g0, J0, mode, disorder=disorder, seed=seed
    )
    return params


def random_params_with_info(N, g0, J0, family, rng, seed_offset=0):
    """
    Build one random frequency realisation for ensemble-style comparison.

    incommensurate:
        well-approximable irrational ratios near randomly perturbed integer
        ladders.  The perturbations are rational at the construction level.

    badly_approximable:
        true vector-level algebraic simultaneous badly approximable family,
        with a random irreducible trinomial parameter selected through the
        supplied seed.

    metallic_ba_inspired:
        legacy metallic quadratic family with random skipped metallic indices.
    """
    if N <= 0:
        return [], _frequency_info(family, [], "empty", "No branches.")

    if family == "incommensurate":
        primes = np.array(_first_primes(N), dtype=float)
        ladder = np.arange(1, N + 1, dtype=float)
        eps_num = int(rng.integers(15, 71))
        jitter_num = rng.integers(-10, 11, size=N)
        eps = eps_num / 1000.0
        jitter = jitter_num / 1000.0
        ratios = ladder + eps * np.sqrt(primes) + jitter
        ratios = np.array(_normalise_to_unit_mean(ratios), dtype=float)
        info = _frequency_info(
            family, ratios,
            "random rationally-jittered n + eps*sqrt(p_n) ladder",
            "Irrational and non-commensurate, deliberately well approximable.",
            eps_num=int(eps_num),
        )

    elif family == "badly_approximable":
        # The seed is generated outside from the trial number to keep runs
        # reproducible while still sampling different irreducible trinomials.
        algebraic_seed = int(rng.integers(0, 2**31 - 1)) + int(seed_offset)
        ratios, metadata = _algebraic_simultaneously_badly_ratios(
            N, seed=algebraic_seed
        )
        ratios = np.array(ratios, dtype=float)
        info = _frequency_info(
            family, ratios,
            "random algebraic power basis from irreducible x^N-a*x-1",
            "Vector-level simultaneous badly-approximable construction.",
            algebraic_seed=algebraic_seed,
            **metadata,
        )

    elif family == "metallic_ba_inspired":
        n_values = rng.choice(np.arange(1, max(4 * N + 4, 12)),
                              size=N, replace=False)
        n_values = np.sort(n_values).astype(float)
        ratios = 0.5 * (n_values + np.sqrt(n_values * n_values + 4.0))
        ratios = np.array(_normalise_to_unit_mean(ratios), dtype=float)
        info = _frequency_info(
            family, ratios,
            "random skipped metallic quadratic irrationals",
            "Individually BA entries; simultaneous vector-level BA not certified."
        )

    else:
        ratios = np.ones(N)
        info = _frequency_info(
            family, ratios,
            "fallback identical ratios",
            "Unknown random family."
        )

    return _params_from_ratios(g0, J0, ratios), info


def random_params(N, g0, J0, family, rng):
    """Backward-compatible wrapper: return only branch couplings."""
    params, _ = random_params_with_info(N, g0, J0, family, rng)
    return params


def revival_score(t_grid, y, cut_fraction=0.08):
    """
    Maximal near-revival after removing the initial neighbourhood of t = 0.

    Larger value means the curve came closer to recohering inside the plotted
    time window.  The early-time cut prevents the trivial value D^2(0)=1 from
    dominating the score.
    """
    if len(t_grid) < 2:
        return float(np.max(y))

    t_cut = t_grid[0] + cut_fraction * (t_grid[-1] - t_grid[0])
    mask = t_grid > t_cut
    if not np.any(mask):
        return float(np.max(y))
    return float(np.max(y[mask]))


# =====================================================================
# Decoherence curves -- ALL analytic, vectorised in time
# =====================================================================

def curve_S_only(params, ax, ay, az, t_grid):
    """ D^2_S(t) = prod_i |gamma_i|^2 """
    log_sum = np.zeros_like(t_grid)
    for g, J in params:
        log_sum += log_sum_clip(gamma_abs2_vec(g, J, ax, ay, az, t_grid))
    return np.exp(log_sum)


def curve_S_plus_L1(params, ax, ay, az, t_grid):
    log_sum = np.zeros_like(t_grid)
    for g, J in params:
        log_sum += log_sum_clip(nu_L1_sq_vec(g, J, ax, ay, az, t_grid))
    return np.exp(log_sum)


def curve_S_plus_L2(params, ax, ay, az, t_grid):
    log_sum = np.zeros_like(t_grid)
    for g, J in params:
        log_sum += log_sum_clip(nu_L2_sq_vec(g, J, ax, ay, az, t_grid))
    return np.exp(log_sum)


def curve_S_plus_observed_full(params, ax, ay, az, t_grid, observed_full_count):
    """Observed full chains contribute 1; the rest contribute |gamma|^2."""
    N = len(params)
    n_obs = max(0, min(N, observed_full_count))
    log_sum = np.zeros_like(t_grid)
    for i, (g, J) in enumerate(params):
        if i >= n_obs:
            log_sum += log_sum_clip(gamma_abs2_vec(g, J, ax, ay, az, t_grid))
    return np.exp(log_sum)


def curve_combination(params, ax, ay, az, t_grid,
                      n_unobserved, n_only_L1, n_only_L2):
    N = len(params)
    n_unobserved = max(0, n_unobserved)
    n_only_L1    = max(0, n_only_L1)
    n_only_L2    = max(0, n_only_L2)
    n_full       = N - n_unobserved - n_only_L1 - n_only_L2
    if n_full < 0:
        return None

    log_sum = np.zeros_like(t_grid)
    for i, (g, J) in enumerate(params):
        if i < n_unobserved:
            log_sum += log_sum_clip(gamma_abs2_vec(g, J, ax, ay, az, t_grid))
        elif i < n_unobserved + n_only_L1:
            log_sum += log_sum_clip(nu_L1_sq_vec(g, J, ax, ay, az, t_grid))
        elif i < n_unobserved + n_only_L1 + n_only_L2:
            log_sum += log_sum_clip(nu_L2_sq_vec(g, J, ax, ay, az, t_grid))
        # full chains -> factor 1, add log(1) = 0
    return np.exp(log_sum)


def compute_selected_curve(case_name, params, ax, ay, az, t_grid,
                           observed_full_count,
                           n_unobserved, n_only_L1, n_only_L2):
    if case_name == "S only":
        return curve_S_only(params, ax, ay, az, t_grid)
    if case_name == "S + L1":
        return curve_S_plus_L1(params, ax, ay, az, t_grid)
    if case_name == "S + L2":
        return curve_S_plus_L2(params, ax, ay, az, t_grid)
    if case_name == "S + observed full chains":
        return curve_S_plus_observed_full(
            params, ax, ay, az, t_grid, observed_full_count
        )
    if case_name == "Combination":
        return curve_combination(
            params, ax, ay, az, t_grid,
            n_unobserved, n_only_L1, n_only_L2,
        )
    return curve_S_only(params, ax, ay, az, t_grid)


# =====================================================================
# Numerical verification (only used inside the verification tab)
# =====================================================================

def compute_selected_curve_numeric(case_name, params, ax, ay, az, t_grid,
                                   observed_full_count,
                                   n_unobserved, n_only_L1, n_only_L2):
    """Reference brute-force route via SVD and partial traces."""
    y = np.zeros_like(t_grid)
    N = len(params)

    for ti, t in enumerate(t_grid):
        log_acc = 0.0
        for i, (g, J) in enumerate(params):
            ga, n1, n2, nf = branch_factors_numeric(g, J, ax, ay, az, t)

            if case_name == "S only":
                f = ga * ga

            elif case_name == "S + L1":
                f = n1 * n1

            elif case_name == "S + L2":
                f = n2 * n2

            elif case_name == "S + observed full chains":
                if i < min(observed_full_count, N):
                    f = nf * nf      # ~ 1
                else:
                    f = ga * ga

            elif case_name == "Combination":
                if i < n_unobserved:
                    f = ga * ga
                elif i < n_unobserved + n_only_L1:
                    f = n1 * n1
                elif i < n_unobserved + n_only_L1 + n_only_L2:
                    f = n2 * n2
                else:
                    f = nf * nf
            else:
                f = ga * ga

            log_acc += np.log(max(f, 1e-300))
        y[ti] = np.exp(log_acc)
    return y


# =====================================================================
# Explanations / formulas / star-model diagram
# =====================================================================

def case_formula(case_name):
    if case_name == "S only":
        return (r"\mathcal D_S^2(t)=\prod_i |\gamma_i(t)|^2"
                r"=\prod_i\bigl[(1-2g_i^2s_i^2)^2"
                r"+4g_i^2s_i^2(c_i a_z+J_is_ia_ya_z)^2\bigr]")
    if case_name == "S + L1":
        return (r"\mathcal D_{L1}^2(t)=\prod_i\bigl[S_i^{(1)}"
                r"+2\sqrt{(X_i^{(1)})^2+(Y_i^{(1)})^2}\,\bigr]")
    if case_name == "S + L2":
        return (r"\mathcal D_{L2}^2(t)=\prod_j\bigl[S_j^{(2)}"
                r"+2\sqrt{(X_j^{(2)})^2+(Y_j^{(2)})^2}\,\bigr]")
    if case_name == "S + observed full chains":
        return (r"\mathcal D_{\mathcal O}^2(t)"
                r"=\prod_{k\in\mathcal U}|\gamma_k(t)|^2")
    if case_name == "Combination":
        return (r"\mathcal D_{\mathrm{comb}}^2(t)"
                r"=\!\!\prod_{k\in\mathcal U}\!\!|\gamma_k|^2"
                r"\!\prod_{i\in\mathcal L_1}\!(\nu_i^{(L1)})^2"
                r"\!\prod_{j\in\mathcal L_2}\!(\nu_j^{(L2)})^2")
    return ""


def case_description(case_name):
    if case_name == "S only":
        return ("All environmental branches are traced out. The "
                "coherence is controlled by scalar factors "
                r"$\gamma_i(t)$.")
    if case_name == "S + L1":
        return ("Keep the first layer L1 of every branch; trace out "
                "L2. Direct dephasing channel.")
    if case_name == "S + L2":
        return ("Keep the second layer L2; trace out L1. S influences "
                "L2 only indirectly through L1 and the coupling J.")
    if case_name == "S + observed full chains":
        return ("Some full chains are kept (each contributing a "
                "trace-norm factor 1); the rest are completely traced "
                "out.")
    if case_name == "Combination":
        return ("Different branches are treated differently: some are "
                "traced out, some keep only L1, some keep only L2, "
                "some are fully observed.")
    return ""


def what_to_expect(case_name, mode_name):
    return FREQUENCY_MODE_TEXT[mode_name]["expect"]


def statuses_for_case(case_name, N, observed_full_count,
                      n_unobserved, n_only_L1, n_only_L2):
    if case_name == "S only":
        return ["unobserved"] * N
    if case_name == "S + L1":
        return ["only_l1"] * N
    if case_name == "S + L2":
        return ["only_l2"] * N
    if case_name == "S + observed full chains":
        return [
            "full_observed" if i < observed_full_count else "unobserved"
            for i in range(N)
        ]
    if case_name == "Combination":
        out = []
        for i in range(N):
            if i < n_unobserved:
                out.append("unobserved")
            elif i < n_unobserved + n_only_L1:
                out.append("only_l1")
            elif i < n_unobserved + n_only_L1 + n_only_L2:
                out.append("only_l2")
            else:
                out.append("full_observed")
        return out
    return ["unobserved"] * N


def draw_star_model(statuses):
    max_draw = min(len(statuses), 16)
    shown = statuses[:max_draw]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_aspect("equal")
    ax.axis("off")

    color_map = {
        "unobserved":    "#9e9e9e",
        "only_l1":       "#2e7d32",
        "only_l2":       "#ef6c00",
        "full_observed": "#1565c0",
    }

    ax.scatter([0], [0], s=950, color="#6a1b9a", zorder=5)
    ax.text(0, 0, "S", color="white", ha="center", va="center",
            fontsize=12, weight="bold")

    r1, r2 = 0.9, 1.65
    if max_draw == 1:
        angles = [np.pi / 2]
    else:
        angles = np.linspace(0, 2 * np.pi, max_draw, endpoint=False)

    for i, stype in enumerate(shown):
        th = angles[i]
        x1, y1 = r1 * np.cos(th), r1 * np.sin(th)
        x2, y2 = r2 * np.cos(th), r2 * np.sin(th)

        ax.plot([0, x1], [0, y1], color="black", lw=1.0, alpha=0.45)
        ax.plot([x1, x2], [y1, y2], color="black", lw=1.0, alpha=0.45)

        if stype == "unobserved":
            c1 = c2 = color_map["unobserved"]
        elif stype == "only_l1":
            c1, c2 = color_map["only_l1"], color_map["unobserved"]
        elif stype == "only_l2":
            c1, c2 = color_map["unobserved"], color_map["only_l2"]
        else:
            c1 = c2 = color_map["full_observed"]

        ax.scatter([x1], [y1], s=320, color=c1, zorder=4)
        ax.scatter([x2], [y2], s=280, color=c2, zorder=4)
        ax.text(x1, y1, f"L1_{i+1}", ha="center", va="center",
                fontsize=8, color="white")
        ax.text(x2, y2, f"L2_{i+1}", ha="center", va="center",
                fontsize=8, color="white")

        if stype == "unobserved":
            ax.text(x1, y1 - 0.20, "Tr", ha="center", va="center", fontsize=7)
            ax.text(x2, y2 - 0.20, "Tr", ha="center", va="center", fontsize=7)
        elif stype == "only_l1":
            ax.text(x2, y2 - 0.20, "Tr", ha="center", va="center", fontsize=7)
        elif stype == "only_l2":
            ax.text(x1, y1 - 0.20, "Tr", ha="center", va="center", fontsize=7)

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label="Fully observed",
               markerfacecolor="#1565c0", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Only L1 kept",
               markerfacecolor="#2e7d32", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Only L2 kept",
               markerfacecolor="#ef6c00", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Traced out",
               markerfacecolor="#9e9e9e", markersize=10),
    ]
    ax.legend(handles=legend_elems, loc="upper center", ncol=4,
              frameon=False, bbox_to_anchor=(0.5, 1.14))

    note = ""
    if len(statuses) > max_draw:
        note = (f"Diagram shows the first {max_draw} branches out of "
                f"N = {len(statuses)}.")
    return fig, note


# =====================================================================
# Plot helpers
# =====================================================================

def plot_main_curve(t, y, case_name, mode_name,
                    y_mode="auto", y_manual=1.0):
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.plot(t, y, lw=2.2)
    ax.set_xlabel("time t")
    ax.set_ylabel("decoherence factor squared")
    ax.set_title(f"{case_name}  |  {mode_name.replace('_', ' ')}  "
                 f"(analytic)")
    ax.grid(True, alpha=0.3)

    if y_mode == "full":
        ax.set_ylim(0.0, 1.02)
    elif y_mode == "manual":
        ax.set_ylim(0.0, max(0.05, y_manual))
    else:
        ymin = max(0.0, float(np.min(y)) - 0.03)
        ymax = min(1.02, float(np.max(y)) + 0.03)
        if ymax - ymin < 0.15:
            ymax = min(1.02, ymax + 0.08)
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    return fig


def plot_standard_comparison(t, curves):
    names = list(curves.keys())
    n = len(names)
    cols = 2 if n <= 4 else 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.1 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, name in zip(axes, names):
        y = curves[name]
        ax.plot(t, y, lw=2.0)
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("t")
        ax.set_ylabel("D^2")
        ymin = max(0.0, float(np.min(y)) - 0.03)
        ymax = min(1.02, float(np.max(y)) + 0.03)
        if ymax - ymin < 0.12:
            ymax = min(1.02, ymax + 0.06)
        ax.set_ylim(ymin, ymax)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Standard cases", fontsize=13)

    plt.tight_layout()
    return fig


def plot_log_curve(t, y, case_name, mode_name, floor=1e-300):
    y_safe = np.clip(y, floor, None)
    log_y = -np.log(y_safe)
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.plot(t, log_y, lw=2.2)
    ax.set_xlabel("time t")
    ax.set_ylabel("- ln D^2(t)")
    ax.set_title(f"Logarithmic decoherence: {case_name} | "
                 f"{mode_name.replace('_', ' ')}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, log_y


def plot_verification(t, y_analytic, y_numeric, case_name):
    diff = np.abs(y_analytic - y_numeric)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)

    axes[0].plot(t, y_analytic, lw=2.2, label="analytic closed-form",
                 alpha=0.9)
    axes[0].plot(t, y_numeric, "--", lw=2.0, label="numeric SVD reference",
                 alpha=0.85)
    axes[0].set_ylabel("D^2(t)")
    axes[0].set_title(f"Analytic verification: {case_name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, diff, lw=1.8)
    axes[1].set_xlabel("time t")
    axes[1].set_ylabel("|analytic - numeric|")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, diff


# =====================================================================
# App header
# =====================================================================

st.title("Decoherence in a Two-Layer Spin-Chain Environment")
st.caption("Closed-form analytic build  •  no brute-force inside the "
           "main loop  •  numerical verification available")

st.markdown(
    "Central two-level system $S$ coupled to $N$ independent two-spin "
    "chains $E_i = L_1^{(i)} \\otimes L_2^{(i)}$.  Interaction "
    "Hamiltonian:"
)
st.latex(r"""
H_{\mathrm{int}}
= \sigma_z^{(S)}\sum_{i=1}^N g_i\sigma_z^{(L1,i)}
+ \sum_{i=1}^N J_i \sigma_x^{(L1,i)}\sigma_z^{(L2,i)}.
""")

with st.expander("Analytic per-branch building blocks"):
    st.markdown("**Pointer-frame frequency**")
    st.latex(r"\omega_i = \sqrt{g_i^2 + J_i^2},\quad "
             r"c_i=\cos(\omega_i t),\quad s_i=\sin(\omega_i t)/\omega_i")
    st.markdown("**Scalar coherence factor (unobserved branch)**")
    st.latex(r"|\gamma_i|^2 = (1-2g_i^2s_i^2)^2 + 4g_i^2s_i^2"
             r"(c_ia_z + J_is_ia_ya_z)^2")
    st.markdown("**Trace-norm of the L1 block**")
    st.latex(r"\bigl(\nu_i^{(L1)}\bigr)^2 = S_i^{(1)} "
             r"+ 2\sqrt{(X_i^{(1)})^2 + (Y_i^{(1)})^2}")
    st.markdown("**Trace-norm of the L2 block**")
    st.latex(r"\bigl(\nu_i^{(L2)}\bigr)^2 = S_i^{(2)} "
             r"+ 2\sqrt{(X_i^{(2)})^2 + (Y_i^{(2)})^2}")
    st.markdown("Explicit $S$, $X$, $Y$ expressions are coded "
                "verbatim from the LaTeX document.")


# =====================================================================
# Sidebar controls
# =====================================================================

st.sidebar.header("Controls")

case_name = st.sidebar.radio(
    "Reduced description",
    ["S only", "S + L1", "S + L2",
     "S + observed full chains", "Combination"],
)

mode_name = st.sidebar.radio(
    "Frequency mode",
    ["identical", "integer_commensurate", "rational_commensurate",
     "incommensurate", "badly_approximable",
     "metallic_ba_inspired", "weak_disorder"],
)

N  = st.sidebar.slider("Number of branches N", 1, 30, 12, 1)
g0 = st.sidebar.slider("Base coupling g", 0.1, 2.5, 1.0, 0.1)
J0 = st.sidebar.slider("Base coupling J", 0.1, 2.5, 1.0, 0.1)

st.sidebar.subheader("Initial single-spin state")
ax0 = st.sidebar.slider("a_x", -1.0, 1.0, 0.0, 0.1)
ay0 = st.sidebar.slider("a_y", -1.0, 1.0, 0.0, 0.1)
az0 = st.sidebar.slider("a_z", -1.0, 1.0, 0.0, 0.1)

if ax0**2 + ay0**2 + az0**2 > 1.0001:
    st.sidebar.warning("Physical Bloch states require |a| ≤ 1.")

disorder, seed = 0.06, 123
if mode_name == "weak_disorder":
    disorder = st.sidebar.slider("Disorder strength", 0.0, 0.20, 0.06, 0.01)
    seed = st.sidebar.number_input("Random seed", 0, 999999, 123, 1)
elif mode_name == "badly_approximable":
    seed = st.sidebar.number_input("Algebraic seed", 0, 999999, 123, 1)
    st.sidebar.caption(
        "The seed selects an irreducible trinomial x^N - a x - 1. "
        "With SymPy installed, irreducibility is checked over Q."
    )

omega0 = np.sqrt(g0**2 + J0**2)
T0     = np.pi / omega0 if omega0 > 0 else 1.0

st.sidebar.subheader("Time window")
time_preset = st.sidebar.radio(
    "Time scale preset",
    ["very short", "short", "medium", "long", "very long", "custom"],
    index=2,
)
time_unit = st.sidebar.radio(
    "Time unit", ["pi / omega0", "1 / omega0", "absolute t"], index=0,
)

preset_windows = {
    "very short": (0.0, 0.20, 1200),
    "short":      (0.0, 1.0, 1200),
    "medium":     (0.0, 8.0, 1400),
    "long":       (0.0, 30.0, 1800),
    "very long":  (0.0, 100.0, 2500),
}

if time_preset == "custom":
    t_start_units = st.sidebar.number_input("t start", value=0.0, step=0.1)
    t_end_units   = st.sidebar.number_input("t end",   value=8.0, step=0.5)
    default_points = 1400
else:
    t_start_units, t_end_units, default_points = preset_windows[time_preset]
    st.sidebar.caption(f"Window: {t_start_units:g} to {t_end_units:g} "
                       f"in units of {time_unit}")

n_points = st.sidebar.slider("Number of time points",
                             300, 5000, default_points, 100)

if time_unit == "pi / omega0":
    time_scale = T0
elif time_unit == "1 / omega0":
    time_scale = 1.0 / omega0 if omega0 > 0 else 1.0
else:
    time_scale = 1.0

t_start = float(t_start_units) * time_scale
t_end   = float(t_end_units)   * time_scale

if t_end <= t_start:
    st.sidebar.error("t end must be larger than t start. Fallback used.")
    t_start, t_end = 0.0, 8.0 * T0

observed_full_count = 0
if case_name == "S + observed full chains":
    observed_full_count = st.sidebar.slider(
        "Number of fully observed chains", 0, N, N // 2, 1
    )

n_unobserved = n_only_L1 = n_only_L2 = 0
n_full_comb = N
if case_name == "Combination":
    st.sidebar.markdown(
        "The last sector is fully observed and set by the remainder."
    )
    n_unobserved = st.sidebar.slider("Unobserved full chains", 0, N, N // 4, 1)
    n_only_L1    = st.sidebar.slider("Branches with only L1 kept", 0, N, N // 4, 1)
    n_only_L2    = st.sidebar.slider("Branches with only L2 kept", 0, N, N // 4, 1)
    n_full_comb  = N - n_unobserved - n_only_L1 - n_only_L2
    if n_full_comb < 0:
        st.sidebar.error("Counts exceed N -- reduce one of them.")
    else:
        st.sidebar.success(f"Fully observed chains = {n_full_comb}")

y_mode   = st.sidebar.radio("Vertical axis mode", ["auto", "full", "manual"])
y_manual = 1.0
if y_mode == "manual":
    y_manual = st.sidebar.slider("Manual y max", 0.05, 1.2, 0.5, 0.05)

show_comparison = st.sidebar.checkbox(
    "Also show the standard cases together", value=False,
)


# =====================================================================
# Build params and time grid
# =====================================================================

params, freq_info = build_params_with_info(
    N, g0, J0, mode_name, disorder=disorder, seed=seed
)
t_grid = np.linspace(t_start, t_end, n_points)

invalid_combination = (case_name == "Combination" and n_full_comb < 0)


# =====================================================================
# Settings summary
# =====================================================================

st.header("Selected physical setting")
colA, colB = st.columns([1.15, 1.0])

with colA:
    st.subheader("Observation pattern")
    statuses = statuses_for_case(case_name, N, observed_full_count,
                                 n_unobserved, n_only_L1, n_only_L2)
    fig_star, note_star = draw_star_model(statuses)
    st.pyplot(fig_star)
    if note_star:
        st.caption(note_star)

with colB:
    st.subheader("What this choice means")
    st.markdown(f"**Case:** {case_name}")
    st.markdown(case_description(case_name))

    st.markdown(f"**Frequency mode:** "
                f"{FREQUENCY_MODE_TEXT[mode_name]['title']}")
    st.markdown(FREQUENCY_MODE_TEXT[mode_name]["desc"])

    st.markdown("**What you should expect:**")
    st.info(what_to_expect(case_name, mode_name))

    if mode_name in (
        "incommensurate", "badly_approximable", "metallic_ba_inspired"
    ):
        st.caption(
            "Irrational modes are separated by Diophantine strength: "
            "incommensurate = well-approximable irrational ladder; "
            "badly_approximable = algebraic simultaneous BA power basis; "
            "metallic_ba_inspired = legacy individual quadratic-irrational "
            "baseline without a vector-level certificate."
        )

    st.markdown("**Selected analytic formula:**")
    st.latex(case_formula(case_name))


# =====================================================================
# Main analytic plot
# =====================================================================

st.header("Selected plot  (analytic closed form)")

if invalid_combination:
    st.error("Combination counts exceed N.")
else:
    with st.spinner("Evaluating analytic formula..."):
        y_selected = compute_selected_curve(
            case_name, params, ax0, ay0, az0, t_grid,
            observed_full_count, n_unobserved, n_only_L1, n_only_L2,
        )

    fig_main = plot_main_curve(
        t_grid, y_selected, case_name, mode_name,
        y_mode=y_mode, y_manual=y_manual,
    )
    st.pyplot(fig_main)

    # ---- a small text panel with branch frequencies (debug aid)
    with st.expander("Branch frequencies used and Diophantine diagnostic"):
        omegas = np.array([np.sqrt(g**2 + J**2) for (g, J) in params])
        ratios_to_first = omegas / omegas[0] if omegas[0] != 0 else omegas
        unit_mean_ratios = np.array(freq_info.get("ratios", []), dtype=float)

        st.write({
            "omega_i": [float(f"{w:.8g}") for w in omegas],
            "omega_i / omega_1": [float(f"{r:.8g}") for r in ratios_to_first],
            "unit_mean_ratio_used_for_g_i_J_i": [
                float(f"{r:.8g}") for r in unit_mean_ratios
            ],
        })

        st.markdown("**Frequency construction**")
        st.write({
            "mode": freq_info.get("mode"),
            "construction": freq_info.get("construction"),
            "claim": freq_info.get("claim"),
            "polynomial": freq_info.get("polynomial"),
            "irreducible_over_Q": freq_info.get("irreducible_over_Q"),
            "certificate": freq_info.get("certificate"),
            "beta": freq_info.get("beta"),
        })

        Q_window = suggested_Qmax_from_time_window(t_grid, params)
        diagnostic = simultaneous_badness_score_from_ratios(
            unit_mean_ratios, Q_window
        )
        Q_plot = suggested_Qmax_strictly_inside_time_window(t_grid, params)
        rephasing = rephasing_candidate_from_ratios(
            unit_mean_ratios, Q_plot
        )
        st.markdown("**Finite-window simultaneous Diophantine diagnostic**")
        st.write(diagnostic)
        st.markdown("**Exact/near rephasing candidate inside the plotted time window**")
        st.write(rephasing)
        st.caption(
            "c_Q = min_{1<=q<=Qmax} q^(1/d) max_j ||q theta_j||, "
            "theta_j = omega_{j+1}/omega_1. The rephasing candidate uses "
            "t = q*pi/omega_1 and also reports t/(pi/omega0), restricted "
            "to the current plotted time window. Exact_q is shown only when "
            "the ratios are commensurate within numerical tolerance. Larger "
            "c_Q means harder finite-time simultaneous rephasing."
        )

    st.markdown("**Interpretation:** values near 1 mean strong "
                "coherence; values near 0 mean strong decoherence.")


# =====================================================================
# Advanced tabs
# =====================================================================

if not invalid_combination:
    st.header("Advanced analysis")
    tab_log, tab_verify, tab_random = st.tabs(
        ["Logarithmic decoherence", "Analytic verification",
         "Random frequency trials"]
    )

    # ---- log view ---------------------------------------------------
    with tab_log:
        st.markdown(
            "When the number of branches is large, the product "
            "structure can push $\\mathcal D^2(t)$ extremely close "
            "to zero.  The variable $\\chi_2(t) = -\\ln \\mathcal "
            "D^2(t)$ resolves the structure that an ordinary linear "
            "plot collapses to zero."
        )
        log_floor = st.select_slider(
            "Numerical floor before taking the log",
            options=[1e-300, 1e-250, 1e-200, 1e-150, 1e-100, 1e-80,
                     1e-60, 1e-40, 1e-30, 1e-20, 1e-12],
            value=1e-300,
            format_func=lambda x: f"{x:.0e}",
        )
        fig_log, log_y = plot_log_curve(
            t_grid, y_selected, case_name, mode_name, floor=log_floor
        )
        st.pyplot(fig_log)
        c1, c2, c3 = st.columns(3)
        c1.metric("min -ln D²", f"{float(np.min(log_y)):.4g}")
        c2.metric("max -ln D²", f"{float(np.max(log_y)):.4g}")
        c3.metric("median -ln D²", f"{float(np.median(log_y)):.4g}")

    # ---- verification ----------------------------------------------
    with tab_verify:
        st.markdown(
            "Cross-check between two independent code paths:\n\n"
            "1. **Analytic** -- the closed-form expressions derived in "
            "the LaTeX document and used by all the plots above.\n"
            "2. **Numeric reference** -- explicit construction of "
            "$\\Gamma_{+-}^{(i)}$, partial traces, and SVD-based trace "
            "norms.\n\n"
            "If the analytic derivation is implemented correctly, the "
            "solid and dashed curves should be indistinguishable."
        )
        verify_points = st.slider(
            "Number of verification time points",
            50, 1500, min(350, int(n_points)), 50,
            key="verify_points_slider",
        )
        st.caption("Use fewer points first for large N or long t.")

        if st.button("Run verification"):
            t_verify = np.linspace(t_start, t_end, verify_points)
            with st.spinner("Computing analytic and numeric curves..."):
                y_a = compute_selected_curve(
                    case_name, params, ax0, ay0, az0, t_verify,
                    observed_full_count,
                    n_unobserved, n_only_L1, n_only_L2,
                )
                y_n = compute_selected_curve_numeric(
                    case_name, params, ax0, ay0, az0, t_verify,
                    observed_full_count,
                    n_unobserved, n_only_L1, n_only_L2,
                )

            fig_v, diff = plot_verification(t_verify, y_a, y_n, case_name)
            st.pyplot(fig_v)

            max_abs = float(np.max(diff))
            rms     = float(np.sqrt(np.mean(diff**2)))
            denom   = max(float(np.max(np.abs(y_a))), 1e-15)
            max_rel = max_abs / denom

            c1, c2, c3 = st.columns(3)
            c1.metric("max abs error",   f"{max_abs:.3e}")
            c2.metric("RMS error",       f"{rms:.3e}")
            c3.metric("max rel error",   f"{max_rel:.3e}")

            if max_abs < 1e-9:
                st.success("Analytic and numeric routes agree to "
                           "numerical precision.")
            elif max_abs < 1e-6:
                st.info("Curves agree well; tiny residuals are "
                        "consistent with finite-precision products.")
            else:
                st.warning("Discrepancy is larger than expected. "
                           "Check parameters.")


    # ---- random frequency trials ------------------------------------
    with tab_random:
        st.markdown(
            "This panel samples many random frequency realisations and "
            "plots the decoherence curve for the currently selected case. "
            "The incommensurate samples are well-approximable irrational "
            "ladders; the badly-approximable samples use the algebraic "
            "simultaneous BA construction; the metallic option is kept as "
            "a legacy BA-inspired baseline.  This is useful when small N "
            "gives misleading single-sample revival behaviour."
        )

        random_families = st.multiselect(
            "Families to sample",
            ["incommensurate", "badly_approximable", "metallic_ba_inspired"],
            default=["incommensurate", "badly_approximable"],
            key="random_families_select",
        )

        n_trials = st.slider(
            "Number of random realisations per family",
            1, 100, 20, 1,
            key="random_trials_slider",
        )

        random_seed = st.number_input(
            "Random trial seed",
            0, 999999, 2026, 1,
            key="random_trial_seed_input",
        )

        cut_fraction = st.slider(
            "Ignore early-time fraction for revival score",
            0.01, 0.30, 0.08, 0.01,
            key="random_cut_fraction_slider",
        )

        st.caption(
            "revival_score = max D²(t) after the early-time cut. "
            "Larger values mean stronger near-revival."
        )

        if st.button("Generate random frequency trials"):
            family_offsets = {
                "incommensurate": 0,
                "badly_approximable": 100_000,
                "metallic_ba_inspired": 200_000,
            }

            fig, ax = plt.subplots(figsize=(10, 4.4))
            rows = []
            summary_rows = []

            for family in random_families:
                family_scores = []
                offset = family_offsets.get(family, 200_000)

                for trial in range(n_trials):
                    rng_trial = np.random.default_rng(
                        int(random_seed) + offset + trial
                    )
                    params_trial, info_trial = random_params_with_info(
                        N, g0, J0, family, rng_trial, seed_offset=trial
                    )

                    y_trial = compute_selected_curve(
                        case_name, params_trial, ax0, ay0, az0, t_grid,
                        observed_full_count,
                        n_unobserved, n_only_L1, n_only_L2,
                    )

                    score = revival_score(
                        t_grid, y_trial, cut_fraction=cut_fraction
                    )
                    family_scores.append(score)

                    label = family if trial == 0 else None
                    ax.plot(
                        t_grid, y_trial,
                        lw=1.1,
                        alpha=0.28,
                        label=label,
                    )

                    omegas = np.array([
                        np.sqrt(g**2 + J**2)
                        for (g, J) in params_trial
                    ])
                    omega_ratios = omegas / np.mean(omegas)

                    Q_trial = suggested_Qmax_from_time_window(
                        t_grid, params_trial
                    )
                    dio = simultaneous_badness_score_from_ratios(
                        info_trial.get("ratios", omega_ratios), Q_trial
                    )

                    rows.append({
                        "family": family,
                        "trial": trial + 1,
                        "revival_score": score,
                        "c_Q": dio["c_Q"],
                        "q_best": dio["q_best"],
                        "Qmax": dio["Qmax"],
                        "polynomial": info_trial.get("polynomial"),
                        "irreducible_over_Q": info_trial.get("irreducible_over_Q"),
                        "min_omega_ratio": float(np.min(omega_ratios)),
                        "max_omega_ratio": float(np.max(omega_ratios)),
                        "ratio_spread": float(
                            np.max(omega_ratios) - np.min(omega_ratios)
                        ),
                    })

                if family_scores:
                    scores = np.array(family_scores)
                    summary_rows.append({
                        "family": family,
                        "mean_revival": float(np.mean(scores)),
                        "median_revival": float(np.median(scores)),
                        "max_revival": float(np.max(scores)),
                        "min_revival": float(np.min(scores)),
                    })

            ax.set_xlabel("time t")
            ax.set_ylabel("decoherence factor squared")
            ax.set_title(
                f"Random frequency trials | {case_name} | N = {N}"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()

            if y_mode == "full":
                ax.set_ylim(0.0, 1.02)
            elif y_mode == "manual":
                ax.set_ylim(0.0, max(0.05, y_manual))

            plt.tight_layout()
            st.pyplot(fig)

            if summary_rows:
                st.subheader("Family summary")
                st.dataframe(summary_rows, use_container_width=True)

            if rows:
                st.subheader("Individual realisations")
                rows_sorted = sorted(
                    rows,
                    key=lambda r: r["revival_score"],
                    reverse=True,
                )
                st.dataframe(rows_sorted, use_container_width=True)


# =====================================================================
# Comparison panel
# =====================================================================

if show_comparison and not invalid_combination:
    st.header("Comparison panel: standard cases")
    with st.spinner("Computing standard cases analytically..."):
        curves = {
            "S only":   curve_S_only(params, ax0, ay0, az0, t_grid),
            "S + L1":   curve_S_plus_L1(params, ax0, ay0, az0, t_grid),
            "S + L2":   curve_S_plus_L2(params, ax0, ay0, az0, t_grid),
            "S + observed full chains":
                curve_S_plus_observed_full(
                    params, ax0, ay0, az0, t_grid,
                    observed_full_count=max(1, N // 2),
                ),
            "Combination":
                curve_combination(
                    params, ax0, ay0, az0, t_grid,
                    n_unobserved=N // 4,
                    n_only_L1=N // 4,
                    n_only_L2=N // 4,
                ),
        }
    fig_compare = plot_standard_comparison(t_grid, curves)
    st.pyplot(fig_compare)

    st.markdown(
        "Layer 1 is directly coupled to S, Layer 2 only indirectly; "
        "fully observed chains contribute factor 1; fully unobserved "
        "chains contribute $|\\gamma_i|$.  There is no universal "
        "ordering between $S+L_1$ and $S+L_2$."
    )


# =====================================================================
# Footer
# =====================================================================

st.markdown("---")
st.markdown("### Quick summary")
st.markdown(
    "- Identical / commensurate frequencies: exact revival when all branch "
    "phases share a common period.\n"
    "- Rational commensurate frequencies: exact revival, but its time depends "
    "on the denominator structure after normalisation.\n"
    "- Incommensurate frequencies: no exact revival; well-approximable "
    "choices can still show earlier finite-time near-revivals.\n"
    "- Badly approximable frequencies: algebraic simultaneous BA construction "
    "with suppressed finite-time rephasing diagnostics.\n"
    "- Weak disorder: clean revival washed out continuously with disorder "
    "strength.\n"
    "- All plotted curves are produced by closed-form analytic expressions; "
    "the numeric SVD route is only invoked inside the verification tab."
)
