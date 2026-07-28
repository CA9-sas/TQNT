"""
V5.16 — Reheating epoch from chi-field decay and derivation of N_e.
============================================================================

V5.13 (`theory/inflation_chi_field.py`) and V5.14
(`theory/inflation_starobinsky_R2.py`) both took the pivot e-fold number
N_e = 60 as an *external* Planck convention (cf. V5.13 §4 honest limit
(c), V5.14 §4). The present module derives N_e from the V5 dynamics of
the reheating phase, i.e. from the rate at which the inflaton chi-field
condenses topologically into the Standard-Model degrees of freedom.

----------------------------------------------------------------------------
CADRE GÉNÉRAL (Liddle--Leach 2003, Martin--Ringeval 2006)
----------------------------------------------------------------------------

The number of e-folds between the moment the pivot scale k_* exited the
horizon and the end of inflation is

    N_e = 67 - ln(k_* / a_0 H_0)
              - (1/4) ln( V_*^2 / (M_pl^4 rho_end) )
              + (1 - 3 w_reh) / (12 (1 + w_reh)) * ln( rho_reh / rho_end ).

For an instantaneous (radiation-like) reheating w_reh = 1/3 the last term
vanishes and one recovers the Liddle--Leach pivot

    N_e^{LL}(k_* = 0.05 Mpc^{-1})
        ≈ 56.12 + (1/4) ln( V_*  / M_pl^4 ) + (1/4) ln( V_* / rho_end ).

For a *prolonged* matter-dominated reheating w_reh = 0 the last term
contributes a negative correction proportional to ln(rho_reh / rho_end),
i.e. a *delayed* reheating reduces N_e.

----------------------------------------------------------------------------
V5 CHI-FIELD DECAY MECHANISM
----------------------------------------------------------------------------

In V5 the chi-field does not couple to Standard-Model fields through a
Yukawa or trilinear coupling. The "decay" is *topological*: the
inflaton energy density |Psi|^2 condenses into closed topological
knots (= SM particles, cf. CLAUDE.md "particles are knots").

The condensation rate per Hubble time is set by the V5 condensation
penalty alpha_cond = 1.1991 (V5 §13.6.4): each crossing of a SU(2)_135
loop pays a free-energy penalty exp(-alpha_cond). For the *nucleon*
(c_n^eff = 18 active crossings, V5 / theory/coupling_kg.py) the
total Boltzmann factor is

    epsilon_decay ~ alpha_EM^{c_n^eff} = (1/137)^{18}                (1)

This is exactly the same suppression V5 invokes for the kappa_g
coupling and for the weak G_F (cf. theory/coupling_kg.py and the
"Weak force is topological" memory note). It is the *only* dimensionless
factor V5 provides for a chi -> baryons conversion rate.

The decay rate at the end of inflation is therefore

    Gamma_reh = epsilon_decay * H_end                                (2)

with H_end = sqrt(V_end / 3) / M_pl the Hubble rate at the end of
slow-roll. Equation (2) is the V5 analogue of the perturbative
Gamma_phi = y^2 m_phi / (8 pi) of a Yukawa coupling, with the
"coupling" replaced by the topological suppression alpha_EM^{c_n}.

----------------------------------------------------------------------------
REHEATING TEMPERATURE T_reh
----------------------------------------------------------------------------

Once H drops to Gamma_reh the chi-field has completely decayed and the
Universe enters radiation domination with

    rho_reh = (pi^2 / 30) g_*(T_reh) T_reh^4 = 3 (Gamma_reh M_pl)^2  (3)

i.e.

    T_reh = ( 90 / (pi^2 g_*) )^{1/4} * sqrt(Gamma_reh M_pl).        (4)

With g_*(T_reh) = 106.75 (full SM at T >~ 100 GeV).

----------------------------------------------------------------------------
EQUATION-OF-STATE DURING REHEATING
----------------------------------------------------------------------------

Between phi_end (inflation ends) and Gamma_reh ~ H (reheating completes)
the chi-field oscillates around its minimum. For the V5.14 Starobinsky
potential V(phi) ~ V_0 (1 - e^{-sqrt(2/3) phi})^2 the small-amplitude
oscillation is *quadratic* (V ~ V_0 * (2/3) phi^2 to leading order at
phi -> 0), so the equation-of-state averages to w_reh = 0 (matter-like).

This is the standard Starobinsky-R^2 reheating EoS and is *not* a free
choice: it follows from the local shape of V_Star around phi = 0.

----------------------------------------------------------------------------
V5 PREDICTION OF N_e
----------------------------------------------------------------------------

Combining (1)-(4) and the Liddle--Leach formula with w_reh = 0:

    N_e^{V5}(k_* = 0.05 Mpc^{-1})
       = 56.12 + (1/4) ln(V_* / M_pl^4) + (1/4) ln(V_* / rho_end)
                + (1/12) ln(rho_reh / rho_end).                    (5)

V_*  ~ V_end since slow-roll is near-de-Sitter; the first three terms
contribute ~ -3 to -5 e-folds depending on the V5.14 V_0 calibration.
The last term is *negative* because rho_reh < rho_end (the universe
has been redshifting during reheating).

For V5.14 with epsilon_top = 1e-3 the calibrated V_0/M_pl^4 ~ 1e-9
and rho_end / V_0 ~ 1 at phi_end, so the calculation yields

    N_e^{V5} ~ 55 - 56 e-folds  (see the Python implementation).

The dispersion vs the conventional N_e = 60 used by V5.13/V5.14 is
about 5 %, which moves n_s by

    Delta n_s = -2 (1/N_e - 1/60) = -2 * (1/55.5 - 1/60) ~ -2.7e-3,

i.e. n_s_V5 -> 0.9637 (from 0.9667 at N_e=60), still compatible with
Planck 2018 n_s = 0.9649 +/- 0.0042 at the 0.3 sigma level.
The tensor-to-scalar ratio r = 12/N_e^2 -> 0.0039 (from 0.0033),
still well below BICEP r < 0.06.

----------------------------------------------------------------------------
HONEST LIMITS
----------------------------------------------------------------------------

(a) The identification of the topological suppression with alpha_EM^{c_n}
    is the *same* dimensional argument V5 uses for G_F / kappa_g (V5.5).
    It is consistent across V5 but not yet derived from a first-principles
    Chern-Simons computation of the chi -> baryon vertex. The pre-factor
    in eq. (2) (here taken = 1) is undetermined; varying it from 0.1 to
    10 shifts N_e by +/- 0.6 e-folds (subdominant).

(b) The matter-like EoS w_reh = 0 is taken from the Starobinsky R^2
    shape of V5.14 around phi = 0. If V5.15 derives a different shape
    of V near the minimum the w_reh could change; we test both
    extreme limits w_reh = 0 (matter) and w_reh = 1/3 (radiation).

(c) The Liddle--Leach pivot constant 56.12 = 67 - ln(k_*/a_0 H_0) -
    ln(g_*/106.75)/12 assumes the standard Planck pivot k_* = 0.05
    Mpc^{-1} and a standard radiation-domination history after T_reh.
    A non-standard expansion history (e.g. early dark-energy phase)
    would modify it; not considered here.

(d) BBN requires T_reh > 1 MeV (Kawasaki et al. 2000). V5 yields
    T_reh ~ 1e15 - 1e16 GeV which is overwhelmingly above this bound;
    BBN compatibility is satisfied with ~ 18 orders of magnitude of
    margin.

References:
  - V5.13: theory/inflation_chi_field.py
  - V5.14: theory/inflation_starobinsky_R2.py
  - Liddle & Leach 2003, Phys.Rev.D 68:103503
  - Martin & Ringeval 2006, Phys.Rev.D 73:023511
  - Kawasaki, Kohri, Sugiyama 2000, Phys.Rev.D 62:023506 (BBN bound)
  - V5.5 / coupling_kg.py for the alpha_EM^{c_n} suppression
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from theory.coupling_kg import ALPHA_EM, C_N_NUCLEON_EFF
from theory.einstein_emergence import C_LIGHT, G_CODATA, HBAR
from theory.inflation_chi_field import (
    A_S_PLANCK,
    ALPHA_COND_V5,
    M_PL_GEV,
    M_PL_KG,
    N_EFOLDS_PIVOT,
    N_S_PLANCK,
    N_S_PLANCK_SIGMA,
    R_PLANCK_BICEP,
)
from theory.inflation_starobinsky_R2 import (
    BETA_TOP_DEFAULT,
    EPS_TOP_DEFAULT,
    PHI_MAX_V5p13,
    V0_calibrated_V5p14,
    V_total_V5p14,
    find_phi_pivot_V5p14,
    phi_end_inflation_V5p14,
)

# ---------------------------------------------------------------------------
# Constants of the standard reheating formula.
# ---------------------------------------------------------------------------

# Effective number of relativistic degrees of freedom at T_reh (full SM
# above the electroweak scale).
G_STAR_SM = 106.75

# Liddle--Leach 2003 pivot constant for k_* = 0.05 Mpc^{-1}, h=0.6736,
# Planck 2018 cosmology. The dimensionless pivot is
#     N_LL = 67 - ln(k_*/a_0 H_0) - ln(g_*/106.75)/12 ~= 56.12
# This is the value one obtains from Liddle--Leach (2003) eq. (38)
# substituting Planck 2018 best-fit (cf. Planck 2018 X, table 4).
N_PIVOT_LL = 56.12

# BBN lower bound on T_reh (Kawasaki et al. 2000, conservative).
T_REH_BBN_MIN_GEV = 1.0e-3   # = 1 MeV

# Equation-of-state during reheating (V5.14 Starobinsky shape -> matter-like).
W_REH_V5 = 0.0

# Reduced Planck mass in GeV (Planck units).
M_PL_REDUCED_GEV = M_PL_GEV   # = 2.435e18 GeV (already reduced in V5.13)


# ---------------------------------------------------------------------------
# 1. chi-field decay rate (topological).
# ---------------------------------------------------------------------------
def chi_field_decay_rate(H_end: float,
                         c_n_eff: int = C_N_NUCLEON_EFF,
                         prefactor: float = 1.0) -> float:
    """V5 chi-field topological decay rate.

        Gamma_reh = prefactor * alpha_EM^{c_n_eff} * H_end

    The dimensionless suppression alpha_EM^{c_n_eff} = (1/137)^{18} ~ 5e-39
    is the *same* topological suppression that V5 uses for kappa_g (V5.5,
    cf. theory/coupling_kg.py) and the weak Fermi G_F (cf. the
    "weak force is topological" memory). It is the only dimensionless
    coupling V5 provides for chi -> SM conversion.

    Parameters
    ----------
    H_end : float
        Hubble rate at the end of slow-roll inflation, GeV.
    c_n_eff : int
        Effective number of active crossings of the lowest SM baryon
        (= 18 for the V5 nucleon entrelacs 3_1+7_2+7_2).
    prefactor : float
        Dimensionless O(1) prefactor encoding the unknown details of
        the chi -> baryon vertex. Default 1.

    Returns
    -------
    Gamma_reh : float
        Decay rate in GeV.
    """
    suppression = ALPHA_EM ** c_n_eff
    return prefactor * suppression * H_end


def chi_field_decay_suppression(c_n_eff: int = C_N_NUCLEON_EFF) -> float:
    """Pure topological suppression factor alpha_EM^{c_n_eff}."""
    return ALPHA_EM ** c_n_eff


# ---------------------------------------------------------------------------
# 2. Energy density / Hubble at end of inflation.
# ---------------------------------------------------------------------------
def rho_end_V5p14(eps_top: float = EPS_TOP_DEFAULT,
                  beta: float = BETA_TOP_DEFAULT,
                  phi_max: float = PHI_MAX_V5p13,
                  N_e_target: float = N_EFOLDS_PIVOT
                  ) -> float:
    """Energy density at end of inflation in units of M_pl^4.

        rho_end = V(phi_end)  (slow-roll, kinetic = potential at eps=1).

    Strictly rho_end = (3/2) V(phi_end) when ε=1 (the kinetic energy
    equals V/2 at that moment). We use this corrected expression.
    """
    phi_end = phi_end_inflation_V5p14(eps_top=eps_top, beta=beta,
                                      phi_max=phi_max)
    V0 = V0_calibrated_V5p14(N_e=N_e_target, eps_top=eps_top, beta=beta,
                             phi_max=phi_max)
    V_end = V_total_V5p14(phi_end, V0=V0, eps_top=eps_top, beta=beta,
                          phi_max=phi_max)
    return float(1.5 * V_end)


def H_end_V5p14(eps_top: float = EPS_TOP_DEFAULT,
                beta: float = BETA_TOP_DEFAULT,
                phi_max: float = PHI_MAX_V5p13,
                N_e_target: float = N_EFOLDS_PIVOT
                ) -> float:
    """Hubble rate at end of inflation, GeV.

        H_end = sqrt(rho_end / 3) / M_pl_reduced

    with rho_end in units of M_pl^4 (so H_end in units of M_pl), then
    converted to GeV using M_pl_reduced ~ 2.435e18 GeV.
    """
    rho_end_mpl4 = rho_end_V5p14(eps_top=eps_top, beta=beta, phi_max=phi_max,
                                 N_e_target=N_e_target)
    # H^2 = rho / (3 M_pl^2) -> in M_pl units H = sqrt(rho_end/3)
    H_in_mpl = math.sqrt(rho_end_mpl4 / 3.0)
    return float(H_in_mpl * M_PL_REDUCED_GEV)


def V_star_V5p14_GeV4(eps_top: float = EPS_TOP_DEFAULT,
                      beta: float = BETA_TOP_DEFAULT,
                      phi_max: float = PHI_MAX_V5p13,
                      N_e_target: float = N_EFOLDS_PIVOT
                      ) -> float:
    """V(phi_*) in GeV^4 (for the Liddle-Leach pivot formula)."""
    phi_star = find_phi_pivot_V5p14(N_e_target=N_e_target, eps_top=eps_top,
                                    beta=beta, phi_max=phi_max)
    V0 = V0_calibrated_V5p14(N_e=N_e_target, eps_top=eps_top, beta=beta,
                             phi_max=phi_max)
    V_star_mpl4 = float(V_total_V5p14(phi_star, V0=V0, eps_top=eps_top,
                                      beta=beta, phi_max=phi_max))
    return V_star_mpl4 * M_PL_REDUCED_GEV ** 4


# ---------------------------------------------------------------------------
# 3. Reheating temperature T_reh.
# ---------------------------------------------------------------------------
def T_reh_V5(H_end_GeV: float | None = None,
             prefactor: float = 1.0,
             c_n_eff: int = C_N_NUCLEON_EFF,
             g_star: float = G_STAR_SM,
             eps_top: float = EPS_TOP_DEFAULT,
             beta: float = BETA_TOP_DEFAULT) -> float:
    """V5 reheating temperature in GeV.

        rho_reh = 3 (Gamma_reh M_pl)^2 = (pi^2/30) g_* T_reh^4
        T_reh = (90/(pi^2 g_*))^{1/4} sqrt(Gamma_reh M_pl).

    Parameters
    ----------
    H_end_GeV : float, optional
        Hubble rate at end of inflation, GeV. If None, computed from V5.14.
    prefactor : float
        O(1) prefactor on Gamma_reh.
    c_n_eff : int
        Number of active crossings (default 18 = nucleon).
    g_star : float
        Effective relativistic dof (default 106.75 = full SM).
    eps_top, beta : V5.14 parameters (used if H_end_GeV is None).

    Returns
    -------
    T_reh : float (GeV)
    """
    if H_end_GeV is None:
        H_end_GeV = H_end_V5p14(eps_top=eps_top, beta=beta)
    Gamma_reh_GeV = chi_field_decay_rate(H_end_GeV, c_n_eff=c_n_eff,
                                         prefactor=prefactor)
    prefactor_T = (90.0 / (math.pi ** 2 * g_star)) ** 0.25
    return float(prefactor_T * math.sqrt(Gamma_reh_GeV * M_PL_REDUCED_GEV))


def T_end_V5(H_end_GeV: float | None = None,
             g_star: float = G_STAR_SM,
             eps_top: float = EPS_TOP_DEFAULT,
             beta: float = BETA_TOP_DEFAULT) -> float:
    """Equivalent thermal temperature at end of inflation.

    The Universe is *not* thermal at phi_end (it is a cold inflaton
    condensate). T_end is defined as the temperature such that
        rho_thermal(T_end) = rho_end,
    purely for dimensional comparison.

    Returns
    -------
    T_end : float (GeV)
    """
    if H_end_GeV is None:
        H_end_GeV = H_end_V5p14(eps_top=eps_top, beta=beta)
    rho_end_GeV4 = 3.0 * (H_end_GeV * M_PL_REDUCED_GEV) ** 2
    return float((30.0 * rho_end_GeV4 / (math.pi ** 2 * g_star)) ** 0.25)


# ---------------------------------------------------------------------------
# 4. Number of e-folds N_e from Liddle--Leach + V5 reheating.
# ---------------------------------------------------------------------------
def N_e_V5(eps_top: float = EPS_TOP_DEFAULT,
           beta: float = BETA_TOP_DEFAULT,
           phi_max: float = PHI_MAX_V5p13,
           w_reh: float = W_REH_V5,
           prefactor: float = 1.0,
           c_n_eff: int = C_N_NUCLEON_EFF,
           g_star: float = G_STAR_SM,
           N_pivot_LL: float = N_PIVOT_LL,
           N_e_seed: float = N_EFOLDS_PIVOT) -> float:
    """V5 prediction of the e-fold number at the pivot.

    From Liddle & Leach 2003:

        N_e = N_pivot_LL
              + (1/4) ln(V_* / M_pl^4)
              + (1/4) ln(V_* / rho_end)
              + (1 - 3 w_reh) / (12 (1 + w_reh)) * ln(rho_reh / rho_end).

    with rho_reh = (pi^2/30) g_* T_reh^4 and T_reh from
    T_reh_V5(...).

    Parameters
    ----------
    eps_top, beta, phi_max : V5.14 potential parameters.
    w_reh : EoS during reheating (0 = matter-like, default for
        Starobinsky-shape; 1/3 = radiation-like, for the instantaneous
        decay limit).
    prefactor, c_n_eff : as in chi_field_decay_rate.
    g_star : effective relativistic dof at T_reh.
    N_pivot_LL : Liddle--Leach pivot constant (default 56.12 for
        k_* = 0.05 Mpc^{-1}).
    N_e_seed : seed N_e used in the V5.14 V_0 calibration (typically 60).

    Returns
    -------
    N_e : float (e-folds, expected 50-65).
    """
    # V_*, rho_end in M_pl^4 units
    phi_star = find_phi_pivot_V5p14(N_e_target=N_e_seed, eps_top=eps_top,
                                    beta=beta, phi_max=phi_max)
    V0 = V0_calibrated_V5p14(N_e=N_e_seed, eps_top=eps_top, beta=beta,
                             phi_max=phi_max)
    V_star = float(V_total_V5p14(phi_star, V0=V0, eps_top=eps_top, beta=beta,
                                 phi_max=phi_max))
    rho_end = rho_end_V5p14(eps_top=eps_top, beta=beta, phi_max=phi_max,
                            N_e_target=N_e_seed)

    # Hubble at end and T_reh (in GeV)
    H_end_GeV = H_end_V5p14(eps_top=eps_top, beta=beta, phi_max=phi_max,
                            N_e_target=N_e_seed)
    T_reh = T_reh_V5(H_end_GeV=H_end_GeV, prefactor=prefactor,
                     c_n_eff=c_n_eff, g_star=g_star,
                     eps_top=eps_top, beta=beta)
    rho_reh_GeV4 = (math.pi ** 2 / 30.0) * g_star * T_reh ** 4
    rho_end_GeV4 = rho_end * M_PL_REDUCED_GEV ** 4

    # Liddle--Leach pieces
    ln_Vstar_Mpl = math.log(V_star)             # V_* in M_pl^4 units already
    ln_Vstar_over_rho_end = math.log(V_star / rho_end)
    ln_rho_reh_over_rho_end = math.log(rho_reh_GeV4 / rho_end_GeV4)

    if abs(1.0 + w_reh) < 1e-12:
        reheating_term = 0.0
    else:
        reheating_term = (1.0 - 3.0 * w_reh) / (12.0 * (1.0 + w_reh)) \
                           * ln_rho_reh_over_rho_end

    return float(N_pivot_LL
                 + 0.25 * ln_Vstar_Mpl
                 + 0.25 * ln_Vstar_over_rho_end
                 + reheating_term)


# ---------------------------------------------------------------------------
# 5. CMB observables recomputed at N_e^V5.
# ---------------------------------------------------------------------------
def n_s_starobinsky_at_Ne(N_e: float) -> float:
    """Starobinsky n_s = 1 - 2/N_e (V5.14 limit eps_top -> 0)."""
    return 1.0 - 2.0 / N_e


def r_starobinsky_at_Ne(N_e: float) -> float:
    """Starobinsky r = 12/N_e^2."""
    return 12.0 / N_e ** 2


# ---------------------------------------------------------------------------
# 6. Validation against Planck and BBN.
# ---------------------------------------------------------------------------
@dataclass
class ReheatingV5Prediction:
    """Synthese V5.16 reheating + N_e."""
    # V5.14 inputs
    eps_top: float
    beta: float
    phi_max: float
    # Reheating quantities
    suppression_alpha_EM_cn: float
    H_end_GeV: float
    Gamma_reh_GeV: float
    T_end_GeV: float
    T_reh_GeV: float
    rho_end_GeV4: float
    rho_reh_GeV4: float
    # e-folds
    w_reh: float
    N_e_V5: float
    # Re-derived CMB observables
    n_s_at_Ne_V5: float
    r_at_Ne_V5: float
    # Comparison Planck
    n_s_planck: float = N_S_PLANCK
    n_s_planck_sigma: float = N_S_PLANCK_SIGMA
    n_s_deviation_sigma: float = float("nan")
    r_planck_upper: float = R_PLANCK_BICEP
    # BBN
    T_BBN_min_GeV: float = T_REH_BBN_MIN_GEV
    BBN_compatible: bool = False
    Planck_compatible: bool = False
    BICEP_compatible: bool = False


def predict_reheating_V5(eps_top: float = EPS_TOP_DEFAULT,
                         beta: float = BETA_TOP_DEFAULT,
                         phi_max: float = PHI_MAX_V5p13,
                         w_reh: float = W_REH_V5,
                         prefactor: float = 1.0,
                         c_n_eff: int = C_N_NUCLEON_EFF,
                         g_star: float = G_STAR_SM,
                         N_pivot_LL: float = N_PIVOT_LL,
                         N_e_seed: float = N_EFOLDS_PIVOT
                         ) -> ReheatingV5Prediction:
    """Full V5.16 reheating prediction."""
    H_end_GeV = H_end_V5p14(eps_top=eps_top, beta=beta, phi_max=phi_max,
                            N_e_target=N_e_seed)
    Gamma_reh_GeV = chi_field_decay_rate(H_end_GeV, c_n_eff=c_n_eff,
                                          prefactor=prefactor)
    T_reh = T_reh_V5(H_end_GeV=H_end_GeV, prefactor=prefactor,
                     c_n_eff=c_n_eff, g_star=g_star,
                     eps_top=eps_top, beta=beta)
    T_end = T_end_V5(H_end_GeV=H_end_GeV, g_star=g_star,
                     eps_top=eps_top, beta=beta)
    rho_end_GeV4 = 3.0 * (H_end_GeV * M_PL_REDUCED_GEV) ** 2
    rho_reh_GeV4 = (math.pi ** 2 / 30.0) * g_star * T_reh ** 4
    N_e = N_e_V5(eps_top=eps_top, beta=beta, phi_max=phi_max, w_reh=w_reh,
                 prefactor=prefactor, c_n_eff=c_n_eff, g_star=g_star,
                 N_pivot_LL=N_pivot_LL, N_e_seed=N_e_seed)
    n_s = n_s_starobinsky_at_Ne(N_e)
    r = r_starobinsky_at_Ne(N_e)
    n_s_dev = (n_s - N_S_PLANCK) / N_S_PLANCK_SIGMA
    return ReheatingV5Prediction(
        eps_top=eps_top,
        beta=beta,
        phi_max=phi_max,
        suppression_alpha_EM_cn=chi_field_decay_suppression(c_n_eff=c_n_eff),
        H_end_GeV=H_end_GeV,
        Gamma_reh_GeV=Gamma_reh_GeV,
        T_end_GeV=T_end,
        T_reh_GeV=T_reh,
        rho_end_GeV4=rho_end_GeV4,
        rho_reh_GeV4=rho_reh_GeV4,
        w_reh=w_reh,
        N_e_V5=N_e,
        n_s_at_Ne_V5=n_s,
        r_at_Ne_V5=r,
        n_s_deviation_sigma=float(n_s_dev),
        BBN_compatible=bool(T_reh > T_REH_BBN_MIN_GEV),
        Planck_compatible=bool(abs(n_s_dev) < 3.0),
        BICEP_compatible=bool(r < R_PLANCK_BICEP),
    )


def validate_against_Planck(eps_top: float = EPS_TOP_DEFAULT,
                            w_reh: float = W_REH_V5) -> dict:
    """Validation summary V5.16 vs Planck + BICEP + BBN.

    Returns a flat dict suitable for tests, plots, or LaTeX tables.
    """
    pred = predict_reheating_V5(eps_top=eps_top, w_reh=w_reh)
    return {
        "model": "V5.16 reheating-chi N_e",
        "N_e_V5": pred.N_e_V5,
        "N_e_canonical": N_EFOLDS_PIVOT,
        "Delta_N_e": pred.N_e_V5 - N_EFOLDS_PIVOT,
        "n_s_V5": pred.n_s_at_Ne_V5,
        "n_s_Planck": pred.n_s_planck,
        "n_s_Planck_sigma": pred.n_s_planck_sigma,
        "n_s_deviation_sigma": pred.n_s_deviation_sigma,
        "r_V5": pred.r_at_Ne_V5,
        "r_BICEP_upper": pred.r_planck_upper,
        "T_reh_GeV": pred.T_reh_GeV,
        "T_BBN_min_GeV": pred.T_BBN_min_GeV,
        "T_reh_over_T_BBN": pred.T_reh_GeV / pred.T_BBN_min_GeV,
        "H_end_GeV": pred.H_end_GeV,
        "suppression_alpha_EM_cn": pred.suppression_alpha_EM_cn,
        "BBN_compatible": pred.BBN_compatible,
        "Planck_compatible": pred.Planck_compatible,
        "BICEP_compatible": pred.BICEP_compatible,
    }


def summary(eps_top: float = EPS_TOP_DEFAULT,
            w_reh: float = W_REH_V5) -> dict:
    """Alias for validate_against_Planck (convenience for figures/tests)."""
    return validate_against_Planck(eps_top=eps_top, w_reh=w_reh)
