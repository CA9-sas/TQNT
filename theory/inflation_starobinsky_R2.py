"""
V5.14 — Correction Chern-Simons R² (type Starobinsky) au potentiel V5.13.
============================================================================

V5.13 (`theory/inflation_chi_field.py`) a dérivé un potentiel topologique pur
en loi de puissance inverse V(φ) ∝ φ^{-q}, q = 2 α_cond ≈ 2,398.
Verdict honnête (Planck 2018 + BICEP/Keck 2021) :
  - A_s = 2,10·10⁻⁹ reproduit par calibration V_0
  - n_s = 0,9967 vs Planck 0,9649 ± 0,0042 → tension +7,6 σ
  - r   = 0,158 vs BICEP < 0,06 (95 % CL) → violation ×2,6

La V5.13 est donc structurellement FALSIFIÉE par les données primordiales :
les potentiels purement en loi de puissance inverse produisent toujours des
tilts spectraux BLEUS, incompatibles avec le tilt rouge observé.

V5.14 — IDÉE PHYSIQUE
---------------------

L'action gravitationnelle V5 contient, en plus du terme Einstein-Hilbert,
une correction de courbure d'origine Chern-Simons :

    S = ∫ d⁴x √(-g) [ R/(16πG) + a · R² / (384πG) + L_χ ]

Le coefficient `a` a la dimension d'une longueur². Starobinsky (1980,
*Phys.Lett.B* **91** 99) a montré qu'après une transformation conforme
en jauge d'Einstein, R + a R²/(384πG) est équivalent à un champ scalaire
canonique avec potentiel

    V_Starobinsky(φ) = (3 M_pl² / 4a) · [1 - exp(-√(2/3) φ/M_pl)]²

Ce potentiel produit, pour N_e = 60 e-folds :
    n_s ≈ 1 - 2/N_e = 0,967    ⟵ compatible Planck (0,965 ± 0,004)
    r   ≈ 12/N_e² = 0,0033     ⟵ compatible BICEP (r < 0,06)

V5.14 conjecture que le terme R² *coexiste* avec la composante topologique
V5.13 et la *domine* à bas champ. Le potentiel total devient alors

    V_total(φ) = V_Star(φ) · [1 + ε_top · (φ/φ_max)^β]    (forme A — multiplicative)

ou, de manière équivalente à l'ordre dominant à bas champ,

    V_total(φ) = V_Star(φ) + V_top(φ)                     (forme B — additive)

où :
    V_Star(φ) = V_0 · (1 - exp(-√(2/3) φ/M_pl))²    ⟵ plateau dominant
    V_top(φ)  = V_0 · ε_top · (φ/φ_max)^{-q}        ⟵ résidu topologique V5.13
    ε_top  ≪ 1    : petit paramètre de résidu topologique
    φ_max ~ 17 M_pl  : échelle V5.13 (= φ_★(N=60) V5.13)
    β     ~ -q = -2,40 ou choisi pour préserver le plateau

Forme retenue : MULTIPLICATIVE (A), parce que :
  (i)  elle préserve le zéro V(0) = 0 du modèle Starobinsky (graceful exit)
  (ii) elle se réduit *exactement* à Starobinsky pour ε_top → 0
  (iii) ε_top mesure la fraction du potentiel d'origine topologique pure

Avec β > 0 (par défaut β = +q = 2,398), le résidu (φ/φ_max)^β :
  - CROÎT à grand φ (régime UV) : on récupère asymptotiquement la
    structure V5.13 pour φ ≫ φ_max
  - DÉCROÎT à bas φ (régime IR) : le plateau Starobinsky est préservé
    intact dans la fenêtre slow-roll (φ_★ ≈ 5,5 M_pl ≪ φ_max ≈ 17 M_pl)

----------------------------------------------------------------------------
DÉRIVATION DU COEFFICIENT a (HONNÊTE)
----------------------------------------------------------------------------

V5 prédit la structure SU(2)_135. Le coefficient a du terme R² dans une
théorie Chern-Simons à niveau k est, sur la base des arguments
dimensionnels et de l'identification κ_g = α_EM^c_n × κ₀ (V5.5) :

    a_V5 ~ M_pl⁻² × α_EM^{-x}   ?

Une dérivation rigoureuse depuis l'action TQNT-V5 n'est PAS encore
disponible. À ce stade, V5.14 paramétrise honnêtement :

    V_0 = échelle absolue (calibrée sur A_s = 2,10·10⁻⁹, comme en V5.13)
    ε_top = fraction de résidu V5.13 (PARAMÈTRE EMPIRIQUE pour l'instant)
    β = exposant de résidu (PARAMÈTRE EMPIRIQUE)

L'identification rigoureuse a_V5 = f(κ_g, α_EM, k=135) est laissée comme
travail futur (V5.15). Cette honnêteté est essentielle : V5.14 ne PRÉTEND
PAS dériver Starobinsky depuis V5, elle montre qu'une correction R² est
*compatible* avec la structure V5 et *suffit* pour reconcilier Planck.

----------------------------------------------------------------------------
PRÉDICTIONS V5.14 (cible)
----------------------------------------------------------------------------

À ε_top = 0 (limite pure Starobinsky) :
    n_s(N_e=60) = 0,9667        — compatible Planck (0,4 σ)
    r(N_e=60)   = 0,0033        — compatible BICEP

Avec ε_top petit (correction topologique sous-dominante) :
    n_s glisse légèrement     (test : ≲ 1 σ Planck pour ε_top < 0,1)
    r reste très inférieur à 0,06

----------------------------------------------------------------------------
LIMITES HONNÊTES V5.14
----------------------------------------------------------------------------

(a) Le coefficient a (Starobinsky) n'est PAS encore dérivé depuis V5.
    À ε_top = 0, V_0 est calibré sur A_s ⇒ a = 3 M_pl² / (4 V_0) = libre.
(b) ε_top et β sont actuellement EMPIRIQUES (deux paramètres libres
    supplémentaires par rapport à V5.13 — mais nécessaires pour ouvrir
    le couloir de réconciliation Planck/BICEP).
(c) La dérivation du terme R² depuis l'action Chern-Simons V5 reste à
    faire (V5.15). On donne ici uniquement le cadre phénoménologique.
(d) Le passage du plateau Starobinsky au régime V5.13 à grand φ se fait
    naturellement par la multiplication 1 + ε_top (φ/φ_max)^β.

Référence : V5.13 `theory/inflation_chi_field.py`,
TQNT_detaillee_V5_FR.tex §17 (gravité émergente P5/P19),
Starobinsky 1980 *Phys.Lett.B* 91:99.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from theory.inflation_chi_field import (
    A_S_PLANCK,
    A_S_PLANCK_SIGMA,
    ALPHA_COND_V5,
    M_PL_GEV,
    M_PL_KG,
    N_EFOLDS_PIVOT,
    N_S_PLANCK,
    N_S_PLANCK_SIGMA,
    Q_INFLATION_V5,
    R_PLANCK_BICEP,
    find_psi_pivot,
)

# ---------------------------------------------------------------------------
# Constantes V5.14 — paramètres de l'interpolation Starobinsky + résidu V5.13.
# ---------------------------------------------------------------------------

# Coefficient Starobinsky canonique : V_Star(φ) = V_0 (1 - e^{-c φ})² avec
# c = √(2/3), forme universelle d'une transformée conforme R + a R².
C_STAROBINSKY = math.sqrt(2.0 / 3.0)

# Échelle de "transition" résidu topologique → plateau : on prend φ_max =
# φ_★(V5.13) à N_e = 60, soit le point pivot V5.13 historique.
PHI_MAX_V5p13 = find_psi_pivot(N_e_target=N_EFOLDS_PIVOT,
                               alpha_cond=ALPHA_COND_V5)
# ≈ 17,05 M_pl

# Fraction de résidu topologique V5.13 (paramètre empirique honnête).
# ε_top = 0 ⇒ Starobinsky pur ; ε_top > 0 ⇒ déformation au-delà du plateau.
# Choix par défaut conservatif : 0,001 (résidu de 0,1 % au point φ = φ_max).
EPS_TOP_DEFAULT = 1.0e-3

# Exposant du résidu V5.13 dans l'enveloppe multiplicative.
# β > 0 : le résidu CROÎT à grand φ (on récupère V5.13 dans l'UV) et
#        DÉCROÎT à bas φ (plateau Starobinsky préservé).
# Choix par défaut : β = +q = 2,398 (symétrie avec V5.13).
BETA_TOP_DEFAULT = Q_INFLATION_V5

# Cible Starobinsky standard à N_e = 60.
N_S_STAROBINSKY_NE60 = 1.0 - 2.0 / N_EFOLDS_PIVOT       # ≈ 0,9667
R_STAROBINSKY_NE60 = 12.0 / N_EFOLDS_PIVOT ** 2          # ≈ 0,00333


# ---------------------------------------------------------------------------
# 1. Potentiel Starobinsky R² standard.
# ---------------------------------------------------------------------------
def V_starobinsky_R2(phi: float | np.ndarray,
                     V0: float = 1.0) -> float | np.ndarray:
    """Potentiel Starobinsky R² standard en jauge d'Einstein.

        V(φ) = V_0 · (1 - exp(-√(2/3) φ/M_pl))²

    En unités M_pl = 1. Le coefficient √(2/3) est UNIVERSEL pour une
    action R + a R² après transformation conforme (Starobinsky 1980 ;
    Whitt 1984 ; Maeda 1988).

    V_0 = 3 M_pl² / (4 a)  où a est le coefficient du terme a R² dans
    l'action.

    Paramètres
    ----------
    phi : float | array
        Champ inflaton en unités M_pl.
    V0 : float
        Échelle d'énergie absolue (calibrée par A_s).

    Retour
    ------
    V : float | array (mêmes unités que V_0).
    """
    phi = np.asarray(phi, dtype=float)
    return V0 * (1.0 - np.exp(-C_STAROBINSKY * phi)) ** 2


def Vprime_starobinsky_R2(phi: float | np.ndarray,
                          V0: float = 1.0) -> float | np.ndarray:
    """dV/dφ analytique du potentiel Starobinsky."""
    phi = np.asarray(phi, dtype=float)
    e = np.exp(-C_STAROBINSKY * phi)
    return V0 * 2.0 * C_STAROBINSKY * (1.0 - e) * e


def Vsecond_starobinsky_R2(phi: float | np.ndarray,
                           V0: float = 1.0) -> float | np.ndarray:
    """d²V/dφ² analytique du potentiel Starobinsky."""
    phi = np.asarray(phi, dtype=float)
    e = np.exp(-C_STAROBINSKY * phi)
    c2 = C_STAROBINSKY ** 2
    # d/dφ [2c (1-e) e] = 2c · (-(-c)e · e + (1-e)(-c)e)
    #                  = 2c² e · (e - (1-e))
    #                  = 2c² e · (2e - 1)
    return V0 * 2.0 * c2 * e * (2.0 * e - 1.0)


# ---------------------------------------------------------------------------
# 2. Potentiel V5.14 = Starobinsky × (1 + ε_top résidu V5.13).
# ---------------------------------------------------------------------------
def envelope_V5p13_residual(phi: float | np.ndarray,
                            eps_top: float = EPS_TOP_DEFAULT,
                            beta: float = BETA_TOP_DEFAULT,
                            phi_max: float = PHI_MAX_V5p13
                            ) -> float | np.ndarray:
    """Enveloppe multiplicative de résidu V5.13.

        E(φ) = 1 + ε_top · (φ/φ_max)^β

    β = -q par défaut ⇒ E(φ) = 1 + ε_top (φ_max/φ)^q
    Le résidu décroît à grand φ (récupère le plateau Starobinsky pur à
    asymptotique) et croît à bas φ (mais ne déforme pas le minimum
    V(0)=0 du potentiel Starobinsky car celui-ci vaut 0 quoi que vaille
    l'enveloppe).
    """
    phi = np.asarray(phi, dtype=float)
    # Pour β < 0, le pôle à φ → 0 est inoffensif car multiplié par
    # V_Star(φ) qui tend vers 0 (avec φ² puisque (1-e^{-cφ})² ~ c²φ²).
    # On régularise quand même pour éviter NaN à φ = 0 exactement.
    phi_safe = np.where(phi > 1e-12, phi, 1e-12)
    return 1.0 + eps_top * (phi_safe / phi_max) ** beta


def V_total_V5p14(phi: float | np.ndarray,
                  V0: float = 1.0,
                  eps_top: float = EPS_TOP_DEFAULT,
                  beta: float = BETA_TOP_DEFAULT,
                  phi_max: float = PHI_MAX_V5p13
                  ) -> float | np.ndarray:
    """Potentiel V5.14 — Starobinsky modulé par résidu topologique V5.13.

        V_total(φ) = V_0 · [1 - exp(-√(2/3) φ)]² · [1 + ε_top · (φ/φ_max)^β]

    À ε_top = 0 : Starobinsky pur (n_s, r compatibles Planck/BICEP).
    À ε_top > 0 : déformation par résidu topologique V5.13.

    Paramètres
    ----------
    phi : float | array
        Champ inflaton en unités M_pl.
    V0 : float
        Échelle d'énergie (calibrée par A_s).
    eps_top : float
        Fraction de résidu V5.13 (défaut 0,01).
    beta : float
        Exposant de l'enveloppe (défaut -q = -2,398).
    phi_max : float
        Échelle de transition (défaut φ_★ V5.13 ≈ 17 M_pl).
    """
    V_s = V_starobinsky_R2(phi, V0=V0)
    E = envelope_V5p13_residual(phi, eps_top=eps_top, beta=beta,
                                phi_max=phi_max)
    return V_s * E


def Vprime_total_V5p14(phi: float | np.ndarray,
                       V0: float = 1.0,
                       eps_top: float = EPS_TOP_DEFAULT,
                       beta: float = BETA_TOP_DEFAULT,
                       phi_max: float = PHI_MAX_V5p13
                       ) -> float | np.ndarray:
    """dV_total/dφ analytique.

    V_total = V_Star · E avec E = 1 + ε_top (φ/φ_max)^β.
    V_total' = V_Star' · E + V_Star · E'
             = V_Star' · E + V_Star · (ε_top β/φ) · (φ/φ_max)^β
    """
    phi = np.asarray(phi, dtype=float)
    phi_safe = np.where(phi > 1e-12, phi, 1e-12)
    V_s = V_starobinsky_R2(phi_safe, V0=V0)
    Vp_s = Vprime_starobinsky_R2(phi_safe, V0=V0)
    E = envelope_V5p13_residual(phi_safe, eps_top=eps_top, beta=beta,
                                phi_max=phi_max)
    Eprime = eps_top * beta / phi_safe * (phi_safe / phi_max) ** beta
    return Vp_s * E + V_s * Eprime


def Vsecond_total_V5p14(phi: float | np.ndarray,
                        V0: float = 1.0,
                        eps_top: float = EPS_TOP_DEFAULT,
                        beta: float = BETA_TOP_DEFAULT,
                        phi_max: float = PHI_MAX_V5p13,
                        h: float = 1e-4
                        ) -> float | np.ndarray:
    """d²V_total/dφ² par différence finie centrée (suffisant pour slow-roll)."""
    phi = np.asarray(phi, dtype=float)
    Vp = V_total_V5p14(phi + h, V0=V0, eps_top=eps_top, beta=beta,
                       phi_max=phi_max)
    V0_ = V_total_V5p14(phi, V0=V0, eps_top=eps_top, beta=beta,
                        phi_max=phi_max)
    Vm = V_total_V5p14(phi - h, V0=V0, eps_top=eps_top, beta=beta,
                       phi_max=phi_max)
    return (Vp - 2.0 * V0_ + Vm) / (h ** 2)


# ---------------------------------------------------------------------------
# 3. Slow-roll de V_total V5.14.
# ---------------------------------------------------------------------------
def slow_roll_epsilon_V5p14(phi: float | np.ndarray,
                            eps_top: float = EPS_TOP_DEFAULT,
                            beta: float = BETA_TOP_DEFAULT,
                            phi_max: float = PHI_MAX_V5p13
                            ) -> float | np.ndarray:
    """ε(φ) = (1/2) (V'/V)² du potentiel V5.14, M_pl = 1."""
    V_ = V_total_V5p14(phi, V0=1.0, eps_top=eps_top, beta=beta,
                       phi_max=phi_max)
    Vp = Vprime_total_V5p14(phi, V0=1.0, eps_top=eps_top, beta=beta,
                            phi_max=phi_max)
    return 0.5 * (Vp / V_) ** 2


def slow_roll_eta_V5p14(phi: float | np.ndarray,
                        eps_top: float = EPS_TOP_DEFAULT,
                        beta: float = BETA_TOP_DEFAULT,
                        phi_max: float = PHI_MAX_V5p13
                        ) -> float | np.ndarray:
    """η(φ) = V''/V du potentiel V5.14, M_pl = 1."""
    V_ = V_total_V5p14(phi, V0=1.0, eps_top=eps_top, beta=beta,
                       phi_max=phi_max)
    Vpp = Vsecond_total_V5p14(phi, V0=1.0, eps_top=eps_top, beta=beta,
                              phi_max=phi_max)
    return Vpp / V_


# ---------------------------------------------------------------------------
# 4. Champ de fin d'inflation, e-folds, point pivot.
# ---------------------------------------------------------------------------
def phi_end_inflation_V5p14(eps_top: float = EPS_TOP_DEFAULT,
                            beta: float = BETA_TOP_DEFAULT,
                            phi_max: float = PHI_MAX_V5p13) -> float:
    """φ_end tel que ε(φ_end) = 1 (fin slow-roll)."""
    # Recherche par bissection : ε(φ) décroissante de l'ordre exp à grand φ,
    # croît à bas φ. On cherche la plus petite racine > 0.
    # Test : ε(φ) ~ 4 c² e^{-2cφ} / (cφ)⁴ ... à bas φ V_Star ~ c²φ²
    # ⇒ V' ~ 2 c² φ, V ~ c² φ², ε ~ (1/2)(2/φ)² = 2/φ² (Starobinsky bas φ).
    # ε = 1 ⇒ φ ~ √2.
    from scipy.optimize import brentq

    def f(phi):
        return slow_roll_epsilon_V5p14(phi, eps_top=eps_top, beta=beta,
                                       phi_max=phi_max) - 1.0

    # Bornes : ε(0.1) ≫ 1, ε(10) ≪ 1.
    return float(brentq(f, 0.1, 10.0, xtol=1e-8))


def N_efolds_V5p14(phi_initial: float,
                   phi_final: float | None = None,
                   eps_top: float = EPS_TOP_DEFAULT,
                   beta: float = BETA_TOP_DEFAULT,
                   phi_max: float = PHI_MAX_V5p13,
                   n_points: int = 1000) -> float:
    """Nombre d'e-folds entre phi_initial et phi_final (intégration trapèze).

        N_e = ∫_{φ_f}^{φ_i} (V/V') dφ
    """
    if phi_final is None:
        phi_final = phi_end_inflation_V5p14(eps_top=eps_top, beta=beta,
                                            phi_max=phi_max)
    if phi_initial <= phi_final:
        return 0.0
    phi_grid = np.linspace(phi_final, phi_initial, n_points)
    V_ = V_total_V5p14(phi_grid, V0=1.0, eps_top=eps_top, beta=beta,
                       phi_max=phi_max)
    Vp = Vprime_total_V5p14(phi_grid, V0=1.0, eps_top=eps_top, beta=beta,
                            phi_max=phi_max)
    integrand = V_ / Vp
    return float(np.trapezoid(integrand, phi_grid))


def find_phi_pivot_V5p14(N_e_target: float = N_EFOLDS_PIVOT,
                         eps_top: float = EPS_TOP_DEFAULT,
                         beta: float = BETA_TOP_DEFAULT,
                         phi_max: float = PHI_MAX_V5p13) -> float:
    """φ_★ tel que N_e(φ_★ → φ_end) = N_e_target.

    Bissection sur φ_★ entre φ_end et 20 M_pl.
    """
    from scipy.optimize import brentq

    phi_end = phi_end_inflation_V5p14(eps_top=eps_top, beta=beta,
                                      phi_max=phi_max)

    def f(phi):
        return (N_efolds_V5p14(phi, phi_final=phi_end, eps_top=eps_top,
                               beta=beta, phi_max=phi_max)
                - N_e_target)

    # φ_★ Starobinsky pur à N=60 ≈ ½ ln(4N) + ½ ln(3) ≈ 5,5 M_pl.
    return float(brentq(f, phi_end + 0.1, 20.0, xtol=1e-6))


def find_phi_pivot_N60(eps_top: float = EPS_TOP_DEFAULT,
                       beta: float = BETA_TOP_DEFAULT,
                       phi_max: float = PHI_MAX_V5p13) -> float:
    """Alias commodité : φ_★(N_e=60)."""
    return find_phi_pivot_V5p14(N_e_target=60.0, eps_top=eps_top, beta=beta,
                                phi_max=phi_max)


# ---------------------------------------------------------------------------
# 5. Observables CMB V5.14.
# ---------------------------------------------------------------------------
def n_s_V5p14(N_e: float = N_EFOLDS_PIVOT,
              eps_top: float = EPS_TOP_DEFAULT,
              beta: float = BETA_TOP_DEFAULT,
              phi_max: float = PHI_MAX_V5p13) -> float:
    """Tilt spectral scalaire V5.14.

        n_s = 1 - 6 ε(φ_★) + 2 η(φ_★)

    À ε_top = 0 : n_s = 1 - 2/N_e ≈ 0,967 (Starobinsky pur).
    """
    phi_star = find_phi_pivot_V5p14(N_e_target=N_e, eps_top=eps_top,
                                    beta=beta, phi_max=phi_max)
    eps = slow_roll_epsilon_V5p14(phi_star, eps_top=eps_top, beta=beta,
                                  phi_max=phi_max)
    et = slow_roll_eta_V5p14(phi_star, eps_top=eps_top, beta=beta,
                             phi_max=phi_max)
    return float(1.0 - 6.0 * eps + 2.0 * et)


def r_V5p14(N_e: float = N_EFOLDS_PIVOT,
            eps_top: float = EPS_TOP_DEFAULT,
            beta: float = BETA_TOP_DEFAULT,
            phi_max: float = PHI_MAX_V5p13) -> float:
    """Rapport tenseur-scalaire V5.14.

        r = 16 ε(φ_★)

    À ε_top = 0 : r = 12/N_e² ≈ 0,003 (Starobinsky pur).
    """
    phi_star = find_phi_pivot_V5p14(N_e_target=N_e, eps_top=eps_top,
                                    beta=beta, phi_max=phi_max)
    eps = slow_roll_epsilon_V5p14(phi_star, eps_top=eps_top, beta=beta,
                                  phi_max=phi_max)
    return float(16.0 * eps)


def V0_calibrated_V5p14(N_e: float = N_EFOLDS_PIVOT,
                        eps_top: float = EPS_TOP_DEFAULT,
                        beta: float = BETA_TOP_DEFAULT,
                        phi_max: float = PHI_MAX_V5p13,
                        A_s_target: float = A_S_PLANCK) -> float:
    """V_0/M_pl⁴ calibré pour reproduire A_s_target.

        A_s = V(φ_★) / (24 π² ε(φ_★) M_pl⁴)
        ⇒ V_0/M_pl⁴ = 24 π² ε(φ_★) A_s / [V(φ_★)/V_0]
    """
    phi_star = find_phi_pivot_V5p14(N_e_target=N_e, eps_top=eps_top,
                                    beta=beta, phi_max=phi_max)
    eps = slow_roll_epsilon_V5p14(phi_star, eps_top=eps_top, beta=beta,
                                  phi_max=phi_max)
    V_over_V0 = V_total_V5p14(phi_star, V0=1.0, eps_top=eps_top, beta=beta,
                              phi_max=phi_max)
    return float(24.0 * math.pi ** 2 * eps * A_s_target / V_over_V0)


# ---------------------------------------------------------------------------
# 6. Synthèse.
# ---------------------------------------------------------------------------
@dataclass
class InflationV5p14Prediction:
    """Synthèse des prédictions V5.14."""
    eps_top: float
    beta: float
    phi_max: float
    N_e: float
    phi_end: float
    phi_star: float
    epsilon: float
    eta: float
    n_s: float
    r: float
    V0_over_Mpl4: float
    A_s: float
    # Comparaison Planck/BICEP
    n_s_planck: float = N_S_PLANCK
    n_s_planck_sigma: float = N_S_PLANCK_SIGMA
    n_s_deviation_sigma: float = float("nan")
    r_upper_planck: float = R_PLANCK_BICEP
    r_passes_upper: bool = False
    A_s_planck: float = A_S_PLANCK
    compatible_planck_n_s: bool = False
    compatible_BICEP_r: bool = False


def predict_inflation_V5p14(N_e: float = N_EFOLDS_PIVOT,
                            eps_top: float = EPS_TOP_DEFAULT,
                            beta: float = BETA_TOP_DEFAULT,
                            phi_max: float = PHI_MAX_V5p13
                            ) -> InflationV5p14Prediction:
    """Synthèse complète V5.14."""
    phi_end = phi_end_inflation_V5p14(eps_top=eps_top, beta=beta,
                                      phi_max=phi_max)
    phi_star = find_phi_pivot_V5p14(N_e_target=N_e, eps_top=eps_top,
                                    beta=beta, phi_max=phi_max)
    eps = slow_roll_epsilon_V5p14(phi_star, eps_top=eps_top, beta=beta,
                                  phi_max=phi_max)
    et = slow_roll_eta_V5p14(phi_star, eps_top=eps_top, beta=beta,
                             phi_max=phi_max)
    n_s = 1.0 - 6.0 * float(eps) + 2.0 * float(et)
    r = 16.0 * float(eps)
    V0 = V0_calibrated_V5p14(N_e=N_e, eps_top=eps_top, beta=beta,
                             phi_max=phi_max)
    deviation = (n_s - N_S_PLANCK) / N_S_PLANCK_SIGMA
    return InflationV5p14Prediction(
        eps_top=eps_top,
        beta=beta,
        phi_max=phi_max,
        N_e=N_e,
        phi_end=phi_end,
        phi_star=phi_star,
        epsilon=float(eps),
        eta=float(et),
        n_s=n_s,
        r=r,
        V0_over_Mpl4=V0,
        A_s=A_S_PLANCK,
        n_s_deviation_sigma=float(deviation),
        r_passes_upper=bool(r < R_PLANCK_BICEP),
        compatible_planck_n_s=bool(abs(deviation) < 3.0),
        compatible_BICEP_r=bool(r < R_PLANCK_BICEP),
    )


def summary_V5p14(N_e: float = N_EFOLDS_PIVOT,
                  eps_top: float = EPS_TOP_DEFAULT,
                  beta: float = BETA_TOP_DEFAULT,
                  phi_max: float = PHI_MAX_V5p13) -> dict:
    """Dictionnaire récapitulatif V5.14 (pour figures/tests/docs)."""
    pred = predict_inflation_V5p14(N_e=N_e, eps_top=eps_top, beta=beta,
                                   phi_max=phi_max)
    return {
        "model": "V5.14 Starobinsky + résidu V5.13",
        "eps_top": pred.eps_top,
        "beta": pred.beta,
        "phi_max": pred.phi_max,
        "N_e": pred.N_e,
        "phi_end_Mpl": pred.phi_end,
        "phi_star_Mpl": pred.phi_star,
        "epsilon_star": pred.epsilon,
        "eta_star": pred.eta,
        "n_s_V5p14": pred.n_s,
        "n_s_Planck": pred.n_s_planck,
        "n_s_deviation_sigma": pred.n_s_deviation_sigma,
        "r_V5p14": pred.r,
        "r_Planck_upper": pred.r_upper_planck,
        "r_passes_upper": pred.r_passes_upper,
        "A_s_target": pred.A_s,
        "V0_over_Mpl4_calibrated": pred.V0_over_Mpl4,
        "compatible_planck_n_s": pred.compatible_planck_n_s,
        "compatible_BICEP_r": pred.compatible_BICEP_r,
    }
