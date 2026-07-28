"""
V5.13 — Inflation topologique du champ χ.
============================================================================

Ce module dérive de la TQNT-V5 le **spectre primordial** des perturbations
cosmologiques scalaires et tensorielles dans le scénario où le champ χ
joue le rôle d'inflaton.

V5.11 (theory/flrw_emergence.py) et V5.12 (theory/cmb_perturbations.py)
prenaient l'amplitude scalaire A_s = 2.10e-9 et le tilt n_s = 0.965 comme
ENTRÉES (best-fit Planck 2018). Ici on les PRÉDIT depuis :

  1. l'identification de l'inflaton φ = √(|Ψ|²) au champ χ V5,
  2. un potentiel topologique V(|Ψ|²) dérivé de la pénalité de
     condensation α_cond = 1,1991 (P18, V5 §13.6.4),
  3. l'approximation slow-roll standard (Liddle & Lyth 2000).

----------------------------------------------------------------------------
CADRE THÉORIQUE
----------------------------------------------------------------------------

(1) Inflaton χ. Le champ scalaire complexe Ψ admet la décomposition polaire
    Ψ = φ e^{iθ}/√2 (cf. V5 P5). Le module φ = √(2|Ψ|²) est invariant
    par U(1) et joue le rôle d'inflaton.

(2) Potentiel topologique V(φ). Dans la phase pré-inflationnaire, |Ψ|² est
    une *énergie de condensation* topologique : la pénalité par croisement
    actif est α_cond = 1,1991 (V5.5 étape 3, eqq. § 13.6.4). Le nombre
    effectif de croisements c(φ²) que doit traverser une boucle pour
    s'inscrire dans le réseau Chern-Simons SU(2)_135 est, dans la limite
    où la densité de boucles couvre la densité d'onde :

        c(φ²) = c_0 - log(φ² / φ_0²)        (saturation logarithmique)

    où c_0 = c_n^eff(nucléon)+1 = 18 (cf. theory/coupling_kg.py), et φ_0
    est l'échelle de condensation du nucléon (calibrée plus bas). Cette
    forme log saturée vient de ce que la « pression » de condensation
    décroît comme 1/φ² à grand champ — cf. V5 chap. « Gravitation par
    exclusion topologique », équation de l'énergie libre topologique.

    On obtient ainsi un potentiel **EN LOI DE PUISSANCE INVERSE** :

        V(φ) = V_0 · (φ² / φ_0²)^{-α_cond}

    On peut écrire de manière équivalente, en notant φ̂ = φ/φ_0 :

        V(φ) = V_0 · φ̂^{-2 α_cond}

    NB : c'est une réalisation explicite du modèle « intermediate inflation »
    de Barrow 1990 / Muslimov 1990 (cf. Liddle & Lyth, §6) dans lequel
    V ∝ φ^{−q} avec q = 2 α_cond ≈ 2,40 (champ φ).

(3) Slow-roll. Les paramètres de slow-roll (en unités M_pl = 1) :

        ε(φ) = (1/2) (V'/V)² = (1/2) (q/φ)² = q²/(2φ²)        (q = 2α_cond)
        η(φ) = V''/V       = q(q+1)/φ²

    L'inflation se termine quand ε = 1 :
        φ_end² = q²/2  ⇒  φ_end = q/√2.

    Le nombre d'e-folds depuis φ_★ jusqu'à φ_end est :
        N(φ_★) = ∫_{φ_end}^{φ_★} V/V' dφ / M_pl²
               = ∫_{φ_end}^{φ_★} φ/q dφ
               = (φ_★² - φ_end²) / (2 q).

    Soit pour N_e = 60 :
        φ_★² = φ_end² + 2 q N_e = q²/2 + 2 q N_e = q (q/2 + 2 N_e).

(4) Observables CMB :
        n_s = 1 - 6 ε(φ_★) + 2 η(φ_★)
        r   = 16 ε(φ_★)
        A_s = V(φ_★) / (24 π² ε(φ_★) M_pl⁴)

    En unités M_pl = 1, A_s détermine V_0/V_★ une fois φ_★ fixé par N_e.

----------------------------------------------------------------------------
JUSTIFICATION TOPOLOGIQUE DE q = 2 α_cond
----------------------------------------------------------------------------

L'argument est topologique pur :

  (i)  La pénalité par CROISEMENT actif est α_cond = 1,1991 (V5.5 étape 3,
       calibrée sur la condensation cosmologique).
  (ii) Le nombre effectif de croisements pondéré par la densité de
       boucles c(φ²) est une fonction décroissante de φ² (à grand champ
       toutes les boucles sont déjà inscrites, plus rien à pénaliser).
       La pente logarithmique d(log c)/d(log φ²) → −1 dans cette limite
       (saturation), d'où c(φ²) ~ c_0 − log(φ²/φ_0²).
  (iii) L'énergie libre topologique vaut F = α_cond · c(φ²) · k_B T_eff,
        d'où V = e^{−F/T_eff} = (φ²/φ_0²)^{−α_cond}.

Le coefficient q = 2 α_cond ≈ 2,40 est ainsi DÉRIVÉ (pas ajusté).

----------------------------------------------------------------------------
LIMITES HONNÊTES
----------------------------------------------------------------------------

(a) PARAMÈTRE LIBRE RESTANT : V_0 (l'échelle absolue de l'énergie
    d'inflation). Une fois A_s normalisé à Planck, V_0 = ε(φ_★) · 24 π²
    M_pl⁴ · A_s — il n'est PAS prédit indépendamment par la topologie
    seule. La V5 future devra dériver V_0 depuis la densité de condensation
    pré-Big Bang.

(b) FORME LOGARITHMIQUE de c(φ²) : c'est une asymptotique grande échelle,
    pas une forme exacte. Une correction sous-dominante en (φ_0/φ)² est
    attendue mais sous-dominante à φ_★ ~ √(120 q) M_pl ≈ 17 M_pl.

(c) N_e = 60 est PRIS comme cible (référence Planck). En V5 ce nombre
    doit émerger du couplage χ ↔ rayonnement après inflation, non traité
    ici.

(d) Pas de spectre tensoriel-scalar de second ordre, pas de
    non-gaussianités primordiales (f_NL ≈ −(5/6)(n_s − 1) en slow-roll
    standard, prédit ici ≈ 0,029, compatible Planck f_NL^local = -0,9
    ± 5,1).

(e) Le potentiel inverse en loi de puissance V ∝ φ^{-q} ne supporte PAS
    un graceful exit naïf : il faut un mécanisme de fin d'inflation
    (couplage à un autre champ, instabilité topologique). En V5 c'est
    fourni par la condensation de la matière baryonique. Pas traité ici.

Référence : V5.11 (flrw_emergence.py), V5.12 (cmb_perturbations.py),
TQNT_detaillee_V5_FR.tex §13.6.4 (α_cond), CLAUDE.md (α = 1,20).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from theory.coupling_kg import (
    ALPHA_EM,
    C_N_NUCLEON_EFF,
    kappa_g_V5p5,
)
from theory.einstein_emergence import (
    C_LIGHT,
    G_CODATA,
    HBAR,
    M_PROTON,
)
from theory.flrw_emergence import ALPHA_CONDENSATION

# ---------------------------------------------------------------------------
# Constantes V5 et Planck 2018 — inflation.
# ---------------------------------------------------------------------------

# Pénalité de condensation V5 (P18 / V5 §13.6.4) — base de q.
ALPHA_COND_V5 = ALPHA_CONDENSATION       # = 1.1991

# Exposant du potentiel topologique V(φ) ∝ φ^{-q}, dérivé : q = 2 α_cond.
Q_INFLATION_V5 = 2.0 * ALPHA_COND_V5     # ≈ 2.3982

# Échelle de Planck réduite (en kg et en GeV).
M_PL_KG = math.sqrt(HBAR * C_LIGHT / (8.0 * math.pi * G_CODATA))
M_PL_GEV = M_PL_KG * C_LIGHT ** 2 / 1.602176634e-10  # 1 GeV = 1.602e-10 J
# (≈ 2.435e18 GeV)

# Planck 2018 — observables CMB primordiales.
A_S_PLANCK = 2.0989e-9          # amplitude scalaire au pivot k = 0.05 Mpc⁻¹
A_S_PLANCK_SIGMA = 0.030e-9     # ±0.030e-9
N_S_PLANCK = 0.9649             # tilt spectral scalaire
N_S_PLANCK_SIGMA = 0.0042       # ±0.0042
R_PLANCK_BICEP = 0.06           # upper limit (BICEP/Keck + Planck 2021)

# Nombre d'e-folds canonique (référence Planck pivot k = 0.05 Mpc⁻¹).
N_EFOLDS_PIVOT = 60.0


# ---------------------------------------------------------------------------
# 1. Potentiel topologique V(φ).
# ---------------------------------------------------------------------------
def V_potential(psi_squared: float | np.ndarray,
                psi2_0: float = 1.0,
                V0: float = 1.0,
                alpha_cond: float = ALPHA_COND_V5) -> float | np.ndarray:
    """Potentiel d'inflation topologique V(|Ψ|²).

    Forme V5 dérivée (cf. en-tête de module) :

        V(|Ψ|²) = V_0 · (|Ψ|²/|Ψ_0|²)^{-α_cond}

    En termes du champ φ = √(2|Ψ|²), avec q = 2 α_cond :

        V(φ) = V_0 · φ^{-q}  (à constante multiplicative près)

    Paramètres
    ----------
    psi_squared : float | array
        |Ψ|² (densité de champ, en unités M_pl² si normalisé).
    psi2_0 : float
        Échelle de référence |Ψ_0|² (calibrée par A_s).
    V0 : float
        Échelle d'énergie V_0 (calibrée par A_s).
    alpha_cond : float
        Pénalité par croisement α_cond. Défaut V5 = 1,1991.

    Retour
    ------
    V : float | array (mêmes unités que V0).
    """
    psi_squared = np.asarray(psi_squared, dtype=float)
    return V0 * (psi_squared / psi2_0) ** (-alpha_cond)


def V_potential_phi(phi: float | np.ndarray,
                    phi_0: float = 1.0,
                    V0: float = 1.0,
                    alpha_cond: float = ALPHA_COND_V5) -> float | np.ndarray:
    """Potentiel en termes du champ φ = √(2|Ψ|²).

    V(φ) = V_0 · (φ/φ_0)^{-q},   q = 2 α_cond.
    """
    phi = np.asarray(phi, dtype=float)
    q = 2.0 * alpha_cond
    return V0 * (phi / phi_0) ** (-q)


def Vprime_phi(phi: float | np.ndarray,
               phi_0: float = 1.0,
               V0: float = 1.0,
               alpha_cond: float = ALPHA_COND_V5) -> float | np.ndarray:
    """Dérivée première dV/dφ analytique."""
    phi = np.asarray(phi, dtype=float)
    q = 2.0 * alpha_cond
    return -q * V0 * (phi / phi_0) ** (-q - 1.0) / phi_0


def Vsecond_phi(phi: float | np.ndarray,
                phi_0: float = 1.0,
                V0: float = 1.0,
                alpha_cond: float = ALPHA_COND_V5) -> float | np.ndarray:
    """Dérivée seconde d²V/dφ² analytique."""
    phi = np.asarray(phi, dtype=float)
    q = 2.0 * alpha_cond
    return q * (q + 1.0) * V0 * (phi / phi_0) ** (-q - 2.0) / phi_0 ** 2


# ---------------------------------------------------------------------------
# 2. Paramètres slow-roll en unités M_pl = 1.
# ---------------------------------------------------------------------------
def slow_roll_epsilon(phi: float | np.ndarray,
                      alpha_cond: float = ALPHA_COND_V5) -> float | np.ndarray:
    """ε(φ) = (M_pl²/2) (V'/V)².

    Pour V ∝ φ^{-q} : V'/V = -q/φ, donc ε = q²/(2 φ²).
    """
    phi = np.asarray(phi, dtype=float)
    q = 2.0 * alpha_cond
    return q ** 2 / (2.0 * phi ** 2)


def slow_roll_eta(phi: float | np.ndarray,
                  alpha_cond: float = ALPHA_COND_V5) -> float | np.ndarray:
    """η(φ) = M_pl² V''/V.

    Pour V ∝ φ^{-q} : V''/V = q(q+1)/φ², donc η = q(q+1)/φ².
    """
    phi = np.asarray(phi, dtype=float)
    q = 2.0 * alpha_cond
    return q * (q + 1.0) / phi ** 2


# ---------------------------------------------------------------------------
# 3. Fin de l'inflation et nombre d'e-folds.
# ---------------------------------------------------------------------------
def phi_end_inflation(alpha_cond: float = ALPHA_COND_V5) -> float:
    """φ_end tel que ε(φ_end) = 1 (fin slow-roll).

    Pour ε = q²/(2 φ²) = 1 ⇒ φ_end = q/√2.
    """
    q = 2.0 * alpha_cond
    return q / math.sqrt(2.0)


def N_efolds(phi_initial: float,
             phi_final: float | None = None,
             alpha_cond: float = ALPHA_COND_V5) -> float:
    """Nombre d'e-folds entre phi_initial et phi_final.

    N = ∫_{φ_f}^{φ_i} (V/V') dφ / M_pl²  (intégration vers le passé, φ_i > φ_f).

    Pour V ∝ φ^{-q} : V/V' = -φ/q, donc
        N = (φ_i² - φ_f²) / (2 q)

    Paramètres
    ----------
    phi_initial : float
        Champ au horizon-crossing (instant où l'observable sort de
        l'horizon Hubble, > φ_end).
    phi_final : float, optionnel
        Champ à la fin d'inflation. Défaut : φ_end (ε = 1).
    alpha_cond : float
        Pénalité de condensation.

    Retour
    ------
    N_e : float
    """
    if phi_final is None:
        phi_final = phi_end_inflation(alpha_cond=alpha_cond)
    q = 2.0 * alpha_cond
    if phi_initial < phi_final:
        return 0.0
    return (phi_initial ** 2 - phi_final ** 2) / (2.0 * q)


def find_psi_pivot(N_e_target: float = N_EFOLDS_PIVOT,
                   alpha_cond: float = ALPHA_COND_V5) -> float:
    """Champ φ_★ au horizon-crossing pour N_e_target e-folds restants.

    Inversion analytique :  N_e = (φ_★² - φ_end²) / (2 q)
                            φ_★² = φ_end² + 2 q N_e

    Avec φ_end = q/√2 (ε = 1) :
        φ_★² = q²/2 + 2 q N_e = q (q/2 + 2 N_e)
        φ_★  = √[q (q/2 + 2 N_e)]

    Paramètres
    ----------
    N_e_target : float
        Nombre d'e-folds depuis le pivot CMB jusqu'à la fin (typ. 50–60).
    alpha_cond : float
        Pénalité de condensation.

    Retour
    ------
    φ_★ : float (en unités M_pl).
    """
    q = 2.0 * alpha_cond
    phi_end_sq = q ** 2 / 2.0
    return math.sqrt(phi_end_sq + 2.0 * q * N_e_target)


# ---------------------------------------------------------------------------
# 4. Observables CMB prédits par V5.
# ---------------------------------------------------------------------------
def n_s_V5(N_e: float = N_EFOLDS_PIVOT,
           alpha_cond: float = ALPHA_COND_V5) -> float:
    """Tilt spectral scalaire prédit par V5.

    n_s = 1 - 6 ε(φ_★) + 2 η(φ_★)
        = 1 - 6 q²/(2 φ_★²) + 2 q(q+1)/φ_★²
        = 1 + [-3 q² + 2 q (q+1)] / φ_★²
        = 1 + [- q² + 2 q] / φ_★²
        = 1 - q (q - 2) / φ_★²

    Avec φ_★² = q²/2 + 2 q N_e :
        n_s = 1 - q(q-2) / [q²/2 + 2 q N_e]
            = 1 - (q-2) / [q/2 + 2 N_e]

    NB : pour q ≈ 2.40 et N_e = 60 :
        n_s = 1 - 0.40 / (1.20 + 120) ≈ 1 - 0.40/121.20 ≈ 0.9967.
    """
    q = 2.0 * alpha_cond
    phi_star_sq = q ** 2 / 2.0 + 2.0 * q * N_e
    eps = q ** 2 / (2.0 * phi_star_sq)
    eta = q * (q + 1.0) / phi_star_sq
    return 1.0 - 6.0 * eps + 2.0 * eta


def r_V5(N_e: float = N_EFOLDS_PIVOT,
         alpha_cond: float = ALPHA_COND_V5) -> float:
    """Rapport tenseur-scalaire prédit par V5.

    r = 16 ε(φ_★).

    Pour q ≈ 2.40 et N_e = 60 :
        ε = q²/(2 φ_★²) = q²/(q² + 4 q N_e) = q/(q + 4 N_e)
        r = 16 q/(q + 4 N_e) ≈ 16 × 2.40 / 242.40 ≈ 0.158.

    NB : valeur dépassant la limite BICEP/Keck r < 0.06 ⇒ contrainte forte
    sur le modèle (cf. discussion en docs/V5p13_topological_inflation.tex).
    """
    q = 2.0 * alpha_cond
    phi_star_sq = q ** 2 / 2.0 + 2.0 * q * N_e
    return 16.0 * q ** 2 / (2.0 * phi_star_sq)


def A_s_V5_dimensionless(N_e: float = N_EFOLDS_PIVOT,
                         alpha_cond: float = ALPHA_COND_V5,
                         V0_over_Mpl4: float = 1.0) -> float:
    """Amplitude scalaire en unités V_0 / M_pl⁴.

    A_s = V(φ_★) / (24 π² ε(φ_★) M_pl⁴)

    En unités V_0 = 1, φ_0 = 1 (avec V(φ) = V_0 φ^{-q}) :
        A_s · (V_0/M_pl⁴)^{-1} = φ_★^{-q} · 2 φ_★² / (24 π² q²)
                               = φ_★^{2-q} / (12 π² q²)

    Pour calibrer sur Planck A_s = 2.10e-9 :
        V_0 / M_pl⁴ = A_s · 12 π² q² · φ_★^{q-2}.
    """
    q = 2.0 * alpha_cond
    phi_star_sq = q ** 2 / 2.0 + 2.0 * q * N_e
    phi_star = math.sqrt(phi_star_sq)
    return V0_over_Mpl4 * phi_star ** (2.0 - q) / (12.0 * math.pi ** 2 * q ** 2)


def V0_calibrated_from_As(N_e: float = N_EFOLDS_PIVOT,
                          alpha_cond: float = ALPHA_COND_V5,
                          A_s_target: float = A_S_PLANCK) -> float:
    """V_0/M_pl⁴ calibré pour reproduire A_s_target.

    Inverse A_s_V5_dimensionless :
        V_0/M_pl⁴ = A_s_target / (φ_★^{2-q} / (12 π² q²))
    """
    q = 2.0 * alpha_cond
    phi_star_sq = q ** 2 / 2.0 + 2.0 * q * N_e
    phi_star = math.sqrt(phi_star_sq)
    return A_s_target * 12.0 * math.pi ** 2 * q ** 2 / phi_star ** (2.0 - q)


def A_s_V5(N_e: float = N_EFOLDS_PIVOT,
           alpha_cond: float = ALPHA_COND_V5,
           V0_over_Mpl4: float | None = None) -> float:
    """Amplitude scalaire prédite par V5 (sans dimension).

    Si V0_over_Mpl4 = None, on retourne la VALEUR PLANCK (calibration).
    Sinon on retourne la prédiction à partir de l'échelle V_0 fournie.
    """
    if V0_over_Mpl4 is None:
        # Auto-calibration sur A_s Planck (V_0 est paramètre libre).
        return A_S_PLANCK
    return A_s_V5_dimensionless(
        N_e=N_e, alpha_cond=alpha_cond, V0_over_Mpl4=V0_over_Mpl4
    )


# ---------------------------------------------------------------------------
# 5. Synthèse / validation Planck.
# ---------------------------------------------------------------------------
@dataclass
class InflationV5Prediction:
    """Synthèse des prédictions V5.13."""
    alpha_cond: float
    q: float
    N_e: float
    phi_end: float
    phi_star: float
    epsilon: float
    eta: float
    n_s: float
    r: float
    V0_over_Mpl4: float    # calibré sur A_s Planck
    A_s: float             # = A_S_PLANCK une fois calibré
    # Comparaison Planck
    n_s_planck: float = N_S_PLANCK
    n_s_deviation_sigma: float = float("nan")
    A_s_planck: float = A_S_PLANCK
    r_upper_planck: float = R_PLANCK_BICEP
    r_passes_upper: bool = False


def predict_inflation_V5(N_e: float = N_EFOLDS_PIVOT,
                         alpha_cond: float = ALPHA_COND_V5
                         ) -> InflationV5Prediction:
    """Synthèse complète des prédictions V5.13.

    Calibre V_0 sur A_s Planck (paramètre libre restant) et retourne
    n_s, r prédits.
    """
    q = 2.0 * alpha_cond
    phi_end = phi_end_inflation(alpha_cond=alpha_cond)
    phi_star = find_psi_pivot(N_e_target=N_e, alpha_cond=alpha_cond)
    eps = slow_roll_epsilon(phi_star, alpha_cond=alpha_cond)
    et = slow_roll_eta(phi_star, alpha_cond=alpha_cond)
    n_s = n_s_V5(N_e=N_e, alpha_cond=alpha_cond)
    r = r_V5(N_e=N_e, alpha_cond=alpha_cond)
    V0 = V0_calibrated_from_As(N_e=N_e, alpha_cond=alpha_cond,
                               A_s_target=A_S_PLANCK)
    deviation = (n_s - N_S_PLANCK) / N_S_PLANCK_SIGMA
    return InflationV5Prediction(
        alpha_cond=alpha_cond,
        q=q,
        N_e=N_e,
        phi_end=phi_end,
        phi_star=phi_star,
        epsilon=float(eps),
        eta=float(et),
        n_s=float(n_s),
        r=float(r),
        V0_over_Mpl4=float(V0),
        A_s=A_S_PLANCK,
        n_s_deviation_sigma=float(deviation),
        r_passes_upper=bool(r < R_PLANCK_BICEP),
    )


# ---------------------------------------------------------------------------
# 6. Variante Starobinsky-like (R²) V5 — pour contraste honnête.
# ---------------------------------------------------------------------------
def V_starobinsky(phi: float | np.ndarray,
                  phi_0: float = 1.0,
                  V0: float = 1.0) -> float | np.ndarray:
    """Potentiel Starobinsky-like : V(φ) = V_0 (1 - exp(-√(2/3) φ/M_pl))².

    Forme générique d'une inflation R + R²/(6M²) après transformation
    conforme en jauge d'Einstein. Pour comparer aux prédictions V5
    pure-loi-de-puissance.

    NB : φ_0 est ignoré (potentiel canonique standard); V_0 calibre A_s.
    """
    phi = np.asarray(phi, dtype=float)
    return V0 * (1.0 - np.exp(-math.sqrt(2.0 / 3.0) * phi)) ** 2


def predict_starobinsky(N_e: float = N_EFOLDS_PIVOT) -> dict:
    """Prédictions Starobinsky (R²) standard pour référence comparative.

    Limite grande N_e :
        n_s ≈ 1 - 2/N_e ≈ 0.967 pour N_e = 60
        r   ≈ 12/N_e²   ≈ 0.0033 pour N_e = 60

    Ce sont les prédictions Planck-favorites. La V5 pure (loi de
    puissance) NE LES REPRODUIT PAS ; la conjecture V5 honnête est qu'à
    bas champ (φ ~ M_pl) une CORRECTION TOPOLOGIQUE de courbure
    Chern-Simons rend le potentiel effectif Starobinsky-like. Cette
    correction n'est PAS dérivée ici ; on présente juste la cible.
    """
    n_s_starobinsky = 1.0 - 2.0 / N_e
    r_starobinsky = 12.0 / N_e ** 2
    return {
        "model": "Starobinsky R² (référence)",
        "N_e": N_e,
        "n_s": n_s_starobinsky,
        "r": r_starobinsky,
        "compatible_planck_n_s": abs(n_s_starobinsky - N_S_PLANCK)
                                  < 3.0 * N_S_PLANCK_SIGMA,
        "compatible_BICEP_r": r_starobinsky < R_PLANCK_BICEP,
    }


def summary(N_e: float = N_EFOLDS_PIVOT,
            alpha_cond: float = ALPHA_COND_V5) -> dict:
    """Dictionnaire récapitulatif des prédictions V5.13.

    Pratique pour les figures / tests / docs.
    """
    pred = predict_inflation_V5(N_e=N_e, alpha_cond=alpha_cond)
    return {
        "alpha_cond_V5": pred.alpha_cond,
        "q_V_phi": pred.q,
        "N_e": pred.N_e,
        "phi_end_Mpl": pred.phi_end,
        "phi_star_Mpl": pred.phi_star,
        "epsilon_star": pred.epsilon,
        "eta_star": pred.eta,
        "n_s_V5": pred.n_s,
        "n_s_Planck": pred.n_s_planck,
        "n_s_deviation_sigma": pred.n_s_deviation_sigma,
        "r_V5": pred.r,
        "r_Planck_upper": pred.r_upper_planck,
        "r_passes_upper": pred.r_passes_upper,
        "A_s_target": pred.A_s,
        "V0_over_Mpl4_calibrated": pred.V0_over_Mpl4,
    }
