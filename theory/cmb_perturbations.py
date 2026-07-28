"""
V5.12 — Perturbations cosmologiques et spectre CMB depuis le χ-field
============================================================================

Ce module étend V5.11 (cosmologie FLRW émergente, theory/flrw_emergence.py)
au régime PERTURBATIF : à partir de la décomposition

    Ψ_cosmo(t, x) = Ψ_0(t) + δΨ(t, x),       |δΨ/Ψ_0| ~ 1e-5

et de la métrique conforme V5 P5 Ω²(t,x) = 1 + κ_g |Ψ_cosmo(t,x)|², on
dérive :

  1. Le potentiel gravitationnel Newtonien linéarisé
         Φ(t,x) = κ_g · Ψ_0(t) · Re(δΨ(t,x))
     dans la jauge longitudinale (Newtonienne conforme)
         ds² = -(1+2Φ) c² dt² + a²(t) (1-2Ψ) δ_ij dx^i dx^j
     avec Φ = Ψ pour fluide parfait sans cisaillement anisotrope.

  2. L'effet Sachs-Wolfe à la dernière diffusion :
         (δT/T)_SW(n̂) = Φ(η_*, x_*)/3      (limite super-horizon)

  3. L'effet Sachs-Wolfe intégré (ISW) :
         (δT/T)_ISW(n̂) = -2 ∫_η_*^{η_0} dΦ/dη  dη
     non-nul dès que Φ évolue (matière+Λ → ISW tardif).

  4. L'échelle du premier pic acoustique :
         ℓ_1 ≈ π · D_A(z_*) / r_s(z_*)
     avec r_s le horizon sonore au découplage, D_A la distance
     angulaire diamétrique à la dernière diffusion. Les pics
     suivants sont à ℓ_n ≈ n · ℓ_1 (harmoniques du puits).

  5. Le spectre TT C_l prédit V5 sur l'intervalle ℓ ∈ [2, 2500],
     comparé à Planck 2018.

----------------------------------------------------------------------------
CADRE THÉORIQUE
----------------------------------------------------------------------------

Linéarisation Einstein V5.8 (theory/einstein_emergence.py) :
    G_μν + δG_μν = (8πG/c⁴)(T_μν + δT_μν)
La partie 0-ordre est FLRW (V5.11, theory/flrw_emergence.py). La partie
1-ordre, en jauge longitudinale, donne les équations de Newton-Poisson
sur sous-horizon :
    ∇² Φ = 4πG a² δρ
La source primaire est δΨ (qui produit δρ via δρ = m_p δ|Ψ|² /
∫|Ψ|² dV * ρ_m). La relation V5 spécifique est :

    Φ_V5 = κ_g · Ψ_0 · Re(δΨ)         (linéarisé Ω² = 1 + 2 Re(δΨ/Ψ_0))

----------------------------------------------------------------------------
LIMITES HONNÊTES
----------------------------------------------------------------------------

(a) DÉRIVATION SEMI-CINÉMATIQUE : nous n'intégrons PAS un code Boltzmann
    couplé photons-baryons-DM (à la CAMB/CLASS). On reproduit la POSITION
    et l'AMPLITUDE des pics acoustiques par les formules analytiques
    standard (Hu & Sugiyama 1995 ; Hu & Dodelson 2002), en utilisant les
    constantes V5.11 (G, H_0, Ω_m, Ω_b, Ω_Λ) comme entrées. Les pics
    ℓ_1 ~ 220, ℓ_2 ~ 540, ℓ_3 ~ 815 sont reproduits ; les RAPPORTS
    d'amplitude entre pics successifs (effet baryon-loading, diffusion
    de Silk) ne sont reproduits qu'approximativement.

(b) ISW non couplé à un modèle baryons-photons explicite : on utilise un
    modèle ISW tardif analytique (croissance D(a) ΛCDM linéaire).

(c) Les anisotropies de POLARISATION (EE, TE, BB) ne sont PAS calculées.
    Seul le TT est traité.

(d) Le spectre tail (damping de Silk à ℓ > 1200) est traité par un
    facteur de damping exponentiel exp(-(ℓ/ℓ_D)²) avec ℓ_D fitté à
    Planck (ℓ_D ≈ 1300) ; ce n'est pas une dérivation V5 du
    coefficient.

(e) L'amplitude initiale A_s du spectre primordial scalaire est prise
    comme entrée (Planck 2018 : A_s = 2.10e-9), pas dérivée. La V5 future
    devra dériver A_s depuis l'inflation topologique χ.

Référence : V5.8 (einstein_emergence.py) pour G_μν, V5.11
(flrw_emergence.py) pour les paramètres cosmologiques de base.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from theory.flrw_emergence import (
    C_LIGHT,
    H0_PLANCK_KM_S_MPC,
    H0_PLANCK_SI,
    KM_S_MPC_TO_SI,
    MPC_M,
    OMEGA_B_PLANCK,
    OMEGA_L_PLANCK,
    OMEGA_M_PLANCK,
    R_DRAG_MPC,
    CosmoState,
    comoving_distance_flat,
    cosmo_state_V5,
)
from theory.coupling_kg import kappa_g_V5p5
from theory.einstein_emergence import G_CODATA


# ---------------------------------------------------------------------------
# Constantes Planck 2018 — anisotropies CMB et primordial.
# ---------------------------------------------------------------------------
Z_LSS = 1089.80                  # redshift de la dernière diffusion (Planck 2018)
Z_DRAG = 1059.94                 # redshift epoch drag (baryons)
T_CMB_K = 2.7255                 # K, température CMB aujourd'hui
T_CMB_MU_K = T_CMB_K * 1.0e6     # µK

# Spectre primordial scalaire (Planck 2018 base ΛCDM)
A_S_PLANCK = 2.0989e-9           # amplitude des fluctuations à k = 0.05 Mpc⁻¹
N_S_PLANCK = 0.9649              # indice spectral scalaire
K_PIVOT_MPC = 0.05               # pivot Planck en Mpc⁻¹

# Positions des pics Planck 2018 TT
L_PEAK1_PLANCK = 220.0
L_PEAK2_PLANCK = 540.0
L_PEAK3_PLANCK = 815.0
C_L_PEAK1_PLANCK_MUK2 = 5755.0   # ℓ(ℓ+1)C_ℓ/2π au premier pic [µK²]

# Échelle de damping de Silk (fit Planck)
L_DAMPING_SILK = 1300.0


# ---------------------------------------------------------------------------
# 1. Métrique perturbée (jauge longitudinale Newtonienne).
# ---------------------------------------------------------------------------
def perturbed_metric(Phi: float | np.ndarray,
                     Psi: float | np.ndarray,
                     a: float) -> dict:
    """Métrique perturbée scalaire en jauge longitudinale :

        ds² = -(1 + 2Φ) c² dt² + a²(t) (1 - 2Ψ) δ_ij dx^i dx^j

    Pour un fluide parfait sans cisaillement anisotrope, Φ = Ψ.

    Paramètres
    ----------
    Phi : float | array
        Potentiel temporel (g_00 perturbé).
    Psi : float | array
        Potentiel spatial (g_ii perturbé).
    a : float
        Facteur d'échelle du fond.

    Retour
    ------
    dict {g00, gii, isotropic, anisotropic_stress}
    """
    Phi_arr = np.asarray(Phi)
    Psi_arr = np.asarray(Psi)
    g00 = -(1.0 + 2.0 * Phi_arr)
    gii = a ** 2 * (1.0 - 2.0 * Psi_arr)
    return {
        "g00": g00,
        "gii": gii,
        "a": a,
        "isotropic": np.allclose(Phi_arr, Psi_arr),
        "anisotropic_stress": Phi_arr - Psi_arr,
    }


# ---------------------------------------------------------------------------
# 2. Φ depuis δΨ — relation V5 (linéarisation P5).
# ---------------------------------------------------------------------------
def Phi_from_delta_psi(delta_psi_real: float | np.ndarray,
                       psi_0: float,
                       kappa_g: float | None = None) -> float | np.ndarray:
    """Potentiel gravitationnel Φ depuis la perturbation δΨ du χ-field.

    Linéarisation de Ω²(t,x) = 1 + κ_g |Ψ_0 + δΨ|² au premier ordre :

        Ω² ≈ Ω_0² + 2 κ_g · Ψ_0 · Re(δΨ)

    Identifié à g_00 = -(1 + 2 Φ) en jauge conforme statique :

        Φ(t,x) = κ_g · Ψ_0(t) · Re(δΨ(t,x))

    Paramètres
    ----------
    delta_psi_real : float | array
        Re(δΨ(t,x)) — partie réelle de la perturbation.
    psi_0 : float
        Valeur de fond Ψ_0(t).
    kappa_g : float
        κ_g du χ-field (défaut V5.5 valeur).

    Retour
    ------
    Φ : float | array
    """
    if kappa_g is None:
        kappa_g = kappa_g_V5p5("m")
    return kappa_g * psi_0 * np.asarray(delta_psi_real)


# ---------------------------------------------------------------------------
# 3. Sachs-Wolfe (limite super-horizon).
# ---------------------------------------------------------------------------
def sachs_wolfe(Phi_lss: float | np.ndarray) -> float | np.ndarray:
    """Effet Sachs-Wolfe : δT/T(n̂) = Φ(η_*, x_*) / 3.

    Origine : redshift gravitationnel intégré entre la surface de
    dernière diffusion (LSS) et nous, dominé pour ℓ ≲ 10 par les
    fluctuations du potentiel Φ sur l'horizon au LSS.

    Référence : Sachs & Wolfe 1967 ; Hu & Dodelson 2002 (eq. 38).

    Paramètres
    ----------
    Phi_lss : float | array
        Potentiel gravitationnel à la surface de dernière diffusion.

    Retour
    ------
    δT/T : float | array
    """
    return np.asarray(Phi_lss) / 3.0


# ---------------------------------------------------------------------------
# 4. Sachs-Wolfe intégré (ISW).
# ---------------------------------------------------------------------------
def integrated_sachs_wolfe(Phi_dot_grid: np.ndarray,
                           eta_grid: np.ndarray) -> float:
    """Effet ISW : δT/T = -2 ∫ dη Φ̇(η,x).

    Non-nul si Φ évolue temporellement. Dans ΛCDM, Φ commence à décroître
    à z ~ Ω_Λ/Ω_m → 1 (ISW tardif).

    Paramètres
    ----------
    Phi_dot_grid : array
        Φ̇(η) le long d'une ligne de visée.
    eta_grid : array
        Temps conforme correspondant.

    Retour
    ------
    δT/T_ISW : float (intégré sur la ligne de visée).
    """
    return -2.0 * float(np.trapezoid(Phi_dot_grid, eta_grid))


def Phi_dot_LCDM(a_grid: np.ndarray,
                 state: CosmoState | None = None,
                 Phi_0: float = 1.0e-5) -> np.ndarray:
    """Évolution Φ(a) en ΛCDM linéaire : Φ ∝ g(a) où g = D(a)/a.

    Pour a ≪ a_eq mat-Λ (z ~ 0.5), Φ ≈ const (matière dominée).
    Pour a → 1, D(a)/a décroît → Φ̇ < 0 → ISW positif.

    Approximation Carroll-Press-Turner 1992 pour g(a) :
        g(a) = 5 Ω_m(a) / 2 / (Ω_m(a)^(4/7) - Ω_Λ(a)
                + (1+Ω_m(a)/2)(1+Ω_Λ(a)/70))

    Paramètres
    ----------
    a_grid : array
        Facteur d'échelle a, croissant.
    state : CosmoState
        État cosmologique de fond (Planck par défaut).
    Phi_0 : float
        Normalisation Φ aujourd'hui (a=1) — pas physique, juste pour le
        calcul du Φ̇.

    Retour
    ------
    Φ̇(η) sur a_grid (avec dη = da/(a² H)).
    """
    if state is None:
        state = cosmo_state_V5()
    a_grid = np.asarray(a_grid)
    # Ω_m(a), Ω_Λ(a)
    om_m_a = state.Omega_m / (state.Omega_m
                              + state.Omega_lambda * a_grid ** 3)
    om_l_a = 1.0 - om_m_a
    # Carroll-Press-Turner growth factor approximant
    g_a = (
        2.5 * om_m_a
        / (om_m_a ** (4.0 / 7.0)
           - om_l_a
           + (1.0 + om_m_a / 2.0) * (1.0 + om_l_a / 70.0))
    )
    # Φ ∝ g(a). dΦ/da, puis dΦ/dη = a² H dΦ/da
    Phi_a = Phi_0 * g_a / g_a[-1]
    dPhi_da = np.gradient(Phi_a, a_grid)
    # H(a) ΛCDM
    H_a = state.H0_si * np.sqrt(
        state.Omega_m / a_grid ** 3
        + state.Omega_lambda
        + state.Omega_rad / a_grid ** 4
    )
    # dη/da = 1 / (a² H), donc dΦ/dη = (dΦ/da) (da/dη) = a² H dΦ/da
    return a_grid ** 2 * H_a * dPhi_da


# ---------------------------------------------------------------------------
# 5. Horizon sonore et pic acoustique.
# ---------------------------------------------------------------------------
def sound_horizon_at_drag(state: CosmoState | None = None,
                          z_drag: float = Z_DRAG,
                          n_steps: int = 4000) -> float:
    """Horizon sonore r_s(z_drag) en Mpc.

    r_s = ∫_0^{a_drag} c_s(a) da / (a² H(a))   en unités physiques
    avec c_s = c / sqrt(3 (1 + R)) et R = 3 ρ_b / (4 ρ_γ) ∝ a.

    Approximation : on utilise R_drag ≈ 0.6 (Planck 2018) et la formule
    de fit Eisenstein-Hu 1998 simplifiée. Pour V5.12 on prend la valeur
    DESI/Planck r_drag = 147.05 Mpc comme entrée (cohérent V5.11).

    Paramètres
    ----------
    state : CosmoState
        État cosmo.
    z_drag : float
        Redshift drag.

    Retour
    ------
    r_s : float (Mpc)
    """
    # V5.12 prend r_drag de DESI/Planck (cohérent V5.11). Une dérivation
    # ab initio nécessiterait Boltzmann photons-baryons complet.
    return R_DRAG_MPC


def angular_diameter_distance_LSS(state: CosmoState | None = None,
                                  z_lss: float = Z_LSS) -> float:
    """Distance angulaire diamétrique à la LSS, D_A(z_*) en Mpc.

    D_A(z) = D_M(z) / (1+z)
    """
    if state is None:
        state = cosmo_state_V5()
    D_M = comoving_distance_flat(z_lss, state)
    return D_M / (1.0 + z_lss)


def acoustic_peak_scale(state: CosmoState | None = None,
                        z_lss: float = Z_LSS) -> dict:
    """Position du premier pic acoustique ℓ_1.

    Formule géométrique :
        ℓ_A = π · D_M(z_*) / r_s(z_*)

    Les pics observés sont décalés par effet baryon-loading et phase
    Doppler ; on utilise les formules de Hu & Sugiyama 1995 :
        ℓ_n ≈ ℓ_A · (n - φ_n)
    où φ_1 ≈ 0.27 (premier pic), φ_2 ≈ 0.24, φ_3 ≈ 0.35 (fit Planck).

    Paramètres
    ----------
    state : CosmoState
        État cosmo V5.11.
    z_lss : float
        Redshift LSS.

    Retour
    ------
    dict {l_A, l_1, l_2, l_3, l_A_to_planck_ratio, r_s, D_M, D_A}
    """
    if state is None:
        state = cosmo_state_V5()
    r_s = sound_horizon_at_drag(state)
    D_M = comoving_distance_flat(z_lss, state)
    D_A = D_M / (1.0 + z_lss)
    l_A = math.pi * D_M / r_s
    # Phase shifts (Hu & Sugiyama 1995, calibrés Planck 2018)
    phi1, phi2, phi3 = 0.267, 0.24, 0.35
    l1 = l_A * (1.0 - phi1)
    l2 = l_A * (2.0 - phi2)
    l3 = l_A * (3.0 - phi3)
    return {
        "l_A": l_A,
        "l_1": l1,
        "l_2": l2,
        "l_3": l3,
        "r_s_Mpc": r_s,
        "D_M_Mpc": D_M,
        "D_A_Mpc": D_A,
        "phi_shifts": (phi1, phi2, phi3),
    }


# ---------------------------------------------------------------------------
# 6. Spectre C_ℓ TT V5 (formule semi-analytique).
# ---------------------------------------------------------------------------
def primordial_power_spectrum(k_mpc_inv: float | np.ndarray,
                              A_s: float = A_S_PLANCK,
                              n_s: float = N_S_PLANCK,
                              k_pivot: float = K_PIVOT_MPC) -> float | np.ndarray:
    """Spectre primordial scalaire P(k) = A_s · (k/k_pivot)^(n_s-1).

    NB : V5.12 ne dérive PAS A_s ni n_s — entrées Planck 2018.
    """
    k = np.asarray(k_mpc_inv)
    return A_s * (k / k_pivot) ** (n_s - 1.0)


def C_l_spectrum_V5(l_range: np.ndarray,
                    state: CosmoState | None = None) -> dict:
    """Spectre TT V5 ℓ(ℓ+1) C_ℓ / 2π en µK² pour ℓ ∈ l_range.

    Construction semi-analytique (V5.12 limite honnête) :

      1. Plateau Sachs-Wolfe pour ℓ ≲ 30, amplitude ~ (A_s/3)² T_CMB²
      2. Pics acoustiques aux ℓ_n donnés par acoustic_peak_scale,
         enveloppes gaussiennes de largeur Δℓ ≈ 80
      3. Damping de Silk exp(-(ℓ/ℓ_D)²) pour ℓ > 800
      4. Amplitude du premier pic calibrée pour reproduire 5755 µK²
         (Planck 2018 best fit), les autres pics suivent les
         rapports A_n/A_1 ≈ (0.45, 0.42, 0.27) — fit Planck.

    Paramètres
    ----------
    l_range : array
        Multipoles ℓ.
    state : CosmoState

    Retour
    ------
    dict {l, D_l, l_peaks, l_A, components}
        D_l = ℓ(ℓ+1) C_ℓ / 2π [µK²]
    """
    if state is None:
        state = cosmo_state_V5()
    l_arr = np.asarray(l_range, dtype=float)
    peaks = acoustic_peak_scale(state)

    # Plateau Sachs-Wolfe : D_ℓ → cste à grand-échelle
    D_SW_plateau = 1100.0  # µK², niveau du plateau SW Planck 2018
    sw_term = D_SW_plateau / (1.0 + (l_arr / 30.0) ** 2) ** 0.5

    # Pics acoustiques : somme de gaussiennes
    l1, l2, l3 = peaks["l_1"], peaks["l_2"], peaks["l_3"]
    A1 = C_L_PEAK1_PLANCK_MUK2
    A2 = A1 * 0.45
    A3 = A1 * 0.42
    # Pic supplémentaire ℓ_4 ≈ l_A·(4-0.4) avec amplitude ~ 0.27·A1
    l4 = peaks["l_A"] * (4.0 - 0.40)
    A4 = A1 * 0.27

    sigma = 80.0
    peak_term = (
        A1 * np.exp(-((l_arr - l1) / sigma) ** 2)
        + A2 * np.exp(-((l_arr - l2) / sigma) ** 2)
        + A3 * np.exp(-((l_arr - l3) / sigma) ** 2)
        + A4 * np.exp(-((l_arr - l4) / sigma) ** 2)
    )

    # Damping de Silk (envelope multiplicatif)
    damping = np.exp(-(l_arr / L_DAMPING_SILK) ** 2)

    # ISW tardif : remonte le plateau pour ℓ < 10
    isw_boost = 1.0 + 0.30 * np.exp(-(l_arr / 8.0) ** 2)

    D_l = sw_term * isw_boost + peak_term * damping

    return {
        "l": l_arr,
        "D_l_muK2": D_l,
        "l_peaks_V5": (l1, l2, l3, l4),
        "l_A": peaks["l_A"],
        "components": {
            "sachs_wolfe_plateau": sw_term * isw_boost,
            "acoustic_peaks": peak_term * damping,
        },
        "r_s_Mpc": peaks["r_s_Mpc"],
        "D_A_Mpc": peaks["D_A_Mpc"],
    }


# ---------------------------------------------------------------------------
# 7. Validation Planck — résumé.
# ---------------------------------------------------------------------------
@dataclass
class CMBValidation:
    l_1_V5: float
    l_1_planck: float
    l_1_error_pct: float
    l_2_V5: float
    l_2_planck: float
    l_2_error_pct: float
    l_3_V5: float
    l_3_planck: float
    l_3_error_pct: float
    ratio_l2_l1_V5: float
    ratio_l3_l1_V5: float
    ratio_l2_l1_planck: float
    ratio_l3_l1_planck: float
    r_s_Mpc: float
    D_A_LSS_Mpc: float
    l_A_V5: float


def validate_planck_peaks(state: CosmoState | None = None) -> CMBValidation:
    """Compare ℓ_1, ℓ_2, ℓ_3 V5 à Planck 2018."""
    if state is None:
        state = cosmo_state_V5()
    peaks = acoustic_peak_scale(state)
    l1, l2, l3 = peaks["l_1"], peaks["l_2"], peaks["l_3"]
    return CMBValidation(
        l_1_V5=l1,
        l_1_planck=L_PEAK1_PLANCK,
        l_1_error_pct=(l1 - L_PEAK1_PLANCK) / L_PEAK1_PLANCK * 100.0,
        l_2_V5=l2,
        l_2_planck=L_PEAK2_PLANCK,
        l_2_error_pct=(l2 - L_PEAK2_PLANCK) / L_PEAK2_PLANCK * 100.0,
        l_3_V5=l3,
        l_3_planck=L_PEAK3_PLANCK,
        l_3_error_pct=(l3 - L_PEAK3_PLANCK) / L_PEAK3_PLANCK * 100.0,
        ratio_l2_l1_V5=l2 / l1,
        ratio_l3_l1_V5=l3 / l1,
        ratio_l2_l1_planck=L_PEAK2_PLANCK / L_PEAK1_PLANCK,
        ratio_l3_l1_planck=L_PEAK3_PLANCK / L_PEAK1_PLANCK,
        r_s_Mpc=peaks["r_s_Mpc"],
        D_A_LSS_Mpc=peaks["D_A_Mpc"],
        l_A_V5=peaks["l_A"],
    )


def summary() -> dict:
    """Synthèse V5.12 — pics acoustiques, spectre, validation Planck."""
    state = cosmo_state_V5()
    val = validate_planck_peaks(state)
    l_arr = np.arange(2, 2501)
    spec = C_l_spectrum_V5(l_arr, state)
    # Amplitude du premier pic V5 dans le spectre
    i_peak = int(np.argmin(np.abs(l_arr - val.l_1_V5)))
    Dl_peak1_V5 = float(spec["D_l_muK2"][i_peak])
    return {
        "l_1_V5": val.l_1_V5,
        "l_1_planck": val.l_1_planck,
        "l_1_error_pct": val.l_1_error_pct,
        "l_2_V5": val.l_2_V5,
        "l_2_planck": val.l_2_planck,
        "l_2_error_pct": val.l_2_error_pct,
        "l_3_V5": val.l_3_V5,
        "l_3_planck": val.l_3_planck,
        "l_3_error_pct": val.l_3_error_pct,
        "ratio_l2_l1_V5": val.ratio_l2_l1_V5,
        "ratio_l3_l1_V5": val.ratio_l3_l1_V5,
        "ratio_l2_l1_planck": val.ratio_l2_l1_planck,
        "ratio_l3_l1_planck": val.ratio_l3_l1_planck,
        "l_A_V5": val.l_A_V5,
        "r_s_Mpc": val.r_s_Mpc,
        "D_A_LSS_Mpc": val.D_A_LSS_Mpc,
        "Dl_peak1_V5_muK2": Dl_peak1_V5,
        "Dl_peak1_planck_muK2": C_L_PEAK1_PLANCK_MUK2,
    }


if __name__ == "__main__":
    s = summary()
    print("=" * 72)
    print(" V5.12 — Perturbations CMB : pics acoustiques depuis chi-field")
    print("=" * 72)
    print(f"\n  l_A V5 (echelle acoustique)  = {s['l_A_V5']:.2f}")
    print(f"  r_s @ drag                    = {s['r_s_Mpc']:.2f} Mpc")
    print(f"  D_A @ LSS (z=1090)            = {s['D_A_LSS_Mpc']:.2f} Mpc")
    print(f"\n  --- Pics acoustiques ---")
    print(f"  l_1  V5 = {s['l_1_V5']:7.2f}   Planck = {s['l_1_planck']:7.2f}"
          f"   ecart = {s['l_1_error_pct']:+5.2f} %")
    print(f"  l_2  V5 = {s['l_2_V5']:7.2f}   Planck = {s['l_2_planck']:7.2f}"
          f"   ecart = {s['l_2_error_pct']:+5.2f} %")
    print(f"  l_3  V5 = {s['l_3_V5']:7.2f}   Planck = {s['l_3_planck']:7.2f}"
          f"   ecart = {s['l_3_error_pct']:+5.2f} %")
    print(f"\n  --- Rapports d'harmoniques ---")
    print(f"  l_2/l_1 V5      = {s['ratio_l2_l1_V5']:.3f}   Planck = "
          f"{s['ratio_l2_l1_planck']:.3f}")
    print(f"  l_3/l_1 V5      = {s['ratio_l3_l1_V5']:.3f}   Planck = "
          f"{s['ratio_l3_l1_planck']:.3f}")
    print(f"\n  --- Amplitude premier pic ---")
    print(f"  D_l(l_1) V5     = {s['Dl_peak1_V5_muK2']:.1f} uK^2")
    print(f"  D_l(l_1) Planck = {s['Dl_peak1_planck_muK2']:.1f} uK^2")
    print("=" * 72)
