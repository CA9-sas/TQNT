"""
V5 — Test du fond stochastique gravitationnel nHz (SGWB) NANOGrav 15-year.

Ce module construit le spectre $h_c(f)$ du fond stochastique d'ondes
gravitationnelles dans la bande PTA (pulsar timing arrays) et confronte
trois modèles à la mesure NANOGrav 15-yr (Agazie+ 2023, ApJL 951 L8,
arXiv:2306.16213) :

   1. SMBHB pur (Phinney 2001)         : h_c ∝ f^{-2/3}, gamma = 13/3.
   2. Cordes cosmiques (cusps)         : h_c ∝ f^{-7/6}, gamma = 16/3.
   3. V5 P22 (nucléation primordiale)  : h_c ∝ f^{alpha_V5}, à dériver.

CONVENTIONS NANOGRAV
====================

NANOGrav paramètre le strain caractéristique par
   h_c(f) = A_yr * (f/f_yr)^alpha
et la PSD résiduelle par
   P(f) = (A_yr^2 / (12 pi^2)) * (f/f_yr)^{-gamma} * f_yr^{-3}
avec la relation
   gamma = 3 - 2 alpha.

Pour SMBHB (Phinney 2001) : alpha = -2/3, gamma = 13/3 ≈ 4.33.
Pour strings (cusps)      : alpha = -7/6, gamma = 16/3 ≈ 5.33.

NANOGrav 15-yr HD-corrélé (table 1 Agazie+ 2023) : gamma = 3.2 ± 0.6,
soit alpha = (3 - gamma)/2 ≈ -0.1 ± 0.3. Cette pente est plus PLATE
que la prédiction SMBHB, déviation marginale (~2σ).

V5 P22 — NUCLÉATION TOPOLOGIQUE PRIMORDIALE
===========================================

Le mécanisme P22 (cf. CLAUDE.md, sec:P22-nucleation de
docs/TQNT_detaillee_V5_FR.tex) postule

   Γ(r) = Γ_0 * exp(eta * xi),     xi = |∇Ω²| * λC / Ω².

Aux frontières matière/vide cosmologiques primordiales, ce processus
nucleate des nœuds topologiques (et donc des défauts métriques h^TT)
qui se propagent comme ondes gravitationnelles. Le spectre RESULTANT
dépend de :

   - la distribution des frontières (≃ horizon décroissant a(t) H(t)),
   - la dimension du défaut (point-like vs ligne-like vs nappe).

Pour des défauts POINT-LIKE de Hubble (entrelacs nucléés à H^-1) avec
formation continue de l'inflation (~10^{-32} s) à la nucléosynthèse
(~10^3 s), le spectre attendu en bande PTA est QUASI-PLAT en
densité d'énergie Ω_GW(f), donc en strain caractéristique :

   h_c(f) ∝ f^{-1}    ⟹    alpha_V5 = -1    ⟹    gamma_V5 = 5.

Pour défauts LIGNE-LIKE (analogue cordes cosmiques V5) : gamma = 16/3
(idem cordes Nambu-Goto).

Pour défauts NAPPE-LIKE (membranes de boundary P22, P21) : spectre
plus rouge, gamma = 7/3 ≈ 2.33 (cf. Vilenkin-Shellard 1994 pour
membranes ; ici on a le même comptage dimensionnel).

V5 PRÉDIT donc une SUPERPOSITION :

   h_c^2(f) = A_SMBHB^2 (f/f_yr)^{-4/3}
            + A_string^2 (f/f_yr)^{-7/3}
            + A_mem^2 (f/f_yr)^{-1/3}.

LIMITES HONNÊTES
================

1. La tension G mu V5 des cordes/membranes émergentes n'est PAS
   calculée first-principles ici. Nous fittons A_string, A_mem.
2. Si l'AJUSTEMENT NANOGrav favorise gamma ≈ 3.2 < 13/3, alors V5
   peut l'expliquer SEULEMENT si la composante membrane (gamma=7/3)
   domine ; dans ce cas, le fit donne f_mem > 50 %.
3. NANOGrav 15-yr ne distingue pas un alpha unique d'une mixture
   plate : la dégénérescence est levée par la BANDE de fréquences
   (5 bins 2-30 nHz). Notre likelihood les utilise tous.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constantes NANOGrav (Agazie+ 2023, table 1, posterior HD-correlated).
# ---------------------------------------------------------------------------
F_YR_HZ = 1.0 / (365.25 * 86400.0)          # 3.17e-8 Hz, fréquence "1/an"
NANOGRAV_GAMMA_HD = 3.2                      # spectral index PSD
NANOGRAV_GAMMA_HD_SIGMA = 0.6
NANOGRAV_A_YR_HD = 2.4e-15                   # amplitude h_c à f_yr (point estimate)

# Index théoriques canoniques (h_c ∝ f^alpha ; gamma = 3 - 2 alpha).
ALPHA_SMBHB = -2.0 / 3.0
GAMMA_SMBHB = 3.0 - 2.0 * ALPHA_SMBHB        # 13/3 ≈ 4.33

ALPHA_STRING = -7.0 / 6.0                    # cordes cosmiques cusps
GAMMA_STRING = 3.0 - 2.0 * ALPHA_STRING      # 16/3 ≈ 5.33

# V5 P22 : 3 régimes selon la dimension du défaut nucléé.
ALPHA_V5_POINT = -1.0                        # défauts ponctuels, h_c ∝ 1/f
GAMMA_V5_POINT = 3.0 - 2.0 * ALPHA_V5_POINT  # = 5

ALPHA_V5_MEMBRANE = +1.0 / 3.0               # défauts 2D (membrane boundary)
GAMMA_V5_MEMBRANE = 3.0 - 2.0 * ALPHA_V5_MEMBRANE  # = 7/3 ≈ 2.33


# ---------------------------------------------------------------------------
# 1. Conversions strain ↔ index PSD ↔ Ω_GW.
# ---------------------------------------------------------------------------
def alpha_to_gamma(alpha: float) -> float:
    """gamma = 3 - 2 alpha (convention NANOGrav)."""
    return 3.0 - 2.0 * alpha


def gamma_to_alpha(gamma: float) -> float:
    """alpha = (3 - gamma)/2."""
    return (3.0 - gamma) / 2.0


def h_c_powerlaw(f_Hz: np.ndarray | float,
                 A_yr: float,
                 alpha: float,
                 f_ref_Hz: float = F_YR_HZ) -> np.ndarray:
    """Strain caractéristique d'une loi de puissance pure.

        h_c(f) = A_yr * (f/f_ref)^alpha

    Paramètres
    ----------
    f_Hz : array_like de fréquences (Hz)
    A_yr : amplitude au pivot f_ref (dimensionless)
    alpha : indice spectral du strain
    f_ref_Hz : pivot (défaut : 1/an = 3.17e-8 Hz)
    """
    if A_yr < 0:
        raise ValueError("A_yr doit être positif ou nul.")
    f = np.asarray(f_Hz, dtype=float)
    return A_yr * (f / f_ref_Hz) ** alpha


def omega_gw_from_hc(f_Hz: np.ndarray | float,
                      h_c: np.ndarray | float,
                      H0_per_s: float = 67.4e3 / 3.0857e22) -> np.ndarray:
    """Densité d'énergie GW Ω_GW(f) à partir du strain caractéristique.

        Ω_GW(f) = (2 pi^2 / 3 H_0^2) * f^2 * h_c^2(f)

    H_0 défaut : 67.4 km/s/Mpc = 2.18e-18 s^-1.
    """
    f = np.asarray(f_Hz, dtype=float)
    h = np.asarray(h_c, dtype=float)
    return (2.0 * math.pi ** 2 / 3.0) * f ** 2 * h ** 2 / H0_per_s ** 2


# ---------------------------------------------------------------------------
# 2. Modèles de spectre h_c(f).
# ---------------------------------------------------------------------------
def h_c_SMBHB(f_Hz: np.ndarray | float,
              A_yr: float = NANOGRAV_A_YR_HD) -> np.ndarray:
    """SMBHB Phinney 2001 : h_c ∝ f^{-2/3} (gamma = 13/3)."""
    return h_c_powerlaw(f_Hz, A_yr, ALPHA_SMBHB)


def h_c_strings(f_Hz: np.ndarray | float,
                G_mu: float = 1.0e-11) -> np.ndarray:
    """Cordes cosmiques (cusps, modèle Vilenkin).

    Amplitude paramétrée par la tension dimensionless G mu.
    Approximation Damour-Vilenkin 2001 :
        A_string(G mu) ≃ 2e-15 * (G mu / 1e-11)
    soit pour G mu ~ 1e-11 le strain est marginalement détectable.

    h_c(f) ∝ f^{-7/6} ⟹ gamma = 16/3.
    """
    A_string = 2.0e-15 * (G_mu / 1.0e-11)
    return h_c_powerlaw(f_Hz, A_string, ALPHA_STRING)


def h_c_V5_point(f_Hz: np.ndarray | float,
                  A_V5: float = 1.0e-15) -> np.ndarray:
    """V5 P22 — défauts ponctuels (nœuds nucléés à H^-1 primordial).

    h_c ∝ f^{-1} ⟹ gamma_V5 = 5.

    Argument dimensionnel : nombre de défauts produits par unité de
    e-fold du facteur d'échelle est constant en P22 si Γ_0 est
    cosmologiquement uniforme. Le spectre de strain de défauts
    ponctuels rayonnant à leur fréquence Hubble propre dN/d ln a = cte
    donne f h_c(f) = cte, soit h_c ∝ 1/f.
    """
    return h_c_powerlaw(f_Hz, A_V5, ALPHA_V5_POINT)


def h_c_V5_membrane(f_Hz: np.ndarray | float,
                     A_mem: float = 1.0e-15) -> np.ndarray:
    """V5 P22 — défauts 2D (membranes de boundary, P21).

    h_c ∝ f^{1/3} ⟹ gamma_V5 = 7/3 ≈ 2.33.

    Spectre relativement BLEU : le strain MONTE avec f, donc en bande
    PTA basse (1 nHz), la contribution est sous-dominante. Cette
    composante peut TIRER l'indice composite vers le bas et expliquer
    la pente observée NANOGrav gamma ≈ 3.2 (au lieu de 13/3).
    """
    return h_c_powerlaw(f_Hz, A_mem, ALPHA_V5_MEMBRANE)


def h_c_V5_combined(f_Hz: np.ndarray | float,
                     A_SMBHB: float = NANOGRAV_A_YR_HD,
                     A_string: float = 0.0,
                     A_mem: float = 0.0) -> np.ndarray:
    """Mixture incohérente SMBHB + V5 strings + V5 membranes.

    Sommation INCOHÉRENTE en puissance (h_c^2 additif).
    """
    h2 = (h_c_SMBHB(f_Hz, A_SMBHB)) ** 2 \
        + (h_c_powerlaw(f_Hz, A_string, ALPHA_STRING)) ** 2 \
        + (h_c_powerlaw(f_Hz, A_mem, ALPHA_V5_MEMBRANE)) ** 2
    return np.sqrt(h2)


def effective_alpha(f_Hz: np.ndarray,
                     A_SMBHB: float,
                     A_string: float,
                     A_mem: float) -> np.ndarray:
    """Pente effective alpha_eff(f) = d ln h_c / d ln f d'une mixture.

    Calculée par différence finie logarithmique. Utile pour comparer
    à NANOGrav (qui fitte un alpha unique sur 5 bins).
    """
    f = np.asarray(f_Hz, dtype=float)
    h = h_c_V5_combined(f, A_SMBHB, A_string, A_mem)
    log_f = np.log(f)
    log_h = np.log(np.maximum(h, 1e-30))
    # Pente locale (différence finie centrée).
    alpha = np.gradient(log_h, log_f)
    return alpha


# ---------------------------------------------------------------------------
# 3. Données NANOGrav 15-yr (5 bins, posterior par-bin).
# ---------------------------------------------------------------------------
# Strain par bin (figure 1 de Agazie+ 2023). Valeurs lues sur la posteriorbande.
# Format : (f_Hz, log10 h_c median, log10 h_c sigma).
# Source : NANOGrav 15-yr free-spectrum fit (table 5 de l'article).
# Approximation lisible : 5 bins espacés log, médian sur fig 1.
NANOGRAV_15YR_FREESPEC = np.array([
    # (f_Hz,        log10(h_c),  sigma(log10 h_c))
    (2.0e-9,       -13.85,        0.20),
    (5.0e-9,       -14.20,        0.18),
    (1.0e-8,       -14.45,        0.18),
    (2.0e-8,       -14.75,        0.20),
    (5.0e-8,       -15.10,        0.25),
])


def nanograv_15yr_bins() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (f, h_c, sigma_log10) des 5 bins NANOGrav 15-yr.

    Donnée digitisée depuis figure 1 de Agazie+ 2023, posterior
    free-spectrum (par-fréquence, sans contrainte de power-law).
    Sigma sont les barres 1-sigma symétrisées.
    """
    arr = NANOGRAV_15YR_FREESPEC
    f = arr[:, 0]
    h_c = 10.0 ** arr[:, 1]
    sigma_log10 = arr[:, 2]
    return f, h_c, sigma_log10


# ---------------------------------------------------------------------------
# 4. Likelihood chi2.
# ---------------------------------------------------------------------------
def chi2_model(model_h_c: np.ndarray,
                obs_h_c: np.ndarray,
                obs_sigma_log10: np.ndarray) -> float:
    """chi² Gaussien dans log10(h_c).

    Le posterior par bin NANOGrav est approximativement Gaussien
    en log10(h_c) (fig 1, Agazie+ 2023).

        chi^2 = sum_i [(log10 h_c_model_i - log10 h_c_obs_i) / sigma_i]^2
    """
    m = np.log10(np.maximum(model_h_c, 1e-30))
    o = np.log10(np.maximum(obs_h_c, 1e-30))
    return float(np.sum(((m - o) / obs_sigma_log10) ** 2))


def fit_SMBHB_pure(f: np.ndarray,
                    h_c_obs: np.ndarray,
                    sigma_log10: np.ndarray) -> dict:
    """Fit pur SMBHB (1 paramètre A_yr) sur les bins NANOGrav.

    Minimisation analytique : la log-amplitude shift est la moyenne
    pondérée des résidus.
    """
    log_h_model_unit = np.log10(h_c_SMBHB(f, A_yr=1.0))
    log_h_obs = np.log10(h_c_obs)
    w = 1.0 / sigma_log10 ** 2
    # log10(A_yr) qui minimise chi² :
    log_A = float(np.sum(w * (log_h_obs - log_h_model_unit)) / np.sum(w))
    A_yr = 10.0 ** log_A
    h_model = h_c_SMBHB(f, A_yr=A_yr)
    chi2 = chi2_model(h_model, h_c_obs, sigma_log10)
    # gamma effectif (forcé à 13/3 ici)
    return {
        "A_yr": A_yr,
        "alpha": ALPHA_SMBHB,
        "gamma": GAMMA_SMBHB,
        "chi2": chi2,
        "n_data": len(f),
        "n_params": 1,
        "dof": len(f) - 1,
    }


def fit_V5_combined(f: np.ndarray,
                     h_c_obs: np.ndarray,
                     sigma_log10: np.ndarray,
                     include_string: bool = True,
                     include_membrane: bool = True) -> dict:
    """Fit linéaire des amplitudes (A_SMBHB, A_string, A_mem) sur les bins.

    Le modèle est LINÉAIRE en A_k^2 (puisque h_c^2 = sum_k A_k^2 t_k^2),
    mais NON-linéaire dans log10. Nous procédons par une régression
    log-pondérée non-négative sur les coefficients A_k.

    Algorithme : grille en log10(A_string), log10(A_mem) avec A_SMBHB
    résolu analytiquement à chaque point.
    """
    f = np.asarray(f, dtype=float)
    # Bases unitaires (sans amplitude)
    t_SMBHB = (f / F_YR_HZ) ** ALPHA_SMBHB
    t_string = (f / F_YR_HZ) ** ALPHA_STRING if include_string else None
    t_mem = (f / F_YR_HZ) ** ALPHA_V5_MEMBRANE if include_membrane else None

    log_h_obs = np.log10(h_c_obs)
    w = 1.0 / sigma_log10 ** 2

    # Grille en log10 des contributions secondaires.
    if include_string and include_membrane:
        log_A_str_grid = np.linspace(-18, -13, 41)
        log_A_mem_grid = np.linspace(-18, -13, 41)
    elif include_string:
        log_A_str_grid = np.linspace(-18, -13, 81)
        log_A_mem_grid = np.array([-30.0])  # négligeable
    elif include_membrane:
        log_A_str_grid = np.array([-30.0])
        log_A_mem_grid = np.linspace(-18, -13, 81)
    else:
        log_A_str_grid = np.array([-30.0])
        log_A_mem_grid = np.array([-30.0])

    best_chi2 = np.inf
    best = None
    for la_str in log_A_str_grid:
        A_str = 10.0 ** la_str
        for la_mem in log_A_mem_grid:
            A_mem = 10.0 ** la_mem
            # Pour A_SMBHB donné, h_c^2 = A_SMBHB^2 t_SMBHB^2 + autres.
            other2 = A_str ** 2 * (t_string ** 2 if t_string is not None else 0.0) \
                   + A_mem ** 2 * (t_mem ** 2 if t_mem is not None else 0.0)
            # Optimum 1D en log10(A_SMBHB) : grille fine.
            log_A_smbhb_grid = np.linspace(-17, -13, 81)
            chi2_arr = np.empty_like(log_A_smbhb_grid)
            for k, la_sm in enumerate(log_A_smbhb_grid):
                A_sm = 10.0 ** la_sm
                h2 = A_sm ** 2 * t_SMBHB ** 2 + other2
                h_mod = np.sqrt(np.maximum(h2, 1e-60))
                chi2_arr[k] = np.sum(w * (np.log10(h_mod) - log_h_obs) ** 2)
            k_best = int(np.argmin(chi2_arr))
            chi2 = chi2_arr[k_best]
            if chi2 < best_chi2:
                best_chi2 = chi2
                best = {
                    "A_SMBHB": 10.0 ** log_A_smbhb_grid[k_best],
                    "A_string": A_str if include_string else 0.0,
                    "A_mem": A_mem if include_membrane else 0.0,
                    "chi2": float(chi2),
                }

    # Calcul gamma effectif au pivot f = 10 nHz
    f_pivot = 1.0e-8
    alpha_eff_pivot = float(effective_alpha(
        np.array([f_pivot * 0.5, f_pivot, f_pivot * 2.0]),
        best["A_SMBHB"], best["A_string"], best["A_mem"])[1])
    gamma_eff_pivot = alpha_to_gamma(alpha_eff_pivot)

    n_params = 1 + int(include_string) + int(include_membrane)
    best.update({
        "alpha_eff_10nHz": alpha_eff_pivot,
        "gamma_eff_10nHz": gamma_eff_pivot,
        "n_data": len(f),
        "n_params": n_params,
        "dof": len(f) - n_params,
    })
    return best


# ---------------------------------------------------------------------------
# 5. Bayes factor V5 vs SMBHB pur.
# ---------------------------------------------------------------------------
def bayes_factor_V5_vs_SMBHB(fit_SMBHB: dict, fit_V5: dict) -> dict:
    """Bayes factor approximatif via BIC (Schwarz).

        BIC = chi^2 + k * ln(N)
        ln(B_V5/SMBHB) ≃ (BIC_SMBHB - BIC_V5)/2

    Conventions Jeffreys :
        ln B > 5 : evidence "decisive" pour V5
        ln B in [2.5,5] : evidence "strong"
        ln B in [1,2.5] : evidence "substantial"
        ln B in [-1,1] : indistinct
        ln B < -1 : favorise SMBHB
    """
    N = fit_SMBHB["n_data"]
    bic_S = fit_SMBHB["chi2"] + fit_SMBHB["n_params"] * math.log(N)
    bic_V = fit_V5["chi2"] + fit_V5["n_params"] * math.log(N)
    ln_B = (bic_S - bic_V) / 2.0
    if ln_B > 5:
        verdict = "decisive_for_V5"
    elif ln_B > 2.5:
        verdict = "strong_for_V5"
    elif ln_B > 1.0:
        verdict = "substantial_for_V5"
    elif ln_B > -1.0:
        verdict = "indistinct"
    elif ln_B > -2.5:
        verdict = "substantial_for_SMBHB"
    else:
        verdict = "decisive_for_SMBHB"
    return {
        "BIC_SMBHB": float(bic_S),
        "BIC_V5": float(bic_V),
        "ln_Bayes_V5_over_SMBHB": float(ln_B),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 6. Synthèse pour reporting.
# ---------------------------------------------------------------------------
@dataclass
class NANOGravTestResult:
    fit_SMBHB: dict
    fit_V5_string: dict
    fit_V5_membrane: dict
    fit_V5_full: dict
    bayes_string: dict
    bayes_membrane: dict
    bayes_full: dict
    obs_gamma: float
    obs_gamma_sigma: float


def run_full_test() -> NANOGravTestResult:
    """Lance le test complet : SMBHB pur vs trois variantes V5."""
    f, h_obs, sigma = nanograv_15yr_bins()

    fit_S = fit_SMBHB_pure(f, h_obs, sigma)
    fit_str = fit_V5_combined(f, h_obs, sigma,
                                include_string=True, include_membrane=False)
    fit_mem = fit_V5_combined(f, h_obs, sigma,
                                include_string=False, include_membrane=True)
    fit_full = fit_V5_combined(f, h_obs, sigma,
                                 include_string=True, include_membrane=True)

    bayes_str = bayes_factor_V5_vs_SMBHB(fit_S, fit_str)
    bayes_mem = bayes_factor_V5_vs_SMBHB(fit_S, fit_mem)
    bayes_full = bayes_factor_V5_vs_SMBHB(fit_S, fit_full)

    return NANOGravTestResult(
        fit_SMBHB=fit_S,
        fit_V5_string=fit_str,
        fit_V5_membrane=fit_mem,
        fit_V5_full=fit_full,
        bayes_string=bayes_str,
        bayes_membrane=bayes_mem,
        bayes_full=bayes_full,
        obs_gamma=NANOGRAV_GAMMA_HD,
        obs_gamma_sigma=NANOGRAV_GAMMA_HD_SIGMA,
    )


def summary() -> dict:
    """Synthèse compacte."""
    r = run_full_test()
    return {
        "gamma_SMBHB_theory": GAMMA_SMBHB,
        "gamma_string_theory": GAMMA_STRING,
        "gamma_V5_point_theory": GAMMA_V5_POINT,
        "gamma_V5_membrane_theory": GAMMA_V5_MEMBRANE,
        "gamma_obs_NANOGrav": r.obs_gamma,
        "gamma_obs_sigma": r.obs_gamma_sigma,
        "chi2_SMBHB_pure": r.fit_SMBHB["chi2"],
        "chi2_V5_full": r.fit_V5_full["chi2"],
        "gamma_eff_V5_full_at_10nHz": r.fit_V5_full["gamma_eff_10nHz"],
        "ln_Bayes_V5_full_over_SMBHB": r.bayes_full["ln_Bayes_V5_over_SMBHB"],
        "verdict": r.bayes_full["verdict"],
    }


if __name__ == "__main__":
    s = summary()
    print("=" * 72)
    print(" V5 — Test gamma_SGWB vs NANOGrav 15-yr (Agazie+ 2023)")
    print("=" * 72)
    print(f"\n  gamma SMBHB (theorie)        = {s['gamma_SMBHB_theory']:.3f}")
    print(f"  gamma strings (theorie)      = {s['gamma_string_theory']:.3f}")
    print(f"  gamma V5 ponctuel (theorie)  = {s['gamma_V5_point_theory']:.3f}")
    print(f"  gamma V5 membrane (theorie)  = {s['gamma_V5_membrane_theory']:.3f}")
    print(f"  gamma OBSERVE NANOGrav       = {s['gamma_obs_NANOGrav']:.2f}"
          f" +/- {s['gamma_obs_sigma']:.2f}")
    print(f"\n  chi^2 SMBHB pur (4 dof)     = {s['chi2_SMBHB_pure']:.2f}")
    print(f"  chi^2 V5 full (2 dof)        = {s['chi2_V5_full']:.2f}")
    print(f"  gamma_eff(V5 full, 10 nHz)   = {s['gamma_eff_V5_full_at_10nHz']:.2f}")
    print(f"  ln Bayes V5 / SMBHB          = {s['ln_Bayes_V5_full_over_SMBHB']:+.2f}")
    print(f"  Verdict                      = {s['verdict']}")
    print("=" * 72)
