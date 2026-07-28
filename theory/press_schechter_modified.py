"""
V5.5 ÉTAPE 7 — Press-Schechter modifié par TQNT-V5.

Dérive |Ψ_halo|²(M_h) et β = d log|Ψ_halo|² / d log M_h depuis Press-
Schechter standard + modification V5 (P22 nucléation aux bords +
condensation χ-field primordiale).

OBJECTIF :
Fermer la chaîne V5.5 en dérivant β depuis premiers principes. La
formule V5.5 étape 5 (Bertschinger CGQ-modifié, commit 8069a7a) :

    Δ_CGQ = -(1/2) · h/(1+h) · β
    h = κ_g · |Ψ_halo|²

requiert β ≈ -6.29 pour reproduire la pente W-B (-0.10) sur SPARC.
Avec κ_g dérivé V5.5 étape 6 (commit pending) à 0.05 %, il reste à
montrer que la dynamique Press-Schechter modifiée par V5 produit
naturellement β = -6.29 ± 5 %.

CADRE THÉORIQUE :

1. PRESS-SCHECHTER STANDARD :
   n(M, z) dM = (ρ_m,0 / M) · |d log σ(M)/d log M| · f_PS(δ_c/σ(M)) dM/M
   avec f_PS(ν) = sqrt(2/π) · ν · exp(-ν²/2) (Press-Schechter 1974)

2. MODIFICATION V5 P22 :
   À petite masse (M < 10⁹ M_sun), la nucléation aux frontières
   matière/vide supprime la HMF par ~5 % :
     n_V5(M, z) = n_PS(M, z) · (1 - 0.05 · exp(-(M/M_*)^β_P22))
   avec M_* = 10⁹ M_sun et β_P22 ~ 0.5.

   À grande masse (M > 10¹⁴ M_sun), la HMF est identique à ΛCDM (~1 %).

3. DENSITÉ HALO |Ψ_halo|²(M_h) :
   Pour un halo NFW de masse M_h :
   ρ_NFW(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
   |Ψ_halo|²(r) ∝ ρ_NFW(r) (V5 P5)
   |Ψ_halo|²(r_{1/2,b}) dépend de r_{1/2,b}/r_s = α(M_h)

4. β = d log|Ψ_halo|² / d log M_h :
   Combinaison de :
     - Concentration c(M_h) : Wechsler-Bullock c ∝ M_h^(-0.10)
     - r_vir ∝ M_h^(1/3)
     - r_s = r_vir/c ∝ M_h^(0.43)
     - r_{1/2,b}/r_s scaling avec M_h (Mo-Mao-White)

REFERENCES
----------
- Press & Schechter 1974, ApJ 187, 425
- Bond et al. 1991, ApJ 379, 440 (modèle d'excursion sets)
- V5.tex §11.2-11.4 (HMF prédictions V5)
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

# Constantes cosmologiques (Planck 2018)
RHO_CRIT_MSUN_KPC3 = 1.39e8  # M_sun/kpc³
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685
H0 = 67.4  # km/s/Mpc

# Press-Schechter
DELTA_C = 1.686  # seuil de collapse linéaire (EdS)
N_EFF_SCALE = -1.5  # indice spectral effectif à l'échelle galactique

# V5.5 étape 6 résultats (Agent 1)
KAPPA_G_V5 = 3.1348e-53  # m, formule canonique (96π/7)·α_EM^18·λbar_C(proton)


def variance_M(M_Msun: float) -> float:
    """Variance σ²(M) du champ de densité lissé à l'échelle M.

    Approximation : σ²(M) ∝ M^(-(n+3)/3) avec n = -1.5 pour les
    échelles galactiques.

    Normalisation : σ(M = 10¹² M_sun) ≈ 0.9 (Planck σ_8 = 0.811).
    """
    # σ_8 ≈ 0.81 pour Planck. Conversion vers σ(M):
    # σ²(M)/σ²(M_8) = (M_8/M)^((n+3)/3)
    M_8 = 5.97e14  # M_sun (masse contenue dans sphère de rayon 8 Mpc/h)
    sigma_8 = 0.811
    exponent = (N_EFF_SCALE + 3) / 3  # = 0.5
    return sigma_8 ** 2 * (M_8 / M_Msun) ** exponent


def press_schechter_HMF(M_Msun: float, z: float = 0.0) -> float:
    """Mass function Press-Schechter standard à redshift z.

    n(M) dM = (ρ_m,0 / M) · |dν/dM| · f(ν) dM
    avec ν = δ_c / σ(M)
    f(ν) = sqrt(2/π) ν exp(-ν²/2)

    Retourne dn/dlogM en M_sun⁻¹ Mpc⁻³.
    """
    sigma = math.sqrt(variance_M(M_Msun))
    nu = DELTA_C / sigma
    f_PS = math.sqrt(2 / math.pi) * nu * math.exp(-nu**2 / 2)
    # |d log σ / d log M| pour σ²(M) ∝ M^(-α), |d log σ / d log M| = α/2
    dlog_sigma = (N_EFF_SCALE + 3) / 6
    rho_m = OMEGA_M * 2.78e11 * (H0/100)**2  # M_sun/Mpc³ at z=0
    return rho_m / M_Msun * f_PS * dlog_sigma / sigma


def V5_HMF_correction(M_Msun: float, z: float = 0.0) -> float:
    """Facteur correctif V5 P22 à la HMF.

    Suppression à petite masse (M < 10⁹) par ~5%.
    Amplification à haut z (M > 10¹⁰ et z > 5) par ~50%.
    À grande masse (M > 10¹⁴) : ~1, identique à ΛCDM.
    """
    M_star_low = 1e9
    M_star_high = 1e14
    suppression_low = 0.05 * math.exp(-(M_Msun / M_star_low) ** 0.5)
    if z > 5:
        boost_high_z = 1 + 0.5 * math.exp(-((math.log10(M_Msun) - 10)**2 / 4))
    else:
        boost_high_z = 1.0
    correction = (1 - suppression_low) * boost_high_z
    return correction


def V5_HMF(M_Msun: float, z: float = 0.0) -> float:
    """Mass function TQNT-V5 corrigée."""
    return press_schechter_HMF(M_Msun, z) * V5_HMF_correction(M_Msun, z)


def concentration_WB(M_Msun: float) -> float:
    """Concentration Wechsler-Bullock : c = 9.6 · (M_h/10¹² M_sun)^(-0.10)."""
    return 9.6 * (M_Msun / 1e12) ** (-0.10)


def r_virial_kpc(M_Msun: float) -> float:
    """Rayon viriel : r_vir = (3 M_h / (4π × 200 ρ_crit))^(1/3)."""
    return (3 * M_Msun / (4 * math.pi * 200 * RHO_CRIT_MSUN_KPC3)) ** (1/3)


def r_half_baryon_kpc(M_Msun: float) -> float:
    """Rayon de mi-masse baryonique (Mo-Mao-White 1998).

    r_d ≈ 0.0185 × r_vir × λ (où λ ≈ 0.04 = spin parameter)
    r_{1/2,b} ≈ 1.68 × r_d (exponential disk)
    """
    r_vir = r_virial_kpc(M_Msun)
    spin = 0.04  # Bullock+2001
    r_d = 0.0185 * r_vir * spin * 10  # facteur ~10 pour matcher empirique
    return 1.68 * r_d


def psi_squared_halo(M_Msun: float) -> float:
    """Densité du champ χ |Ψ_halo|²(r_{1/2,b}) ∝ ρ_NFW.

    Pour NFW :
        ρ_NFW(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
    Au r_{1/2,b} avec α = r_{1/2,b}/r_s :
        ρ_NFW(α·r_s) = ρ_s / [α(1+α)²]
    """
    c = concentration_WB(M_Msun)
    r_vir = r_virial_kpc(M_Msun)
    r_s = r_vir / c
    r_half = r_half_baryon_kpc(M_Msun)
    alpha = r_half / r_s
    # ρ_s = ρ_crit × δ_NFW(c)
    # δ_NFW(c) = (200/3) · c³ / [ln(1+c) - c/(1+c)]
    delta_NFW = (200 / 3) * c**3 / (math.log(1 + c) - c / (1 + c))
    rho_s = RHO_CRIT_MSUN_KPC3 * delta_NFW
    rho_NFW = rho_s / (alpha * (1 + alpha)**2)
    return rho_NFW  # M_sun/kpc³, ∝ |Ψ|² (V5 P5)


def beta_psi(M_Msun: float, dlog_M: float = 0.01) -> float:
    """β = d log|Ψ_halo|² / d log M_h."""
    log_M = math.log10(M_Msun)
    psi1 = psi_squared_halo(10 ** (log_M - dlog_M))
    psi2 = psi_squared_halo(10 ** (log_M + dlog_M))
    return (math.log10(psi2) - math.log10(psi1)) / (2 * dlog_M)


def h_at_M(M_Msun: float) -> float:
    """h(M_h) = κ_g · |Ψ_halo|²(M_h).

    Note : conversion M_sun/kpc³ → SI (kg/m³) puis multiplication par
    κ_g qui est en m (longueur). Le produit donne un nombre sans dimension.
    """
    psi_Msun_kpc3 = psi_squared_halo(M_Msun)
    # Conversion : 1 M_sun/kpc³ = 6.77e-23 kg/m³
    psi_SI = psi_Msun_kpc3 * 6.77e-23  # kg/m³
    return KAPPA_G_V5 * psi_SI  # adimensionnel (κ_g a la dimension d'une longueur)


def delta_CGQ_predicted(M_Msun: float) -> float:
    """Δ_CGQ (formule V5.5 étape 5)."""
    h = h_at_M(M_Msun)
    beta = beta_psi(M_Msun)
    return -0.5 * h / (1 + h) * beta


def summary(M_Msun: float = 1e12) -> dict:
    """Diagnostic à une masse de halo de référence."""
    return {
        "M_h_Msun": M_Msun,
        "c_WB": concentration_WB(M_Msun),
        "r_vir_kpc": r_virial_kpc(M_Msun),
        "r_half_baryon_kpc": r_half_baryon_kpc(M_Msun),
        "psi_squared_M_sun_kpc3": psi_squared_halo(M_Msun),
        "h_dim": h_at_M(M_Msun),
        "beta_psi": beta_psi(M_Msun),
        "delta_CGQ_predicted": delta_CGQ_predicted(M_Msun),
        "delta_CGQ_target_V5p5_step5": 0.41,
    }


if __name__ == "__main__":
    print("=" * 78)
    print("V5.5 ÉTAPE 7 — Press-Schechter modifié + |Ψ_halo|²(M_h)")
    print("=" * 78)

    print(f"\nKappa_g (V5.5 étape 6) : {KAPPA_G_V5:.4e} m\n")

    # Tableau récapitulatif sur plusieurs M_h
    print(f"  {'M_h (Msun)':>12s} {'c_WB':>8s} {'r_s (kpc)':>10s} "
          f"{'|Ψ|² (Msun/kpc³)':>20s} {'h':>10s} {'β':>10s} {'Δ_CGQ pred':>12s}")
    for log_M in [9, 10, 11, 12, 13, 14, 15]:
        M = 10 ** log_M
        s = summary(M)
        print(f"  {M:>12.0e} {s['c_WB']:>8.3f} "
              f"{r_virial_kpc(M)/s['c_WB']:>10.2f} "
              f"{s['psi_squared_M_sun_kpc3']:>20.3e} "
              f"{s['h_dim']:>10.3e} {s['beta_psi']:>+10.4f} "
              f"{s['delta_CGQ_predicted']:>+12.4f}")

    print()
    print("=" * 78)
    print("ANALYSE DE β")
    print("=" * 78)
    print(f"""
  Pour des halos NFW + WB c(M_h) + Mo-Mao-White r_{{1/2,b}}(M_h) :
    β = d log|Ψ_halo|² / d log M_h ≈ valeur observée

  La cible V5.5 étape 5 (Bertschinger CGQ-modifié) était β = -6.29 avec
  h = 0.15. Pure NFW + WB donne β ~ -0.3 à -1.0, soit 10-20× plus faible.

  HYPOTHÈSES POUR β STEEP :

  (i)  Profil de halo PLUS STEEP que NFW : Einasto ou ψ²-cumulée avec
       cusp marqué donnerait β ~ -1 à -2.

  (ii) Effet P22 multiplicatif : enhancement de |Ψ|² près des bords
       baryoniques. Échelle de couche P22 ~ λ_C/Ω². Multiplicateur
       ~ exp(η ξ) avec ξ ∝ ∇Ω²/Ω². Pour halos massifs (Ω² grand),
       ξ → 1 et le multiplicateur sature.

  (iii) Self-similarité Bertschinger PURE : si l'on prend ρ ∝ t^(-2/3)
        et t ∝ M_h^(1/2), alors |Ψ|² ∝ M_h^(-1/3) → β = -0.33.

  (iv) Correction conformelle Ω² : si Ω² varie comme M_h^q' avec q'
       ~ 0.5, le facteur Ω² renormalise localement Ψ². Multiplicateur
       additionnel à β.

  ÉCART OBSERVÉ : β_NFW ~ -1, β_V5.5_required = -6.29. Facteur 6×
  d'amplification structurelle reste à dériver.

  PISTE V5.5 ÉTAPE 8 :
    Coupler Press-Schechter modifié avec le profil Einasto ψ²-cumulé +
    P22 boundary enhancement. La combinaison devrait donner β ~ -6
    naturellement (ou identifier que la formule Δ_CGQ doit être révisée).
""")
