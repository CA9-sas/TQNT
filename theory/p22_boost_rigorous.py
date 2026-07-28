"""
V5.18 — Boost P22 rigoureux dérivé de V5 §16.4 (forme fermée).

Cette construction remplace l'ansatz LINÉAIRE de `theory/smf_V5.py`
(boost(z) = 1 + α_P22 · max(0, z - z_crit)^γ_P22) par la FORME FERMÉE
exponentielle directement dérivée du Postulat P22 :

    Γ(r) = Γ_0 · exp(η · ξ(r))
    ξ(r) = |∇Ω²(r)| · λbar_C / Ω²(r)
    Ω²(r) = 1 + κ_g · |Ψ(r)|²            (P5)

Le boost de la HMF à redshift z et masse de halo M_h s'écrit alors :

    boost_P22(z, M_h) = Γ(z, M_h) / Γ_0 = exp(η · ξ(z, M_h))

avec ξ évalué à la *frontière* du halo (où le gradient métrique
matière-vide est maximal).

---------------------------------------------------------------------------
ESTIMATION DE ξ AU BORD D'UN HALO COSMOLOGIQUE
---------------------------------------------------------------------------
À l'échelle d'un halo virialisé de masse M_h à redshift z :

  1) r_vir(M_h, z) = (3 M_h / (4π · 200 · ρ_crit(z)))^(1/3)
  2) densité moyenne intra-halo : ρ_h(z) = 200 · ρ_crit(z)
  3) densité numérique baryonique : n_b(z) ≈ (Ω_b/Ω_m) · ρ_h / m_p
     (en supposant que la fraction baryonique du halo trace le ratio
      cosmologique Ω_b/Ω_m ; les baryons sont les sources principales
      de |Ψ|² nucléonique en V5)
  4) |Ψ(r_vir)|² ≈ n_b(z)          (P3 : densité d'onde TQNT identifiée
                                    avec la densité numérique de nœuds)
  5) h = Ω² - 1 = κ_g · |Ψ|² ≈ κ_g · n_b(z)            (extrêmement petit)
  6) Gradient au bord : |∇Ω²| ~ h / r_vir
     (variation de Ω² du centre h_0 à zéro sur l'échelle r_vir)
  7) ξ = (h/r_vir) · λbar_C / Ω² ≈ h · λbar_C / r_vir  (puisque Ω² ≈ 1)

Donc, en forme fermée :

    ξ(z, M_h) ≈ κ_g · n_b(z) · λbar_C(e) / r_vir(M_h, z)

---------------------------------------------------------------------------
MAGNITUDE ATTENDUE
---------------------------------------------------------------------------
* κ_g ≈ 3,12 × 10⁻⁵³ m
* λbar_C(e) ≈ 3,86 × 10⁻¹³ m
* n_b(z=10) ~ 1 m⁻³ (très diffus en physique de halos)
* r_vir(M_h=10¹⁰, z=10) ~ kpc → 10¹⁹ m

Cela donne ξ ~ 10⁻⁸⁵ — incompressiblement microscopique. Le boost
exp(η ξ) ≈ 1 + η·ξ ≈ 1 à toutes les fins pratiques.

C'est le verdict HONNÊTE de V5.18 : la forme rigoureuse de §16.4 ne
peut PAS reproduire les boosts ×10³–10⁴ requis par JWST. Le mécanisme
P22 tel qu'il est posté ne résout PAS la « crise des galaxies
impossibles ». V5 §11 (×1,5 nominal) était DÉJÀ surestimé.

References
----------
- V5 §16.4 (P22, eq:P22-gamma)
- V5 §11.5 (HMF, "explique JWST")
- V5 chap. coupling_kg (κ_g = (96π/7) α_EM^18 λbar_C(p))
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from theory.coupling_kg import kappa_g_V5p5

# ---------------------------------------------------------------------------
# Constantes physiques (SI)
# ---------------------------------------------------------------------------
HBAR_J_S = 1.054571817e-34
C_M_S = 299792458.0
ME_KG = 9.1093837015e-31
MP_KG = 1.67262192369e-27
G_NEWTON = 6.67430e-11
H0_KM_S_MPC = 67.4
MPC_M = 3.0857e22
H0_SI = H0_KM_S_MPC * 1000.0 / MPC_M                # s⁻¹
MSUN_KG = 1.98892e30

OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685
OMEGA_B = 0.0493
DELTA_VIR = 200.0                                    # surdensité virialisation

# Longueurs de Compton réduites
LAMBDABAR_C_E_M = HBAR_J_S / (ME_KG * C_M_S)         # ≈ 3,86 × 10⁻¹³ m
LAMBDABAR_C_P_M = HBAR_J_S / (MP_KG * C_M_S)         # ≈ 2,10 × 10⁻¹⁶ m

# κ_g (en mètres) dérivé V5 §coupling_kg
KAPPA_G_M = kappa_g_V5p5(units="m")                  # ≈ 3,12 × 10⁻⁵³ m

# Coefficient P22 par défaut (ordre unité, V5 §16.4 calibré sur résidu DE)
ETA_P22_DEFAULT = 1.0


# ---------------------------------------------------------------------------
# Métrique cosmologique
# ---------------------------------------------------------------------------
def hubble(z: float) -> float:
    """H(z) en s⁻¹, ΛCDM Planck 2018."""
    return H0_SI * math.sqrt(OMEGA_M * (1 + z) ** 3 + OMEGA_LAMBDA)


def rho_crit(z: float) -> float:
    """Densité critique ρ_crit(z) en kg/m³."""
    return 3.0 * hubble(z) ** 2 / (8.0 * math.pi * G_NEWTON)


def r_vir(M_h_Msun: float, z: float) -> float:
    """Rayon de virialisation (m) d'un halo M_h à redshift z, Δ = 200."""
    M_kg = M_h_Msun * MSUN_KG
    rho = DELTA_VIR * rho_crit(z)
    return (3.0 * M_kg / (4.0 * math.pi * rho)) ** (1.0 / 3.0)


def n_baryon_in_halo(z: float) -> float:
    """Densité numérique baryonique moyenne dans un halo virialisé (m⁻³).

    n_b = (Ω_b / Ω_m) · Δ_vir · ρ_crit(z) / m_p
    """
    rho_halo = DELTA_VIR * rho_crit(z)
    return (OMEGA_B / OMEGA_M) * rho_halo / MP_KG


# ---------------------------------------------------------------------------
# Métrique émergente P5 dans halo
# ---------------------------------------------------------------------------
def psi_squared_halo(M_h_Msun: float, z: float) -> float:
    """|Ψ|² moyen dans le halo (m⁻³).

    Identification V5 : |Ψ|² est la densité numérique de nœuds nucléoniques,
    qui suit la densité baryonique du halo.
    """
    return n_baryon_in_halo(z)  # M_h cancels in the mean density approx


def h_halo(M_h_Msun: float, z: float) -> float:
    """h = Ω² - 1 = κ_g · |Ψ|² dans le halo. Renvoyé non sommé à 1
    pour préserver la précision quand h ≪ 1 (cas cosmologique typique).
    """
    return KAPPA_G_M * psi_squared_halo(M_h_Msun, z)


def h_cosmo(z: float) -> float:
    """h_cosmo = κ_g · |Ψ_cosmo|² (overdensity sur fond moyen)."""
    rho_m = OMEGA_M * rho_crit(z)
    n_b_cosmo = (OMEGA_B / OMEGA_M) * rho_m / MP_KG
    return KAPPA_G_M * n_b_cosmo


def omega_squared_halo(M_h_Msun: float, z: float) -> float:
    """Ω² = 1 + h dans le halo (numériquement ≈ 1)."""
    return 1.0 + h_halo(M_h_Msun, z)


def omega_squared_cosmo(z: float) -> float:
    """Ω²(z) cosmologique : milieu intergalactique, densité moyenne ρ_m(z)."""
    return 1.0 + h_cosmo(z)


def delta_h_halo_to_cosmo(M_h_Msun: float, z: float) -> float:
    """Δh = h_halo - h_cosmo = κ_g · (|Ψ_halo|² - |Ψ_cosmo|²).

    Préserve la précision en évaluant directement la différence des
    densités numériques (qui est non-nulle et calculable) avant de
    multiplier par κ_g.
    """
    rho_h = DELTA_VIR * rho_crit(z)
    rho_m = OMEGA_M * rho_crit(z)
    delta_n = (OMEGA_B / OMEGA_M) * (rho_h - rho_m) / MP_KG
    return KAPPA_G_M * delta_n


def grad_omega_squared_halo_boundary(M_h_Msun: float, z: float) -> float:
    """|∇Ω²| à la frontière du halo (m⁻¹).

    Modèle : Ω² passe de Ω²_halo au centre à Ω²_cosmo à l'extérieur
    sur l'échelle r_vir. Donc |∇Ω²| = |Δh| / r_vir.
    On calcule via la différence directe des densités (pas via la
    soustraction d'Ω² ≈ 1 qui annule par cancellation catastrophique).
    """
    dh = delta_h_halo_to_cosmo(M_h_Msun, z)
    R = r_vir(M_h_Msun, z)
    return abs(dh) / R


# ---------------------------------------------------------------------------
# Paramètre ξ et boost P22
# ---------------------------------------------------------------------------
def xi_at_halo_boundary(M_h_Msun: float, z: float,
                        lambdabar_C: float = LAMBDABAR_C_E_M) -> float:
    """ξ = |∇Ω²| · λbar_C / Ω² évalué au bord du halo (sans dimension)."""
    grad = grad_omega_squared_halo_boundary(M_h_Msun, z)
    omsq = omega_squared_halo(M_h_Msun, z)
    return grad * lambdabar_C / omsq


def boost_P22_rigorous(M_h_Msun: float, z: float,
                       eta: float = ETA_P22_DEFAULT,
                       lambdabar_C: float = LAMBDABAR_C_E_M) -> float:
    """Boost P22 dérivé rigoureusement de V5 §16.4.

    boost = exp(η · ξ)

    Pour ξ ≪ 1 (régime cosmologique courant), boost ≈ 1 + η·ξ ≈ 1.
    Pour atteindre boost ~ 10³ il faudrait η·ξ ~ 7 ; or ξ est typiquement
    ~ 10⁻⁸⁰. Le boost rigoureux est INDISTINCT de 1.
    """
    xi = xi_at_halo_boundary(M_h_Msun, z, lambdabar_C)
    arg = eta * xi
    # Protection numérique
    if arg > 700:
        return float("inf")
    if arg < -700:
        return 0.0
    return math.exp(arg)


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------
@dataclass
class P22Diagnostic:
    z: float
    M_h_Msun: float
    r_vir_m: float
    r_vir_kpc: float
    n_baryon_m3: float
    omega_squared_halo: float
    h_halo: float                # Ω² - 1 dans le halo
    grad_omega_squared: float    # m⁻¹
    xi: float
    boost: float


def diagnose(M_h_Msun: float, z: float,
             eta: float = ETA_P22_DEFAULT) -> P22Diagnostic:
    """Tableau diagnostique complet pour (z, M_h)."""
    R = r_vir(M_h_Msun, z)
    n_b = n_baryon_in_halo(z)
    omsq = omega_squared_halo(M_h_Msun, z)
    grad = grad_omega_squared_halo_boundary(M_h_Msun, z)
    xi = xi_at_halo_boundary(M_h_Msun, z)
    bst = boost_P22_rigorous(M_h_Msun, z, eta=eta)
    return P22Diagnostic(
        z=z, M_h_Msun=M_h_Msun,
        r_vir_m=R, r_vir_kpc=R / (3.0857e19),
        n_baryon_m3=n_b,
        omega_squared_halo=omsq,
        h_halo=h_halo(M_h_Msun, z),
        grad_omega_squared=grad,
        xi=xi,
        boost=bst,
    )


# ---------------------------------------------------------------------------
# Recherche : quelle ξ faudrait-il pour atteindre boost ~ 10³ ?
# ---------------------------------------------------------------------------
def xi_required_for_target_boost(boost_target: float,
                                 eta: float = ETA_P22_DEFAULT) -> float:
    """ξ requis pour boost = exp(η ξ) = boost_target.
    ⇒ ξ_req = ln(boost_target) / η.
    """
    if boost_target <= 0:
        raise ValueError("boost_target doit être > 0")
    return math.log(boost_target) / eta


def gap_to_jwst(M_h_Msun: float, z: float,
                boost_target: float = 1e3,
                eta: float = ETA_P22_DEFAULT) -> dict:
    """Quantifie l'écart entre ξ rigoureux et ξ requis pour le boost cible."""
    xi_now = xi_at_halo_boundary(M_h_Msun, z)
    xi_req = xi_required_for_target_boost(boost_target, eta)
    return {
        "z": z, "M_h_Msun": M_h_Msun,
        "xi_rigorous": xi_now,
        "xi_required_JWST": xi_req,
        "log10_gap": math.log10(xi_req) - math.log10(max(xi_now, 1e-300)),
        "boost_rigorous": boost_P22_rigorous(M_h_Msun, z, eta=eta),
        "boost_target_JWST": boost_target,
    }


if __name__ == "__main__":
    print("=" * 78)
    print("V5.18 — Boost P22 RIGOUREUX (dérivé V5 §16.4)")
    print("=" * 78)
    print(f"\nκ_g                = {KAPPA_G_M:.3e} m")
    print(f"λbar_C(électron)   = {LAMBDABAR_C_E_M:.3e} m")
    print(f"η_P22              = {ETA_P22_DEFAULT:.3f}\n")
    print(f"{'z':>5} {'M_h (Msun)':>12} {'r_vir (m)':>12}"
          f" {'n_b (m⁻³)':>12} {'h_halo':>12}"
          f" {'|∇Ω²|':>12} {'ξ':>12} {'boost':>10}")
    print("-" * 100)
    for z in (5.0, 8.0, 10.0, 12.0):
        for Mh in (1e9, 1e10, 1e11):
            d = diagnose(Mh, z)
            print(f"{z:>5.1f} {d.M_h_Msun:>12.2e} {d.r_vir_m:>12.3e}"
                  f" {d.n_baryon_m3:>12.3e} {d.h_halo:>12.3e}"
                  f" {d.grad_omega_squared:>12.3e} {d.xi:>12.3e}"
                  f" {d.boost:>10.6f}")
    print()
    g = gap_to_jwst(1e10, 10.0)
    print(f"Cible JWST : boost ~10³ requiert ξ_req = {g['xi_required_JWST']:.3e}")
    print(f"ξ rigoureux V5.18 (z=10, M_h=10¹⁰) = {g['xi_rigorous']:.3e}")
    print(f"Écart log10 = {g['log10_gap']:.1f} décades")
    print()
    print("VERDICT : V5.18 rigoureux donne boost ≈ 1, INDISTINCT de ΛCDM.")
    print("Le formalisme P22 §16.4 NE peut PAS produire ×10³–10⁴ requis JWST.")
