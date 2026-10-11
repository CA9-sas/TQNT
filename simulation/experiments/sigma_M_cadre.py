"""sigma(M) calculé DANS le cadre, sans ancre empruntée à LambdaCDM.

Entrées, toutes internes ou dérivées :
  A_s  = 1,89e-9   amplitude scalaire DÉRIVÉE sans paramètre libre (V5.15)
  n_s  = 1 - 2/N   inflation de Starobinsky (F1 fermée par V5.14), N e-folds
  h    = 0,672     H0 sous platitude (théorème Omega_tot = 1)
  Omega_m, Omega_b : registre cosmologique du cadre
  croissance D(a) : théorème de croissance (indiscernable de LambdaCDM)

Transfert : Eisenstein & Hu 1998, forme « sans oscillations » (few %).
C'est le seul emprunt, et il est de la physique linéaire standard, pas
une normalisation : il porte la FORME, l'amplitude vient du cadre.

Contrôle : sigma_8 prédit, à confronter au registre (S8 = 0,826 +- 0,011).
"""
from __future__ import annotations
import math

# ---- constantes du cadre ------------------------------------------------
A_S = 1.89e-9          # V5.15, dérivé
K_PIVOT = 0.05         # Mpc^-1 (convention déclarée)
N_EFOLDS = 55.0
N_S = 1.0 - 2.0 / N_EFOLDS
H = 0.672
OM_M, OM_B = 0.315, 0.0493
THETA27 = 2.7255 / 2.7
C_KMS = 2.998e5        # km/s
H0 = 100.0 * H         # km/s/Mpc
RHO_M = 2.775e11 * H * H * OM_M     # M_sun / Mpc^3


def transfer(k):
    """T(k), Eisenstein-Hu 1998 sans oscillations ; k en Mpc^-1."""
    om, ob = OM_M * H * H, OM_B * H * H
    fb = OM_B / OM_M
    s = 44.5 * math.log(9.83 / om) / math.sqrt(1.0 + 10.0 * ob ** 0.75)
    a_g = (1.0 - 0.328 * math.log(431.0 * om) * fb
           + 0.38 * math.log(22.3 * om) * fb * fb)
    g_eff = OM_M * H * (a_g + (1.0 - a_g) / (1.0 + (0.43 * k * s) ** 4))
    q = k * THETA27 * THETA27 / g_eff
    l0 = math.log(2.0 * math.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return l0 / (l0 + c0 * q * q)


def delta2_m(k):
    """Variance adimensionnée de la densité de matière AUJOURD'HUI."""
    # delta_m = (2/5) (k c / H0)^2 / Omega_m * T(k) * R_curvature
    pref = (2.0 / 5.0) * (k * C_KMS / H0) ** 2 / OM_M * transfer(k)
    return pref * pref * A_S * (k / K_PIVOT) ** (N_S - 1.0)


def sigma_R(R, nbin=900):
    """sigma(R) avec fenêtre top-hat, integration log en k."""
    lk1, lk2 = math.log(1e-5 / R if R > 1e-3 else 1e-4), math.log(300.0 / R)
    tot = 0.0
    dl = (lk2 - lk1) / nbin
    for i in range(nbin):
        lk = lk1 + (i + 0.5) * dl
        k = math.exp(lk)
        x = k * R
        w = 3.0 * (math.sin(x) - x * math.cos(x)) / x ** 3
        tot += delta2_m(k) * w * w * dl
    return math.sqrt(tot)


def R_of_M(M):
    return (3.0 * M / (4.0 * math.pi * RHO_M)) ** (1.0 / 3.0)


def sigma_M(M):
    return sigma_R(R_of_M(M))


if __name__ == "__main__":
    print("=== ENTRÉES DU CADRE ===")
    print(f"  A_s = {A_S:.3e} (dérivé, V5.15)   n_s = 1 - 2/{N_EFOLDS:.0f} = {N_S:.4f}")
    print(f"  h = {H}   Omega_m = {OM_M}   Omega_b = {OM_B}")
    print(f"  rho_m = {RHO_M:.3e} M_sun/Mpc^3\n")
    print("=== CONTRÔLE : sigma_8 ===")
    s8 = sigma_R(8.0 / H)
    print(f"  sigma_8 prédit = {s8:.4f}")
    print(f"  S8 = sigma_8 sqrt(Om/0,3) = {s8*math.sqrt(OM_M/0.3):.4f}"
          f"   (registre : 0,826 +- 0,011)")
    print(f"  Planck A_s = 2,10e-9 donnerait {s8*math.sqrt(2.10/1.89):.4f}\n")
    print("=== sigma(M) CALCULÉ ===")
    print(f"  {'M (M_sun)':>12s} {'R (Mpc)':>10s} {'sigma(M)':>10s}")
    for M in (1e14, 1e12, 1e10, 1e8, 1e7, 1e6, 1e5):
        print(f"  {M:12.0e} {R_of_M(M):10.2e} {sigma_M(M):10.3f}")
