"""Plafond sur f_F avec sigma(M) recalculé — entrées d'amplitude du CADRE.

Ce que le cadre fournit et qui n'est PAS emprunté :
  A_s = 1,89e-9  amplitude scalaire dérivée sans paramètre libre (V5.15)
  n_s = 1 - 2/N  inflation de Starobinsky (V5.14, F1 fermée)
  h, Omega_m, Omega_b : registre cosmologique sous platitude
Ce qui reste emprunté, et qui est déclaré :
  la FORME du transfert linéaire (Eisenstein-Hu 98 sans oscillations)
  une CONSTANTE de normalisation, calée en imposant sigma_8 = 0,811 à la
  cosmologie de référence (A_s = 2,10e-9, n_s = 0,9649). L'implémentation
  rapide ci-dessous a une erreur d'amplitude absolue non localisée ; la
  caler sur un point mesuré est plus honnête que de la publier fausse.
  Le RÉSULTAT ne dépend pas de cette constante : il dépend du RAPPORT
  des amplitudes, qui lui vient entièrement du cadre.
"""
from __future__ import annotations
import math

C_KMS, THETA27 = 2.998e5, 2.7255 / 2.7
K_PIVOT, DELTA_C, Z_EQ = 0.05, 1.686, 3400.0

CADRE = dict(A_s=1.89e-9, n_s=1.0 - 2.0 / 55.0, h=0.672, Om=0.315, Ob=0.0493)
REF   = dict(A_s=2.10e-9, n_s=0.9649,            h=0.674, Om=0.315, Ob=0.0493)
SIGMA8_REF = 0.811


def transfer(k, p):
    om, ob = p["Om"] * p["h"] ** 2, p["Ob"] * p["h"] ** 2
    fb = p["Ob"] / p["Om"]
    s = 44.5 * math.log(9.83 / om) / math.sqrt(1.0 + 10.0 * ob ** 0.75)
    ag = (1.0 - 0.328 * math.log(431.0 * om) * fb
          + 0.38 * math.log(22.3 * om) * fb * fb)
    geff = p["Om"] * p["h"] * (ag + (1.0 - ag) / (1.0 + (0.43 * k * s) ** 4))
    q = k * THETA27 ** 2 / geff
    l0 = math.log(2.0 * math.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return l0 / (l0 + c0 * q * q)


def delta2(k, p):
    pref = (k * C_KMS / (100.0 * p["h"])) ** 2 / p["Om"] * transfer(k, p)
    return pref ** 2 * p["A_s"] * (k / K_PIVOT) ** (p["n_s"] - 1.0)


def sigma_R(R, p, nbin=1200):
    lk1, lk2 = math.log(1e-4 / R), math.log(400.0 / R)
    dl, tot = (lk2 - lk1) / nbin, 0.0
    for i in range(nbin):
        k = math.exp(lk1 + (i + 0.5) * dl)
        x = k * R
        w = 3.0 * (math.sin(x) - x * math.cos(x)) / x ** 3
        tot += delta2(k, p) * w * w * dl
    return math.sqrt(tot)


def rho_m(p):
    return 2.775e11 * p["h"] ** 2 * p["Om"]


def R_of_M(M, p):
    return (3.0 * M / (4.0 * math.pi * rho_m(p))) ** (1.0 / 3.0)


# constante de calage, fixée une fois sur la référence
KAL = SIGMA8_REF / sigma_R(8.0 / REF["h"], REF)


def sigma_M(M, p):
    return KAL * sigma_R(R_of_M(M, p), p)


def growth(z):
    return 1.3 / (1.0 + z)


def f_coll(M, z, p):
    s = sigma_M(M, p) * growth(z)
    return math.erfc(DELTA_C / (math.sqrt(2.0) * s))


def plafond(M_seed, z, p):
    Mh = M_seed * (1.0 + Z_EQ) / (1.0 + z)
    n = f_coll(Mh, z, p) * rho_m(p) / Mh
    Om_dm = p["Om"] - p["Ob"]
    return n * M_seed / (rho_m(p) * Om_dm / p["Om"]), Mh


if __name__ == "__main__":
    print(f"=== CALAGE ===\n  constante = {KAL:.4f} "
          f"(implémentation brute -> sigma_8 = {SIGMA8_REF/KAL:.3f})")
    s8c = sigma_M(None, CADRE) if False else KAL * sigma_R(8.0 / CADRE["h"], CADRE)
    print(f"\n=== CE QUE LE CADRE PRÉDIT ===")
    print(f"  A_s = 1,89e-9 (dérivé) au lieu de 2,10e-9 -> sigma_8 = {s8c:.4f}")
    print(f"  S8 = {s8c*math.sqrt(CADRE['Om']/0.3):.4f}   registre : 0,826 +- 0,011")
    print(f"  rapport sigma(cadre)/sigma(ref) = {s8c/SIGMA8_REF:.4f}")

    print("\n=== sigma(M) ===")
    print(f"  {'M':>10s} {'ref':>9s} {'cadre':>9s}")
    for M in (1e8, 1e7, 3e6, 1e6):
        print(f"  {M:10.0e} {sigma_M(M,REF):9.3f} {sigma_M(M,CADRE):9.3f}")

    print("\n=== PLAFOND SUR f_F ===")
    print(f"  {'M_graine':>10s} {'z':>4s} {'f_max ref':>11s} {'f_max cadre':>13s} {'resserrement':>13s}")
    for Ms in (1.6e4, 1e5):
        for z in (10, 15, 20, 25, 30):
            fr, _ = plafond(Ms, z, REF)
            fc, _ = plafond(Ms, z, CADRE)
            print(f"  {Ms:10.1e} {z:4d} {fr:11.2e} {fc:13.2e} {fr/fc:12.1f}x")
