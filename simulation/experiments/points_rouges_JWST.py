"""Petits points rouges du JWST : le coeur sombre transparent (Classe F).

Teste les deux candidats du « milieu » entre étoile à trou noir et étoile
à matière noire : (1) le cusp d'éremnions contracté adiabatiquement,
(2) un défaut de Classe F comme masse centrale transparente aux baryons.
"""
from __future__ import annotations
import math

Msun, G, c, mp, AU, pc, Rsun = 1.989e30, 6.6710e-11, 2.998e8, 1.673e-27, 1.496e11, 3.086e16, 6.96e8
SIG_T, ERG, GeV, KB = 6.652e-29, 1e-7, 1.7827e-27, 1.381e-23
M_ER = 1.8 * GeV
SIGMA_ER_H = 1e-48 * 1e-4          # m^2, éremnion-nucléon
Q_GEL = 5.156e-17                  # densité d'espace des phases au gel (kg m^-6 s^3)


def L_annihilation(M_dm, R, sv_cm3s):
    """Luminosité d'annihilation d'un coeur sombre uniforme (W)."""
    n = M_dm / ((4 * math.pi / 3) * R ** 3) / M_ER
    return n ** 2 * (sv_cm3s * 1e-6) * M_ER * c ** 2 * (4 * math.pi / 3) * R ** 3


def contraction_adiabatique(M_b, R_env, rho_s=1e-19, rs=100 * pc):
    """Matière noire ramenée sous R_env par condensation de M_b (invariant r M)."""
    M_i = lambda r: 2 * math.pi * rho_s * rs * r ** 2
    lo, hi = 1e10, 1e20
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if mid * M_i(mid) < R_env * (M_b + M_i(mid)):
            lo = mid
        else:
            hi = mid
    return M_i(lo)


def transparence(M_F, R_F):
    """Profondeur optique du gaz à travers un défaut F uniforme."""
    n = M_F / ((4 * math.pi / 3) * R_F ** 3) / M_ER
    return n * 2 * R_F * SIGMA_ER_H


def L_eddington(M):
    return 4 * math.pi * G * M * mp * c / SIG_T


def t_kelvin_helmholtz(M_tot, M_env, R, L):
    return G * M_tot * M_env / (R * L)


def R_min_liouville(M):
    return (3 / (4 * math.pi * Q_GEL) * M ** -0.5 * G ** -1.5) ** (2 / 3)


if __name__ == "__main__":
    print("annihilation, 1e5 Msun dans 100 UA :")
    for lab, sv in (("WIMP", 3e-26), ("alpha^5", 3e-26 * 137 ** -5), ("alpha^22", 3e-26 * 137 ** -22)):
        print(f"  {lab:9s} L = {L_annihilation(1e5*Msun, 100*AU, sv)/ERG:.1e} erg/s")
    print("contraction adiabatique, 1e6 Msun de gaz :")
    for R in (100, 1000, 10000):
        print(f"  R_env={R:6d} UA : MN = {contraction_adiabatique(1e6*Msun, R*AU)/Msun:.1e} Msun")
    print("transparence du défaut F :")
    for M, R in ((1.6e4, Rsun), (1e5, Rsun), (1e5, 10 * Rsun)):
        print(f"  {M:.1e} Msun, {R/Rsun:4.0f} Rsun : tau = {transparence(M*Msun, R):.1e}")
    for M_env in (1e5, 1e6):
        Mt = (M_env + 1e5) * Msun
        L = L_eddington(Mt)
        print(f"  enveloppe {M_env:.0e} : L_Edd = {L/ERG:.1e} erg/s, "
              f"t_KH(1000 UA) = {t_kelvin_helmholtz(Mt, M_env*Msun, 1000*AU, L)/3.156e7:.0e} ans")
    print(f"R_min Liouville pour 1e5 Msun : {R_min_liouville(1e5*Msun)/Rsun:.1f} Rsun")
