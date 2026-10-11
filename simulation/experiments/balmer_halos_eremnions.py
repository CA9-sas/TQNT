"""Raies de Balmer : ce qui mime la chaleur (Stark, diffusion électronique,
gradient gravitationnel) et ce que les halos d'éremnions n'y font pas."""
from __future__ import annotations
import math

kB, mp, me, c, G, Msun = 1.381e-23, 1.673e-27, 9.109e-31, 2.998e8, 6.6710e-11, 1.989e30
LAM_HA, LAM_HB = 656.28, 486.13   # nm
_2SQL2 = 2.0 * math.sqrt(math.log(2.0))


def fwhm_thermique(T, lam=LAM_HA, m=mp):
    """Largeur Doppler à mi-hauteur (nm)."""
    return _2SQL2 * math.sqrt(2 * kB * T / m) / c * lam


def T_lue(fwhm_nm, lam=LAM_HA):
    """Température qu'on infère en lisant une largeur comme thermique (H)."""
    v = fwhm_nm / (_2SQL2 * lam) * c
    return v ** 2 * mp / (2 * kB)


def fwhm_stark_hbeta(n_e):
    """Stark linéaire, H-beta : ~0,5 nm à 1e14 cm^-3, loi en n_e^(2/3)."""
    return 0.5 * (n_e / 1e14) ** (2.0 / 3.0)


def z_grav(M, R):
    return G * M / (R * c ** 2)


if __name__ == "__main__":
    for T in (1e2, 1e4, 1e6):
        print(f"T={T:.0e} K : FWHM(Ha)={fwhm_thermique(T)*1e3:.0f} pm")
    for ne in (1e12, 1e14, 1e16):
        w = fwhm_stark_hbeta(ne)
        print(f"n_e={ne:.0e} : Stark {w*1e3:.0f} pm -> T lue {T_lue(w, LAM_HB):.1e} K")
    print(f"illusion diffusion électronique : m_p/m_e = {mp/me:.0f}")
    for nom, M, R in [("mini-halo 1e4 Msun 13 pc", 1e4 * Msun, 13 * 3.086e16),
                      ("perle 2e-3 Msun 21 km", 2.03e-3 * Msun, 21e3),
                      ("M_Ch 0,5 Msun 10 km", 0.5 * Msun, 1e4)]:
        print(f"{nom:28s} z_g = {z_grav(M, R):.2e}")
