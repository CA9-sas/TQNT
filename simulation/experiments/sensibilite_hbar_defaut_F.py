"""Que change une variation de hbar au problème de constitution du défaut F ?
On multiplie hbar par un facteur k et on recalcule chaque brique du verdict."""
from __future__ import annotations
import math

G, c, kB, e_ch, eps0 = 6.6710e-11, 2.998e8, 1.381e-23, 1.602e-19, 8.854e-12
HBAR0, ME = 1.0546e-34, 9.109e-31
Msun, AU, pc, Rsun = 1.989e30, 1.496e11, 3.086e16, 6.96e8
GeV = 1.7827e-27
m_er, M_F, R_F = 1.8 * GeV, 3e5 * Msun, 20 * AU
Q_GEL = 5.156e-17
SIG_M0 = 3e-25            # cm2/g, éremnion


def briques(k):
    hb = k * HBAR0
    out = {}
    # (1) Liouville classique : Q = rho/sigma^3 — aucun hbar
    rho = M_F / (4 * math.pi / 3 * R_F ** 3); sv = math.sqrt(G * M_F / R_F)
    out["Q/Q_gel"] = (rho / sv ** 3) / Q_GEL
    # (2) borne de Pauli : g m^4 / h^3
    h = 2 * math.pi * hb
    out["Q/Q_Pauli"] = (rho / sv ** 3) / (2 * m_er ** 4 / h ** 3)
    # (3) longueur de de Broglie des éremnions dans l'objet
    out["lambda_dB / R_F"] = (hb / (m_er * sv)) / R_F
    # (4) énergie d'assemblage : purement gravitationnelle
    out["E_bind(20UA)/E_bind(190pc)"] = (190 * pc) / R_F
    # (5) dissipation : R_K ∝ hbar (rayon de Compton) -> sigma_pol ∝ R_K^2 ∝ hbar^2 ; requis 2,4e3 au viriel
    out["sigma/m (cm2/g)"] = SIG_M0 * k ** 2
    out["écart au requis"] = 2.4e3 / (SIG_M0 * k ** 2)
    # (6) ce que le même k fait aux atomes : Bohr ∝ hbar^2, alpha ∝ 1/hbar
    out["a_Bohr x"] = k ** 2
    out["alpha x"] = 1 / k
    return out


if __name__ == "__main__":
    print("=== LE DÉCRET ===")
    print("  Quatuor de la trame : h = la dent, invariant ABSOLU ; E_dent = h f_local varie,")
    print("  h jamais. Observations : dérive de alpha < 1e-17/an (horloges), Rydberg constant")
    print("  à tous les redshifts (quasars). Toute variation de hbar serait vue. On calcule quand même.\n")
    ks = (1e-3, 1e-1, 1, 1e1, 1e3, 1e7, 1e14)
    rows = briques(1).keys()
    print(f"  {'hbar x':>14s}" + "".join(f"{k:>10.0e}" for k in ks))
    for r in rows:
        print(f"  {r:>14s}" + "".join(f"{briques(k)[r]:>10.1e}" for k in ks))
    print("\n=== LECTURE ===")
    print("  Liouville (ligne 1) et l'énergie d'assemblage (ligne 4) sont STRICTEMENT constantes :")
    print("  l'obstacle est classique — conservation de l'énergie en dynamique sans collision —")
    print("  et aucune constante quantique n'y entre. La borne de Pauli (2) et de Broglie (3)")
    print("  bougent, mais l'objet n'est ni dégénéré ni ondulatoire : il faudrait hbar x 1e26")
    print("  pour que la longueur d'onde atteigne 20 UA. La dissipation (5) ne rejoint le requis")
    print("  qu'à hbar x 1e14 — où le rayon de Bohr vaut 1e28 fois le sien : plus d'atomes,")
    print("  plus d'hydrogène, plus de points rouges à expliquer.")
    print("\n  => Faire varier h ne touche pas l'obstacle ; il ne fait que détruire ce qu'on cherche à expliquer.")
