r"""
Rapport de Koide dépendant du secteur, via le niveau de Chern-Simons k.

Le facteur 2/3 vient de l'amplitude A = d_{1/2}(k=2) = √2 (derive_koide_23.py).
Si CHAQUE secteur vit à un niveau CS k différent, son rapport de Koide est
    Q(k) = (1 + A²/2)/3 = (2 + cos(2π/(k+2)))/3,   A = d_{1/2}(k) = 2cos(π/(k+2)).
On teste où tombent leptons / up / down.

RÉSULTAT :
  - leptons (e,μ,τ)  : Q=0,6667 -> k = 2,00  EXACT (niveau Ising/fort, √2).
  - up (u,c,t)       : Q=0,849  -> k = 4,33 ; proche k=4 (Q=0,833), écart 1,8 %.
  - down (d,s,b)     : Q=0,731  -> k = 2,57 ; entre k=2 et k=3 (moins propre).

LECTURE : les leptons sont EXACTEMENT à k=2. Les up-quarks tombent près de k=4
(bien mieux que k=2 qui donnait 27 %). Le rapport de Koide est donc plausiblement
SECTORIEL (niveau CS par secteur), les leptons étant le cas propre (k=2). Réserve :
pas d'entier propre pour les down (k=2,57) ; suggestif, non dérivé. La sensibilité
du secteur up au m_t (173 GeV) rend son k incertain.
"""
from __future__ import annotations
import numpy as np


def koide(masses):
    m = np.array(masses, float)
    return float(m.sum() / (np.sqrt(m).sum() ** 2))


def Q_of_k(k):
    """Rapport de Koide au niveau CS k : Q = (2 + cos(2π/(k+2)))/3."""
    return (2 + np.cos(2 * np.pi / (k + 2))) / 3


def k_of_Q(Q):
    """Niveau effectif : cos(2π/(k+2)) = 3Q−2."""
    c = 3 * Q - 2
    if abs(c) > 1:
        return None
    return 2 * np.pi / np.arccos(c) - 2


SECTORS = {
    "leptons (e,μ,τ)": [0.51099895, 105.6583755, 1776.86],
    "up (u,c,t)": [2.16, 1270.0, 172760.0],
    "down (d,s,b)": [4.67, 93.4, 4180.0],
}


def main():
    print("Q(k) = (2 + cos(2π/(k+2)))/3   [rapport de Koide au niveau CS k]\n")
    print("Niveaux entiers :")
    for k in (0, 1, 2, 3, 4, 5, 135):
        print(f"  k={k:3d} : Q = {Q_of_k(k):.4f}")
    print("\nSecteurs mesurés → niveau effectif :")
    for nm, m in SECTORS.items():
        Q = koide(m)
        k = k_of_Q(Q)
        kn = round(k)
        tag = " EXACT" if abs(k - round(k)) < 0.05 else \
            f"  (proche k={kn}: Q={Q_of_k(kn):.4f}, écart {abs(Q - Q_of_k(kn))/Q*100:.1f} %)"
        print(f"  {nm:18s} Q={Q:.4f} → k={k:.2f}{tag}")
    print("\n  Leptons EXACTEMENT k=2 ; up ≈ k=4 (1,8 %) ; down k≈2,6 (moins propre).")
    print("  Rapport de Koide plausiblement SECTORIEL (niveau CS par secteur).")


if __name__ == "__main__":
    main()
