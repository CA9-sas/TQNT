r"""
L'équation de la masse : m = √σ_χ × S(catégorie).

Réponse consolidée à « quelle est l'équation de la masse ? ». Il n'y a PAS une
formule universelle (le fit universel plafonne à 34%, mélange des physiques). Il y
a une FORME UNIQUE : m = √σ_χ · S, une seule échelle √σ_χ (l'unique nombre
dimensionnel) fois un facteur topologique SANS DIMENSION S propre à la catégorie
(nombre de brins) :
  BASE (p=1, γ,ν)      : S = 0 (pas de nœud → pas de masse)
  LEPTONS (p=2)        : S_k = (3/4)(1+√2 cos(2/9+2πk/3))²  [Koide]
  HADRONS (≥2 confinés): S = √(2π(J−α_0))  [Regge] (+ saveur Z₃)
Vérif : leptons −0,2% ; proton 938 et ρ 775 MeV exacts ; base 0.
La forme est UNE ; le contenu S est la topologie. Statut de S : voir
brouillon_masse_absolue.tex (dérivé A=√2 ; suggestif 3/4=1/2Q, 2/9 ; physique α_0).
"""
from __future__ import annotations
import numpy as np

RS = np.sqrt(1.096 / (2 * np.pi)) * 1000   # √σ_χ, MeV


def S_lepton(k):
    return 0.75 * (1 + np.sqrt(2) * np.cos(2 / 9 + 2 * np.pi * k / 3)) ** 2


def S_hadron(J, alpha0):
    return np.sqrt(2 * np.pi * (J - alpha0))


def main():
    print(f"m = √σ_χ · S ;  √σ_χ = {RS:.1f} MeV\n")
    print("LEPTONS S_k=(3/4)(1+√2 cos(2/9+2πk/3))² :")
    for nm, S, e in zip("eμτ", np.sort([S_lepton(k) for k in range(3)]),
                        [0.511, 105.66, 1776.86]):
        print(f"  m_{nm} = {RS * S:8.2f}  (exp {e:8.2f}, {(RS*S/e-1)*100:+.1f}%)")
    print("\nHADRONS S=√(2π(J−α_0)) :")
    for nm, J, a, e in [("proton", 0.5, -0.303, 938.3), ("ρ", 1, 0.452, 775.3)]:
        print(f"  {nm:6s} = {RS * S_hadron(J, a):.0f}  (exp {e:.0f})")
    print("\nBASE : S=0 → m=0.")


if __name__ == "__main__":
    main()
