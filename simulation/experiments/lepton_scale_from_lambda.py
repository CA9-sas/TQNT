r"""
L'échelle leptonique M ≈ (3/2)Λ_χ = Λ_χ/Q — les 3 masses leptoniques depuis Λ_χ seul.

Demande utilisateur : « justifie topologiquement l'échelle M ≈ 1,5 Λ_χ ».

M (échelle Koide leptonique, √M = moyenne des √m) = 313,8 MeV ; M/Λ_χ = 1,503.
Écritures équivalentes de 3/2 (car Q=2/3, A²=2) :
    M = Λ_χ/Q = (3/2)Λ_χ = 3Λ_χ/A²   ;   équivalent Σm_lepton = 9Λ_χ = Z₃²·Λ_χ.

RÉSULTAT MAJEUR — les 3 masses leptoniques ABSOLUES depuis Λ_χ SEUL (+ Ising) :
  M=(3/2)Λ_χ, θ=2/9, A=√2, 0 paramètre libre (Λ_χ = unique input dimensionnel) :
    m_e = 0,510 vs 0,511 (-0,20%) ; m_μ = 105,45 vs 105,66 (-0,20%) ;
    m_τ = 1773,5 vs 1776,9 (-0,19%). RMS 0,19%.
  → tout le spectre leptonique (3 masses en MeV) sort de Λ_χ + quantités d'Ising.

STATUT (honnête) :
  - Le décalage −0,2% est SYSTÉMATIQUE (les 3 masses basses) : M/Λ_χ=1,503, pas
    exactement 3/2=1,500. Compatible avec M=(3/2)Λ_χ à l'incertitude de Λ_χ près
    (Λ_χ vient de la pente de Regge, ~0,2% d'incertitude).
  - M=Λ_χ/Q : QUE des quantités dérivées (Q d'Ising), tight — SUGGESTIF. MAIS
    aucun argument ne FORCE le 1/Q (même statut que θ=2/9=Q/3, [[koide]]).
  - TENSION physique : Λ_χ est l'échelle FORTE (pente de Regge hadronique) ; le
    lepton est EM. Pourquoi M_lepton = Λ_χ_fort/Q ? Seulement si Λ_χ est l'échelle
    χ UNIVERSELLE (un seul champ χ, secteurs fort ET lepton en dérivant) — c'est
    le postulat du cadre, pas une dérivation.

SYNTHÈSE (le tableau complet après cette série) : UNE seule échelle dimensionnelle
irréductible, Λ_χ (pente de Regge). Tout le reste = quantités d'Ising/topologiques :
  proton : m_p = √(2π·D²·(J-α_0)) Λ_χ = 4,5 Λ_χ (Regge, α_0 physique) → 936 MeV.
  leptons: M = Λ_χ/Q, spectre par θ=2/9 + A=√2 → 3 masses à 0,2%.
Donc le spectre proton+leptons sort de Λ_χ + Ising à ~0,2-0,5%. Le RESTE OUVERT :
la justification rigoureuse des facteurs (α_0, 1/Q, 2/9) — tous 'suggestifs, que
des quantités dérivées, mais non forcés'. L'échelle Λ_χ elle-même est l'unique
input dimensionnel incompressible (comme dans toute théorie).
"""
from __future__ import annotations
import numpy as np

LAM = np.sqrt((1.096 / (2 * np.pi)) / 4.0) * 1000   # Λ_χ, MeV
A, Q = np.sqrt(2), 2 / 3
THETA = Q / 3
EXP = np.sort([0.51099895, 105.6583755, 1776.86])


def main():
    M = 1.5 * LAM
    print(f"Λ_χ={LAM:.1f} ; M=(3/2)Λ_χ={M:.1f} MeV = Λ_χ/Q  (M_obs/Λ_χ=1,503)\n")
    print(f"Σm_lepton = {EXP.sum():.1f} vs 9Λ_χ = {9*LAM:.1f} MeV "
          f"({(EXP.sum()/(9*LAM)-1)*100:+.2f}%)  [= Z₃²·Λ_χ]\n")
    sM = np.sqrt(M)
    pred = np.sort([(sM * (1 + A * np.cos(THETA + 2 * np.pi * k / 3))) ** 2 for k in range(3)])
    print("Les 3 masses leptoniques ABSOLUES depuis Λ_χ seul (+ Ising, 0 param) :")
    for nm, p, e in zip(["e", "μ", "τ"], pred, EXP):
        print(f"  m_{nm} = {p:9.3f} MeV  vs {e:9.3f}  ({(p/e-1)*100:+.2f}%)")
    rms = np.sqrt(np.mean(((pred - EXP) / EXP * 100) ** 2))
    print(f"  RMS = {rms:.2f}%\n")
    print("Statut : M=Λ_χ/Q suggestif (quantités dérivées, 0,2%) mais 1/Q non forcé ;")
    print("  tension Λ_χ fort vs lepton EM (OK si Λ_χ = échelle χ universelle). OUVERT.")


if __name__ == "__main__":
    main()
