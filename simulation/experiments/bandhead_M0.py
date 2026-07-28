r"""
Nature de la tête de bande M₀ (masse au repos, J minimal) — caractérisation honnête.

M₀ = énergie au repos de la configuration χ nouée = valeur propre fondamentale E₀
du puits χ (opérateur de Schrödinger, potentiel de Jones). C'est une quantité
SPECTRALE, calculable NUMÉRIQUEMENT (solveur FEM), corr(E₀, complexité) = -0,99
(mass_chi_well_eigenvalue.py).

Est-elle réductible à une forme close ? NON. Sur le nucléon (939 MeV) :
  m_N/Λ_χ = 4,35 ; m_N/√σ_χ = 2,21 ; m_N²/σ_χ = 4,90 — des O(quelques) non 'propres'.

PARALLÈLE QCD (décisif) : la masse du nucléon en QCD est calculée par le RÉSEAU à
~1 % mais N'EST PAS dérivable analytiquement de Λ_QCD (m_N/√σ ≈ 2,1, m_N/Λ_QCD ≈ 3
sont des nombres du réseau). C'est LE problème de la masse hadronique au repos —
non-perturbatif, sans formule fermée. En TQNT, M₀ = E₀ a EXACTEMENT ce statut.

CONCLUSION : M₀ est DÉRIVÉE au sens d'une quantité spectrale bien définie (E₀),
calculable numériquement, échelle fixée par Λ_χ, ordre reproduit (corr -0,99).
Elle N'EST PAS dérivable en forme close : c'est un nombre non-perturbatif (comme
m_N au réseau QCD). Le O(1) (m_N ≈ 4,3 Λ_χ) n'est pas réductible.

BILAN de l'équation de masse : deux entrées NON-PERTURBATIVES à l'échelle
hadronique — σ_χ (pente de Regge) et M₀ (tête de bande) — toutes deux fixées par
Λ_χ et calculables numériquement, pas en forme close ; le RESTE est dérivé
(structure de Regge M²∝J kinématique, Koide 2/3, incrément Z₃ de saveur).
C'est la MÊME structure que la QCD : un seul paramètre d'échelle (Λ_χ), le
spectre étant non-perturbatif.
"""
from __future__ import annotations
import numpy as np

LAMBDA_CHI, SIGMA_CHI, M_N = 0.216, 0.18, 0.9389  # GeV


def ratios():
    return {
        "m_N/Λ_χ": M_N / LAMBDA_CHI,
        "m_N/√σ_χ": M_N / np.sqrt(SIGMA_CHI),
        "m_N²/σ_χ": M_N ** 2 / SIGMA_CHI,
    }


def main():
    print("M₀ = E₀ (valeur propre du puits χ) — SPECTRAL, numérique, corr(E₀,cr)=-0,99.\n")
    print("Réductible à une forme close ? Test nucléon (939 MeV) :")
    for k, v in ratios().items():
        print(f"  {k:10s} = {v:.2f}")
    print("  → O(quelques), non 'propres'. Comme QCD : m_N/√σ≈2,1, m_N/Λ_QCD≈3 (réseau).\n")
    print("PARALLÈLE QCD : m_N calculée au RÉSEAU (~1 %), JAMAIS dérivée analytiquement")
    print("de Λ_QCD. M₀=E₀ en TQNT a ce statut : non-perturbatif, sans forme close.\n")
    print("CONCLUSION : M₀ dérivée comme quantité spectrale (E₀) calculable, échelle Λ_χ,")
    print("ordre reproduit ; PAS en forme close (nombre non-perturbatif, comme m_N réseau).")
    print("Équation de masse = structure QCD : un paramètre d'échelle Λ_χ ; σ_χ et M₀")
    print("non-perturbatifs ; Regge M²∝J + Koide 2/3 dérivés.")


if __name__ == "__main__":
    main()
