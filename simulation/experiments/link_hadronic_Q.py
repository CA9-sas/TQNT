r"""
Le lien « énergie de saveur hadronique ↔ Koide Q » : test honnête (RÉFUTÉ direct).

On avait noté E_Z₃^had ≈ (2/3)Λ_χ (décuplet 147 vs 144 MeV). Le 2/3 y est-il
la valeur de Koide Q ? Test : les baryons suivent-ils vraiment Q=2/3 ?

Formule Z₃ universelle (derive_koide_23.py) : Q = (1 + A²/2)/3, donc
A = √(2(3Q−1)). A=√2 ⇒ Q=2/3 (grand écart de masses) ; A=0 ⇒ Q=1/3 (masses égales).

RÉSULTAT :
  - triplets d'étrangeté BARYONIQUES (N,Λ,Ξ), (N,Σ,Ξ), décuplet : Q ≈ 1/3
    (A ≈ 0,06–0,10, minuscule). L'étrangeté est une PETITE perturbation sur la
    base topologique (~939 MeV), PAS la structure Koide 2/3.
  - triplet de GÉNÉRATION quark (d,s,b) : Q = 0,73 (A ≈ √2). C'EST le Koide 2/3
    (approximatif). Le s-quark vit dans un triplet ~Koide.

VERDICT : le lien DIRECT E_Z₃ = Q_baryon · Λ_χ est FAUX — Q_baryon ≈ 1/3, pas 2/3.
Le match (2/3)Λ_χ ≈ 147 était COÏNCIDENTIEL. Le vrai 2/3 gouverne le triplet de
GÉNÉRATION (d,s,b), d'où hérite m_s ; la masse baryonique = base topologique
(dominante) + petite perturbation de saveur (A-petit) portée par m_s.

Statut : la saveur LEPTONIQUE (et quark-génération) est Koide 2/3 (dérivé/approx) ;
l'incrément baryonique n'est PAS directement (2/3)Λ_χ — il hérite de m_s. Le lien
énergie↔Q demandé N'EST PAS un lien direct.
"""
from __future__ import annotations
import numpy as np


def koide(masses):
    m = np.array(masses, float)
    return float(m.sum() / (np.sqrt(m).sum() ** 2))


def amplitude_from_Q(Q):
    """A tel que Q=(1+A²/2)/3 ⇒ A=√(2(3Q−1))."""
    return np.sqrt(max(2 * (3 * Q - 1), 0.0))


BARYON_TRIPLETS = {
    "octet (N,Λ,Ξ)": [938.9, 1115.7, 1318.3],
    "octet (N,Σ,Ξ)": [938.9, 1193.2, 1318.3],
    "décuplet (Δ,Σ*,Ξ*)": [1232.0, 1385.0, 1533.0],
    "décuplet (Δ,Ξ*,Ω)": [1232.0, 1533.0, 1672.0],
}
GENERATION_TRIPLETS = {
    "leptons (e,μ,τ)": [0.51099895, 105.6583755, 1776.86],
    "quarks down (d,s,b)": [4.67, 93.4, 4180.0],
    "quarks up (u,c,t)": [2.16, 1270.0, 172760.0],
}
LAMBDA_CHI = 216.0


def main():
    print("Formule Z₃ : Q=(1+A²/2)/3 ; A=√2→Q=2/3 (grand écart) ; A=0→Q=1/3 (égales)\n")
    print("=== Baryons : suivent-ils Koide 2/3 ? (triplets d'étrangeté) ===")
    for nm, m in BARYON_TRIPLETS.items():
        Q = koide(m)
        print(f"  {nm:22s} Q={Q:.4f}  A={amplitude_from_Q(Q):.3f}")
    print("  → Q≈1/3, A≈0.1 : étrangeté = PETITE perturbation, PAS Koide 2/3.")

    print("\n=== Triplets de GÉNÉRATION : le vrai 2/3 ===")
    for nm, m in GENERATION_TRIPLETS.items():
        Q = koide(m)
        print(f"  {nm:22s} Q={Q:.4f}  A={amplitude_from_Q(Q):.3f}")
    print("  → leptons exact 2/3 ; down 0.73 (~2/3). Le s-quark est dans un triplet ~Koide.")

    print("\n=== Verdict ===")
    dec = list(BARYON_TRIPLETS["décuplet (Δ,Σ*,Ξ*)"])
    inc = (dec[1] - dec[0] + dec[2] - dec[1]) / 2
    print(f"  incrément décuplet {inc:.0f} MeV vs (2/3)Λ_χ={2/3*LAMBDA_CHI:.0f} : match numérique")
    print("  MAIS Q_baryon≈1/3 ≠ 2/3 → le 2/3 de (2/3)Λ_χ n'est PAS le Koide baryonique.")
    print("  Lien DIRECT énergie↔Q RÉFUTÉ. Le 2/3 réel = triplet génération (d,s,b),")
    print("  d'où hérite m_s ; la base topologique domine la masse baryonique.")


if __name__ == "__main__":
    main()
