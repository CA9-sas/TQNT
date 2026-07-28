r"""Ancrage CS de l'amplitude de reconnexion ΔLk=±1 (mise en place du calcul).

Voir docs/brouillon_amplitude_reconnexion_chi.tex. Montre :
- lepton = trèfle 3_1 = fermeture de σ_1³ → 3 croisements (l'origine du '3')
- CS pure = phases (topologique unitaire) : R hors-diag ~ sin(π/(k+2)) ≈ πα, |.|=1
  → la magnitude α³ est DYNAMIQUE (tension χ + jauge), pas topologique
- cible G_F Λ_χ² = c·α³, c=1.31 ; candidats 4/3 (1/Casimir spin-1/2) ou 5π/12 (phase-space)
- objet à calculer : amplitude du vertex de reconnexion à UN croisement 𝒜_1 = κ·α
"""
import numpy as np

K = 135
ALPHA = 1 / (K + 2)
GF = 1.1663787e-5
LAM = np.sqrt(1.096 / (2 * np.pi)) / 2


def target_coefficient():
    return GF * LAM ** 2 / ALPHA ** 3


def R_offdiagonal():
    return np.sin(np.pi / (K + 2))       # ≈ πα ; c'est une PHASE en CS pur


def conformal_weight_half():
    return 0.75 / (K + 2)                # h_{1/2} = C_2/(k+2), C_2 = 3/4


if __name__ == '__main__':
    print(f"lepton = trèfle σ_1³ → 3 croisements")
    print(f"R hors-diag = sin(π/(k+2)) = {R_offdiagonal():.5f} ≈ πα = {np.pi*ALPHA:.5f} (phase)")
    print(f"h_(1/2) = (3/4)α = {conformal_weight_half():.5f}")
    c = target_coefficient()
    print(f"cible c = G_F Λ_χ²/α³ = {c:.4f}  (4/3={4/3:.4f}, 5π/12={5*np.pi/12:.4f})")
