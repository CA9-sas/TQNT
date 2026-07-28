r"""Première évaluation de l'amplitude de reconnexion ΔLk±1 (champ χ).

Voir docs/brouillon_amplitude_reconnexion_chi.tex §Première évaluation. Résultats :
(i) PERTURBATIF, pas instanton : S_requis=-ln(G_F Λ²)=14.5 ≈ 3 ln(1/α)=3 ln(k+2)=14.8
    → suppression = α³=(1/(k+2))³ = 3e ordre perturbatif (instanton de tension→exp(-4), trop grand)
(ii) le '3' = writhe du trèfle-lepton 3_1=T(2,3) (self-linking 3) ; α par croisement
     → 𝒜~α^writhe=α³, universel (e,μ,τ tous trèfles). EM writhe 0→α~1 ; fort writhe 0→α⁰
(iii) coef c=κ_1³×phase-space, κ_1≈1.09 (O(1)/croisement) ; 4/3 vs 5π/12 non discriminable
"""
import numpy as np

K = 135
ALPHA = 1 / (K + 2)
GF = 1.1663787e-5
LAM = np.sqrt(1.096 / (2 * np.pi)) / 2


def required_action():
    return -np.log(GF * LAM ** 2)


def perturbative_action():
    return 3 * np.log(1 / ALPHA)          # 3 × ln(k+2), 3 = writhe du trèfle


def tension_instanton_action():
    sig = 4 * LAM ** 2
    return sig / LAM ** 2                  # ~4 → exp(-4)~1e-2, trop grand


def vertex_normalization():
    c = GF * LAM ** 2 / ALPHA ** 3
    return c ** (1 / 3)                     # κ_1 par croisement


if __name__ == '__main__':
    print(f"S requis = {required_action():.1f} ; 3 ln(1/α) = {perturbative_action():.1f} (writhe trèfle=3)")
    print(f"instanton tension = {tension_instanton_action():.1f} → exp(-.) trop grand : EXCLU")
    print(f"→ α³ PERTURBATIF (3e ordre), pas instanton. κ_1 = {vertex_normalization():.3f} (O(1)/croisement)")
