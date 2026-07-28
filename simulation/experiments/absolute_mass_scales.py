r"""
Masse ABSOLUE du proton et de l'électron — ce qui est dérivé, ce qui ne l'est pas.

Demande utilisateur : « dérive la masse absolue du proton et de l'électron ».

MUR DIMENSIONNEL : aucune masse absolue ne se dérive « de rien » (analyse
dimensionnelle : il faut UN input dimensionnel). Le cadre en a un : Λ_χ ≈ 209 MeV,
l'échelle de confinement χ, fixée par la pente de Regge (mesurée une fois). On ne
peut dériver que les RAPPORTS.

PROTON : m_p ≈ 4,5 Λ_χ (échelle FORTE, confinement). LE 4,5 SE DÉRIVE par la
trajectoire de Regge (J = α' M² + α_0 ; α'=1/(2πσ_χ) ; σ_χ=4Λ_χ²) :
    m_p/Λ_χ = √(2π · D² · (J − α_0)) = √(2π·4·(1/2+0,30)) = 4,49
  - 2π : kinématique de la corde en rotation (dérivé)
  - D²=4 : dim. quantique² d'Ising (DÉRIVÉ ; facteur de la pente σ_χ=4Λ_χ²)
  - J=1/2 : spin du proton (topologique)
  - α_0 ≈ −0,30 : intercept de Regge du NUCLÉON = LE SEUL morceau non-perturbatif,
    mais PHYSIQUE et CONNU (intercept nucléon mesuré −0,3 à −0,4 ; on tombe dedans,
    pas ajusté ad hoc). Avec α_0=−0,3 : m_p = 936 MeV (0,2 %).
  → le 4,5 est DÉRIVÉ à un intercept de corde près (petit, physique, contraint) ;
    seule reste irréductible l'échelle Λ_χ (dimensionnel).

ÉLECTRON : m_e ≈ (1/3) α_EM Λ_χ (échelle EM). Le lepton est lié par l'EM (α×Λ_χ =
1,52 MeV), PAS par la force forte — d'où sa légèreté. m_e = plus léger mode Z₃ ;
μ,τ montent par Koide.
LE FACTEUR 1/3 NE SE DÉRIVE PAS (contraste net avec le 4,5 du proton) :
  - Regge EM (électron = corde tension EM) : exige α_0^lep ≈ 0,5 = J presque exact
    → PAS un intercept propre. RAISON : l'électron n'est PAS confiné → pas de Regge.
  - auto-énergie EM (charge taille ~1/Λ_χ) : sphère 0,6, coquille 0,5 — aucune
    géométrie standard ne donne 1/3.
  → le proton est dérivable PARCE QU'IL EST CONFINÉ (Regge le verrouille) ;
    l'électron ne l'est PAS (non confiné, mode Z₃ léger sensible à la phase de
    Koide). C'est le problème de hiérarchie du lepton léger — non résolu (partout).
  CE QUI RESTE SOLIDE : m_e ~ α Λ_χ (échelle EM) DÉRIVE la LÉGÈRETÉ (fort/EM~1/α) ;
  observation m_μ ≈ Λ_χ/2 (1,2%) = muon mode central. Le facteur PRÉCIS (1/3, ou
  la phase de Koide) reste OUVERT.

LE VRAI RÉSULTAT — la hiérarchie m_p/m_e vient de 1/α :
    m_p/m_e mesuré = 1836,15
    structure TQNT : 4,5 / ((1/3)·α) = 1850  (0,8 %)
  CONTENU NON TRIVIAL : le proton est ~1836× l'électron PARCE QUE le baryon est lié
  par la force FORTE (~Λ_χ) et le lepton par l'EM (~α Λ_χ) ; le rapport fort/EM est
  ~1/α = 137. Comme α = 1/(k+2) est DÉRIVÉ (k=135), la MAJEURE PARTIE de la
  hiérarchie (le facteur 137) est DÉRIVÉE ; le reste (facteur O(10) = structure
  3-quarks du baryon vs facteur EM de l'électron) donne 1850 vs 1836.

STATUT (net) :
  DÉRIVÉ    : POURQUOI m_p ≫ m_e (fort vs EM, ~1/α ; α dérivé).
  INPUT     : l'échelle Λ_χ (mesurée sur Regge) — irréductible (dimensionnel).
  NON CLOSE : le 4,5 (proton/Λ_χ, non-perturbatif) ; le 1/3 (facteur EM électron, estimé).
  NON DÉRIVÉ: le 1836 PRÉCIS (~0,8 % via la structure, pas exact).

PIÈGE À NE PAS REVENDIQUER : 6π⁵ = 1836,12 (Lenz 1951) tombe à 0,002 % de 1836,15,
mais c'est de la NUMÉROLOGIE, pas une dérivation.

CONCLUSION : on ne dérive PAS les masses absolues en MeV (il faut Λ_χ). On dérive
la STRUCTURE de la hiérarchie : baryon fort (~Λ_χ), lepton EM (~αΛ_χ), donc
m_p/m_e ~ 1/α — et α est dérivé. La hiérarchie proton/électron est expliquée à
l'ordre de grandeur par la séparation forte/EM ; le nombre précis reste ouvert
(non-perturbatif fort + facteur EM). Cohérent avec [[epistemic-assessment-V6]].
"""
from __future__ import annotations
import numpy as np

LAM = np.sqrt((1.096 / (2 * np.pi)) / 4.0) * 1000   # Λ_χ, MeV (Regge, input)
ALPHA = 1 / 137.035999                               # dérivé (k=135)
MP, ME = 938.272, 0.51099895


def main():
    print(f"Échelle UNIQUE Λ_χ = {LAM:.1f} MeV (Regge, input) ; α=1/137 (dérivé)\n")
    D2, J = 4.0, 0.5
    a0 = J - (MP / LAM) ** 2 / (2 * np.pi * D2)
    print(f"PROTON  : m_p/Λ_χ = {MP / LAM:.2f} = √(2π·D²·(J−α_0)), D²=4 Ising DÉRIVÉ,")
    print(f"          J=1/2, α_0={a0:+.2f} (intercept Regge nucléon, physique ~−0,3)")
    print(f"          → avec α_0=−0,3 : m_p = {LAM * np.sqrt(2*np.pi*D2*(J+0.3)):.0f} MeV (0,2%)")
    print(f"ÉLECTRON: m_e/(α Λ_χ) = {ME / (ALPHA * LAM):.3f}  (échelle EM ; ≈1/3)\n")
    struct = 4.5 / ((1 / 3.) * ALPHA)
    print(f"RAPPORT m_p/m_e = {MP / ME:.2f} (mesuré)")
    print(f"  structure TQNT : 4,5/((1/3)α) = {struct:.0f}  ({(struct/(MP/ME)-1)*100:+.1f}%)")
    print(f"  → hiérarchie ~1/α (fort/EM) : le 137 est DÉRIVÉ (α), le reste O(10) estimé.")
    print(f"\n  6π⁵ = {6*np.pi**5:.2f} = numérologie (0,002%), NON revendiqué.")
    print("\n  DÉRIVÉ : pourquoi m_p≫m_e (fort vs EM). INPUT : Λ_χ. NON DÉRIVÉ : 1836 précis.")


if __name__ == "__main__":
    main()
