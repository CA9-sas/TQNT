r"""
Dériver m_s depuis le triplet de génération (d,s,b) via Koide — test honnête.

Le strange = mode Z₃ gen-2 du secteur down-type. Le Koide (Q=2/3, A=√2 dérivé)
prédit une masse depuis les deux autres (puissance historique : m_τ depuis
m_e, m_μ à ~ppm). On l'applique à (d,s,b).

RÉSULTAT :
  - m_s prédit depuis (m_d, m_b) via Q=2/3 exact : ~205 MeV, vs réel 93,4 MeV
    → FACTEUR 2,2 d'écart. Le bon ORDRE DE GRANDEUR, pas la valeur précise.
  - Cause : le Koide down-quark vaut Q ≈ 0,73-0,75 à TOUTE échelle cohérente
    (2 GeV, M_Z, 1 TeV), soit ~10 % au-dessus de 2/3. Le running ne le ramène
    PAS à 2/3. Or le Koide est HYPERSENSIBLE : 10 % sur Q → facteur ~2 sur la
    masse prédite.

CONCLUSION : m_s n'est PAS dérivable précisément du triplet (d,s,b) — seulement
à un facteur ~2 (ordre de grandeur ~100-200 MeV, correct). Le Koide LEPTONIQUE
est exact (m_τ à ~ppm) ; le Koide QUARK est seulement approximatif (~10 % sur Q,
~facteur 2 sur les masses). Le mécanisme Z₃/Koide donne la bonne STRUCTURE
(3 générations, ordre, ordre de grandeur) mais pas les masses de quarks précises.
Cohérent avec « Koide-Z₃ = le seul indice sérieux, ~15-25 % ».
"""
from __future__ import annotations
import numpy as np


def koide(masses):
    m = np.array(masses, float)
    return float(m.sum() / (np.sqrt(m).sum() ** 2))


def predict_third(m_two):
    """3e masse via Q=2/3 exact (√m3 = 2S ± √(6S²−3P)). Renvoie (racine_basse, haute)."""
    a, b = sorted(m_two)
    S = np.sqrt(a) + np.sqrt(b)
    P = a + b
    disc = 6 * S ** 2 - 3 * P
    if disc < 0:
        return None
    return (2 * S - np.sqrt(disc)) ** 2, (2 * S + np.sqrt(disc)) ** 2


# masses down-type MS-bar à échelles COHÉRENTES
DSB_SCALES = {
    "PDG (échelles mixtes)": [4.67, 93.4, 4180.0],
    "à 2 GeV": [4.67, 93.4, 4900.0],
    "à M_Z (runnées)": [2.70, 53.8, 2860.0],
    "à 1 TeV": [2.6, 52.0, 2430.0],
}


def main():
    md, ms, mb = 4.67, 93.4, 4180.0
    print("=== Prédiction de Koide (Q=2/3, A=√2 dérivé) pour m_s ===")
    lo, hi = predict_third([md, mb])
    print(f"  m_s depuis (m_d={md}, m_b={mb}) : {lo:.0f} MeV  (réel {ms} → facteur {lo/ms:.1f})")
    lo2, hi2 = predict_third([md, ms])
    print(f"  contrôle m_b depuis (d,s) : {hi2:.0f} MeV (réel {mb} → facteur {hi2/mb:.1f})")

    print("\n=== Pourquoi imprécis : Koide quark ≈ 0.73 à toute échelle ===")
    for nm, m in DSB_SCALES.items():
        print(f"  {nm:22s} Q(d,s,b) = {koide(m):.4f}")
    print("  ~10 % au-dessus de 2/3, robuste ; le running ne ramène PAS à 2/3.")
    print("  Koide hypersensible : 10 % sur Q → facteur ~2 sur la masse.")

    print("\n=== Verdict ===")
    print("  m_s dérivé du triplet (d,s,b) : ~205 MeV = bon ORDRE DE GRANDEUR,")
    print("  facteur 2 d'écart. NON précis. Koide EXACT leptons / APPROX quarks.")
    print("  Le Z₃/Koide donne la structure (3 gén., ordre) pas les masses précises.")


if __name__ == "__main__":
    main()
