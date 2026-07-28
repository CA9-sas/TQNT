"""Prédiction hors-échantillon TQNT-V6 + candidats matière noire les plus probables.

(1) TEST DE CRÉDIBILITÉ — la relation de Koide (Q=2/3, DÉRIVÉE de la Z₃ du trèfle)
    prédit m_τ à partir des SEULS m_e, m_μ mesurés (m_τ non utilisé) :
        3(m_e+m_μ+m_τ) = 2(√m_e+√m_μ+√m_τ)²  →  m_τ = (2A+√(6A²−3B))²
    → m_τ = 1776,97 MeV (mesuré 1776,86 ; +61 ppm).
    Honnête : la FORME de Koide fut trouvée historiquement en ajustant les 3 leptons ;
    l'apport TQNT est de DÉRIVER le 2/3 de Z₃. C'est donc un test fort de la structure
    Z₃-Koide (qu'elle passe), pas la prédiction d'une quantité jamais mesurée.

(2) MATIÈRE NOIRE — candidats Q=0 amphichiraux stables, classés par probabilité de
    composition P(L). Le plus probable : 4_1+8_3 à 16,01 GeV (pair distinct, pas de
    pénalité de symétrie, faible tension). Prédictions VRAIMENT hors-échantillon
    (DM non détectée) testables par les expériences futures.
"""
import math
from simulation.experiments.composition_kinetics_v6 import P_combination

ME, MMU, MTAU_EXP = 0.51099895, 105.6583755, 1776.86


def predict_mtau():
    se, smu = math.sqrt(ME), math.sqrt(MMU)
    A, B = se + smu, ME + MMU
    return (2 * A + math.sqrt(6 * A**2 - 3 * B)) ** 2


# 6 candidats DM STRICT amphichiraux (Q=0) : config, masse GeV
DM_STRICT = [("6_3+6_3", 1.55), ("6_3+6_3+6_3", 4.70), ("4_1+8_3", 16.01),
             ("8_3+8_3", 17.24), ("6_3+8_3", 26.48), ("4_1+4_1+8_3", 51.78)]


def rank_dm():
    out = []
    for cfg, m in DM_STRICT:
        n = len(cfg.split("+"))
        out.append((cfg, m, P_combination(cfg, Lk=0, Lk_opt=n)))  # Q=0 → Lk=0
    return sorted(out, key=lambda x: -x[2])


if __name__ == "__main__":
    mt = predict_mtau()
    print(f"(1) Koide → m_τ = {mt:.2f} MeV (mesuré {MTAU_EXP}, {(mt/MTAU_EXP-1)*1e6:+.0f} ppm)")
    print("(2) DM la plus probable :")
    r = rank_dm(); Pmax = r[0][2]
    for cfg, m, P in r:
        print(f"    {cfg:>14} {m:>6.2f} GeV  P/Pmax={P/Pmax:.3f}")
