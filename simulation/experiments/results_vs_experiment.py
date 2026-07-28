r"""
Prédictions STRUCTURELLES de la masse redéfinie vs expérience.

Le cadre prédit les STRUCTURES dérivées (Koide, Regge, incrément Z₃) ; les masses
ABSOLUES (têtes de bande M₀) sont non-perturbatives (entrées, pas prédites en
forme close — cf. bandhead_M0.py). Ce script quantifie les prédictions.

RÉSULTATS :
  1. Koide leptonique — m_τ prédit depuis m_e, m_μ (Q=2/3 dérivé, SANS paramètre) :
       1776,97 MeV vs exp 1776,86 → 61 ppm (0,006 %).
  2. Regge — pente universelle 2πσ_χ=1,096 + tête de bande → membres excités :
       écart moyen 1,5 % sur 10 états (0,2 à 4 %).
  3. Saveur Z₃ — incrément d'étrangeté décuplet : 147 MeV vs (2/3)Λ_χ=144 → 1,8 %.

À NE PAS confondre avec l'ancien « 0,7 % RMS sur 118/122 PDG » de la V5/V6 : celui-là
était PHÉNOMÉNOLOGIQUE (calibré per-nœud sur les courbes fausses), reproduisant les
masses absolues par calibration. Ici : trois prédictions STRUCTURELLES vérifiables,
dont le Koide à 61 ppm (seul résultat ppm sans paramètre de la TQNT).
"""
from __future__ import annotations
import numpy as np

SLOPE = 1.096       # 2π σ_χ (GeV²/J), pente de Regge mesurée
LAMBDA_CHI = 0.216  # GeV


def koide_predict_mtau(m_e=0.51099895, m_mu=105.6583755):
    """m_τ prédit par Q=2/3 depuis m_e, m_μ (racine physique)."""
    a = np.sqrt(m_e) + np.sqrt(m_mu)
    b = m_e + m_mu
    s_tau = 2 * a + np.sqrt(6 * a ** 2 - 3 * b)
    return s_tau ** 2


REGGE = {
    "ρ (I=1)": [("ρ", 0.7753, 1), ("a₂", 1.3183, 2), ("ρ₃", 1.689, 3), ("a₄", 1.967, 4)],
    "ω/f": [("ω", 0.7827, 1), ("f₂", 1.2755, 2), ("ω₃", 1.667, 3), ("f₄", 2.018, 4)],
    "N": [("N", 0.9389, 0.5), ("N(1680)", 1.685, 2.5), ("N(2220)", 2.25, 4.5)],
    "Δ": [("Δ", 1.232, 1.5), ("Δ(1950)", 1.930, 3.5), ("Δ(2420)", 2.40, 5.5)],
}


def regge_predictions():
    """Prédit les membres excités depuis la tête de bande + pente universelle."""
    rows, errs = [], []
    for name, states in REGGE.items():
        _, M0, J0 = states[0]
        for nm, Mx, J in states[1:]:
            Mp = np.sqrt(M0 ** 2 + SLOPE * (J - J0)) * 1000
            e = (Mp - Mx * 1000) / (Mx * 1000) * 100
            rows.append((name, nm, J, Mp, Mx * 1000, e))
            errs.append(abs(e))
    return rows, float(np.mean(errs))


def z3_strangeness():
    dec = [1232, 1385, 1533, 1672]
    inc = float(np.mean(np.diff(dec)))
    pred = 2 / 3 * LAMBDA_CHI * 1000
    return inc, pred, abs(inc - pred) / inc * 100


def main():
    mt, mt_exp = koide_predict_mtau(), 1776.86
    print(f"1. KOIDE : m_τ = {mt:.2f} vs exp {mt_exp} → {(mt-mt_exp)/mt_exp*1e6:.0f} ppm (SANS paramètre)")
    print("\n2. REGGE (pente universelle 1,096 + tête de bande) :")
    rows, mean = regge_predictions()
    for name, nm, J, Mp, Mx, e in rows:
        print(f"   {name:8s} {nm:10s} J={J:<4} M_préd={Mp:5.0f}  exp={Mx:5.0f}  {e:+.1f}%")
    print(f"   écart moyen : {mean:.1f}%")
    inc, pred, e = z3_strangeness()
    print(f"\n3. Z₃ étrangeté (décuplet) : {inc:.0f} MeV vs (2/3)Λ_χ={pred:.0f} → {e:.1f}%")
    print("\nMasses ABSOLUES (M₀) : non-perturbatives (entrées), pas prédites en forme close.")


if __name__ == "__main__":
    main()
