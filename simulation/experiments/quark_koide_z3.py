"""Masses de quarks : même forme Z₃/Koide que les leptons, déformée par la couleur.

√m_k = √m₀(1 + r·cos(δ + 2πk/3)) pour chaque triplet (up u,c,t ; down d,s,b).
Leptons : r=√2, δ=2/9 (trèfle PUR, incolore). Quarks : r>√2, δ<2/9 (trèfle COLORÉ).
Pattern suggestif des phases : δ ≈ 2/9, 1/9, 2/27 (= 2/(9n), n=1,2,3).
Honnête : masses de quarks courantes (schéma-dépendantes) → paramètres incertains.
"""
import numpy as np, math

SECTORS = {
    "leptons (e,mu,tau)": [0.51099895, 105.6583755, 1776.86],
    "up (u,c,t)":         [2.16, 1270.0, 172760.0],
    "down (d,s,b)":       [4.67, 93.4, 4180.0],
}


def koide_params(ms):
    sq = np.array([math.sqrt(m) for m in ms]); m0 = (sq.sum() / 3)**2
    Q = sum(ms) / sq.sum()**2
    r = math.sqrt(6 * (Q - 1/3))
    cosv = np.clip((sq / math.sqrt(m0) - 1) / r, -1, 1)
    return m0, Q, r, math.acos(cosv.max())


if __name__ == "__main__":
    for nm, ms in SECTORS.items():
        m0, Q, r, d = koide_params(ms)
        print(f"{nm:>20}: m0={m0:9.1f}  Q={Q:.4f}  r={r:.3f} (r/√2={r/math.sqrt(2):.2f})  δ={d:.4f} (2/9·{d/(2/9):.2f})")
