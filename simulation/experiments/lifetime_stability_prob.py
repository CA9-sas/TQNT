"""
Durée de vie par PROBABILITÉ DE STABILITÉ — les brins ne doivent pas se toucher.

Idée (L. Cariou) : un nœud/entrelacs n'a sa stabilité propre que pour une forme
particulière. Si deux brins (ou deux boucles) sont trop PROCHES, la probabilité
de maintenir la configuration stable chute -> durée de vie courte. Ce n'est pas
une barrière d'énergie (écartée), mais une probabilité GÉOMÉTRIQUE.

Mesure calculable : la DISTANCE MINIMALE entre brins non-adjacents de la courbe
(l'« épaisseur » du nœud ; petit = brins serrés = fragile). On en tire une
proximité scale-free et on teste si la durée de vie en dépend.

Modèle visé (reste en loi de puissance, compatible avec les données) :
    Γ ∝ ω · P_proche  =>  τ ∝ δ^p / M,   δ = distance min normalisée.

Usage :  python3 -m simulation.experiments.lifetime_stability_prob
"""
from __future__ import annotations

import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import geometric_integrals
from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium
from theory.topological_lifetime import TAU_EXP_DATA

WEAK = ["d", "s", "mu-", "c", "tau-", "b"]
_CACHE: dict[str, dict] = {}


def min_strand_distance(name: str, n: int = 600) -> dict:
    """Distance minimale entre points NON-ADJACENTS de la courbe (épaisseur).

    Exclut les points proches le long de l'arc (séparation < 12 % de L) pour
    ne garder que les rapprochements entre brins distincts.
    """
    if name in _CACHE:
        return _CACHE[name]
    pts = knot_curve(name, n)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    N = len(pts)
    # longueur d'arc cumulée
    seg = np.linalg.norm(np.diff(pts, axis=0, append=pts[:1]), axis=1)
    L = float(seg.sum())
    s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    # distances par paires, masquées par séparation d'arc le long de la courbe
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.linalg.norm(diff, axis=2)
    ds = np.abs(s[:, None] - s[None, :])
    ds = np.minimum(ds, L - ds)             # arc le plus court (courbe fermée)
    mask = ds > 0.12 * L                     # garder seulement les brins distincts
    D_masked = np.where(mask, D, np.inf)
    d_min = float(D_masked.min())
    R = L / (2 * np.pi)                       # rayon caractéristique (si cercle)
    out = {"L": L, "d_min": d_min,
           "delta": d_min / R,                # proximité scale-free (brins/rayon)
           "ropelength": L / (d_min / 2)}     # ~ longueur / épaisseur
    _CACHE[name] = out
    return out


def main():
    print("=" * 92)
    print("  DURÉE DE VIE vs PROBABILITÉ DE STABILITÉ (distance min entre brins)")
    print("=" * 92)

    # mesures par nœud du catalogue (info)
    print("\n  Proximité par nœud (δ = d_min/R ; petit = brins serrés = fragile) :")
    seen = set()
    for p in SM_ASSIGNMENTS:
        for kn in p["knots"]:
            if kn in seen:
                continue
            seen.add(kn)
    for kn in sorted(seen, key=lambda k: (int(k.split('_')[0]), k)):
        m = min_strand_distance(kn)
        print(f"    {kn:>5} : d_min={m['d_min']:.3f}  δ={m['delta']:.3f}  "
              f"ropelength={m['ropelength']:.1f}")

    # agrégation par particule : produit des stabilités (composantes indép.)
    # log P_stable_tot = Σ log δ_i ; weakest-link = min δ_i
    print("\n  Par particule (faible) — agrégats de proximité + résidu/τ :")
    print(f"  {'part.':>6} {'n':>2} {'M_raw':>9} {'Σlnδ':>8} {'min δ':>7} "
          f"{'log τ':>8}")
    print("  " + "-" * 50)
    data = []
    for p in SM_ASSIGNMENTS:
        if p["name"] not in WEAK:
            continue
        deltas = [min_strand_distance(kn)["delta"] for kn in p["knots"]]
        sumln = float(np.sum(np.log(deltas)))
        mind = float(min(deltas))
        M = cornell_equilibrium(p["knots"], p["k"])["mass_raw"]
        lt = np.log10(TAU_EXP_DATA[p["name"]])
        data.append({"name": p["name"], "n": len(p["knots"]), "M": M,
                     "sumln": sumln, "mind": mind, "logtau": lt})
        print(f"  {p['name']:>6} {len(p['knots']):>2} {M:>9.1f} {sumln:>8.2f} "
              f"{mind:>7.3f} {lt:>+8.2f}")

    lt = np.array([d["logtau"] for d in data])
    print("\n  Corrélations de log10 τ avec :")
    cands = {
        "Σ ln δ (Π stabilité)":  np.array([d["sumln"] for d in data]),
        "min δ (maillon faible)": np.array([d["mind"] for d in data]),
        "log M":                  np.log10([d["M"] for d in data]),
        "Σlnδ - ν logM":          np.array([d["sumln"] for d in data])
                                  - 7.854 * np.log10([d["M"] for d in data]),
    }
    for lab, x in cands.items():
        print(f"    {lab:>22} : r = {np.corrcoef(x, lt)[0,1]:+.3f}")

    # test calibré : τ = τ0 · exp(γ Σlnδ) / M^ν  (ν=5π/2 fixe), ajuste γ + intercept
    print("\n  Test : log10 τ = c + ν' logM-pente fixe + γ·(Σlnδ) ... (régression libre)")
    X = np.column_stack([np.ones(len(data)),
                         np.log10([d["M"] for d in data]),
                         [d["sumln"] for d in data]])
    coef, *_ = np.linalg.lstsq(X, lt, rcond=None)
    rms = float(np.sqrt(np.mean((lt - X @ coef) ** 2)))
    print(f"    log10 τ = {coef[0]:+.2f} {coef[1]:+.3f}·logM {coef[2]:+.3f}·Σlnδ"
          f"   RMS={rms:.3f}  (réf P15 ~0,30, #par=3)")
    # sans la proximité (juste masse) pour voir l'apport
    X0 = np.column_stack([np.ones(len(data)), np.log10([d["M"] for d in data])])
    c0, *_ = np.linalg.lstsq(X0, lt, rcond=None)
    rms0 = float(np.sqrt(np.mean((lt - X0 @ c0) ** 2)))
    print(f"    sans proximité (masse seule) : RMS={rms0:.3f}  (#par=2)")

    # maillon faible : log10 τ = c + ν' logM + p·log(min δ)   (le vrai signal r=+0,75)
    print("\n  Maillon faible (min δ) en loi de puissance :")
    Xw = np.column_stack([np.ones(len(data)),
                          np.log10([d["M"] for d in data]),
                          np.log10([d["mind"] + 1e-6 for d in data])])
    cw, *_ = np.linalg.lstsq(Xw, lt, rcond=None)
    rmsw = float(np.sqrt(np.mean((lt - Xw @ cw) ** 2)))
    print(f"    log10 τ = {cw[0]:+.2f} {cw[1]:+.3f}·logM {cw[2]:+.3f}·log(minδ)"
          f"   RMS={rmsw:.3f}  (#par=3 ; réf P15 0,30 #par=2)")
    # + nombre de nœuds (proximité remplace-t-elle ou complète-t-elle n ?)
    Xwn = np.column_stack([np.ones(len(data)),
                           np.log10([d["M"] for d in data]),
                           np.log10([d["mind"] + 1e-6 for d in data]),
                           [d["n"] for d in data]])
    cwn, *_ = np.linalg.lstsq(Xwn, lt, rcond=None)
    rmswn = float(np.sqrt(np.mean((lt - Xwn @ cwn) ** 2)))
    print(f"    + n (nœuds) : RMS={rmswn:.3f}  (#par=4)")
    print("=" * 92)


if __name__ == "__main__":
    main()
