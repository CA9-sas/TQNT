r"""
VALIDATION des courbes du catalogue de nœuds (theory/knot_catalog.py).

Recalcule le TYPE de nœud réalisé par chaque courbe paramétrique, via le
DÉTERMINANT (|Δ(−1)|, invariant topologique indépendant du plongement),
calculé depuis la géométrie 3D : projection 2D → croisements (sur/sous par z)
→ matrice de coloriage → |det du mineur (n−1)|. Compare au déterminant vrai
(table de Jones).

Résultat (3 juil. 2026) : les 14 courbes NON-toriques réalisent des nœuds
TRIVIAUX (det=1) ou des TRÈFLES (det=3), PAS les nœuds annoncés. Seules les
courbes toriques (3₁,5₁,7₁,9₁ = T(2,q)) et la figure-8 (4₁) sont valides.
Conséquence : les U_κ=∮κ²ds, U_τ=∮τ²ds calculés depuis ces courbes sont la
géométrie du MAUVAIS nœud → entrée non principielle de la formule de masse.
"""
from __future__ import annotations
import numpy as np
import sympy as sp
from collections import Counter
from theory.omega_ts import _knot_curve, _axiom_data


def _det_from_curve(P: np.ndarray, seed: int) -> int | None:
    """Déterminant du nœud réalisé par la courbe 3D P (projection générique)."""
    P = np.asarray(P, float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    rng = np.random.default_rng(seed)
    A, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = P @ A
    N = len(Q)
    xy, z = Q[:, :2], Q[:, 2]
    P1, D = xy, np.roll(xy, -1, axis=0) - xy
    raw = []
    for i in range(N):
        d1 = D[i]
        js = np.arange(i + 2, N)
        if i == 0:
            js = js[js != N - 1]
        d2 = D[js]
        den = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
        good = np.abs(den) > 1e-12
        dp = P1[js] - P1[i]
        s = np.where(good, (dp[:, 0] * d2[:, 1] - dp[:, 1] * d2[:, 0]) / np.where(good, den, 1), -1)
        t = np.where(good, (dp[:, 0] * d1[1] - dp[:, 1] * d1[0]) / np.where(good, den, 1), -1)
        hit = good & (s > 1e-7) & (s < 1 - 1e-7) & (t > 1e-7) & (t < 1 - 1e-7)
        for k in np.where(hit)[0]:
            j = js[k]
            z1 = z[i] + s[k] * (z[(i + 1) % N] - z[i])
            z2 = z[j] + t[k] * (z[(j + 1) % N] - z[j])
            raw.append((i + s[k], j + t[k], z1 > z2))
    n = len(raw)
    if n < 3:
        return None
    ev = []
    for cid, (p1, p2, o1) in enumerate(raw):
        ev += [(p1, cid, o1), (p2, cid, not o1)]
    ev.sort()
    seq = [(c, o) for _, c, o in ev]
    m = len(seq)
    try:
        u0 = next(k for k in range(m) if not seq[k][1])
    except StopIteration:
        return None
    seq = seq[u0:] + seq[:u0]
    arc, oa, ui = 0, {}, {}
    for cid, ov in seq:
        if ov:
            oa[cid] = arc
        else:
            ai = arc
            arc = (arc + 1) % n
            ui[cid] = (ai, arc)
    if len(oa) != n or len(ui) != n:
        return None
    M = sp.zeros(n, n)
    for c in range(n):
        M[c, oa[c]] += 2
        a, b = ui[c]
        M[c, a] -= 1
        M[c, b] -= 1
    return abs(int(M[: n - 1, : n - 1].det()))


def realized_determinant(knot: str, votes: int = 5, npts: int = 450) -> int | None:
    """Déterminant réalisé (vote majoritaire sur projections aléatoires)."""
    v = Counter()
    for seed in range(votes):
        d = _det_from_curve(_knot_curve(knot, 2048)[:: max(1, 2048 // npts)], seed)
        if d is not None:
            v[d] += 1
    return v.most_common(1)[0][0] if v else None


def validate(knots=None):
    if knots is None:
        knots = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_2", "7_3",
                 "7_4", "7_5", "7_6", "7_7", "8_1", "8_2", "8_3", "10_1"]
    rows = []
    for K in knots:
        true_det = _axiom_data(K)["det"]
        got = realized_determinant(K)
        rows.append((K, true_det, got, got == true_det))
    return rows


if __name__ == "__main__":
    print("VALIDATION courbes catalogue : det(courbe) vs det(vrai)\n")
    rows = validate()
    for K, dt, dc, ok in rows:
        print(f"  {K:5s} vrai={dt:3d}  courbe={str(dc):4s}  {'OK' if ok else 'INVALIDE'}")
    bad = [K for K, _, _, ok in rows if not ok]
    print(f"\n  INVALIDES : {bad}  ({len(bad)}/{len(rows)})")
