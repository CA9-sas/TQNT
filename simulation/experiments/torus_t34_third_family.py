r"""
Construction de 8_19 = T(3,4) et sa place dans les familles de vortex.

Demande utilisateur : « construis 8_19 = T(3,4) » (le plus petit nœud torique
T(3,q>2), absent du catalogue, pour tester s'il forme une 3e famille de vortex).

CONSTRUCTION : courbe torique paramétrique _torus_knot_curve(3, 4) — nœud sur un
tore, 3 tours autour de l'axe, 4 à travers le trou. VALIDÉ par Alexander : polynôme
(1,-1,0,1,0,-1,1), déterminant |Δ(-1)| = 3 = celui de 8_19 (= T(3,4)). C'est bien lui.

RÉSULTAT — 8_19 est CENTRÉ (famille torique), PAS excentré (twist), malgré b=3 :
    nœud   b  anisotropie  κ_conc  clasp%  writhe  classe
    3_1    2  1,66         1,61    25%     -3,4    centré (torus)
    5_1    2  1,66         1,16    41%     +6,9    centré
    7_1    2  1,65         1,10    44%     +10,5   centré
    8_19   3  1,68         1,29    35%     +8,5    CENTRÉ (torus) ← b=3 !
    5_2    3  4,36         8,29    4%      +5,0    excentré (twist)
    7_2    3  4,98         7,07    5%      +6,3    excentré
    8_3    4  5,46         6,73    5%      +7,8    excentré
8_19 a l'anisotropie 1,68 (torus ~1,65), AUCUN clasp (κ_conc 1,29, courbure étalée
sur 35% de l'arc), writhe +8,5 (torique). Il est CENTRÉ comme les T(2,q), PAS
excentré comme les twists 5_2/7_2 de MÊME indice de tresse b=3.

CORRECTION DE LA CLASSIFICATION (le vrai résultat) :
  Le discriminant des familles de vortex est TORIQUE vs NON-TORIQUE, PAS b=2 vs
  b≥3. Le mapping « b=2=torus / b≥3=twist » que j'utilisais était un ARTEFACT du
  catalogue (qui ne contient AUCUN T(3,q)). Dès qu'on ajoute T(3,4) — b=3 mais
  torique — il rejoint la famille CENTRÉE. La géométrie centré/excentré suit le
  fait de reposer sur un TORE (symétrie, courbure étalée), pas l'indice de tresse.

CE QUE 8_19 APPORTE : un nœud b=3, CENTRÉ, à symétrie C_3 (3-fold). C'est une
géométrie naturelle de BARYON (3-fold = 3 lobes/quarks), contrairement au cœur
baryonique actuel 7_2 (twist excentré, pis-aller faute de T(3,q) au catalogue).
Piste : un baryon pourrait être un T(3,q) UNIQUE (C_3) plutôt qu'un entrelacs de
3 nœuds — à explorer (la masse fine reste étrangeté/Z₃, non-géométrique).

NB : le cadre référence déjà T(3, q≥10) comme matière noire (CLAUDE.md) ; les
petits T(3,q) (8_19=T(3,4), 10_124=T(3,5)) n'étaient simplement pas catalogués.
Ce script les rend calculables (via _torus_knot_curve).
"""
from __future__ import annotations
import numpy as np
from theory.knot_catalog import _torus_knot_curve
from theory.omega_ts import _knot_curve
from simulation.experiments.repair_curves_braid import braid_curve, BRAID_WORDS, alexander

FAMILY = [("3_1", 2), ("5_1", 2), ("7_1", 2), ("8_19", 3), ("5_2", 3), ("7_2", 3), ("8_3", 4)]


def get_curve(K, n=600):
    if K == "8_19":
        return np.asarray(_torus_knot_curve(3, 4, n), float)
    if K in ("3_1", "4_1", "5_1", "7_1", "9_1"):
        return np.asarray(_knot_curve(K, n), float)
    w, ns, _ = BRAID_WORDS[K]
    return np.asarray(braid_curve(w, ns)[0], float)


def _resample(P, m=300):
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0], np.cumsum(d)]); Lt = s[-1]
    u = np.linspace(0, Lt, m, endpoint=False)
    return np.column_stack([np.interp(u, s[:-1], P[:, k], period=Lt) for k in range(3)])


def vortex(K):
    Q = _resample(get_curve(K)); n = len(Q)
    Q = Q / np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1).sum()
    R = Q - Q.mean(0); r2 = (R ** 2).sum(1)
    ev = np.linalg.eigvalsh(np.einsum("n,ij->ij", r2, np.eye(3)) - np.einsum("ni,nj->ij", R, R))
    d1 = np.gradient(Q, axis=0); d2 = np.gradient(d1, axis=0)
    kap = np.linalg.norm(np.cross(d1, d2), axis=1) / np.clip(np.linalg.norm(d1, axis=1) ** 3, 1e-9, None)
    cum = np.cumsum(np.sort(kap ** 2)[::-1])
    dl = np.roll(Q, -1, 0) - Q; mid = Q + dl / 2; W = 0.0
    for i in range(n):
        r = mid[i] - mid; rn = np.linalg.norm(r, axis=1); m = rn > 1e-9
        W += np.sum((np.cross(dl[i], dl)[m] * r[m]).sum(1) / rn[m] ** 3)
    return dict(aniso=ev[2] / ev[0], kconc=kap.max() / kap.mean(),
                kloc=np.searchsorted(cum, 0.5 * cum[-1]) / n, writhe=W / (4 * np.pi))


def main():
    a = alexander(get_curve("8_19"))
    print(f"T(3,4) validé par Alexander : {a[0]}, déterminant {a[2]} (8_19 : det 3)\n")
    print(f"{'nœud':6s} {'b':>2s} {'aniso':>6s} {'κ_conc':>7s} {'clasp%':>7s} {'writhe':>7s} {'classe':>10s}")
    for K, b in FAMILY:
        o = vortex(K)
        cls = "centré" if o["aniso"] < 2.5 else "excentré"
        print(f"{K:6s} {b:>2d} {o['aniso']:>6.2f} {o['kconc']:>7.2f} {o['kloc']*100:>6.0f}% "
              f"{o['writhe']:>+7.2f} {cls:>10s}")
    print("\n→ 8_19 (b=3, C_3) est CENTRÉ (torus), pas excentré. Le discriminant des")
    print("  familles est TORIQUE vs NON-TORIQUE, pas b=2 vs b≥3 (artefact du catalogue).")


if __name__ == "__main__":
    main()
