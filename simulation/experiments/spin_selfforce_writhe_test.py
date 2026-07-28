"""Test : l'auto-interaction (attraction/repulsion) du tube de flux est-elle le spin ?

Idee (utilisateur) : le spin serait l'effet auto-attractif/repulsif du tube de flux selon la
topologie du noeud/entrelacs. Mesure de l'auto-interaction = writhe (auto-enlacement de Gauss,
= facon dont le tube s'enroule sur lui-meme ; proxy des forces de Biot-Savart entre brins).

RESULTAT :
  - |writhe| corr croisements = +0.84 (forte) -> MAGNITUDE = complexite = MASSE (self-energie).
  - |writhe| corr spin = -0.32 ; corr b = -0.32 -> ne trace PAS le spin.
  - Torus T(2,q) : tous spin-1/2 mais |writhe| croit 3.4->14.4 avec q.
  - SIGNE du writhe (attractif/repulsif) -> CHIRALITE (4_1 amphichiral -> writhe ~ 0 ;
    writhe flippe sous miroir = matiere/antimatiere). NB writhe geometrique (embedding-dependant) ;
    l'invariant propre est V_K(t) vs V_K(1/t).
=> l'auto-interaction du tube donne MASSE (magnitude) + CHIRALITE (signe), PAS le spin.
   Le spin vient de l'indice de tresse b (arite = nombre de brins), s=(b-1)/2, independant.
"""
import numpy as np
from theory.knot_catalog import knot_curve
from theory.jones import topological_spin, braid_index, jones_polynomial, jones_span
from simulation.experiments.repair_curves_braid import braid_curve, BRAID_WORDS

KNOTS = ['3_1', '5_1', '7_1', '9_1', '4_1', '5_2', '7_2', '8_3']


def writhe(P):
    """Auto-enlacement de Gauss : Wr = (1/4pi) ∮∮ (dl1 x dl2)·(r1-r2)/|r1-r2|^3."""
    P = np.asarray(P, float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    n = len(P)
    dl = np.roll(P, -1, 0) - P
    mid = P + 0.5 * dl
    W = 0.0
    for i in range(n):
        r = mid[i] - mid
        r[i] = 1e9
        cr = np.cross(np.tile(dl[i], (n, 1)), dl)
        W += (( cr * r).sum(1) / np.linalg.norm(r, axis=1) ** 3).sum()
    return W / (4 * np.pi)


def curve(K, npts=1200):
    if K in BRAID_WORDS:
        w, ns, _ = BRAID_WORDS[K]
        P, _, _ = braid_curve(w, ns, pts_per_seg=40)
        return P
    return np.asarray(knot_curve(K, npts), float)


def run():
    print(f"{'K':5}{'b':>3}{'spin':>5}{'cross':>6}{'writhe':>9}")
    bb, ss, cc, ww = [], [], [], []
    for K in KNOTS:
        W = writhe(curve(K))
        b = braid_index(K)
        s = topological_spin(K)
        c = jones_span(jones_polynomial(K))
        bb.append(b); ss.append(s); cc.append(c); ww.append(abs(W))
        print(f"{K:5}{b:>3}{s:>5}{c:>6}{W:>9.2f}")
    ss, cc, bb, ww = map(np.array, (ss, cc, bb, ww))
    print(f"\ncorr(|writhe|, spin)        = {np.corrcoef(ww, ss)[0, 1]:+.3f}")
    print(f"corr(|writhe|, croisements) = {np.corrcoef(ww, cc)[0, 1]:+.3f}  (=> MASSE)")
    print(f"corr(|writhe|, b)           = {np.corrcoef(ww, bb)[0, 1]:+.3f}")
    print("Auto-interaction : magnitude -> masse ; signe -> chiralite ; PAS le spin (= indice de tresse).")


if __name__ == "__main__":
    run()
