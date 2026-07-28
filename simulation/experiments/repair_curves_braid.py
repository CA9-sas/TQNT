r"""
RÉPARATION des courbes de nœuds invalides par FERMETURE DE TRESSE (trace closure),
et VALIDATION du type de nœud (déterminant + polynôme d'Alexander depuis la courbe).

Contexte : `validate_catalog_curves.py` a montré que 14/19 courbes du catalogue
sont topologiquement FAUSSES (réalisent trèfle/nœud trivial). Pour re-mesurer la
signature multipolaire de Biot--Savart des cœurs baryoniques (5₂ décuplet,
7₂ octet/nucléon), on a besoin de VRAIES courbes de 5₂ et 7₂.

Méthode : générer la courbe comme fermeture d'un mot de tresse, avec des arcs de
fermeture IMBRIQUÉS HORS de la boîte de tresse (=> aucun croisement parasite,
contrairement à une fermeture naïve). Chaque courbe est VALIDÉE par :
  - le déterminant |Δ(−1)| (mineur de la matrice de coloriage), et
  - le polynôme d'Alexander Δ(t) complet, calculé numériquement (déterminants
    complexes de la matrice d'Alexander orientée + interpolation polynomiale),
    car det=7 est partagé par 5₂ (span 2) ET 7₁ torique (span 6) — seul le
    span du polynôme d'Alexander les distingue.

Mots de tresse confirmés (3 juil. 2026) :
  5₂ : σ₁³σ₂σ₁⁻¹σ₂          = [1,1,1,2,-1,2]        (3 brins)  Alexander (2,-3,2)
  7₂ : σ₁³σ₂σ₁⁻¹σ₂σ₃σ₂⁻¹σ₃ = [1,1,1,2,-1,2,3,-2,3] (4 brins)  Alexander (3,-5,3)
  8₃ : σ₁⁴σ₂σ₁⁻¹σ₂²        = [1,1,1,1,2,-1,2,2]    (3 brins)  Alexander (2,-4,5,-4,2)
  (bonus 6₂ : [1,1,1,-2,1,-2] -> Alexander (1,-3,3,-3,1))
Le contrôle d'Alexander est indispensable : det=7 partagé par 5₂ (span 2) et 7₁
torique (span 6) ; det=17 partagé par 8₃ (span 4, amphichiral) et 8₂ (span 6).

Résultat multipolaire (cf. brouillon
`docs/brouillon_Ukappa_Utau_geometrie.tex` §Réparation) :

  nœud   classe                     ||Q||/(|m|·R_g)
  3_1    torique C3 (lepton)        0.0000   <- quadrupôle NUL (protégé par C_q)
  5_1    torique C5                 0.0000
  7_1    torique C7                 0.0000
  9_1    torique C9                 0.0000
  4_1    amphichiral C2             0.28
  5_2    chiral C1 (DÉCUPLET)       0.91     <- non nul
  7_2    chiral C1 (OCTET/NUCLÉON)  0.93     <- non nul
  8_3    amphichiral (EW)           0.86     <- non nul

Conclusion (corrigée) : la SUPPRESSION du quadrupôle est gouvernée par la
symétrie ROTATIONNELLE C_q (torique C_{q≥3} -> 0 EXACT), PAS par l'amphichiralité
miroir. Le cœur lepton (torique) est le seul à signature multipolaire robuste
(Q=0 protégé). 8₃ amphichiral (EW) donne O(1) comme les baryons chiraux : le
ratio multipolaire NE sépare PAS baryon/EW. Ce qui distingue l'EW reste
l'AMPHICHIRALITÉ (auto-conjugué), invariant topologique embedding-indépendant,
pas le quadrupôle (l'ancien «8₃=quadrupôle pur» était un artefact d'unknot).
La magnitude reste embedding-dépendante ; l'invariant robuste est le ZÉRO
protégé (torique) vs le NON-NUL permis (non-torique).
"""
from __future__ import annotations
import numpy as np
from collections import Counter


# --------------------------------------------------------------------------- #
#  Génération de courbe par fermeture de tresse propre                        #
# --------------------------------------------------------------------------- #
def braid_curve(word, n_strands, pts_per_seg=60):
    """Courbe 3D fermée réalisant la clôture (trace) du mot de tresse `word`.

    word : liste de générateurs signés (i>0 => σ_i brin gauche DESSUS,
           i<0 => brin gauche DESSOUS), indices 1..n_strands-1.
    Retourne (P, perm, n_components).
    """
    L = len(word)
    occupant = {k: k for k in range(1, n_strands + 1)}
    paths = {s: [(float(s), 0.0, 0.0)] for s in range(1, n_strands + 1)}
    for layer, g in enumerate(word):
        y0, y1 = float(layer), float(layer + 1)
        i = abs(g)
        over_is_left = g > 0
        for slot in range(1, n_strands + 1):
            s = occupant[slot]
            if slot == i:
                for u in np.linspace(0, 1, pts_per_seg)[1:]:
                    z = (0.35 if over_is_left else -0.35) * np.sin(np.pi * u)
                    paths[s].append((i + u, y0 + u * (y1 - y0), z))
            elif slot == i + 1:
                for u in np.linspace(0, 1, pts_per_seg)[1:]:
                    z = (-0.35 if over_is_left else 0.35) * np.sin(np.pi * u)
                    paths[s].append(((i + 1) - u, y0 + u * (y1 - y0), z))
            else:
                for u in np.linspace(0, 1, pts_per_seg)[1:]:
                    paths[s].append((float(slot), y0 + u * (y1 - y0), 0.0))
        occupant[i], occupant[i + 1] = occupant[i + 1], occupant[i]
    top_y = float(L)
    final_slot = {s: slot for slot, s in occupant.items()}
    perm = {b: final_slot[b] for b in range(1, n_strands + 1)}
    seen = set()
    ncomp = 0
    for b in range(1, n_strands + 1):
        if b in seen:
            continue
        ncomp += 1
        c = b
        while c not in seen:
            seen.add(c)
            c = perm[c]

    def closure_arc(slot):
        off = 0.6 + 0.7 * slot
        pts = [np.array([slot, top_y, 0.0]), np.array([slot, top_y + off, 0.0]),
               np.array([-off, top_y + off, 0.0]), np.array([-off, -off, 0.0]),
               np.array([slot, -off, 0.0]), np.array([slot, 0.0, 0.0])]
        A = []
        for a, b in zip(pts[:-1], pts[1:]):
            for u in np.linspace(0, 1, 40)[1:]:
                A.append(tuple(a + u * (b - a)))
        return A

    curve, b, visited = [], 1, set()
    while b not in visited:
        visited.add(b)
        curve += paths[b] if not curve else paths[b][1:]
        f = perm[b]
        curve += closure_arc(f)
        b = f
    return np.array(curve), perm, ncomp


# --------------------------------------------------------------------------- #
#  Polynôme d'Alexander depuis la courbe (numérique, rapide)                  #
# --------------------------------------------------------------------------- #
def _oriented_crossings(P, seed):
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
            over_seg, under_seg = (i, j) if z1 > z2 else (j, i)
            do, du = D[over_seg], D[under_seg]
            sign = 1 if (do[0] * du[1] - do[1] * du[0]) > 0 else -1
            raw.append((i + s[k], j + t[k], z1 > z2, sign))
    return raw


def _alexander_data(P, seed):
    raw = _oriented_crossings(P[:: max(1, len(P) // 450)], seed)
    n = len(raw)
    if n < 3:
        return None
    ev = []
    for cid, (p1, p2, o1, sg) in enumerate(raw):
        ev += [(p1, cid, o1, sg), (p2, cid, not o1, sg)]
    ev.sort()
    seq = [(c, o, sg) for _, c, o, sg in ev]
    m = len(seq)
    try:
        u0 = next(k for k in range(m) if not seq[k][1])
    except StopIteration:
        return None
    seq = seq[u0:] + seq[:u0]
    arc, oa, ui = 0, {}, {}
    for cid, ov, sg in seq:
        if ov:
            oa[cid] = arc
        else:
            ai = arc
            arc = (arc + 1) % n
            ui[cid] = (ai, arc)
    if len(oa) != n or len(ui) != n:
        return None
    return n, oa, ui, [raw[c][3] for c in range(n)]


def _det_at(t, n, oa, ui, signs):
    M = np.zeros((n, n), complex)
    for c in range(n):
        over = oa[c]
        a, b = ui[c]
        if signs[c] > 0:
            M[c, a] += t
            M[c, b] += -1
            M[c, over] += (1 - t)
        else:
            M[c, a] += 1
            M[c, b] += -t
            M[c, over] += (t - 1)
    return np.linalg.det(M[: n - 1, : n - 1])


def alexander(P, votes=5):
    """Polynôme d'Alexander (coeffs entiers haut->bas), span, |Δ(−1)|.

    Interpolation polynomiale de déterminants complexes de la matrice
    d'Alexander orientée. Renvoie (coeffs, span, det) ou None.
    """
    res = Counter()
    for seed in range(votes):
        data = _alexander_data(P, seed)
        if data is None:
            continue
        n, oa, ui, signs = data
        deg = n - 1
        ts = 1.3 * np.exp(2j * np.pi * np.arange(deg + 1) / (deg + 1))
        vals = np.array([_det_at(t, n, oa, ui, signs) for t in ts])
        try:
            coef = np.linalg.solve(np.vander(ts, deg + 1), vals)
        except np.linalg.LinAlgError:
            continue
        c = list(np.round(coef.real).astype(int))
        while c and c[0] == 0:
            c = c[1:]
        while c and c[-1] == 0:
            c = c[:-1]
        if not c:
            continue
        if c[0] < 0:
            c = [-x for x in c]
        res[tuple(c)] += 1
    if not res:
        return None
    best = res.most_common(1)[0][0]
    det = abs(sum(v * (-1) ** (len(best) - 1 - i) for i, v in enumerate(best)))
    return best, len(best) - 1, det


# --------------------------------------------------------------------------- #
#  Multipôles magnétiques de Biot--Savart (dipôle + quadrupôle sans trace)    #
# --------------------------------------------------------------------------- #
def multipoles(P):
    """(|m|, ||Q||, R_g, ||Q||/(|m|·R_g)) autour du centroïde.

    m = ½∮ r×dl ; Q_ab = ½∮ [(r×dl)_a r_b + (r×dl)_b r_a] sans trace.
    Le ratio est adimensionné et invariant d'échelle : il vaut 0 (exact)
    pour les toriques C_{q≥3} (protégé par symétrie) et O(1) pour les
    chiraux C₁ (permis).
    """
    P = np.asarray(P, float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    P = P - P.mean(0)
    dl = np.roll(P, -1, axis=0) - P
    rmid = P + 0.5 * dl
    m = 0.5 * np.cross(rmid, dl).sum(0)
    md = np.linalg.norm(m)
    rxdl = np.cross(rmid, dl)
    Q = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            Q[a, b] = 0.5 * np.sum(rxdl[:, a] * rmid[:, b] + rxdl[:, b] * rmid[:, a])
    Q = Q - np.trace(Q) / 3 * np.eye(3)
    Rg = np.sqrt((P ** 2).sum(1).mean())
    return md, np.linalg.norm(Q), Rg, (np.linalg.norm(Q) / (md * Rg) if md > 0 else np.inf)


# mots de tresse validés (Alexander confirmé depuis la courbe)
#
# Famille des nœuds twist (clasp + n demi-torsions) = chaîne de tresse : on
# ajoute un bloc σ_k σ_{k-1}^{-1} σ_k (nouveau brin) par +2 torsions.
#   impairs  5₂,7₂,9₂ : σ₁³σ₂σ₁⁻¹σ₂ [σ₃σ₂⁻¹σ₃][σ₄σ₃⁻¹σ₄]...
#   pairs    4₁,6₁,8₁ : σ₁⁻¹σ₂σ₁⁻¹σ₂ [σ₃σ₂⁻¹σ₃][σ₄σ₃⁻¹σ₄]...
# (indice de tresse croissant : 8₁ a besoin de 5 brins.)
BRAID_WORDS = {
    "5_2": ([1, 1, 1, 2, -1, 2], 3, (2, -3, 2)),            # décuplet
    "7_2": ([1, 1, 1, 2, -1, 2, 3, -2, 3], 4, (3, -5, 3)),  # octet / nucléon
    "8_3": ([1, 1, 1, 1, 2, -1, 2, 2], 3, (2, -4, 5, -4, 2)),  # EW (amphichiral)
    "6_2": ([1, 1, 1, -2, 1, -2], 3, (1, -3, 3, -3, 1)),    # méson diagonal ρ⁰
    "8_1": ([-1, 2, -1, 2, 3, -2, 3, 4, -3, 4], 5, (3, -7, 3)),  # méson diagonal ω
    "6_1": ([-1, 2, -1, 2, 3, -2, 3], 4, (2, -5, 2)),       # chaîne twist (contrôle)
    "9_2": ([1, 1, 1, 2, -1, 2, 3, -2, 3, 4, -3, 4], 5, (4, -7, 4)),  # chaîne twist
    "10_1": ([-1, 2, -1, 2, 3, -2, 3, 4, -3, 4, 5, -4, 5], 6, (4, -9, 4)),  # EW (twist, det 17)
    "8_2": ([1, 1, 1, 1, 1, -2, 1, -2], 3, (1, -3, 3, -3, 3, -3, 1)),  # cœur top (chiral, det 17)
}


if __name__ == "__main__":
    from theory.omega_ts import _knot_curve

    print("VALIDATION Alexander (courbe torique/fig-8) :")
    for K in ["3_1", "4_1", "5_1", "7_1"]:
        print(f"  {K}: {alexander(np.asarray(_knot_curve(K, 2048), float))}")

    print("\nRÉPARATION + validation des cœurs baryoniques :")
    for K, (w, ns, exp) in BRAID_WORDS.items():
        P, _, nc = braid_curve(w, ns, pts_per_seg=45)
        alx = alexander(P)
        ok = alx and tuple(int(x) for x in alx[0]) == exp
        print(f"  {K}: word={w} comp={nc} Alexander={alx} {'OK' if ok else 'MISMATCH'}")

    print("\nSIGNATURE MULTIPOLAIRE (||Q||/(|m|·R_g)) :")
    for K, cls in [("3_1", "torique C3 (lepton)"), ("5_1", "torique C5"),
                   ("7_1", "torique C7"), ("9_1", "torique C9"),
                   ("4_1", "amphichiral C2")]:
        md, qn, rg, r = multipoles(np.asarray(_knot_curve(K, 2048), float))
        print(f"  {K:4s} {r:7.4f}   {cls}")
    for K, cls in [("5_2", "chiral C1 (DÉCUPLET)"), ("7_2", "chiral C1 (OCTET/NUCLÉON)"),
                   ("8_3", "amphichiral (EW W/Z/H/top)"),
                   ("6_2", "chiral (méson diagonal rho0)"),
                   ("8_1", "chiral (méson diagonal omega)")]:
        w, ns, _ = BRAID_WORDS[K]
        P, _, _ = braid_curve(w, ns, pts_per_seg=60)
        md, qn, rg, r = multipoles(P)
        print(f"  {K:4s} {r:7.4f}   {cls}")
    print("  -> suppression du quadrupôle = symétrie ROTATIONNELLE C_q (torique->0 exact).")
    print("     Tout NON-torique (baryon, EW, méson) donne O(1) : le ratio ne distingue")
    print("     QUE le lepton (torique, dipôle pur). Le reste tient sur chiralité/identité.")
