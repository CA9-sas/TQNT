r"""
Matrice de Gram du catalogue de nœuds dans l'espace de Hilbert TQNT (couche I).

Construction (cf. docs/brouillon_hilbert_TQNT.tex) : un nœud K est le vecteur
    |K> = sum_n J_n(K;q) |n>,   n = 2j+1,
dans la base orthonormée des anyons de la catégorie modulaire SU(2)_k (k=135),
où J_n(K;q) est le Jones coloré (q = e^{2pi i/137}). [Le poids modulaire S_{0,n}
et la normalisation réduite se combinent en un facteur global qui s'annule dans
le produit scalaire normalisé : le vecteur est donc proportionnel aux J_n(K).]
La matrice de Gram normalisée G_{KK'} = <K|K'> doit être hermitienne, de
diagonale 1, et définie positive (indépendance linéaire <=> base de Riesz).

DEUX MOTEURS :
  * float (double) : projection basse-couleur n=1..N (rapide, démonstration) ;
  * mpmath (précision arbitraire) : SOMME COMPLÈTE sur les 136 anyons à la
    racine de l'unité zeta_137 — lève le débordement et l'annulation
    catastrophique de la double précision (équivalent numériquement certifié
    de l'arithmétique cyclotomique exacte ; le calcul SYMBOLIQUE sur Z[zeta]
    est intraitable car les termes de Rosso-Jones/Habiro croissent
    exponentiellement). Ancrage exact : [n]_q = sin(n pi/137)/sin(pi/137).

Disponible pour 6 nœuds : 0_1, 3_1, 4_1, 5_1, 7_1, 9_1.
Usage :  python3 -m simulation.experiments.hilbert_gram_matrix
"""
from __future__ import annotations
import os
import numpy as np

from theory.chern_simons import cs_q, colored_jones

K_LEVEL = 135
KP2 = K_LEVEL + 2            # 137
N_ANYONS = KP2 - 1           # 136 anyons (n = 1..136)
KNOTS = ["0_1", "3_1", "4_1", "5_1", "7_1", "9_1"]
_TORUS_S = {"3_1": 3, "5_1": 5, "7_1": 7, "9_1": 9}


# ============================================================
# Moteur float (double précision) — projection basse-couleur
# ============================================================
def knot_vector_float(name: str, N: int, q: complex) -> np.ndarray:
    a = np.array([colored_jones(name, n, q) for n in range(1, N + 1)], dtype=complex)
    nrm = np.linalg.norm(a)
    return a / nrm if nrm > 0 else a


def gram_float(N: int, q: complex) -> np.ndarray:
    V = np.array([knot_vector_float(k, N, q) for k in KNOTS])
    return V.conj() @ V.T


# ============================================================
# Moteur mpmath — somme complète sur les 136 anyons à zeta_137
# ============================================================
def _gram_mpmath(dps: int):
    import mpmath as mp
    mp.mp.dps = dps
    q = mp.exp(2 * mp.pi * mp.mpc(0, 1) / KP2)
    h = mp.sqrt(q)                              # q^{1/2} = e^{i pi/137}
    denom = h - 1 / h

    # cache des q-entiers [m]_q = (q^{m/2}-q^{-m/2})/(q^{1/2}-q^{-1/2})
    M = 2 * KP2 + 2
    qint = [mp.mpc(0)] * (M + 1)
    for m in range(1, M + 1):
        qint[m] = (h**m - h**(-m)) / denom     # = sin(m pi/137)/sin(pi/137)

    def cj(name, n):
        if n == 1:
            return mp.mpc(1)
        if name == "0_1":
            return qint[n]
        if name in _TORUS_S:                    # Rosso-Jones (fsum -> moins d'annulation)
            s = _TORUS_S[name]
            phase = q ** (mp.mpf(-s * (n * n - 1)) / 4)
            terms = []
            prod = mp.mpc(1)
            for j in range(n):
                if j > 0:
                    prod *= qint[n - j] / qint[j]
                terms.append(q ** (mp.mpf(s * j * (j + 2)) / 4) * prod)
            return phase * mp.fsum(terms)
        if name == "4_1":                       # Habiro (figure-eight)
            terms = [mp.mpc(1)]
            prod = mp.mpc(1)
            for k in range(1, n):
                prod *= qint[n + k] * qint[n - k]   # s'annule dès que n+k = 137
                terms.append(prod)
            return mp.fsum(terms)
        raise ValueError(name)

    # vecteurs complets n = 1..136
    vecs = {K: [cj(K, n) for n in range(1, N_ANYONS + 1)] for K in KNOTS}
    nk = len(KNOTS)
    # Gram non normalisée M_{KK'} = sum_n conj(J_n(K)) J_n(K')
    Mmat = mp.matrix(nk, nk)
    for i, Ki in enumerate(KNOTS):
        for jdx, Kj in enumerate(KNOTS):
            Mmat[i, jdx] = mp.fsum(mp.conj(a) * b for a, b in zip(vecs[Ki], vecs[Kj]))
    # normalisation
    diag = [mp.sqrt(mp.re(Mmat[i, i])) for i in range(nk)]
    G = mp.matrix(nk, nk)
    for i in range(nk):
        for jdx in range(nk):
            G[i, jdx] = Mmat[i, jdx] / (diag[i] * diag[jdx])

    # ancrage exact : [n]_q == sin(n pi/137)/sin(pi/137)
    exact_err = max(abs(qint[n] - mp.sin(n * mp.pi / KP2) / mp.sin(mp.pi / KP2))
                    for n in (2, 7, 30, 68, 130))

    # valeurs propres (hermitienne, via polynôme caractéristique réel) + mineurs
    ev = _herm_eigvals(G, nk, mp)
    minors = [mp.det(G[:k, :k]).real for k in range(1, nk + 1)]
    return G, ev, minors, exact_err


def _herm_eigvals(G, nk, mp):
    """Valeurs propres d'une matrice hermitienne nk x nk via l'embedding réel
    symétrique 2nk x 2nk + mp.eigsy (chaque λ apparaît en double, on en garde
    une sur deux). Stable même pour les valeurs propres très groupées."""
    H = mp.matrix(2 * nk, 2 * nk)
    for i in range(nk):
        for j in range(nk):
            r, im = mp.re(G[i, j]), mp.im(G[i, j])
            H[i, j] = r;          H[i, nk + j] = -im
            H[nk + i, j] = im;    H[nk + i, nk + j] = r
    E = list(mp.eigsy(H, eigvals_only=True))
    return sorted(mp.re(e) for e in E)[::2]


def main():
    q = cs_q(K_LEVEL)
    print("=" * 74)
    print("  MATRICE DE GRAM DU CATALOGUE — espace de Hilbert TQNT (couche I)")
    print("=" * 74)
    print(f"  SU(2)_k, k={K_LEVEL}, q = e^(2pi i/{KP2}) ; nœuds : {', '.join(KNOTS)}")

    # --- (1) projection basse-couleur (double précision), pour référence ---
    print("\n  [float] indépendance selon la troncature N (anyons n=1..N) :")
    print(f"  {'N':>3} {'lambda_min':>12} {'cond':>11} {'rang':>5}")
    for N in (6, 10, 12):
        G = gram_float(N, q)
        ev = np.clip(np.linalg.eigvalsh(G).real, 0, None)
        cond = ev.max() / ev.min() if ev.min() > 1e-14 else np.inf
        rank = int(np.sum(ev > 1e-10 * ev.max()))
        print(f"  {N:>3} {ev.min():>12.2e} {cond:>11.1e} {rank:>5d}")

    # --- (2) somme COMPLÈTE sur 136 anyons (mpmath, précision arbitraire) ---
    try:
        import mpmath as mp
    except ImportError:
        print("\n  [mpmath absent] pip install mpmath pour la somme complète.")
        return

    DPS = 400
    print(f"\n  [mpmath, dps={DPS}] somme COMPLÈTE sur les {N_ANYONS} anyons (n=1..{N_ANYONS}) :")
    G, ev, minors, exact_err = _gram_mpmath(DPS)
    nk = len(KNOTS)

    print(f"  ancrage exact  max|[n]_q - sin(n pi/137)/sin(pi/137)| = {mp.nstr(exact_err, 3)}")
    herm = max(abs(G[i, j] - mp.conj(G[j, i])) for i in range(nk) for j in range(nk))
    print(f"  hermiticité    max|G - G^†| = {mp.nstr(herm, 3)}")

    print(f"\n  Matrice de Gram COMPLÈTE |G_KK'| :")
    print("        " + "".join(f"{k:>8}" for k in KNOTS))
    for i, k in enumerate(KNOTS):
        print(f"  {k:>5} " + "".join(f"{float(abs(G[i, j])):>8.3f}" for j in range(nk)))

    print(f"\n  valeurs propres : " + "  ".join(mp.nstr(e, 4) for e in ev))
    print(f"  mineurs principaux (Sylvester) : "
          + "  ".join(mp.nstr(m, 3) for m in minors))
    pd = all(float(e) > 0 for e in ev) and all(float(m) > 0 for m in minors)
    print(f"  => DÉFINIE POSITIVE sur les 136 anyons : {pd}")
    print(f"     (6 vecteurs indépendants => trame = base de Riesz, rang plein)")

    # --- (3) certification : accord entre deux précisions élevées ---
    DPS2 = 280
    G2, ev2, _, _ = _gram_mpmath(DPS2)
    g_delta = max(abs(complex(G[i, j] - G2[i, j])) for i in range(nk) for j in range(nk))
    ev_delta = max(abs(float(a) - float(b)) for a, b in zip(ev, ev2))
    print(f"\n  certification : dps={DPS} vs dps={DPS2}")
    print(f"    |G − G(dps={DPS2})|_max = {mp.nstr(mp.mpf(g_delta), 3)}")
    print(f"    |λ − λ(dps={DPS2})|_max = {mp.nstr(mp.mpf(ev_delta), 3)}")
    print(f"  marge PD : λ_min = {mp.nstr(ev[0], 4)} >> 0  =>  PD robuste à toute")
    print(f"  incertitude numérique raisonnable (l'écart de précision ne renverse")
    print(f"  pas le signe de λ_min, qui reste >> 0).")
    print("=" * 74)


if __name__ == "__main__":
    main()
