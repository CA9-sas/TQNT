# -*- coding: utf-8 -*-
"""A-11 — Entrelacs de trèfles codé à ΔM = 0 (17/09/2026).

Suite de A-10 : coder le qubit entre états de même M dans un entrelacs de
n trèfles de spin ½ couplés par échange. Trois codes :
  n = 2 : singulet / triplet M=0 (S/T0) ;
  n = 3 : sous-système J=½, M=+½ (qubit « tout échange ») ;
  n = 4 : sous-espace J = 0 (deux singulets) — insensible à tout champ
          uniforme, quelle que soit sa direction.
Méthode : diagonalisation exacte de H = Σ J_ij S_i·S_j + γ Σ B_i·S_i
(+ champ de polarisation B0 selon z), Monte-Carlo sur un bruit quasi statique
(champ local vectoriel gaussien σ_B par composante et par site, corrélation c
entre sites ; bruit relatif r sur chaque J). T2* = √2/(2π σ_f).
Fuite hors du code à 300 K bornée par la relaxation de spin : T_fuite ≈ T1/n.
Entrées déclarées (ordres de grandeur expérimentaux) : σ_B = 10 µT (bain
¹³C naturel), r = 3e-3 (bruit de charge relatif sur l'échange), T1 = 6 ms
(spin électronique à 300 K, diamant), B0 = 10 mT.
Lecture TQNT : l'espace J = 0 de n trèfles a pour base les appariements en
singulets sans croisement (nombres de Catalan) — un codage par le MOTIF
D'APPARIEMENT des trèfles. (Valeur de boucle SU(2) classique, d = 2 ; à ne
pas confondre avec le d = √2 du ruban k = 2.)
"""
import numpy as np
from math import comb, log2

rng = np.random.default_rng(20260917)
gamma = 28.025e9                               # Hz/T
sx = np.array([[0, 1], [1, 0]]) / 2
sy = np.array([[0, -1j], [1j, 0]]) / 2
sz = np.array([[1, 0], [0, -1]]) / 2
I2 = np.eye(2)


def op(single, k, n):
    m = np.array([[1.0]])
    for i in range(n):
        m = np.kron(m, single if i == k else I2)
    return m


def build(n):
    S = [[op(s, k, n) for s in (sx, sy, sz)] for k in range(n)]
    Stot = [sum(S[k][a] for k in range(n)) for a in range(3)]
    S2 = sum(Stot[a] @ Stot[a] for a in range(3))
    return S, Stot, S2


def dot(S, i, j):
    return sum(S[i][a] @ S[j][a] for a in range(3))


CODES = {
    2: {"nom": "S/T0 (M=0)", "J": {(0, 1): 1.0}},
    3: {"nom": "J=½, M=+½ (tout échange)", "J": {(0, 1): 1.0, (1, 2): 0.5}},
    4: {"nom": "J=0 (deux singulets)", "J": {(0, 1): 1.0, (2, 3): 1.0, (1, 2): 0.5}},
}


def code_projector(n, J_Hz, B0, S, Stot, S2):
    """Projecteur sur le sous-espace idéal du code (sans bruit)."""
    H0 = gamma * B0 * Stot[2]
    for (i, j), w in CODES[n]["J"].items():
        H0 = H0 + J_Hz * w * dot(S, i, j)
    w0, v0 = np.linalg.eigh(H0)
    s2 = np.real(np.einsum("ij,jk,ki->i", v0.conj().T, S2, v0))
    mz = np.real(np.einsum("ij,jk,ki->i", v0.conj().T, Stot[2], v0))
    if n == 2:
        sel = np.where(np.abs(mz) < 0.25)[0]                                   # S, T0
    elif n == 3:
        sel = np.where((np.abs(s2 - 0.75) < 0.1) & (np.abs(mz - 0.5) < 0.1))[0]  # J=½, M=+½
    else:
        sel = np.where(np.abs(s2) < 0.1)[0]                                    # J = 0
    assert len(sel) == 2, (n, len(sel))
    V = v0[:, sel]
    return V @ V.conj().T


def logical_pair(H, P):
    w, v = np.linalg.eigh(H)
    weight = np.real(np.einsum("ij,jk,ki->i", v.conj().T, P, v))
    top = np.argsort(weight)[-2:]
    top = top[np.argsort(w[top])]
    return w[top[1]] - w[top[0]], weight[top].min()


def sigma_f(n, J_Hz, sB=10e-6, c=0.0, r=3e-3, B0=10e-3, N=1500):
    S, Stot, S2 = build(n)
    P = code_projector(n, J_Hz, B0, S, Stot, S2)
    f = []
    wmin = 1.0
    for _ in range(N):
        common = rng.normal(0, sB, 3)
        H = gamma * B0 * Stot[2]
        for k in range(n):
            loc = np.sqrt(c) * common + np.sqrt(1 - c) * rng.normal(0, sB, 3)
            H = H + gamma * sum(loc[a] * S[k][a] for a in range(3))
        for (i, j), w in CODES[n]["J"].items():
            H = H + J_Hz * w * (1 + r * rng.normal()) * dot(S, i, j)
        df, wt = logical_pair(H, P)
        f.append(df)
        wmin = min(wmin, wt)
    sigma_f.last_weight = wmin
    return np.std(f)


def T2star(s):
    return np.sqrt(2) / (2 * np.pi * s)


T2_single = T2star(gamma * 10e-6)
print(f"Référence : spin seul ΔM=1, σ_B = 10 µT → T2* = {T2_single*1e6:.2f} µs\n")

J_grid = np.array([0.3, 1, 3, 10, 30, 100]) * 1e6
best = {}
for n in (2, 3, 4):
    print("=" * 72)
    print(f"n = {n} trèfles — code {CODES[n]['nom']}")
    print("=" * 72)
    res = []
    for J in J_grid:
        s = sigma_f(n, J)
        res.append((J, s))
        print(f"  J = {J/1e6:6.1f} MHz : σ_f = {s/1e3:8.2f} kHz → T2* = {T2star(s)*1e6:8.1f} µs"
              f"  (poids min. dans le code {sigma_f.last_weight:.3f})")
    Jb, sb = min(res, key=lambda x: x[1])
    best[n] = (Jb, sb)
    print(f"  optimum : J ≈ {Jb/1e6:.0f} MHz, T2* ≈ {T2star(sb)*1e6:.0f} µs"
          f" (×{T2star(sb)/T2_single:.0f} sur le spin seul)")

print("\n" + "=" * 72)
print("Sensibilité à la corrélation c du bruit de champ (J à l'optimum de chaque code)")
print("=" * 72)
for r_ in (3e-3, 0.0):
    print(f"  bruit de charge r = {r_:g}")
    for n in (2, 3, 4):
        Jb = best[n][0]
        line = [f"c={c:4.2f}: {T2star(sigma_f(n, Jb, c=c, r=r_))*1e6:8.0f} µs" for c in (0.0, 0.9, 0.99)]
        print(f"    n = {n} : " + " | ".join(line))

print("\nSensibilité locale au premier ordre : ⟨S_z⟩ de chaque trèfle dans les deux états logiques")
for n in (2, 3, 4):
    S, Stot, S2 = build(n)
    Jb = best[n][0]
    H0 = gamma * 10e-3 * Stot[2]
    for (i, j), w in CODES[n]["J"].items():
        H0 = H0 + Jb * w * dot(S, i, j)
    P = code_projector(n, Jb, 10e-3, S, Stot, S2)
    wv, v = np.linalg.eigh(H0)
    weight = np.real(np.einsum("ij,jk,ki->i", v.conj().T, P, v))
    top = np.argsort(weight)[-2:]
    top = top[np.argsort(wv[top])]
    diffs = []
    for k in range(n):
        e = [np.real(v[:, t].conj() @ S[k][2] @ v[:, t]) for t in top]
        diffs.append(e[1] - e[0])
    print(f"  n = {n} : Δ⟨S_z,i⟩ = " + ", ".join(f"{d:+.3f}" for d in diffs)
          + ("  → insensible au premier ordre" if max(abs(d) for d in diffs) < 1e-6
             else "  → sensible aux écarts de champ entre trèfles"))

print("\n" + "=" * 72)
print("Fuite hors du code à 300 K (relaxation de spin, T1 = 6 ms)")
print("=" * 72)
for n in (2, 3, 4):
    print(f"  n = {n} : T_fuite ≈ T1/n = {6/n:.1f} ms")

print("\n" + "=" * 72)
print("Densité du codage J = 0 : dimension = Catalan(n/2) (appariements sans croisement)")
print("=" * 72)
for n in (2, 4, 6, 8, 10, 12):
    k = n // 2
    cat = comb(2 * k, k) // (k + 1)
    q = log2(cat) if cat > 1 else 0.0
    print(f"  n = {n:>2} : {cat:>3} appariements → {q:.2f} qubit(s), {q/n:.2f} qubit par trèfle")
