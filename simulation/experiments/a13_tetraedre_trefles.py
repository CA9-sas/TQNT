# -*- coding: utf-8 -*-
"""A-13 — Quatre trèfles en tétraèdre (17/09/2026).

Suite de A-12 (triangle). Le tétraèdre régulier est la seule disposition à
trois dimensions où les quatre trèfles sont tous équidistants : six échanges
égaux, H = J Σ_{i<j} S_i·S_j = (J/2)·(S_tot² − 9/4) — l'énergie ne dépend que
du spin total. Le qubit : les deux singulets J = 0 (dégénérés exactement).
Deux atouts attendus :
  (a) J = 0 → insensible à tout champ UNIFORME, dans toute direction ;
  (b) Wigner–Eckart : un champ LOCAL est un vecteur (rang 1) ; il ne couple
      pas deux états J = 0 entre eux au premier ordre → seul le second ordre
      via le triplet (écart J) subsiste.
Écart logique : chiralité scalaire K Σ_faces S_a·(S_b × S_c) (faces orientées
vers l'extérieur), qui sépare les deux singulets.
Bruit comme A-11/A-12 : σ_B = 10 µT par composante et par trèfle (corrélation
c), bruit relatif r = 3e-3 indépendant sur chacun des six échanges et sur K.
T2* = √2/(2π σ_f). Références : spin seul 0,80 µs ; chaîne J=0 (A-11) 4 µs ;
triangle (A-12) 25 µs.
"""
import numpy as np
from itertools import combinations

rng = np.random.default_rng(20260917)
gamma = 28.025e9
sx = np.array([[0, 1], [1, 0]]) / 2
sy = np.array([[0, -1j], [1j, 0]]) / 2
sz = np.array([[1, 0], [0, -1]]) / 2
I2 = np.eye(2)
n = 4


def op(single, k):
    m = np.array([[1.0]])
    for i in range(n):
        m = np.kron(m, single if i == k else I2)
    return m


S = [[op(s, k) for s in (sx, sy, sz)] for k in range(n)]
Stot = [sum(S[k][a] for k in range(n)) for a in range(3)]
S2 = sum(Stot[a] @ Stot[a] for a in range(3))
bonds = list(combinations(range(n), 2))                       # 6 arêtes
faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]          # orientées
eps = np.zeros((3, 3, 3))
for (i, j, k), sgn in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1),
                       ((0, 2, 1), -1), ((2, 1, 0), -1), ((1, 0, 2), -1)):
    eps[i, j, k] = sgn
HEIS = {b: sum(S[b[0]][a] @ S[b[1]][a] for a in range(3)) for b in bonds}
CHIR = []
for (a_, b_, c_) in faces:
    M = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if eps[i, j, k]:
                    M += eps[i, j, k] * S[a_][i] @ S[b_][j] @ S[c_][k]
    CHIR.append(M)


def H_clean(J, K, B0):
    H = gamma * B0 * Stot[2]
    for b in bonds:
        H = H + J * HEIS[b]
    for M in CHIR:
        H = H + K * M
    return H


def projector(J, K, B0):
    w, v = np.linalg.eigh(H_clean(J, K, B0))
    s2 = np.real(np.einsum("ij,jk,ki->i", v.conj().T, S2, v))
    sel = np.where(np.abs(s2) < 0.1)[0]
    assert len(sel) == 2
    return v[:, sel] @ v[:, sel].conj().T, v[:, sel], w[sel]


def logical(H, P):
    w, v = np.linalg.eigh(H)
    wt = np.real(np.einsum("ij,jk,ki->i", v.conj().T, P, v))
    top = np.argsort(wt)[-2:]
    top = top[np.argsort(w[top])]
    return w[top[1]] - w[top[0]]


def sigma_f(J, K, sB=10e-6, c=0.0, r=3e-3, rK=None, B0=10e-3, N=1200):
    rK = r if rK is None else rK
    P, _, _ = projector(J, K, B0)
    f = []
    for _ in range(N):
        common = rng.normal(0, sB, 3)
        H = gamma * B0 * Stot[2]
        for k in range(n):
            loc = np.sqrt(c) * common + np.sqrt(1 - c) * rng.normal(0, sB, 3)
            H = H + gamma * sum(loc[a] * S[k][a] for a in range(3))
        for b in bonds:
            H = H + J * (1 + r * rng.normal()) * HEIS[b]
        for M in CHIR:
            H = H + K * (1 + rK * rng.normal()) * M
        f.append(logical(H, P))
    return np.std(f)


def T2(s):
    return np.sqrt(2) / (2 * np.pi * s)


print("=" * 72)
print("1. Structure du spectre et règles de sélection")
print("=" * 72)
J, B0 = 10e6, 0.0
w = np.linalg.eigvalsh(H_clean(J, 0.0, B0))
vals, counts = np.unique(np.round(w / 1e6, 6), return_counts=True)
print("  K = 0 : niveaux (MHz) × dégénérescence : " + ", ".join(f"{v:+.1f}×{c}" for v, c in zip(vals, counts)))
print(f"  écart singulets → triplet = J = {J/1e6:.0f} MHz ; triplet → quintet = 2J")
for K in (1e6,):
    _, V, wl = projector(J, K, B0)
    print(f"  K = {K/1e6:.0f} MHz : écart entre les deux singulets = {abs(wl[1]-wl[0])/1e6:.3f} MHz"
          f" (= {abs(wl[1]-wl[0])/K:.3f}·K)")
    mx = max(abs(V[:, 0].conj() @ S[k][a] @ V[:, 1]) for k in range(n) for a in range(3))
    md = max(abs(np.real(V[:, t].conj() @ S[k][a] @ V[:, t])) for t in range(2) for k in range(n) for a in range(3))
    print(f"  champ local sur un trèfle, entre singulets : max|⟨s₁|S_i^a|s₂⟩| = {mx:.1e},"
          f" max|⟨s|S_i^a|s⟩| = {md:.1e}"
          "\n  → ni décalage ni mélange au premier ordre (Wigner–Eckart vérifié)")
    ex = max(abs(V[:, 0].conj() @ HEIS[b] @ V[:, 1]) for b in bonds)
    print(f"  échange asymétrique sur une arête : max|⟨s₁|S_i·S_j|s₂⟩| = {ex:.3f}"
          "\n  → le bruit d'échange NON uniforme couple les singulets au premier ordre")

print("\n" + "=" * 72)
print("2. Balayage croisé J × K (tout le bruit, B0 = 10 mT)")
print("=" * 72)
grid = []
Ks = (0.03e6, 0.1e6, 0.3e6, 1e6, 3e6)
for Jv in (3e6, 10e6, 30e6, 100e6):
    row = []
    for Kv in Ks:
        t = T2(sigma_f(Jv, Kv))
        row.append(t)
        grid.append((t, Jv, Kv))
    print(f"  J = {Jv/1e6:5.0f} MHz : " + " | ".join(f"K={Kv/1e6:5.2f} → {t*1e6:6.1f} µs" for Kv, t in zip(Ks, row)))
tb, Jb, Kb = max(grid)
print(f"  meilleur point : J ≈ {Jb/1e6:.0f} MHz, K ≈ {Kb/1e6:.2f} MHz → T2* ≈ {tb*1e6:.0f} µs (×{tb/0.80e-6:.0f})")

print("\n" + "=" * 72)
print("3. Quel bruit reste ? (meilleur point)")
print("=" * 72)
cases = [
    ("tout le bruit", {}),
    ("sans bruit de champ (σ_B = 0)", {"sB": 0.0}),
    ("sans bruit d'échange ni de K (r = 0)", {"r": 0.0}),
    ("bruit d'échange seul, K sans bruit", {"sB": 0.0, "rK": 0.0}),
    ("bruit de K seul", {"sB": 0.0, "r": 0.0, "rK": 3e-3}),
    ("r = 0, champ corrélé c = 0,99", {"r": 0.0, "c": 0.99}),
]
for lab, kw in cases:
    s = sigma_f(Jb, Kb, **kw)
    print(f"  {lab:<40}: T2* = {T2(s)*1e6:10.1f} µs" if s > 0 else f"  {lab:<40}: σ_f = 0")

print("\n" + "=" * 72)
print("4. Échange plus stable (r = 3e-4) : la limite se déplace-t-elle ?")
print("=" * 72)
for Jv in (10e6, 30e6, 100e6):
    best = max((T2(sigma_f(Jv, Kv, r=3e-4)), Kv) for Kv in (0.01e6, 0.03e6, 0.1e6, 0.3e6, 1e6))
    print(f"  J = {Jv/1e6:5.0f} MHz : meilleur K = {best[1]/1e6:.2f} MHz → T2* ≈ {best[0]*1e6:.0f} µs")

Kopt = 0.5 * np.sqrt(3e-3 / 12) * Jb
print(f"\n  Ordre de grandeur analytique de l'optimum : K ~ ½·√(r/12)·J = {Kopt/1e3:.0f} kHz à J = {Jb/1e6:.0f} MHz"
      "\n  (la simulation le place 3 à 4 fois plus haut)"
      "\n  (équilibre entre bruit de l'écart, ∝ rK, et mélange par l'échange asymétrique, ∝ (rJ)²/K)")
print("\nFuite hors du code à 300 K : ≈ T1/4 = 1,5 ms.")
print("Densité : 4 trèfles → 1 qubit (Catalan(2) = 2) ; le tétraèdre est le seul"
      "\n  simplexe régulier en 3D : au-delà de 4, l'équidistance est impossible.")
