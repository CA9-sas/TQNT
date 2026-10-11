# -*- coding: utf-8 -*-
"""A-12 — Trois trèfles en triangle (17/09/2026).

Proposition de l'auteur après A-11 : placer trois trèfles dont les centres
sont sur les côtés d'un triangle, pour enlever le bruit plus facilement.
Triangle équilatéral → trois échanges égaux, symétrie C3. Le qubit est la
CHIRALITÉ DE SPIN : les deux états J=½, M=+½ se transforment comme ω et ω²
(ω = e^{2iπ/3}) sous la rotation d'un tiers de tour.

H = J Σ S_i·S_{i+1} + D Σ (S_i × S_{i+1})_z + γ B0 S_z + bruit
(D : terme de Dzyaloshinskii–Moriya, qui ouvre un écart √3·D entre les deux
chiralités ; D = 0 → doublet dégénéré.)
Même méthode que A-11 : diagonalisation exacte, Monte-Carlo sur un bruit
quasi statique (σ_B = 10 µT par composante et par trèfle, corrélation c ;
bruit relatif r = 3e-3 indépendant sur chaque échange et sur D),
T2* = √2/(2π σ_f). Référence spin seul : 0,80 µs ; chaîne de A-11 : 1 µs.
"""
import numpy as np

rng = np.random.default_rng(20260917)
gamma = 28.025e9
sx = np.array([[0, 1], [1, 0]]) / 2
sy = np.array([[0, -1j], [1j, 0]]) / 2
sz = np.array([[1, 0], [0, -1]]) / 2
I2 = np.eye(2)
n = 3


def op(single, k):
    m = np.array([[1.0]])
    for i in range(n):
        m = np.kron(m, single if i == k else I2)
    return m


S = [[op(s, k) for s in (sx, sy, sz)] for k in range(n)]
Stot = [sum(S[k][a] for k in range(n)) for a in range(3)]
S2 = sum(Stot[a] @ Stot[a] for a in range(3))
bonds = [(0, 1), (1, 2), (2, 0)]


def heis(i, j):
    return sum(S[i][a] @ S[j][a] for a in range(3))


def dm_z(i, j):
    return S[i][0] @ S[j][1] - S[i][1] @ S[j][0]


def hamiltonian(J, D, B0, Jw=(1, 1, 1), Dw=(1, 1, 1)):
    H = gamma * B0 * Stot[2]
    for b, (i, j) in enumerate(bonds):
        H = H + J * Jw[b] * heis(i, j) + D * Dw[b] * dm_z(i, j)
    return H


def projector(J, D, B0):
    w, v = np.linalg.eigh(hamiltonian(J, D, B0))
    s2 = np.real(np.einsum("ij,jk,ki->i", v.conj().T, S2, v))
    mz = np.real(np.einsum("ij,jk,ki->i", v.conj().T, Stot[2], v))
    sel = np.where((np.abs(s2 - 0.75) < 0.1) & (np.abs(mz - 0.5) < 0.1))[0]
    assert len(sel) == 2
    return v[:, sel] @ v[:, sel].conj().T, v[:, sel]


def logical(H, P):
    w, v = np.linalg.eigh(H)
    wt = np.real(np.einsum("ij,jk,ki->i", v.conj().T, P, v))
    top = np.argsort(wt)[-2:]
    top = top[np.argsort(w[top])]
    return w[top[1]] - w[top[0]]


def sigma_f(J, D, sB=10e-6, c=0.0, r=3e-3, B0=10e-3, N=1500, asym_only=False):
    P, _ = projector(J, D, B0)
    f = []
    for _ in range(N):
        common = rng.normal(0, sB, 3)
        H = gamma * B0 * Stot[2]
        for k in range(n):
            loc = np.sqrt(c) * common + np.sqrt(1 - c) * rng.normal(0, sB, 3)
            H = H + gamma * sum(loc[a] * S[k][a] for a in range(3))
        if asym_only:
            g = rng.normal()
            Jw = [1 + r * g] * 3                     # bruit commun seulement
        else:
            Jw = [1 + r * rng.normal() for _ in range(3)]
        Dw = [1 + r * rng.normal() for _ in range(3)]
        for b, (i, j) in enumerate(bonds):
            H = H + J * Jw[b] * heis(i, j) + D * Dw[b] * dm_z(i, j)
        f.append(logical(H, P))
    return np.std(f)


def T2(s):
    return np.sqrt(2) / (2 * np.pi * s)


print("=" * 72)
print("1. Symétrie : spin moyen de chaque trèfle dans les deux chiralités")
print("=" * 72)
J, B0 = 10e6, 10e-3
print("  (à D = 0 les deux chiralités sont dégénérées : la base est arbitraire ;"
      "\n   un petit écart D les sépare et révèle la symétrie)")
for D in (1e6,):
    _, V = projector(J, D, B0)
    for t in range(2):
        e = [np.real(V[:, t].conj() @ S[k][2] @ V[:, t]) for k in range(n)]
        print(f"  D = {D/1e6:.0f} MHz, état {t+1} : ⟨S_z,i⟩ = " + ", ".join(f"{x:+.4f}" for x in e))
Pzc = [V[:, 0].conj() @ S[k][2] @ V[:, 1] for k in range(n)]
print("  élément hors diagonale |⟨χ₁|S_z,i|χ₂⟩| = " + ", ".join(f"{abs(x):.3f}" for x in Pzc)
      + "\n  → diagonale identique (bruit commun et moyen éliminés) ; hors diagonale non"
      "\n    nulle : un écart de champ ENTRE trèfles couple encore les deux chiralités.")

H0 = hamiltonian(J, 1e6, B0)
w = np.linalg.eigvalsh(H0)
P, V = projector(J, 1e6, B0)
wl = sorted(np.real(np.diag(V.conj().T @ H0 @ V)))
print(f"  écart des chiralités à D = 1 MHz : {(wl[1]-wl[0])/1e6:.3f} MHz (√3·D = {np.sqrt(3):.3f} MHz)")
print(f"  écart vers le quartet J=3/2 : 3J/2 = {1.5*J/1e6:.0f} MHz")

print("\n" + "=" * 72)
print("2. Cohérence : balayage de l'écart D (J = 10 MHz)")
print("=" * 72)
res = []
for D in (0.0, 0.1e6, 0.3e6, 1e6, 3e6, 10e6):
    s = sigma_f(J, D)
    res.append((D, s))
    print(f"  D = {D/1e6:5.1f} MHz (écart {np.sqrt(3)*D/1e6:5.2f} MHz) : σ_f = {s/1e3:7.1f} kHz → T2* = {T2(s)*1e6:6.1f} µs")
Db, sb = min(res, key=lambda x: x[1])
print(f"  optimum D ≈ {Db/1e6:.1f} MHz : T2* ≈ {T2(sb)*1e6:.0f} µs"
      f" (×{T2(sb)/0.80e-6:.0f} sur le spin seul ; chaîne A-11 : 1 µs)")

print("\n" + "=" * 72)
print("3. Quel bruit reste ? (J = 10 MHz, D optimal)")
print("=" * 72)
print(f"  tout le bruit                           : T2* = {T2(sigma_f(J, Db))*1e6:7.1f} µs")
print(f"  échange bruité en COMMUN seulement      : T2* = {T2(sigma_f(J, Db, asym_only=True))*1e6:7.1f} µs")
print(f"  sans bruit d'échange ni de D (r = 0)    : T2* = {T2(sigma_f(J, Db, r=0.0))*1e6:7.1f} µs")
for c in (0.9, 0.99):
    print(f"  r = 0, champ corrélé c = {c:4.2f}          : T2* = {T2(sigma_f(J, Db, r=0.0, c=c))*1e6:7.0f} µs")
print(f"  sans bruit de champ (σ_B = 0), r = 3e-3 : T2* = {T2(sigma_f(J, Db, sB=0.0))*1e6:7.1f} µs")

print("\n" + "=" * 72)
print("4. Balayage croisé J × D (tout le bruit)")
print("=" * 72)
grid = []
for Jv in (1e6, 3e6, 10e6, 30e6):
    row = []
    for Dv in (1e6, 3e6, 10e6):
        t = T2(sigma_f(Jv, Dv))
        row.append(t)
        grid.append((t, Jv, Dv))
    print(f"  J = {Jv/1e6:4.0f} MHz : " + " | ".join(f"D={Dv/1e6:4.0f} MHz → {t*1e6:5.1f} µs" for Dv, t in zip((1e6, 3e6, 10e6), row)))
tb, Jb2, Db2 = max(grid)
print(f"  meilleur point : J ≈ {Jb2/1e6:.0f} MHz, D ≈ {Db2/1e6:.0f} MHz → T2* ≈ {tb*1e6:.0f} µs (×{tb/0.80e-6:.0f})")
print(f"  bruit sur l'écart seul (σ_B = 0) à ce point : T2* = {T2(sigma_f(Jb2, Db2, sB=0.0))*1e6:.0f} µs"
      "\n  → le bruit relatif de la source de l'écart devient la limite.")
print("\nFuite hors du code à 300 K : ≈ T1/3 = 2 ms (comme A-11).")
