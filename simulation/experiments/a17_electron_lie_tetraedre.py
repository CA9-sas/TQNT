# -*- coding: utf-8 -*-
"""A-17 — Un électron lié au centre d'un tétraèdre de trèfles (17/09/2026).

Suite de A-13 (tétraèdre, 72 µs) et A-16 (résonance). Un cinquième trèfle,
l'électron lié, au centre du tétraèdre : équidistant des quatre, il conserve
la symétrie. Hamiltonien (5 spins ½, 32 états) :
  H = J Σ_arêtes S_i·S_j + J_c S_0·(S_1+S_2+S_3+S_4) + K·chiralité + γB0·S_z + bruit
Comme J_c ne voit que le spin total du tétraèdre S_T :
  E = (J/2)[S_T(S_T+1) − 3] + (J_c/2)[F(F+1) − S_T(S_T+1) − 3/4]
Deux codes :
  C1 : singulets du tétraèdre (S_T = 0) ⊗ électron central (F = ½, M = +½) —
       l'électron est exactement découplé (S_T = 0) ;
  C2 : horloge « hyperfine » dans le quintet (S_T = 2) : F = 3/2 ↔ F = 5/2,
       M = +½ (ΔM = 0), écart (5/2)·J_c.
Résonance : le code C1 (E = −3J/2) croise l'état S_T = 1, F = ½
(E = −J/2 − J_c) à J_c = J ; le croisement est exact sous la symétrie,
le bruit asymétrique y ouvre une fuite.
Bruit comme A-11..A-13 : σ_B = 10 µT par composante sur les cinq spins, bruit
relatif r = 3e-3 indépendant sur les six arêtes, les quatre liens centraux et K.
"""
import numpy as np
from itertools import combinations

rng = np.random.default_rng(20260917)
gamma = 28.025e9
sx = np.array([[0, 1], [1, 0]]) / 2
sy = np.array([[0, -1j], [1j, 0]]) / 2
sz = np.array([[1, 0], [0, -1]]) / 2
I2 = np.eye(2)
N = 5                                    # site 0 = électron central, 1..4 = tétraèdre


def op(single, k):
    m = np.array([[1.0]])
    for i in range(N):
        m = np.kron(m, single if i == k else I2)
    return m


S = [[op(s, k) for s in (sx, sy, sz)] for k in range(N)]
tet = [1, 2, 3, 4]
ST = [sum(S[k][a] for k in tet) for a in range(3)]
ST2 = sum(ST[a] @ ST[a] for a in range(3))
F = [sum(S[k][a] for k in range(N)) for a in range(3)]
F2 = sum(F[a] @ F[a] for a in range(3))
edges = list(combinations(tet, 2))
centre = [(0, k) for k in tet]
faces = [(1, 2, 3), (1, 4, 2), (1, 3, 4), (2, 4, 3)]
eps = np.zeros((3, 3, 3))
for (i, j, k), sg in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1),
                      ((0, 2, 1), -1), ((2, 1, 0), -1), ((1, 0, 2), -1)):
    eps[i, j, k] = sg
HE = {b: sum(S[b[0]][a] @ S[b[1]][a] for a in range(3)) for b in edges + centre}
CH = []
for (a_, b_, c_) in faces:
    M = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if eps[i, j, k]:
                    M += eps[i, j, k] * S[a_][i] @ S[b_][j] @ S[c_][k]
    CH.append(M)


def H_clean(J, Jc, K, B0):
    H = gamma * B0 * F[2]
    for b in edges:
        H = H + J * HE[b]
    for b in centre:
        H = H + Jc * HE[b]
    for M in CH:
        H = H + K * M
    return H


def expval(v, O):
    return np.real(np.einsum("ij,jk,ki->i", v.conj().T, O, v))


def projector(code, J, Jc, K, B0):
    w, v = np.linalg.eigh(H_clean(J, Jc, K, B0))
    st2, f2, mz = expval(v, ST2), expval(v, F2), expval(v, F[2])
    if code == "C1":
        sel = np.where((np.abs(st2) < 0.1) & (np.abs(mz - 0.5) < 0.1))[0]
    else:
        sel = np.where((np.abs(st2 - 6) < 0.1) & (np.abs(mz - 0.5) < 0.1)
                       & ((np.abs(f2 - 3.75) < 0.1) | (np.abs(f2 - 8.75) < 0.1)))[0]
    assert len(sel) == 2, (code, len(sel))
    return v[:, sel] @ v[:, sel].conj().T, v[:, sel]


def logical(H, P):
    w, v = np.linalg.eigh(H)
    wt = np.real(np.einsum("ij,jk,ki->i", v.conj().T, P, v))
    top = np.argsort(wt)[-2:]
    top = top[np.argsort(w[top])]
    return w[top[1]] - w[top[0]], wt[top].min()


def sigma_f(code, J, Jc, K, sB=10e-6, r=3e-3, B0=10e-3, Nmc=800):
    P, _ = projector(code, J, Jc, K, B0)
    f, wmin = [], 1.0
    for _ in range(Nmc):
        H = gamma * B0 * F[2]
        for k in range(N):
            loc = rng.normal(0, sB, 3)
            H = H + gamma * sum(loc[a] * S[k][a] for a in range(3))
        for b in edges:
            H = H + J * (1 + r * rng.normal()) * HE[b]
        for b in centre:
            H = H + Jc * (1 + r * rng.normal()) * HE[b]
        for M in CH:
            H = H + K * (1 + r * rng.normal()) * M
        df, wt = logical(H, P)
        f.append(df)
        wmin = min(wmin, wt)
    return np.std(f), wmin


def T2(s):
    return np.sqrt(2) / (2 * np.pi * s)


J, K, B0 = 10e6, 0.3e6, 10e-3
print("=" * 72)
print("1. Spectre (J = 10 MHz, K = 0, B0 = 0) et découplage du central (K, B0 non nuls)")
print("=" * 72)
Jc = 3e6
w, v = np.linalg.eigh(H_clean(J, Jc, 0.0, 0.0))
st2, f2 = expval(v, ST2), expval(v, F2)
seen = {}
for e, a, b in zip(w, st2, f2):
    key = (round(a), round(b, 2), round(e / 1e6, 4))
    seen[key] = seen.get(key, 0) + 1
for (a, b, e), c in sorted(seen.items(), key=lambda x: x[0][2]):
    STv = (-1 + np.sqrt(1 + 4 * a)) / 2
    Fv = (-1 + np.sqrt(1 + 4 * b)) / 2
    th = J / 2 * (STv * (STv + 1) - 3) + Jc / 2 * (Fv * (Fv + 1) - STv * (STv + 1) - 0.75)
    print(f"  S_T = {STv:.0f}, F = {Fv:.1f} : E = {e:+8.3f} MHz (formule {th/1e6:+8.3f}) × {c}")
P1, V1 = projector("C1", J, Jc, 0.3e6, B0)   # K et B0 non nuls : base du code bien définie
m = max(abs(V1[:, i].conj() @ S[0][a] @ V1[:, j]) for i in range(2) for j in range(2) for a in range(3) if i != j)
d = max(abs(expval(V1, S[0][a])[0] - expval(V1, S[0][a])[1]) for a in range(3))
print(f"  C1 : électron central entre les deux singulets — hors diagonale {m:.1e}, écart de diagonale {d:.1e}"
      "\n       → le central ne distingue pas les états du code (découplé, et incapable de les lire)")

print("\n" + "=" * 72)
print("2. C1 : le code du tétraèdre avec l'électron central (K = 0,3 MHz)")
print("=" * 72)
for Jcv in (1e6, 3e6, 6e6, 9e6, 10e6, 11e6, 15e6, 30e6):
    s, wm = sigma_f("C1", J, Jcv, K)
    print(f"  J_c = {Jcv/1e6:5.1f} MHz (J_c/J = {Jcv/J:4.2f}) : T2* = {T2(s)*1e6:7.1f} µs"
          f"  (poids min. {wm:.3f})")
print("  Référence A-13 sans électron : ≈ 72 µs. Croisement exact attendu à J_c = J.")

print("\n" + "=" * 72)
print("3. C2 : horloge F = 3/2 ↔ 5/2 dans le quintet S_T = 2 (ΔM = 0)")
print("=" * 72)
_, V2 = projector("C2", J, 3e6, 0.0, B0)
dz = [expval(V2, S[k][2]) for k in range(N)]
print("  Δ⟨S_z⟩ entre les deux états : central {:+.3f}, sommets ".format(dz[0][1] - dz[0][0])
      + ", ".join(f"{dz[k][1]-dz[k][0]:+.3f}" for k in tet)
      + "\n  → champ uniforme sans effet (même M total, g identiques) ;"
      "\n    écart de champ centre ↔ sommets : sensible au premier ordre")
for Jcv in (1e6, 3e6, 25e6, 50e6):   # 10 et 20 MHz tombent sur des croisements exacts
    s, wm = sigma_f("C2", J, Jcv, 0.0)
    print(f"  J_c = {Jcv/1e6:5.1f} MHz (écart {2.5*Jcv/1e6:5.1f} MHz) : T2* = {T2(s)*1e6:7.2f} µs"
          f"  (poids min. {wm:.3f})")

print("\n" + "=" * 72)
print("4. Électron central seul comme qubit (ΔM = 1), tétraèdre en singulet")
print("=" * 72)
print(f"  T2* = √2/(2π γ σ_B) = {T2(gamma*10e-6)*1e6:.2f} µs : le singulet ne blinde pas le champ.")

print("\n" + "=" * 72)
print("SYNTHÈSE")
print("=" * 72)
print("  L'électron lié au centre ne dégrade pas le code du tétraèdre loin de J_c = J,"
      "\n  mais ne l'améliore pas et ne peut pas le lire (découplage exact)."
      "\n  À J_c ≈ J, le code croise S_T = 1, F = ½ : résonance de fuite."
      "\n  L'horloge « hyperfine » F = 3/2 ↔ 5/2 est insensible au champ uniforme"
      "\n  mais sensible à l'écart centre ↔ sommets : elle ne rivalise pas avec C1.")
