# -*- coding: utf-8 -*-
"""A-18 — Un électron lié à une FACE du tétraèdre de trèfles (17/09/2026).

Suite de A-17 (électron au centre : découplé, résonance de fuite à J_c = J,
incapable de lire le code). Sur la face (1,2,3), l'électron est équidistant
de trois sommets et plus loin du quatrième : symétrie réduite de Td à C3v.
  H = J Σ_arêtes S_i·S_j + Σ_i J_i S_0·S_i + K·chiralité + γB0·F_z + bruit
  J_1 = J_f(1+η), J_2 = J_3 = J_f(1−η/2), J_4 = λ·J_f
λ < 1 : sommet opposé plus lointain ; η : déplacement de l'électron vers le
sommet 1 (brise C3v, pour tester la lecture).
Questions : (1) le code des deux singulets reste-t-il protégé ?
(2) l'électron peut-il distinguer les deux états (lecture) ? à quel coût ?
Signal de lecture : décalage conditionnel χ = différence, entre les deux états
logiques, de la fréquence de retournement de l'électron. THÉORÈME (vérifié) :
avec des couplages isotropes (Heisenberg, chiralité), χ = 0 exactement pour
toute géométrie — le spin total est conservé et le doublet F = ½ ne se sépare
que par Zeeman. La lecture exige un couplage ANISOTROPE : on ajoute le couplage
dipolaire électron–sommets et sommets–sommets, calculé sur des positions réelles
(tétraèdre d'arête a, électron au-dessus de la face (1,2,3), déplacé vers le
sommet 1 d'une fraction η du rayon de la face).
Bruit comme A-11..A-17 ; N = 800 tirages.
"""
import numpy as np
from itertools import combinations

rng = np.random.default_rng(20260917)
gamma = 28.025e9
sx = np.array([[0, 1], [1, 0]]) / 2
sy = np.array([[0, -1j], [1j, 0]]) / 2
sz = np.array([[1, 0], [0, -1]]) / 2
I2 = np.eye(2)
N = 5


def op(single, k):
    m = np.array([[1.0]])
    for i in range(N):
        m = np.kron(m, single if i == k else I2)
    return m


S = [[op(s, k) for s in (sx, sy, sz)] for k in range(N)]
tet = [1, 2, 3, 4]
ST = [sum(S[k][a] for k in tet) for a in range(3)]
ST2 = sum(ST[a] @ ST[a] for a in range(3))
Fz = sum(S[k][2] for k in range(N))
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

# sous-espace nu du code : S_T = 0, M = +½
w0, v0 = np.linalg.eigh(ST2 + 10 * (Fz - 0.5 * np.eye(2**N)) @ (Fz - 0.5 * np.eye(2**N)))
bare = v0[:, np.argsort(w0)[:2]]
P_bare = bare @ bare.conj().T

# base nue étiquetée : chiralités du tétraèdre (vecteurs propres de la chiralité
# dans les singulets) ⊗ électron ↑ ou ↓ — pour le décalage conditionnel χ
CHsum = sum(CH)
LAB = {}
for mz in (+0.5, -0.5):
    w1, v1 = np.linalg.eigh(ST2 + 10 * (Fz - mz * np.eye(2**N)) @ (Fz - mz * np.eye(2**N)))
    B = v1[:, np.argsort(w1)[:2]]
    wc_, uc_ = np.linalg.eigh(B.conj().T @ CHsum @ B)
    LAB[mz] = B @ uc_                      # colonnes : chiralité −, +


def couplings(Jf, lam, eta):
    return [Jf * (1 + eta), Jf * (1 - eta / 2), Jf * (1 - eta / 2), lam * Jf]


def H_build(J, Jc, K, B0, noise=None):
    H = gamma * B0 * Fz
    if noise is not None:
        sB, r = noise
        for k in range(N):
            loc = rng.normal(0, sB, 3)
            H = H + gamma * sum(loc[a] * S[k][a] for a in range(3))
    else:
        r = 0.0
    for b in edges:
        H = H + J * (1 + r * rng.normal()) * HE[b]
    for (b, Jv) in zip(centre, Jc):
        H = H + Jv * (1 + r * rng.normal()) * HE[b]
    for M in CH:
        H = H + K * (1 + r * rng.normal()) * M
    return H


def dressed(H, P):
    w, v = np.linalg.eigh(H)
    wt = np.real(np.einsum("ij,jk,ki->i", v.conj().T, P, v))
    top = np.argsort(wt)[-2:]
    top = top[np.argsort(w[top])]
    return w[top], v[:, top], wt[top]


def chi_shift(H):
    """Fréquence de retournement de l'électron selon l'état logique : χ = f_e(1) − f_e(2)."""
    w, v = np.linalg.eigh(H)
    E = {}
    for mz in (+0.5, -0.5):
        for c in range(2):
            ov = np.abs(v.conj().T @ LAB[mz][:, c])**2
            E[(mz, c)] = w[np.argmax(ov)]
    f1 = E[(-0.5, 0)] - E[(+0.5, 0)]
    f2 = E[(-0.5, 1)] - E[(+0.5, 1)]
    return f1 - f2


def T2(s):
    return np.sqrt(2) / (2 * np.pi * s)


def study(J, Jf, lam, eta, K, B0=10e-3, sB=10e-6, r=3e-3, Nmc=800):
    Hc = H_build(J, couplings(Jf, lam, eta), K, B0)
    wc, vc, wtc = dressed(Hc, P_bare)
    Pd = vc @ vc.conj().T
    # écart induit à K = 0 (indépendant du spin de l'électron) et décalage conditionnel χ
    H0 = H_build(J, couplings(Jf, lam, eta), 0.0, B0)
    w0_, v0_, _ = dressed(H0, P_bare)
    delta = w0_[1] - w0_[0]
    dsz = chi_shift(Hc)
    f = []
    wmin = 1.0
    for _ in range(Nmc):
        w, v, wt = dressed(H_build(J, couplings(Jf, lam, eta), K, B0, noise=(sB, r)), Pd)
        f.append(w[1] - w[0])
        wmin = min(wmin, wt.min())
    return T2(np.std(f)), delta, dsz, 1 - wtc.min(), wmin


J, K = 10e6, 0.3e6
print("=" * 72)
print("1. Électron sur la face, symétrie C3v (η = 0) — protection et lecture")
print("=" * 72)
print("  (J = 10 MHz, K = 0,3 MHz ; A-13 sans électron ≈ 72 µs ; A-17 centre ≈ 72 µs hors résonance)")
for Jf in (1e6, 3e6, 6e6, 30e6):
    for lam in (0.3, 0.1):
        t2, d, dsz, adm, wm = study(J, Jf, lam, 0.0, K)
        print(f"  J_f = {Jf/1e6:4.0f} MHz, λ = {lam:.1f} : T2* = {t2*1e6:6.1f} µs ;"
              f" décalage conditionnel χ = {abs(dsz):.1e} Hz ;"
              f" habillage {adm:.1e} ; poids min. {wm:.2f}")
print("  → sous C3v, les deux singulets restent une paire E dégénérée : l'électron ne"
      "\n    les distingue toujours pas (χ = 0) ; mais l'habillage par S_T = 1 croît"
      "\n    avec J_f et la cohérence baisse (contrairement au centre, découplé exactement).")

print("\n" + "=" * 72)
print("2. Recherche de la résonance sur la face (λ = 0,1)")
print("=" * 72)
for Jf in (7e6, 9e6, 10e6, 11e6, 12e6, 14e6):
    t2, d, dsz, adm, wm = study(J, Jf, 0.1, 0.0, K)
    print(f"  J_f = {Jf/1e6:4.0f} MHz : T2* = {t2*1e6:6.1f} µs ; poids min. {wm:.2f}")

print("\n" + "=" * 72)
print("3. Briser C3v pour lire : électron déplacé vers le sommet 1 (J_f = 3 MHz, λ = 0,1)")
print("=" * 72)
for eta in (0.0, 0.01, 0.03, 0.1, 0.3):
    t2, d, dsz, adm, wm = study(J, 3e6, 0.1, eta, K)
    print(f"  η = {eta:4.2f} : écart induit δ = {abs(d)/1e3:8.2f} kHz (sans lecture),"
          f" décalage conditionnel χ = {abs(dsz):9.2f} Hz, T2* = {t2*1e6:6.1f} µs, χ·T2* = {abs(dsz)*t2:.1e}")
print("  Lecture possible si χ·T2* ≳ 1 (l'électron distingue les deux états avant brouillage).")

print("\n" + "=" * 72)
print("4. Avec le couplage dipolaire (anisotrope), positions réelles")
print("=" * 72)
C_dip = 52.04e6 * 1e-27                        # Hz·m³ : μ0/4π·(gμ_B)²/h
verts = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
verts -= verts.mean(axis=0)


def geometry(a_nm, h_frac, eta):
    V = verts / np.linalg.norm(verts[0] - verts[1]) * a_nm * 1e-9
    fc = V[:3].mean(axis=0)
    nrm = fc / np.linalg.norm(fc)
    R = np.linalg.norm(V[0] - fc)
    e = fc + nrm * h_frac * a_nm * 1e-9 + eta * R * (V[0] - fc) / R
    return {0: e, 1: V[0], 2: V[1], 3: V[2], 4: V[3]}


def H_dip(pos, pairs):
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i, j in pairs:
        rv = pos[j] - pos[i]
        r = np.linalg.norm(rv)
        u = rv / r
        Sdot = sum(S[i][k] @ S[j][k] for k in range(3))
        Si_u = sum(u[k] * S[i][k] for k in range(3))
        Sj_u = sum(u[k] * S[j][k] for k in range(3))
        H = H + C_dip / r**3 * (Sdot - 3 * Si_u @ Sj_u)
    return H


pairs = [(0, k) for k in tet] + edges
for a_nm in (2.0, 5.0):
    print(f"  arête a = {a_nm:.0f} nm (dipolaire sur une arête : {C_dip/(a_nm*1e-9)**3/1e6:.2f} MHz)")
    for eta in (0.0, 0.1, 0.3):
        pos = geometry(a_nm, 0.3, eta)
        Hd = H_dip(pos, pairs)
        Hc = H_build(J, couplings(3e6, 0.1, eta), K, 10e-3) + Hd
        chi = chi_shift(Hc)
        _, vc, _ = dressed(Hc, P_bare)
        Pd = vc @ vc.conj().T
        fl = []
        for _ in range(500):
            Hn = H_build(J, couplings(3e6, 0.1, eta), K, 10e-3, noise=(10e-6, 3e-3)) + Hd
            ww, vv, _ = dressed(Hn, Pd)
            fl.append(ww[1] - ww[0])
        t2 = T2(np.std(fl))
        print(f"    η = {eta:.1f} : χ = {abs(chi):9.3f} Hz ; T2* = {t2*1e6:6.1f} µs ; χ·T2* = {abs(chi)*t2:.1e}")
print("  Lecture exploitable si χ·T2* ≳ 1. (À η = 0, χ ≠ 0 car le champ B0 selon z"
      "\n  n'est pas aligné sur l'axe de la face : l'orientation brise déjà C3v.)")
print("  Balayage de la taille (η = 0) : compromis signal / cohérence")
best = None
for a_nm in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
    pos = geometry(a_nm, 0.3, 0.0)
    Hd = H_dip(pos, pairs)
    Hc = H_build(J, couplings(3e6, 0.1, 0.0), K, 10e-3) + Hd
    chi = chi_shift(Hc)
    _, vc, _ = dressed(Hc, P_bare)
    Pd = vc @ vc.conj().T
    fl = []
    for _ in range(500):
        Hn = H_build(J, couplings(3e6, 0.1, 0.0), K, 10e-3, noise=(10e-6, 3e-3)) + Hd
        ww, vv, _ = dressed(Hn, Pd)
        fl.append(ww[1] - ww[0])
    t2 = T2(np.std(fl))
    fom = abs(chi) * t2
    best = max(best or (0, 0, 0, 0), (fom, a_nm, abs(chi), t2))
    print(f"    a = {a_nm:.1f} nm : χ = {abs(chi)/1e3:9.3f} kHz ; T2* = {t2*1e6:6.1f} µs ; χ·T2* = {fom:.2e}")
print(f"  meilleur compromis : a ≈ {best[1]:.1f} nm, χ ≈ {best[2]/1e3:.1f} kHz, T2* ≈ {best[3]*1e6:.1f} µs, χ·T2* ≈ {best[0]:.1f}")

print("\n" + "=" * 72)
print("SYNTHÈSE")
print("=" * 72)
print("  Sur la face, le code reste protégé mais l'habillage coûte de la cohérence."
      "\n  Couplages isotropes : χ = 0 exactement, quelle que soit la géométrie (spin total"
      "\n  conservé) — l'électron ne lit jamais. Seul un couplage anisotrope (dipolaire)"
      "\n  peut créer un signal ; il ne permet de lire (χ·T2* ≳ 1) qu'à a ≈ 2 nm, où la"
      "\n  cohérence retombe au niveau d'un spin nu (~1 µs). Lire = déprotéger."
      "\n  Réserves : échanges fixés indépendamment de a ; étiquetage approximatif à 2 nm.")
