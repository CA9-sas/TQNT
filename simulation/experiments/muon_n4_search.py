r"""
RECHERCHE EXHAUSTIVE — le muon est-il un entrelacs n=4 ?

V5 actuel : μ⁻ = 3_1+6_1 (n=2), Ω_ts = 103.26 MeV, écart +2.27%.
Hypothèse alternative : μ⁻ pourrait être une configuration n=4
mieux ajustée à la masse expérimentale m_μ = 105.6583755 MeV.

Critères de match :
  • Masse Ω_ts dans [102, 110] MeV (±2.5% autour de m_μ)
  • Spin J=1/2 accessible via P7quater (Clebsch-Gordan)
  • Charge Q=−1 atteignable par Lk entier ∈ [−5, +5]

Énumération : combinaisons avec répétition de 4 nœuds parmi 19
            = C(22, 4) = 7315 configurations.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations_with_replacement
from theory.abraham import frenet_geometry
from theory.jones import (jones_polynomial, jones_determinant, braid_index,
                          topological_spin, topological_charge)

# ───────── précalcul ─────────
M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A = np.sqrt(7/(8*np.pi)); P = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); B = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul Frenet…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 50000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)), float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]),
               "n_l": N_LOBES[K], "b": braid_index(K),
               "det": jones_determinant(pj),
               "s": topological_spin(K), "q": topological_charge(K)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]; T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def omega(comps):
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in comps)/P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B))**(ct-1)
    return Phi_n*Phi_p*skel*G**Q_EXP


def J_set(comps):
    """P7quater : J accessibles par couplage Clebsch-Gordan."""
    s_max = sum(TOPO[K]["s"] for K in comps)
    parity = s_max - int(s_max)
    n_steps = int(round(s_max - parity)) + 1
    return {parity + i for i in range(n_steps)}


def Q_topo(comps):
    return sum(TOPO[K]["q"] for K in comps)


# Cible
M_MU = 105.6583755
J_MU = 0.5
Q_MU = -1

# ───────── ÉNUMÉRATION n=4 ─────────
print("\n" + "═"*82)
print("  RECHERCHE EXHAUSTIVE n=4 pour le muon (m=105.66 MeV, J=1/2, Q=-1)")
print("═"*82)

print(f"\n  Φ_n factor :")
for n in [1, 2, 3, 4]:
    phi = ((n+1)/2)**(-X_EXP)
    print(f"    n={n} : Φ_n = ({n+1}/2)^√6 = {phi:.3f}")

print(f"\n  → Pour n=4, Φ_4 = {((5/2)**(-X_EXP)):.3f} (vs n=2 : {((3/2)**(-X_EXP)):.3f})")
print(f"  → m(n=4) ≈ {((5/2)**(-X_EXP)) / ((3/2)**(-X_EXP)):.2f}x m(n=2) pour mêmes nœuds.")
print(f"  → Pour atteindre 105 MeV à n=4, il faut des nœuds très légers.")

print(f"\n  Énumération des configurations n=4 avec répétition…")
n_total = 0; n_in_window = 0
results = []
WIN_LOW = 95; WIN_HIGH = 116  # ±10% autour de m_μ
for combo in combinations_with_replacement(KNOTS, 4):
    cfg = list(combo); n_total += 1
    m = omega(cfg)
    if WIN_LOW <= m <= WIN_HIGH:
        Js = J_set(cfg)
        if J_MU not in Js: continue
        Q_t = Q_topo(cfg)
        residue = Q_MU - Q_t
        lk = round(residue)
        if abs(lk) > 5 or abs(residue - lk) > 0.34: continue
        err = (m - M_MU)/M_MU*100
        n_in_window += 1
        results.append((cfg, m, err, Js, Q_t, lk))

results.sort(key=lambda x: abs(x[2]))
print(f"  Configurations testées : {n_total}")
print(f"  Configurations dans fenêtre m, J, Q : {len(results)}")

# Afficher les meilleures n=4
print(f"\n  ─── TOP 15 CONFIGS n=4 POUR μ⁻ ───")
print(f"  {'config':<26} {'m (MeV)':>10} {'Δ %':>8} {'J_set':>22} {'Q_topo':>8} {'Lk':>4}")
print("  " + "-" * 88)
for cfg, m, err, Js, Qt, lk in results[:15]:
    Js_str = "{" + ", ".join(f"{j:.1f}" for j in sorted(Js)) + "}"
    if len(Js_str) > 22: Js_str = Js_str[:19] + "…}"
    print(f"  {'+'.join(cfg):<26} {m:>10.3f} {err:>+7.2f}% {Js_str:>22} {Qt:>+7.2f} {lk:>+3d}")

# ───────── COMPARAISON RAPIDE n=2 vs n=4 pour le muon ─────────
print(f"\n  ─── COMPARAISON HYPOTHÈSES ───")
m_n2_v5 = omega(["3_1","6_1"])
print(f"  V5 actuelle (n=2) : 3_1+6_1 → {m_n2_v5:.3f} MeV (Δ = {(m_n2_v5-M_MU)/M_MU*100:+.2f}%)")
if results:
    best = results[0]
    print(f"  V6 candidat (n=4) : {'+'.join(best[0])} → {best[1]:.3f} MeV (Δ = {best[2]:+.2f}%)")
    gain_factor = abs(m_n2_v5 - M_MU) / abs(best[1] - M_MU) if abs(best[1] - M_MU) > 0.01 else 1e6
    print(f"  Facteur d'amélioration : {gain_factor:.1f}×")

# ───────── ANALYSE DES CONFIGS PROCHES ─────────
print(f"\n  ─── ANALYSE STRUCTURELLE DES MEILLEURS CANDIDATS ───")
if len(results) >= 3:
    for cfg, m, err, Js, Qt, lk in results[:5]:
        nodes_used = set(cfg)
        c_total = sum(TOPO[K]["cr"] for K in cfg)
        bs = [TOPO[K]["b"] for K in cfg]
        dets = [TOPO[K]["det"] for K in cfg]
        print(f"\n  {'+'.join(cfg)}  → {m:.3f} MeV ({err:+.2f}%)")
        print(f"    nœuds distincts : {len(nodes_used)}")
        print(f"    croisements tot : c = {c_total}")
        print(f"    indices tresse  : {bs}")
        print(f"    déterminants    : {dets}")

# ───────── TEST PHYSIQUE : cohérence avec hiérarchie leptonique ─────────
print("\n" + "─"*82)
print("  COHÉRENCE AVEC LA HIÉRARCHIE LEPTONIQUE TQNT")
print("─"*82)
print(f"""
  V5 actuelle (lepton family) :
    e⁻   = 3_1                    (n=1)  m = 0.511 MeV   ★ exact
    μ⁻   = 3_1+6_1                (n=2)  m = 103.3 MeV   Δ = +2.27%
    τ⁻   = 4_1+5_1+6_1            (n=3)  m = 1776 MeV    Δ = +0.04%

  Pattern : n = 1, 2, 3 pour génération 1, 2, 3 (NATUREL).

  Si μ⁻ devient n=4, on aurait :
    e⁻   = 3_1                    (n=1)
    μ⁻   = [n=4]                  (saut !)
    τ⁻   = 4_1+5_1+6_1            (n=3)

  → Brisure du pattern naturel 1-2-3 → 1-4-3 = ANTI-PHYSIQUE.

  À moins de proposer aussi un nouveau τ⁻ à n=5 ou n=6...
""")

# ───────── ÉCOULEMENT MASSIQUE n=2 vs n=4 ─────────
print("─"*82)
print("  POURQUOI n=4 EST DÉFAVORISÉ POUR DES MASSES LÉGÈRES")
print("─"*82)
print(f"""
  Mass de référence (n=2 simple) :
    M_min n=2 (3_1+3_1)  = {omega(['3_1','3_1']):.2f} MeV
    M_min n=2 (3_1+4_1)  = {omega(['3_1','4_1']):.2f} MeV

  Mass de référence (n=4 simple) :
    M_min n=4 (4 × 3_1) = {omega(['3_1','3_1','3_1','3_1']):.2f} MeV
    M_min n=4 (3_1×3 + 4_1) = {omega(['3_1','3_1','3_1','4_1']):.2f} MeV

  → À cause de Φ_4 ≈ 9.5 et de l'écrantage par paires, la masse
    minimale n=4 est typiquement 3-5× plus grande que la masse
    minimale n=2 pour les mêmes nœuds.

  → Pour avoir m=105 MeV à n=4, il faut une RECETTE TRÈS PARTICULIÈRE
    (peut-être 4 nœuds très simples avec écrantage maximal).
""")
