r"""
TQNT-V6 MILNOR — invariants MULTI-COMPOSANTES pour entrelacs TQNT.

L'analyse Kontsevich par v_n individuels (du polynôme de Jones de
chaque nœud) a échoué (gain 1.4%) parce que ces invariants sont
REDONDANTS avec ce qu'utilise Ω_ts.

Le vrai chemin : capturer l'INTERACTION entre les composantes de
l'entrelacs, c'est-à-dire des quantités qui sont DIFFÉRENTES de la
simple somme Σ f(K_i).

Invariants multi-composantes testés :

  (M1) Asymétrie des composantes :  Δv_2 = v_2_max - v_2_min
       Mesure si les nœuds sont "équilibrés" ou non.

  (M2) Variance des croisements :   Var(c_i) = (1/n) Σ (c_i - <c>)²
       Mesure la dispersion topologique.

  (M3) Entropie de complexité :     S = -Σ p_i log p_i, p_i = c_i/Σc_j
       Mélange topologique des composantes.

  (M4) Produit croisé :             P_cross = Π det(K_i)
       Volume topologique en représentation Jones-multiple.

  (M5) Triple discordance :         |v_3(K1)| + |v_3(K2)| + |v_3(K3)| - |v_3_sum|
       Mesure l'annulation des chiralités opposées.

  (M6) Homogénéité :                δ_hom = 1 si tous K_i identiques, 0 sinon
       Particulier des paires (8_1+8_1) et triples (4_1+4_1+4_1).

  (M7) Triple linking proxy :       Π b_i + Σ b_i b_j   (combinatoire de tresses)
       Approximation algébrique du triple linking.

Test : régression LASSO sur ces 7 features + base V5.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin

# ───────── précalcul ─────────
M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A_ANA = np.sqrt(7/(8*np.pi)); P_ANA = 3/4; Q_ANA = 3/4
X_ANA = -np.sqrt(6); B_ANA = 33/20; RHO_0_ANA = 10 - 3/(8*np.pi)

print("Précalcul…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 50000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)), float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]),
               "n_l": N_LOBES[K], "b": braid_index(K),
               "det": jones_determinant(pj), "s": topological_spin(K),
               "jones": pj}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]; T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def omega(comps):
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_ANA for K in comps]
    skel = float(sum(w**A_ANA for w in ws)**(1/A_ANA))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in comps)/P_31
    Phi_n = ((n+1)/2)**(-X_ANA)
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0_ANA * TOPO[K]["cr"]**(-B_ANA))**(ct-1)
    return Phi_n*Phi_p*skel*G**Q_ANA


def jones_derivs_at_1(poly, order=4):
    derivs = []
    for k in range(order + 1):
        val = 0.0
        for p, c in poly.items():
            fact = 1.0
            for i in range(k):
                fact *= (p - i)
            val += c * fact
        derivs.append(val)
    return derivs

VAS = {}
for K in KNOTS:
    d = jones_derivs_at_1(TOPO[K]["jones"], 4)
    VAS[K] = {"v2": -d[2]/6, "v3": -d[3]/36 - d[2]/12, "v4": d[4]/720 + (d[2]/6)**2/2}


def multi_invariants(comps: list[str]) -> dict[str, float]:
    """Calcule les 7 invariants multi-composantes."""
    n = len(comps)
    cs = [TOPO[K]["cr"] for K in comps]
    bs = [TOPO[K]["b"] for K in comps]
    dets = [TOPO[K]["det"] for K in comps]
    v2s = [VAS[K]["v2"] for K in comps]
    v3s = [VAS[K]["v3"] for K in comps]
    cnt = Counter(comps)

    if n == 1:
        # Pas d'invariants multi-composantes définis
        return {"M1_asym_v2": 0.0, "M2_var_c": 0.0, "M3_entropy": 0.0,
                "M4_prod_det": dets[0], "M5_triple_disc": 0.0,
                "M6_homogeneity": 1.0, "M7_braid_triple": float(bs[0])}

    # (M1) Asymétrie des v_2
    asym_v2 = max(v2s) - min(v2s)

    # (M2) Variance des croisements
    var_c = float(np.var(cs))

    # (M3) Entropie de Shannon des croisements
    total_c = sum(cs)
    p = [c/total_c for c in cs if c > 0]
    entropy = -sum(pi * np.log(pi) for pi in p if pi > 0)

    # (M4) Produit des déterminants (volume topologique)
    prod_det = 1.0
    for d in dets:
        prod_det *= d

    # (M5) Triple discordance (annulation de chiralités)
    abs_sum_v3 = sum(abs(v) for v in v3s)
    sum_v3 = abs(sum(v3s))
    triple_disc = abs_sum_v3 - sum_v3   # > 0 si signes contraires

    # (M6) Homogénéité : 1 si tous identiques, 0 sinon (utilité topologique)
    homogeneity = 1.0 if len(set(comps)) == 1 else (1.0/len(set(comps)))

    # (M7) Triple-braid : produit croisé d'indices de tresses
    if n >= 3:
        braid_triple = bs[0] * bs[1] * bs[2]
    elif n == 2:
        braid_triple = bs[0] * bs[1]
    else:
        braid_triple = bs[0]

    return {
        "M1_asym_v2": asym_v2,
        "M2_var_c": var_c,
        "M3_entropy": entropy,
        "M4_prod_det": prod_det,
        "M5_triple_disc": triple_disc,
        "M6_homogeneity": homogeneity,
        "M7_braid_triple": float(braid_triple),
    }


# ───────── PDG ─────────
PDG = [
    ("e⁻",       ["3_1"],              0.51099895),
    ("μ⁻",       ["3_1","6_1"],        105.66),
    ("τ⁻",       ["4_1","5_1","6_1"],  1776.86),
    ("π⁰",       ["3_1","3_1","5_2"],  134.98),
    ("π±",       ["3_1","3_1","5_2"],  139.57),
    ("K±",       ["4_1","4_1","4_1"],  493.68),
    ("K⁰",       ["6_1","6_1","6_1"],  497.61),
    ("η",        ["3_1","3_1","6_1"],  547.86),
    ("η'",       ["7_7","7_7"],        957.78),
    ("ρ",        ["8_1","8_1"],        775.26),
    ("ω",        ["8_1","8_1"],        782.65),
    ("φ",        ["3_1","5_1","5_2"],  1019.46),
    ("K*(892)",  ["6_1","6_2"],        891.66),
    ("J/ψ",      ["3_1","4_1","6_3"],  3096.9),
    ("ψ(2S)",    ["6_1","8_2"],        3686.1),
    ("Υ(1S)",    ["6_2","7_1","7_3"],  9460.30),
    ("Υ(2S)",    ["6_2","6_2","8_2"],  10023.26),
    ("Υ(3S)",    ["7_2","10_1"],       10355.2),
    ("D⁰",       ["3_1","5_1","7_2"],  1864.84),
    ("D±",       ["3_1","5_1","7_2"],  1869.66),
    ("D_s",      ["3_1","6_1","7_1"],  1968.34),
    ("B±",       ["3_1","6_2","7_7"],  5279.34),
    ("B⁰",       ["3_1","6_2","7_7"],  5279.66),
    ("B_s",      ["6_3","7_6"],        5366.92),
    ("χ_c0",     ["6_1","7_7","7_7"],  3414.71),
    ("χ_c1",     ["7_2","8_1","8_1"],  3510.67),
    ("χ_c2",     ["6_1","8_2"],        3556.17),
    ("p",        ["3_1","7_2","7_2"],  938.27),
    ("n",        ["3_1","7_2","7_2"],  939.57),
    ("Λ",        ["5_1","8_1"],        1115.68),
    ("Σ⁺",       ["4_1","4_1","5_1"],  1189.37),
    ("Σ⁰",       ["4_1","4_1","5_1"],  1192.64),
    ("Σ⁻",       ["4_1","4_1","5_1"],  1197.45),
    ("Ξ⁰",       ["4_1","5_1","5_2"],  1314.86),
    ("Ξ⁻",       ["4_1","5_1","5_2"],  1321.71),
    ("Λ_c",      ["3_1","3_1","9_1"],  2286.46),
    ("Σ_c",      ["4_1","5_1","7_2"],  2453.97),
    ("Ξ_c⁺",     ["3_1","4_1","8_1"],  2467.94),
    ("Ξ_c⁰",     ["3_1","4_1","8_1"],  2470.9),
    ("Ω_c",      ["4_1","4_1","7_7"],  2695.2),
    ("Λ_b",      ["6_3","7_4"],        5619.6),
    ("W±",       ["7_1","7_2","8_3"],  80369),
    ("Z⁰",       ["7_2","8_3","9_1"],  91188),
    ("H",        ["6_3","8_2","8_3"],  125250),
    ("t",        ["6_3","7_1","7_5"],  172690),
]

# ───────── analyse ─────────
print("\n" + "═"*82)
print("  TQNT-V6 MILNOR — invariants MULTI-COMPOSANTES")
print("═"*82)

data = []
for nm, cfg, m_exp in PDG:
    m_pred = omega(cfg)
    res_log = np.log(m_exp / m_pred)
    res_pct = (m_exp - m_pred) / m_exp * 100
    inv = multi_invariants(cfg)
    data.append((nm, cfg, m_exp, m_pred, res_log, res_pct, inv))

res_logs = np.array([d[4] for d in data])
INV_NAMES = ["M1_asym_v2", "M2_var_c", "M3_entropy", "M4_prod_det",
             "M5_triple_disc", "M6_homogeneity", "M7_braid_triple"]

print(f"\n  Sur {len(data)} particules :")
print(f"  Corrélations individuelles avec log(m_exp/Ω_ts) :")
print(f"  {'invariant':<22} {'r de Pearson':>14}")
print("  " + "-" * 38)
for name in INV_NAMES:
    xs = np.array([d[6][name] for d in data])
    if np.var(xs) > 0:
        r = np.corrcoef(res_logs, xs)[0, 1]
        sig = "★" if abs(r) > 0.3 else (" " if abs(r) > 0.15 else "·")
        print(f"  {name:<22} {r:>+13.3f} {sig}")
    else:
        print(f"  {name:<22} {'(constant)':>13}")

# Régression multivariée
print(f"\n  ─── RÉGRESSION MULTIVARIÉE ───")
X = np.column_stack([np.array([d[6][n] for d in data]) for n in INV_NAMES])
# Centrer-réduire
X_mean = X.mean(axis=0); X_std = X.std(axis=0); X_std[X_std == 0] = 1.0
X_norm = (X - X_mean) / X_std

y = res_logs
betas, *_ = np.linalg.lstsq(X_norm, y, rcond=None)
y_pred = X_norm @ betas
res_corr = y - y_pred

rms_before = np.sqrt(np.mean(y**2))
rms_after = np.sqrt(np.mean(res_corr**2))
rms_pct_before = np.sqrt(np.mean(np.array([d[5] for d in data])**2))
rms_pct_after = rms_pct_before * (rms_after/rms_before)

print(f"\n  Coefficients (en log/std-unit) :")
for name, beta in zip(INV_NAMES, betas):
    print(f"    β_{name:<20} = {beta:+.6f}")

print(f"\n  Résultats :")
print(f"    RMS V5 brut            : {rms_pct_before:.3f}%")
print(f"    RMS V5 + Milnor 7 inv. : {rms_pct_after:.3f}%")
print(f"    Réduction              : {(1 - rms_after/rms_before)*100:.1f}%")

# Régressions partielles : ajouter une features à la fois
print(f"\n  ─── RÉGRESSIONS PARTIELLES (greedy forward selection) ───")
selected = []
remaining = list(INV_NAMES)
current_rms = rms_before
print(f"  {'modèle':<60} {'RMS log':>10} {'RMS %':>9} {'gain':>7}")
print(f"  Ω_ts seul (V5){'':<46} {rms_before:>10.5f} {rms_pct_before:>8.3f}% {'+0.0%':>7}")
while remaining:
    best_rms = float('inf'); best_name = None
    for name in remaining:
        idx_list = [INV_NAMES.index(n) for n in selected + [name]]
        X_part = X_norm[:, idx_list]
        b, *_ = np.linalg.lstsq(X_part, y, rcond=None)
        rms = np.sqrt(np.mean((y - X_part @ b)**2))
        if rms < best_rms:
            best_rms = rms; best_name = name
    selected.append(best_name); remaining.remove(best_name)
    gain = (1 - best_rms/rms_before)*100
    rms_p = best_rms / rms_before * rms_pct_before
    sel_str = " + ".join(selected)[:55]
    print(f"  + {sel_str:<58} {best_rms:>10.5f} {rms_p:>8.3f}% {gain:>+6.1f}%")
    current_rms = best_rms

# ─── effets particule par particule (top 5 features) ───
print(f"\n  ─── EFFET sur cas problématiques (top 5 features) ───")
top5_idx = [INV_NAMES.index(n) for n in selected[:5]]
X_top5 = X_norm[:, top5_idx]
b, *_ = np.linalg.lstsq(X_top5, y, rcond=None)
y_pred_top5 = X_top5 @ b
print(f"  {'particule':<10} {'Δ V5 %':>10} {'Δ V6 %':>10} {'gain':>8}")
problematic = ["π⁰","π±","η","η'","φ","ψ(2S)","ρ","K*(892)","D⁰","D±"]
for i, (nm, _, _, _, res_l, res_pct, _) in enumerate(data):
    if nm in problematic:
        res_corr_pct = (np.exp(res_l - y_pred_top5[i]) - 1) * 100
        gain = abs(res_pct) - abs(res_corr_pct)
        print(f"  {nm:<10} {res_pct:>+9.2f}% {res_corr_pct:>+9.2f}% {gain:>+7.2f}")

print("\n" + "═"*82)
print("  CONCLUSION")
print("═"*82)
print(f"""
  Test : 7 invariants MULTI-COMPOSANTES (au-delà de v_n single-knot).

  RÉSULTATS :
    • RMS V5 brut          : {rms_pct_before:.3f}%
    • RMS V5 + 7 invariants: {rms_pct_after:.3f}%
    • Gain                 : {(1 - rms_after/rms_before)*100:.1f}%

  Interprétation :
    Si gain >> Kontsevich (1.4%) : on a trouvé l'info manquante.
    Sinon : il faut aller plus loin (triple linking number, HOMFLY...).

  Top features par sélection greedy :
    1er : {selected[0] if selected else 'aucun'}
    2e  : {selected[1] if len(selected) > 1 else 'aucun'}
    3e  : {selected[2] if len(selected) > 2 else 'aucun'}
""")
