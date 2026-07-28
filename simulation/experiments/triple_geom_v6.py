r"""
TQNT-V6 GÉOMÉTRIQUE — Frenet par composante + combinaisons triples.

Au lieu d'utiliser uniquement Σ κ² et Σ τ² (V5), on utilise les
distributions individuelles par composante :

  Pour chaque K_i : U_κ(K_i), U_τ(K_i)
  Pour l'entrelacs L = ⊔ K_i :
    • Sommes :    Σ U_κ, Σ U_τ                       (déjà en V5)
    • Produits :  Π U_κ, Π U_τ                       (NOUVEAU triple)
    • Asymétries: max U_κ - min U_κ                  (NOUVEAU)
    • Variances : Var(U_κ), Var(U_τ)                 (NOUVEAU)
    • Croisés :   Σ U_κ · Σ U_τ, Π (U_κ + U_τ)       (NOUVEAU)
    • Triple :    Σ_{i<j<k} U_κ(i)U_κ(j)U_κ(k)       (NOUVEAU - triple symétrique)

Ces invariants TRIPLE-COMPOSANTES sont algébriquement indépendants
des sommes utilisées dans V5.

Test : régression LASSO avec validation croisée pour éviter le surfit.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin

M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A_ANA = np.sqrt(7/(8*np.pi)); P_ANA = 3/4; Q_ANA = 3/4
X_ANA = -np.sqrt(6); B_ANA = 33/20; RHO_0_ANA = 10 - 3/(8*np.pi)

print("Précalcul Frenet…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 50000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)), float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]),
               "n_l": N_LOBES[K], "b": braid_index(K),
               "det": jones_determinant(pj), "s": topological_spin(K)}
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


def triple_geometric(comps: list[str]) -> dict[str, float]:
    """Invariants géométriques de l'entrelacs L = ⊔ K_i."""
    Uk = [U_KT[K][0] for K in comps]   # U_κ par composante
    Ut = [U_KT[K][1] for K in comps]   # U_τ par composante
    bs = [TOPO[K]["b"] for K in comps]
    dets = [TOPO[K]["det"] for K in comps]
    cs = [TOPO[K]["cr"] for K in comps]
    n = len(comps)

    result = {}
    # (G1) Produit pur de U_κ (triple si n=3, sinon paire)
    prod_Uk = 1.0
    for u in Uk: prod_Uk *= u
    result["G1_prod_Uk"] = prod_Uk
    # (G2) Produit pur de U_τ
    prod_Ut = 1.0
    for u in Ut: prod_Ut *= u
    result["G2_prod_Ut"] = prod_Ut
    # (G3) Asymétrie U_κ
    result["G3_asym_Uk"] = (max(Uk) - min(Uk)) if n > 1 else 0
    # (G4) Asymétrie U_τ
    result["G4_asym_Ut"] = (max(Ut) - min(Ut)) if n > 1 else 0
    # (G5) Variance U_κ
    result["G5_var_Uk"] = float(np.var(Uk)) if n > 1 else 0
    # (G6) Variance U_τ
    result["G6_var_Ut"] = float(np.var(Ut)) if n > 1 else 0
    # (G7) Cross : Σ U_κ × Σ U_τ
    result["G7_cross_sum"] = sum(Uk) * sum(Ut)
    # (G8) Triple symétrique (n=3 only) : Σ_{i<j<k}
    if n == 3:
        result["G8_triple_sym_Uk"] = Uk[0]*Uk[1]*Uk[2]
        result["G8b_triple_sym_Ut"] = Ut[0]*Ut[1]*Ut[2]
    else:
        result["G8_triple_sym_Uk"] = 0
        result["G8b_triple_sym_Ut"] = 0
    # (G9) Triple braid
    prod_b = 1.0
    for b in bs: prod_b *= b
    result["G9_prod_braid"] = prod_b
    # (G10) Entropie de Shannon des U_κ
    total_uk = sum(Uk)
    p = [u/total_uk for u in Uk if u > 0]
    result["G10_entropy_Uk"] = -sum(pi * np.log(pi) for pi in p if pi > 0) if len(p) > 1 else 0
    # (G11) Coefficient de Pearson Uk-Ut
    if n > 1 and np.std(Uk) > 0 and np.std(Ut) > 0:
        result["G11_corr_kt"] = float(np.corrcoef(Uk, Ut)[0, 1])
    else:
        result["G11_corr_kt"] = 0
    # (G12) Geometric mean
    if all(u > 0 for u in Uk):
        result["G12_gmean_Uk"] = float(np.prod(Uk)**(1/n))
    else:
        result["G12_gmean_Uk"] = 0
    # (G13) Harmonic mean
    if all(u > 0 for u in Uk):
        result["G13_hmean_Uk"] = float(n / sum(1/u for u in Uk))
    else:
        result["G13_hmean_Uk"] = 0
    # (G14) Skewness of Uk (asymétrie statistique)
    if n > 1 and np.std(Uk) > 0:
        m = np.mean(Uk); s = np.std(Uk)
        result["G14_skew_Uk"] = float(np.mean([(u-m)**3 for u in Uk])/s**3) if s > 0 else 0
    else:
        result["G14_skew_Uk"] = 0
    # (G15) Determinant-cross : det₁·U_κ₂ + cyclic
    if n == 3:
        result["G15_det_Uk_cross"] = dets[0]*Uk[1] + dets[1]*Uk[2] + dets[2]*Uk[0]
    elif n == 2:
        result["G15_det_Uk_cross"] = dets[0]*Uk[1] + dets[1]*Uk[0]
    else:
        result["G15_det_Uk_cross"] = dets[0]*Uk[0]
    return result


# PDG
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

print("\n" + "═"*82)
print("  TQNT-V6 GÉOMÉTRIQUE — invariants tri-corps de Frenet")
print("═"*82)

# Construire les features
data = []
for nm, cfg, m_exp in PDG:
    m_pred = omega(cfg)
    res_log = np.log(m_exp / m_pred)
    res_pct = (m_exp - m_pred) / m_exp * 100
    tg = triple_geometric(cfg)
    data.append((nm, cfg, m_exp, m_pred, res_log, res_pct, tg))

FEAT_NAMES = list(data[0][6].keys())
print(f"\n  {len(FEAT_NAMES)} features géométriques testées.")

# Corrélations individuelles
res_logs = np.array([d[4] for d in data])
print(f"\n  Corrélations individuelles avec log(m_exp/Ω_ts) :")
print(f"  {'feature':<28} {'r':>8}")
print("  " + "-" * 36)
ranked = []
for name in FEAT_NAMES:
    xs = np.array([d[6][name] for d in data])
    if np.var(xs) > 0:
        r = np.corrcoef(res_logs, xs)[0, 1]
        ranked.append((name, r))
ranked.sort(key=lambda x: -abs(x[1]))
for name, r in ranked[:10]:
    sig = "★★" if abs(r) > 0.4 else ("★" if abs(r) > 0.25 else " ")
    print(f"  {name:<28} {r:>+7.3f} {sig}")

# Standardiser et faire la régression complète
X = np.column_stack([np.array([d[6][n] for d in data]) for n in FEAT_NAMES])
X_mean = X.mean(axis=0); X_std = X.std(axis=0); X_std[X_std == 0] = 1
X_norm = (X - X_mean) / X_std

# Régression complète
betas, *_ = np.linalg.lstsq(X_norm, res_logs, rcond=None)
y_pred = X_norm @ betas
res_corr = res_logs - y_pred
rms_before = np.sqrt(np.mean(res_logs**2))
rms_after = np.sqrt(np.mean(res_corr**2))
rms_pct_before = np.sqrt(np.mean(np.array([d[5] for d in data])**2))
rms_pct_after = rms_pct_before * (rms_after/rms_before)

print(f"\n  ─── RÉGRESSION COMPLÈTE ({len(FEAT_NAMES)} features) ───")
print(f"    RMS V5 brut             : {rms_pct_before:.3f}%")
print(f"    RMS V5 + 15 géom        : {rms_pct_after:.3f}%")
print(f"    Réduction               : {(1 - rms_after/rms_before)*100:.1f}%")

# Forward selection greedy
print(f"\n  ─── FORWARD SELECTION (top 8 features) ───")
print(f"  {'modèle (cumul)':<60} {'RMS log':>10} {'RMS %':>9} {'gain':>7}")
print(f"  {'Ω_ts seul (V5)':<60} {rms_before:>10.5f} {rms_pct_before:>8.3f}% {'+0.0%':>7}")
selected = []; remaining = list(FEAT_NAMES)
for step in range(8):
    best_rms = float('inf'); best_name = None
    for name in remaining:
        idx_list = [FEAT_NAMES.index(n) for n in selected + [name]]
        X_part = X_norm[:, idx_list]
        b, *_ = np.linalg.lstsq(X_part, res_logs, rcond=None)
        rms = np.sqrt(np.mean((res_logs - X_part @ b)**2))
        if rms < best_rms:
            best_rms = rms; best_name = name
    selected.append(best_name); remaining.remove(best_name)
    gain = (1 - best_rms/rms_before)*100
    rms_p = best_rms / rms_before * rms_pct_before
    sel_str = " + ".join(selected)[:55]
    print(f"  + {sel_str:<58} {best_rms:>10.5f} {rms_p:>8.3f}% {gain:>+6.1f}%")

# Effets particule par particule (top 8)
print(f"\n  ─── EFFET sur cas problématiques (top 8) ───")
top8_idx = [FEAT_NAMES.index(n) for n in selected]
X_top = X_norm[:, top8_idx]
b, *_ = np.linalg.lstsq(X_top, res_logs, rcond=None)
y_pred_top = X_top @ b
print(f"  {'particule':<10} {'Δ V5 %':>10} {'Δ V6 %':>10} {'gain':>8}")
problematic = ["π⁰","π±","η","η'","φ","ψ(2S)","ρ","K*(892)","D⁰","D±","Λ","Σ⁺","B±","χ_c1","Σ_b"]
for i, (nm, _, _, _, rlog, rpct, _) in enumerate(data):
    if nm in problematic:
        res_pct_corr = (np.exp(rlog - y_pred_top[i]) - 1) * 100
        gain = abs(rpct) - abs(res_pct_corr)
        flag = "✓" if gain > 0 else "✗"
        print(f"  {nm:<10} {rpct:>+9.2f}% {res_pct_corr:>+9.2f}% {gain:>+7.2f} {flag}")

# Validation croisée k-fold
print(f"\n  ─── VALIDATION CROISÉE 5-FOLD (pour vérifier le sur-ajustement) ───")
np.random.seed(42)
indices = np.arange(len(data))
np.random.shuffle(indices)
k = 5
folds = np.array_split(indices, k)
cv_errors = []
for n_feat in range(1, min(9, len(FEAT_NAMES)+1)):
    fold_errors = []
    for j in range(k):
        test_idx = folds[j]
        train_idx = np.concatenate([folds[i] for i in range(k) if i != j])
        # Sélection forward sur train
        sel = []
        for _ in range(n_feat):
            best_rms = float('inf'); best_idx = None
            for fi in range(len(FEAT_NAMES)):
                if fi in sel: continue
                idx_list = sel + [fi]
                X_train = X_norm[train_idx][:, idx_list]
                y_train = res_logs[train_idx]
                bb, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
                rms = np.sqrt(np.mean((y_train - X_train @ bb)**2))
                if rms < best_rms:
                    best_rms = rms; best_idx = fi
            sel.append(best_idx)
        # Évaluer sur test
        X_train = X_norm[train_idx][:, sel]
        y_train = res_logs[train_idx]
        bb, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        X_test = X_norm[test_idx][:, sel]
        y_test = res_logs[test_idx]
        err = np.mean((y_test - X_test @ bb)**2)
        fold_errors.append(err)
    cv_rmse = np.sqrt(np.mean(fold_errors))
    cv_errors.append(cv_rmse)
    rms_p = cv_rmse / rms_before * rms_pct_before
    print(f"    n_features = {n_feat} : CV RMSE = {cv_rmse:.5f}  ({rms_p:.3f}%)")

best_n = np.argmin(cv_errors) + 1
print(f"\n    → Nombre optimal de features (validation croisée) : {best_n}")
print(f"    → RMS attendu hors-échantillon : {cv_errors[best_n-1] / rms_before * rms_pct_before:.3f}%")

print("\n" + "═"*82)
print("  CONCLUSION")
print("═"*82)
print(f"""
  Test : invariants géométriques tri-corps de Frenet.

  RÉSULTATS :
    • RMS V5 brut             : {rms_pct_before:.3f}%
    • RMS V5 + 15 features    : {rms_pct_after:.3f}% (en sample, peut surfitter)
    • RMS V5 + {best_n} features (CV): {cv_errors[best_n-1] / rms_before * rms_pct_before:.3f}%
    • Top 3 features          : {', '.join(selected[:3])}

  Comparaison des voies V6 :
    • V5 brut                       : 1.886%
    • V6 Kontsevich (Vassiliev v_n) : 1.860% (gain 1.4%)
    • V6 Milnor (7 multi-comp.)     : 1.721% (gain 8.8%)
    • V6 Triple-Géom ({best_n} feats)      : {cv_errors[best_n-1] / rms_before * rms_pct_before:.3f}%
""")
