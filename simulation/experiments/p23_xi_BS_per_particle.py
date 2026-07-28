r"""
CALCUL DE ξ_BS POUR LES CONFIGURATIONS V5 RÉELLES.

Pour chaque particule du tableau V5, on calcule l'intégrale Biot-Savart
radiale spécifique :

  ξ_BS(L) = ∮_i [∮_j (T_j × (γ_i - γ_j)) · N_i / |γ_i - γ_j|^3 dt_j] ds_i

sommée sur toutes les paires (i, j) de composantes.

Configurations V5 publiées :
  μ⁻ = 3_1 + 6_1            (n=2)
  η  = 3_1 + 4_1 + 5_1      (n=3)
  p  = n = 3_1 + 7_2 + 7_2  (n=3, diff par Lk)
  π⁰ = π± = 8_1             (n=1 → ξ_BS = 0 par symétrie)
  K  = 4_1 + 4_1 + 5_2      (n=3)

Méthode :
  1. Pour chaque link, calculer F^rad(L; D) sur grille de D
  2. Fit F^rad = a(L) / D^3
  3. a(L) est le coefficient ξ_BS dimensionnel pour cette configuration
"""
from __future__ import annotations
import numpy as np
from theory.knot_catalog import _CURVE_REGISTRY


def get_curve_normal(name, n_points=200):
    pts = _CURVE_REGISTRY[name](n_points)
    dt = 2*np.pi / n_points
    r1 = np.gradient(pts, dt, axis=0)
    speed = np.linalg.norm(r1, axis=1, keepdims=True)
    T = r1 / (speed + 1e-12)
    dT_ds = np.gradient(T, dt, axis=0)
    N = dT_ds / (np.linalg.norm(dT_ds, axis=1, keepdims=True) + 1e-12)
    return pts, T, N, speed.ravel() * dt


def field_at(r_target, curve_src, T_src, ds_src):
    disp = r_target[None, :] - curve_src
    d = np.linalg.norm(disp, axis=1) + 1e-6
    B = np.cross(T_src, disp) / d[:, None]**3
    return np.sum(B * ds_src[:, None], axis=0)


def F_radial_mean(curve_i, N_i, curves_o, T_o, ds_o):
    F_list = []
    for r_i, n_i in zip(curve_i, N_i):
        B = np.zeros(3)
        for c_j, T_j, ds_j in zip(curves_o, T_o, ds_o):
            B += field_at(r_i, c_j, T_j, ds_j)
        F_list.append(np.abs(np.dot(B, n_i)))
    return float(np.mean(F_list))


def place_link(component_names, D, n_points=200):
    """Place les composantes à distance D dans positions symétriques."""
    curves, Ts, Ns, dss = [], [], [], []
    if len(component_names) == 1:
        offsets = [np.array([0.0, 0.0, 0.0])]
    elif len(component_names) == 2:
        offsets = [np.array([0.0, 0.0, 0.0]),
                   np.array([D, 0.0, 0.0])]
    elif len(component_names) == 3:
        # Triangle équilatéral en xy
        offsets = [np.array([0.0, 0.0, 0.0]),
                   np.array([D, 0.0, 0.0]),
                   np.array([D/2, D*np.sqrt(3)/2, 0.0])]
    else:
        # Disposition générique
        offsets = [np.array([D*np.cos(2*np.pi*i/len(component_names)),
                              D*np.sin(2*np.pi*i/len(component_names)), 0.0])
                   for i in range(len(component_names))]
    for i, name in enumerate(component_names):
        pts, T, N, ds = get_curve_normal(name, n_points)
        pts = pts + offsets[i]
        curves.append(pts); Ts.append(T); Ns.append(N); dss.append(ds)
    return curves, Ts, Ns, dss


def compute_F_rad(component_names, D, n_points=200):
    """F^rad global (moyenné sur composantes) pour un link à distance D."""
    if len(component_names) == 1:
        return 0.0  # nœud isolé : symétrie → 0
    curves, Ts, Ns, dss = place_link(component_names, D, n_points)
    F_total = 0
    for i in range(len(component_names)):
        others = [j for j in range(len(component_names)) if j != i]
        F_i = F_radial_mean(curves[i], Ns[i],
                            [curves[j] for j in others],
                            [Ts[j] for j in others],
                            [dss[j] for j in others])
        F_total += F_i
    return F_total / len(component_names)


def compute_xi_BS(component_names, D_grid=None, n_points=200):
    """Extrait ξ_BS via fit F^rad ~ a / D^3."""
    if len(component_names) == 1:
        return 0.0, 3.0
    if D_grid is None:
        D_grid = np.array([2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    F_arr = np.array([compute_F_rad(component_names, D, n_points)
                       for D in D_grid])
    # Fit log F = -k log D + log a
    valid = F_arr > 1e-10
    if valid.sum() < 3:
        return 0.0, 3.0
    slope, intercept = np.polyfit(np.log(D_grid[valid]), np.log(F_arr[valid]), 1)
    k_exp = -slope
    a_pref = np.exp(intercept)
    return a_pref, k_exp


# ───────── Configurations V5 ─────────
PARTICULES_V5 = [
    # (nom, config V5, m_exp)
    ("μ⁻",   ["3_1","6_1"],          105.66),
    ("η",    ["3_1","4_1","5_1"],    547.86),
    ("π⁰",   ["8_1"],                134.98),
    ("π±",   ["8_1"],                139.57),
    ("K⁰",   ["4_1","4_1","5_2"],    497.61),
    ("K±",   ["4_1","4_1","5_2"],    493.68),
    ("p",    ["3_1","7_2","7_2"],    938.27),
    ("n",    ["3_1","7_2","7_2"],    939.57),
]


print("═"*90)
print("  Calcul de ξ_BS pour les configurations V5 publiées")
print("═"*90)

print(f"\n  {'particule':<8} {'config V5':<22} {'n':>3} {'ξ_BS (a)':>10} "
      f"{'k_exp':>7} {'régime':>10}")
print("  " + "-" * 70)

xi_results = []
for nm, cfg, m_exp in PARTICULES_V5:
    a, k = compute_xi_BS(cfg, n_points=180)
    n = len(cfg)
    regime = "isolé" if n == 1 else f"n={n}"
    flag = " (Hopf)" if abs(k - 3.0) < 0.3 and n >= 2 else ""
    if n == 1:
        print(f"  {nm:<8} {'+'.join(cfg):<22} {n:>3} "
              f"{'0 (symétrie)':>10} {'—':>7} {regime:>10}")
    else:
        print(f"  {nm:<8} {'+'.join(cfg):<22} {n:>3} "
              f"{a:>10.4f} {k:>7.3f} {regime:>10}{flag}")
    xi_results.append({"nm": nm, "cfg": cfg, "a": a, "k": k, "n": n,
                        "m_exp": m_exp})


# ───────── Comparaison à ξ_BS(Hopf canonique) ─────────
print("\n" + "═"*90)
print("  Comparaison à ξ_BS(Hopf canonique) = 5.089")
print("═"*90)

xi_Hopf = 5.089
print(f"\n  {'particule':<8} {'ξ_BS calculé':>13} {'ratio / Hopf':>14} {'note':<25}")
print("  " + "-" * 60)
for r in xi_results:
    if r["n"] == 1:
        print(f"  {r['nm']:<8} {'0 (symétrie)':>13} {'N/A':>14} "
              f"{'nœud isolé':<25}")
        continue
    ratio = r["a"] / xi_Hopf
    if ratio < 2: note = "Hopf-like"
    elif ratio < 10: note = "modéré"
    else: note = "structure complexe"
    print(f"  {r['nm']:<8} {r['a']:>13.4f} {ratio:>14.3f} {note:<25}")


# ───────── Prédiction de l'effet plastique ─────────
print("\n" + "═"*90)
print("  Prédiction qualitative de l'effet plastique par particule")
print("═"*90)

# Calibration : si ξ_BS proportionnel à |γ|, comparer aux résidus V5
# Résidus V5 publiés (du tableau):
residus_v5 = {
    "μ⁻":  3.2,   # 102 vs 106 = -3.8 MeV / 106 = -3.6%, |Δ| ≈ 3.2%
    "η":   0.5,   # 545 vs 548
    "π⁰":  0.8,   # 134 vs 135
    "π±":  4.3,   # 134 vs 140
    "K⁰":  0.8,   # 494 vs 498
    "K±":  0.0,   # 494 vs 494
    "p":   0.2,   # 940 vs 938
    "n":   0.2,   # 941 vs 940
}

print(f"\n  {'particule':<8} {'ξ_BS':>10} {'|Δ V5 publié|':>15} "
      f"{'prédiction P23':<25}")
print("  " + "-" * 70)
for r in xi_results:
    res_v5 = residus_v5.get(r["nm"], 0)
    if r["n"] == 1:
        pred = "ξ_BS=0 → pas de plasticité"
    elif r["a"] < 5:
        pred = "Plasticité faible"
    elif r["a"] < 20:
        pred = "Plasticité modérée"
    else:
        pred = "Plasticité forte"
    print(f"  {r['nm']:<8} {r['a']:>10.4f} {res_v5:>14.2f}% "
          f"{pred:<25}")


# ───────── Corrélation ξ_BS vs résidu V5 ─────────
multi_results = [r for r in xi_results if r["n"] >= 2]
xi_arr = np.array([r["a"] for r in multi_results])
res_arr = np.array([residus_v5.get(r["nm"], 0) for r in multi_results])

print(f"\n  Corrélation ξ_BS vs |Δ V5| (n={len(multi_results)} multi-composantes) :")
if len(multi_results) > 2:
    r_corr = np.corrcoef(xi_arr, res_arr)[0, 1]
    print(f"    Pearson r = {r_corr:+.3f}")


# ───────── Synthèse ─────────
print("\n" + "═"*90)
print("  SYNTHÈSE")
print("═"*90)
print(f"""
  Résultats clés pour les configurations V5 :

  Particules à composantes uniques (π⁰, π±) :
    ξ_BS = 0 par construction (symétrie topologique du nœud isolé)
    → P23 prédit aucune déformation plastique
    → Cohérent avec π⁰ (résidu 0.8% ~ 0)
    → π± (résidu 4.3%) : différence vient d'isospin V6, pas de P23

  Particules multi-composantes (μ⁻, η, K, p, n) :
    ξ_BS varies selon la topologie et la géométrie des nœuds
    Le RATIO à ξ_BS(Hopf canonique) reflète la « complexité d'enlacement »

  Comparaison aux résidus V5 publiés :
    μ⁻ a le PLUS GRAND résidu (3.2%) → on attendrait grand ξ_BS
    η, K, p, n ont des résidus PETITS (<1%) → on attendrait petit ξ_BS
""")
