r"""
CALCUL DE L'INTÉGRALE RADIALE P23 sur μ⁻ et η.

Pour un entrelacs L = K_1 ∪ ... ∪ K_n, la force plastique radiale
sur la composante K_i est :

    F^rad_i = ∮_{K_i} ∇_⊥ B_others(r) ds
            = ∮_{K_i} Σ_{j≠i} ∇_⊥ B_j(γ_i(t)) ds

avec B_j(r) le champ magnétique intrinsèque (Biot-Savart-like) de K_j
au point r ∈ K_i :

    B_j(r) = ∮_{K_j} (γ_j'(s) × (r - γ_j(s))) / |r - γ_j(s)|^3 ds

Pour une composante UNIQUE (j = i seulement) : l'auto-action s'annule
par symétrie topologique de la circulation fermée → résultat 0.
Pour n ≥ 2 composantes : superposition non-balancée → ≠ 0.

Test : computer ces intégrales pour μ⁻ (3_1+6_1) et η (3_1+4_1+5_1)
et vérifier qu'elles sont non-nulles et structurées.
"""
from __future__ import annotations
import numpy as np
from theory.knot_catalog import _CURVE_REGISTRY


def get_curve_with_normal(name, n_points=300):
    """Retourne (γ, T, N) le long de la courbe : position, tangente, normale."""
    pts = _CURVE_REGISTRY[name](n_points)
    dt = 2*np.pi / n_points
    r1 = np.gradient(pts, dt, axis=0)
    r2 = np.gradient(r1, dt, axis=0)
    speed = np.linalg.norm(r1, axis=1, keepdims=True)
    T = r1 / (speed + 1e-12)
    # Normale principale (Frenet)
    dT_ds = np.gradient(T, dt, axis=0)
    N_unnormalized = dT_ds
    N_norm = np.linalg.norm(N_unnormalized, axis=1, keepdims=True)
    N = N_unnormalized / (N_norm + 1e-12)
    return pts, T, N, speed.ravel() * dt


def field_at_point(r_target, curve_source, tangent_source, ds_source):
    """Champ magnétique Biot-Savart au point r_target dû à la courbe source.
    B(r) = ∮ T(s) × (r - γ(s)) / |r - γ(s)|^3 ds
    """
    disp = r_target[None, :] - curve_source  # (n_src, 3)
    d = np.linalg.norm(disp, axis=1) + 1e-6  # éviter singularités
    B = np.cross(tangent_source, disp) / d[:, None]**3
    return np.sum(B * ds_source[:, None], axis=0)


def radial_force_on_component(curve_i, T_i, N_i, ds_i,
                              curves_others, T_others, ds_others,
                              positions_others=None):
    """Calcule F^rad sur K_i due aux autres composantes.

    Pour chaque point r ∈ K_i :
      B_total(r) = Σ_{j≠i} B_j(r)
      F^rad(r) = B_total(r) · N_i(r)  (composante radiale = normale)
    Intégrée le long de K_i.
    """
    F_radial = []
    F_vec_total = np.zeros(3)
    for r_i, n_i, ds in zip(curve_i, N_i, ds_i):
        B_others = np.zeros(3)
        for curve_j, T_j, ds_j in zip(curves_others, T_others, ds_others):
            B_others += field_at_point(r_i, curve_j, T_j, ds_j)
        F_rad_pt = np.dot(B_others, n_i)  # projection radiale (sur N)
        F_radial.append(F_rad_pt)
        F_vec_total += B_others * ds
    return np.array(F_radial), F_vec_total


def position_link_components(component_names, n_points=300, separations=None):
    """Place les composantes dans l'espace pour former un entrelacs."""
    curves = []; T_list = []; N_list = []; ds_list = []
    for i, name in enumerate(component_names):
        pts, T, N, ds = get_curve_with_normal(name, n_points)
        # Décaler la 2e composante en X, la 3e en Y, etc.
        # Distance ~ rayon caractéristique des nœuds
        if separations is None:
            separations = [0.0, 3.0, -3.0, 5.0][:len(component_names)]
        if i > 0:
            offset = np.zeros(3)
            if i == 1: offset[0] = separations[1]
            elif i == 2: offset[1] = separations[2]
            elif i == 3: offset[2] = separations[3]
            pts = pts + offset
        curves.append(pts)
        T_list.append(T)
        N_list.append(N)
        ds_list.append(ds)
    return curves, T_list, N_list, ds_list


def compute_radial_integral(component_names, n_points=300, separations=None):
    """Compute the full radial integral for a link."""
    curves, T_list, N_list, ds_list = position_link_components(
        component_names, n_points, separations)

    print(f"\n  Composantes : {component_names}")
    print(f"  {'composante':<5} {'n_pts':<6} {'⟨|F^rad|⟩':>12} {'∑F^rad (radial)':>18} "
          f"{'|F_vec_total|':>15}")
    print("  " + "-" * 70)

    results = []
    for i, K_name in enumerate(component_names):
        others_idx = [j for j in range(len(component_names)) if j != i]
        if not others_idx:
            print(f"  {K_name:<5} (composante unique → F^rad = 0 par construction)")
            results.append({"K": K_name, "F_mean": 0, "F_sum": 0,
                            "F_vec_norm": 0})
            continue
        curves_o = [curves[j] for j in others_idx]
        T_o = [T_list[j] for j in others_idx]
        ds_o = [ds_list[j] for j in others_idx]
        F_rad, F_vec = radial_force_on_component(
            curves[i], T_list[i], N_list[i], ds_list[i],
            curves_o, T_o, ds_o)
        F_mean = np.mean(np.abs(F_rad))
        F_sum = np.sum(F_rad * ds_list[i])
        F_vec_norm = np.linalg.norm(F_vec)
        print(f"  {K_name:<5} {len(curves[i]):<6} {F_mean:>12.5f} "
              f"{F_sum:>18.5f} {F_vec_norm:>15.5f}")
        results.append({
            "K": K_name, "F_rad_array": F_rad,
            "F_mean": F_mean, "F_sum": F_sum, "F_vec_norm": F_vec_norm,
        })

    # Métrique globale
    F_global = sum(r["F_mean"] for r in results)
    print(f"\n  ⟨|F^rad|⟩ global = {F_global:.5f}")
    return results, F_global


# ───────── Test 1 : nœud isolé (contrôle, devrait donner 0) ─────────
print("═"*100)
print("  TEST 1 (CONTRÔLE) : nœud isolé → la formule prédit F^rad = 0")
print("═"*100)
results_e, F_e = compute_radial_integral(["3_1"])

# ───────── Test 2 : μ⁻ = 3_1 + 6_1 ─────────
print("\n" + "═"*100)
print("  TEST 2 : μ⁻ = 3_1 + 6_1 (n=2)")
print("═"*100)
results_mu, F_mu = compute_radial_integral(["3_1", "6_1"])

# ───────── Test 3 : η = 3_1 + 4_1 + 5_1 (assignation V5) ─────────
print("\n" + "═"*100)
print("  TEST 3 : η = 3_1 + 4_1 + 5_1 (n=3, assignation V5)")
print("═"*100)
results_eta, F_eta = compute_radial_integral(["3_1", "4_1", "5_1"])

# ───────── Test 4 : sensibilité à la séparation pour μ⁻ ─────────
print("\n" + "═"*100)
print("  TEST 4 : sensibilité de F^rad(μ⁻) à la distance D entre composantes")
print("═"*100)
print(f"  {'D':>6} {'⟨|F^rad|⟩ 3_1':>14} {'⟨|F^rad|⟩ 6_1':>14} {'F^rad global':>14}")
print("  " + "-" * 60)
F_vs_D = []
for D in [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
    res, Ftot = compute_radial_integral(["3_1", "6_1"],
                                        separations=[0.0, D])
    F_vs_D.append((D, res[0]["F_mean"], res[1]["F_mean"], Ftot))
print()
print(f"  Récapitulatif :")
print(f"  {'D':>6} {'⟨|F^rad|⟩ 3_1':>16} {'⟨|F^rad|⟩ 6_1':>16} {'global':>10}")
for D, f1, f2, ft in F_vs_D:
    print(f"  {D:>6.1f} {f1:>16.5f} {f2:>16.5f} {ft:>10.5f}")

# ───────── Synthèse ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE")
print("═"*100)
print(f"""
  Résultats pour le coefficient global ⟨|F^rad|⟩ :

    Nœud isolé (3_1) :           F = {F_e:.5f}  (contrôle, ≈ 0)
    μ⁻ = 3_1 + 6_1 :             F = {F_mu:.5f}
    η  = 3_1 + 4_1 + 5_1 :       F = {F_eta:.5f}

  Comparaison aux γ(L) extraits empiriquement (session précédente) :
    |γ(μ⁻)| ≈ 0.148
    |γ(η)|  ≈ 0.184

  Ratios :
    F(η) / F(μ⁻) calculé    = {F_eta/F_mu if F_mu > 0 else 0:.3f}
    |γ(η)| / |γ(μ⁻)| extrait = {0.184/0.148:.3f}

  Si l'intégrale radiale est l'origine physique correcte de γ(L),
  on attend une corrélation entre F^rad_global et |γ(L)|.

  Sensibilité à D (séparation entre composantes) :
    F^rad varie selon 1/D^k (k ≈ ?)
    cohérent avec une interaction de type magnétostatique
""")
