r"""
DÉFORMATIONS MULTI-AXES PAR COMPOSANTE.

Au lieu d'une seule déformation s_z par composante, on autorise
(s_xy, s_z) par composante = 2 paramètres par K (par symétrie).
Pour n composantes : 2n paramètres.

Contrainte unique : m(L; déformations) = m_exp.
Objectif : minimiser ||s - 1||² (déformation minimale au sens L²).

Cela donne UNE solution unique par particule (Lagrange L²).

Questions :
  1. Les 4 particules sans solution 1D (K⁰, K±, J/ψ, B±)
     deviennent-elles accessibles ?
  2. Les déformations restent-elles physiquement raisonnables
     (~5-10% au lieu de 50%+ ou divergences) ?
  3. Quel pattern apparaît : quelle composante se déforme le plus,
     quel axe se déforme le plus ?
  4. Le pattern est-il cohérent avec un PRINCIPE physique
     (énergie, contrainte de volume, anisotropie de torsion, ...) ?
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize, brentq, NonlinearConstraint
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index
from theory.knot_catalog import _CURVE_REGISTRY

M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A_FREN = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); BB = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 30000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)),
               float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def frenet_deformed_xyz(K, s_x, s_y, s_z, n_points=15000):
    """U_κU_τ pour K avec scale (s_x, s_y, s_z)."""
    pts = _CURVE_REGISTRY[K](n_points)
    pts = pts * np.array([s_x, s_y, s_z])
    dt = 2*np.pi / n_points
    r1 = np.gradient(pts, dt, axis=0)
    r2 = np.gradient(r1, dt, axis=0)
    r3 = np.gradient(r2, dt, axis=0)
    speed = np.linalg.norm(r1, axis=1)
    cross = np.cross(r1, r2)
    kappa_arr = np.linalg.norm(cross, axis=1) / (speed**3 + 1e-12)
    triple = np.einsum('ij,ij->i', cross, r3)
    tau_arr = triple / (np.linalg.norm(cross, axis=1)**2 + 1e-12)
    ds = speed * dt
    Uk = float(np.sum(kappa_arr**2 * ds))
    Ut = float(np.sum(tau_arr**2 * ds))
    return Uk * Ut


_CACHE_3D = {}
def U_xy_z(K, s_xy, s_z):
    key = (K, round(s_xy, 4), round(s_z, 4))
    if key not in _CACHE_3D:
        _CACHE_3D[key] = frenet_deformed_xyz(K, s_xy, s_xy, s_z)
    return _CACHE_3D[key]


def omega_ts_link_2d(components, defs_2d):
    """Ω_ts pour link avec déformations 2D (s_xy, s_z) par composante.
       defs_2d : array shape (n, 2)
    """
    n = len(components)
    ws = []
    Us = []
    for K, (s_xy, s_z) in zip(components, defs_2d):
        Us.append(U_xy_z(K, s_xy, s_z))
        w = M_NAIVE[K] * (TOPO[K]["b"] * TOPO[K]["det"] / T_31)**P_EXP
        ws.append(w)
    skel = float(sum(w**A_FREN for w in ws)**(1/A_FREN))
    G = sum(Us) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(components); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


PARTICULES = [
    ("μ⁻",  ["3_1","6_1"],          105.6584),
    ("τ⁻",  ["4_1","5_1","6_1"],   1776.86),
    ("π⁰",  ["3_1","3_1","5_2"],    134.98),
    ("π±",  ["3_1","3_1","5_2"],    139.57),
    ("K⁰",  ["3_1","3_1","6_2"],    497.61),    # ★ no 1D solution
    ("K±",  ["3_1","3_1","6_2"],    493.68),    # ★ no 1D solution
    ("η",   ["3_1","3_1","6_1"],    547.86),
    ("φ",   ["3_1","5_1","5_2"],   1019.46),
    ("J/ψ", ["3_1","4_1","6_3"],   3096.9),     # ★ no 1D solution
    ("Λ_c", ["3_1","3_1","9_1"],   2286.46),
    ("D⁰",  ["3_1","5_1","7_2"],   1864.84),
    ("p",   ["3_1","7_2","7_2"],    938.27),
    ("n",   ["3_1","7_2","7_2"],    939.57),
    ("B±",  ["3_1","6_2","7_7"],   5279.34),    # ★ no 1D solution
]

NO_1D = ["K⁰", "K±", "J/ψ", "B±"]


def find_minimal_deformation_2d(cfg, m_exp, bounds=(0.3, 3.0)):
    """Trouve déformations 2D minimisant ||s-1||² sous contrainte m=m_exp,
       via méthode de pénalité avec Nelder-Mead (sans gradient)."""
    n = len(cfg)
    # 2 passes : grand lambda pour forcer la contrainte, puis petit pour relaxer
    # Final : minimum L² des déformations respectant m=m_exp à ε près
    best_x = None; best_obj = 1e10
    for x0_seed in [np.ones(2*n), np.array([1.05, 1.05]*n), np.array([0.95, 0.95]*n)]:
        for lam in [1e-3, 1.0]:  # passes : approche grossière puis fine
            def penalty(x):
                if np.any(x < bounds[0]) or np.any(x > bounds[1]):
                    return 1e10
                defs = x.reshape(n, 2)
                try:
                    m_p = omega_ts_link_2d(cfg, defs)
                except:
                    return 1e10
                # Pénalité forte sur (m - m_exp)/m_exp, et obj = ||s-1||²
                return np.sum((x - 1)**2) + lam * ((m_p - m_exp)/m_exp)**2
            try:
                res = minimize(penalty, x0_seed, method='Nelder-Mead',
                               options={'fatol': 1e-10, 'xatol': 1e-6,
                                        'maxiter': 2000})
                x0_seed = res.x  # warm start
            except:
                pass
        # Évaluer si la contrainte est satisfaite et si L² est minimal
        defs = x0_seed.reshape(n, 2)
        m_p = omega_ts_link_2d(cfg, defs)
        rel_err = abs(m_p - m_exp) / m_exp
        if rel_err < 0.001:
            obj_val = np.sum((x0_seed - 1)**2)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = x0_seed.copy()
    if best_x is None:
        return None, None, False
    return best_x.reshape(n, 2), best_obj, True


# ───────── ÉTAPE 1 — Test sur les 14 particules ─────────
print("\n" + "═"*100)
print("  ÉTAPE 1 — Déformations 2D minimales (s_xy, s_z) par composante")
print("═"*100)
print(f"  {'Particule':<8} {'Config':<22} {'Δ V5':>7} {'||s-1||²':>9} "
      f"{'(s_xy, s_z) par composante':<45} {'OK':>3}")
print("  " + "-" * 100)

results = []
for nm, cfg, m_exp in PARTICULES:
    m_v5 = omega_ts_link_2d(cfg, [[1.0, 1.0]]*len(cfg))
    d_v5 = (m_v5 - m_exp)/m_exp*100
    defs, norm, ok = find_minimal_deformation_2d(cfg, m_exp)
    if defs is None:
        print(f"  {nm:<8} {'+'.join(cfg):<22} {d_v5:>+6.2f}% {'fail':>9}")
        continue
    defs_str = " ".join(f"({s[0]:.2f},{s[1]:.2f})" for s in defs)
    flag = "★" if nm in NO_1D else " "
    print(f"  {nm:<8} {'+'.join(cfg):<22} {d_v5:>+6.2f}% {norm:>9.4f} "
          f"{defs_str:<45} {('Y' if ok else 'N'):>3} {flag}")
    results.append((nm, cfg, m_exp, defs, norm, d_v5))


# ───────── ÉTAPE 2 — Analyse des patterns ─────────
print("\n" + "═"*100)
print("  ÉTAPE 2 — Analyse statistique des déformations")
print("═"*100)

# Quel axe est le plus déformé : xy ou z ?
all_xy = []
all_z = []
for nm, cfg, m_exp, defs, norm, d_v5 in results:
    for s in defs:
        all_xy.append(s[0] - 1)
        all_z.append(s[1] - 1)
all_xy = np.array(all_xy); all_z = np.array(all_z)

print(f"\n  Statistiques (sur toutes composantes de toutes particules) :")
print(f"    Nombre de composantes : {len(all_xy)}")
print(f"    s_xy : mean = {np.mean(all_xy):+.4f}, std = {np.std(all_xy):.4f}, "
      f"min = {np.min(all_xy):+.4f}, max = {np.max(all_xy):+.4f}")
print(f"    s_z  : mean = {np.mean(all_z):+.4f}, std = {np.std(all_z):.4f}, "
      f"min = {np.min(all_z):+.4f}, max = {np.max(all_z):+.4f}")
print(f"\n  L'écart-type sur s_z est {np.std(all_z)/np.std(all_xy):.2f}x celui sur s_xy")

# Corrélation s_xy <-> s_z ?
r = np.corrcoef(all_xy, all_z)[0, 1]
print(f"  Corrélation s_xy ↔ s_z : r = {r:+.4f}")

# Conservation du volume ? (s_xy² × s_z)
volumes = []
for nm, cfg, m_exp, defs, norm, d_v5 in results:
    for s in defs:
        volumes.append(s[0]**2 * s[1])
volumes = np.array(volumes)
print(f"\n  Volumes (s_xy² · s_z) :")
print(f"    mean = {np.mean(volumes):.4f} (idéal 1.000), std = {np.std(volumes):.4f}")
print(f"    min  = {np.min(volumes):.4f}, max = {np.max(volumes):.4f}")
if abs(np.mean(volumes) - 1.0) < 0.05 and np.std(volumes) < 0.1:
    print(f"    → Volume approximativement conservé !")
else:
    print(f"    → Volume NON conservé.")

# ───────── ÉTAPE 3 — Comparaison V5 vs Multi-axes ─────────
print("\n" + "═"*100)
print("  ÉTAPE 3 — Comparaison RMS V5 vs après déformations multi-axes")
print("═"*100)

rms_v5_sq = 0; rms_2d_sq = 0
n_ok = 0
for nm, cfg, m_exp, defs, norm, d_v5 in results:
    m_2d = omega_ts_link_2d(cfg, defs)
    d_2d = (m_2d - m_exp)/m_exp*100
    rms_v5_sq += d_v5**2
    rms_2d_sq += d_2d**2
    n_ok += 1
rms_v5 = np.sqrt(rms_v5_sq/n_ok)
rms_2d = np.sqrt(rms_2d_sq/n_ok)
print(f"\n  RMS V5             : {rms_v5:.3f}%")
print(f"  RMS multi-axes 2D  : {rms_2d:.3f}%")
print(f"  Solution trouvée pour {n_ok}/{len(PARTICULES)} particules")

n_no_1d_solved = sum(1 for nm, *_ in results if nm in NO_1D)
print(f"  Parmi les 4 'sans 1D' : {n_no_1d_solved}/4 résolues")

# ───────── ÉTAPE 4 — Recherche du principe ─────────
print("\n" + "═"*100)
print("  ÉTAPE 4 — Recherche d'un pattern (s_xy, s_z) vs invariants")
print("═"*100)

# Pour chaque composante : invariants vs (s_xy, s_z) - 1
component_data = []
for nm, cfg, m_exp, defs, norm, d_v5 in results:
    for i, K in enumerate(cfg):
        s_xy, s_z = defs[i]
        # Invariants partenaires
        U_self = U_KT[K][0] * U_KT[K][1]
        U_others = sum(U_KT[c][0]*U_KT[c][1] for j, c in enumerate(cfg) if j != i)
        component_data.append({
            "nm": nm, "K": K, "i": i,
            "s_xy_dev": s_xy - 1, "s_z_dev": s_z - 1,
            "norm_self": (s_xy-1)**2 + (s_z-1)**2,
            "U_self": U_self, "U_others": U_others,
            "U_ratio": U_others / U_self,
            "b": TOPO[K]["b"], "det": TOPO[K]["det"], "cr": TOPO[K]["cr"],
            "n_comp": len(cfg),
        })

# Régression : (s_z - 1) en fonction de divers invariants
print(f"\n  Pour SZ_DEV = s_z - 1 (par composante, n = {len(component_data)}) :")
features = ["U_self", "U_others", "U_ratio", "b", "det", "cr", "n_comp"]
y = np.array([d["s_z_dev"] for d in component_data])
for f in features:
    v = np.array([d[f] for d in component_data])
    if np.std(v) > 1e-10:
        r = np.corrcoef(v, y)[0, 1]
        marker = "★" if abs(r) > 0.5 else (" " if abs(r) > 0.3 else " ")
        print(f"    {marker} {f:<10} r = {r:+.4f}")

print(f"\n  Pour SXY_DEV = s_xy - 1 :")
y2 = np.array([d["s_xy_dev"] for d in component_data])
for f in features:
    v = np.array([d[f] for d in component_data])
    if np.std(v) > 1e-10:
        r = np.corrcoef(v, y2)[0, 1]
        marker = "★" if abs(r) > 0.5 else (" " if abs(r) > 0.3 else " ")
        print(f"    {marker} {f:<10} r = {r:+.4f}")

# ───────── ÉTAPE 5 — Diagnostic final ─────────
print("\n" + "═"*100)
print("  DIAGNOSTIC FINAL")
print("═"*100)
print(f"""
  Bilan de l'exploration multi-axes (s_xy, s_z) par composante :

  • Solutions trouvées : {n_ok}/{len(PARTICULES)}
  • K⁰/K±/J/ψ/B± : {n_no_1d_solved}/4 maintenant résolues
  • Déformations typiques :
      ⟨s_xy⟩ = {1 + np.mean(all_xy):.4f},  σ(s_xy) = {np.std(all_xy):.4f}
      ⟨s_z⟩  = {1 + np.mean(all_z):.4f},  σ(s_z)  = {np.std(all_z):.4f}

  Pattern observé : {'volume conservé' if abs(np.mean(volumes)-1) < 0.05 else 'volume NON conservé'}

  Implication :
    • Si pattern clair → principe variationnel multi-axes possible
    • Si pas de pattern → calibration multi-paramètres ad hoc
    • Si volume conservé → contrainte d'incompressibilité physique
    • Si pas de volume conservé → autre principe (élasticité)
""")
