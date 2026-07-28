r"""
DÉFORMATIONS MULTI-AXES — APPROCHE GLOBALE.

Au lieu de chercher 2n paramètres (un (s_xy, s_z) par composante avec
régularisation), on teste l'approche GLOBALE :
  • un seul (s_xy, s_z) appliqué à toutes les composantes du link
  • 2 paramètres par particule, 1 contrainte (m = m_exp)
  • Une dimension libre, choix dicté par minimisation L² ||s-1||²

Avantage : optimisation bien posée, robuste.
Question : est-ce que le pattern global (s_xy, s_z) corrèle avec
un invariant ou un nombre quantique ?
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize, brentq
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
    return float(np.sum(kappa_arr**2 * ds)) * float(np.sum(tau_arr**2 * ds))


_CACHE = {}
def U_K(K, s_xy, s_z):
    key = (K, round(s_xy, 4), round(s_z, 4))
    if key not in _CACHE:
        _CACHE[key] = frenet_deformed_xyz(K, s_xy, s_xy, s_z)
    return _CACHE[key]


def omega_global(components, s_xy, s_z):
    n = len(components)
    ws = []; Us = []
    for K in components:
        Us.append(U_K(K, s_xy, s_z))
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
    ("μ⁻",  ["3_1","6_1"],          105.6584, 1, -1, 0.5, 0),
    ("τ⁻",  ["4_1","5_1","6_1"],   1776.86,  3, -1, 0.5, 0),
    ("π⁰",  ["3_1","3_1","5_2"],    134.98,  0,  0, 0.0, 0),
    ("π±",  ["3_1","3_1","5_2"],    139.57,  0, +1, 0.0, 1),
    ("K⁰",  ["3_1","3_1","6_2"],    497.61,  1,  0, 0.0, 0.5),
    ("K±",  ["3_1","3_1","6_2"],    493.68,  1, +1, 0.0, 0.5),
    ("η",   ["3_1","3_1","6_1"],    547.86,  0,  0, 0.0, 0),
    ("φ",   ["3_1","5_1","5_2"],   1019.46,  0,  0, 1.0, 0),
    ("J/ψ", ["3_1","4_1","6_3"],   3096.9,   0,  0, 1.0, 0),
    ("Λ_c", ["3_1","3_1","9_1"],   2286.46,  1,  0, 0.5, 0),
    ("D⁰",  ["3_1","5_1","7_2"],   1864.84,  0,  0, 0.0, 0.5),
    ("p",   ["3_1","7_2","7_2"],    938.27,  1, +1, 0.5, 0.5),
    ("n",   ["3_1","7_2","7_2"],    939.57,  1,  0, 0.5, -0.5),
    ("B±",  ["3_1","6_2","7_7"],   5279.34,  0, +1, 0.0, 0),
]
# columns: nm, cfg, m_exp, baryon#, Q, J, I_3


def find_optimal_xy_z(cfg, m_exp):
    """(s_xy, s_z) global minimisant ||s-1||² s.t. m=m_exp.
       Méthode : grid sur 1D iso-m, choisir le point le plus près de (1,1)."""
    # Grid 2D : pour chaque s_xy, trouver s_z tel que m=m_exp
    # Stocker la liste (s_xy, s_z(s_xy), |s-1|²)
    candidates = []
    for s_xy in np.linspace(0.3, 2.5, 30):
        def err(s_z):
            return omega_global(cfg, s_xy, s_z) - m_exp
        # Vérifier qu'il y a un signe-change
        try:
            sign_low = err(0.1); sign_high = err(5.0)
            if sign_low * sign_high > 0:
                continue
            s_z = brentq(err, 0.1, 5.0, xtol=1e-4)
            norm = (s_xy - 1)**2 + (s_z - 1)**2
            candidates.append((s_xy, s_z, norm))
        except:
            continue
    if not candidates:
        return None, None, None
    # Choisir le point le plus près de (1,1)
    s_xy_opt, s_z_opt, norm_opt = min(candidates, key=lambda x: x[2])
    return s_xy_opt, s_z_opt, norm_opt


# ───────── ÉTAPE 1 — Recherche des déformations optimales ─────────
print("\n" + "═"*100)
print("  Recherche (s_xy, s_z) global minimisant ||s-1||² s.t. m=m_exp")
print("═"*100)
print(f"  {'nm':<6} {'config':<22} {'Δ V5':>8} "
      f"{'s_xy':>7} {'s_z':>7} {'||s-1||':>8} {'volume':>8} {'OK':>3}")
print("  " + "-" * 90)
results = []
for nm, cfg, m_exp, b, Q, J, I3 in PARTICULES:
    m_v5 = omega_global(cfg, 1.0, 1.0)
    d_v5 = (m_v5 - m_exp)/m_exp*100
    s_xy, s_z, norm = find_optimal_xy_z(cfg, m_exp)
    if s_xy is None:
        print(f"  {nm:<6} {'+'.join(cfg):<22} {d_v5:>+7.2f}% {'no_sol':>7}")
        continue
    vol = s_xy**2 * s_z
    flag = "Y"
    print(f"  {nm:<6} {'+'.join(cfg):<22} {d_v5:>+7.2f}% "
          f"{s_xy:>7.4f} {s_z:>7.4f} {np.sqrt(norm):>8.4f} {vol:>8.4f} {flag:>3}")
    results.append({"nm": nm, "cfg": cfg, "m_exp": m_exp,
                    "s_xy": s_xy, "s_z": s_z, "norm": norm,
                    "volume": vol, "d_v5": d_v5,
                    "b_num": b, "Q": Q, "J": J, "I3": I3})


# ───────── ÉTAPE 2 — Analyse des patterns ─────────
print("\n" + "═"*100)
print("  Analyse statistique")
print("═"*100)

if results:
    sxy = np.array([r["s_xy"] for r in results])
    sz = np.array([r["s_z"] for r in results])
    vol = np.array([r["volume"] for r in results])

    print(f"\n  N = {len(results)} particules avec solution (s_xy, s_z) global")
    print(f"  s_xy : range [{sxy.min():.3f}, {sxy.max():.3f}], ⟨s_xy⟩ = {sxy.mean():.4f}")
    print(f"  s_z  : range [{sz.min():.3f}, {sz.max():.3f}], ⟨s_z⟩  = {sz.mean():.4f}")
    print(f"  Volume s_xy²·s_z : range [{vol.min():.3f}, {vol.max():.3f}], ⟨V⟩ = {vol.mean():.4f}")

    if abs(vol.mean() - 1) < 0.05 and vol.std() < 0.1:
        print(f"   → VOLUME APPROXIMATIVEMENT CONSERVÉ")
    else:
        print(f"   → Volume non conservé (σ = {vol.std():.3f})")

    # Corrélations
    print(f"\n  Corrélations Pearson :")
    sxy_dev = sxy - 1
    sz_dev = sz - 1
    log_m = np.log([r["m_exp"] for r in results])
    n_comp = np.array([len(r["cfg"]) for r in results])
    n_naive = np.array([sum(M_NAIVE[K] for K in r["cfg"]) for r in results])
    sum_U = np.array([sum(U_KT[K][0]*U_KT[K][1] for K in r["cfg"]) for r in results])
    Q = np.array([abs(r["Q"]) for r in results])
    J = np.array([r["J"] for r in results])
    b_num = np.array([r["b_num"] for r in results])

    for label, v in [("log(m_exp)", log_m), ("n_components", n_comp.astype(float)),
                     ("M_naive_sum", n_naive), ("Σ U_κU_τ", sum_U),
                     ("|Q|", Q.astype(float)), ("J", J), ("baryon#", b_num.astype(float))]:
        if np.std(v) > 1e-10:
            r_xy = np.corrcoef(v, sxy_dev)[0, 1]
            r_z = np.corrcoef(v, sz_dev)[0, 1]
            r_vol = np.corrcoef(v, vol - 1)[0, 1] if abs(vol.mean()-1) > 0.01 or vol.std() > 0.01 else float('nan')
            f1 = "★" if abs(r_xy) > 0.5 else " "
            f2 = "★" if abs(r_z) > 0.5 else " "
            print(f"   {label:<14} (s_xy-1)={f1}{r_xy:+.3f}  (s_z-1)={f2}{r_z:+.3f}")

# ───────── ÉTAPE 3 — Tester si pattern s_xy = 1/√s_z (volume conservé exact) ─────────
print("\n" + "═"*100)
print("  TEST : hypothèse volume conservé exact s_xy = 1/√s_z (1 paramètre)")
print("═"*100)

print(f"\n  Si vol = 1 exactement, s_xy = 1/√s_z, on a 1 paramètre par particule.")
print(f"  Test : avec contrainte s_xy² · s_z = 1, trouver s_z global tel que m=m_exp.\n")

def find_s_z_iso_vol(cfg, m_exp):
    """Trouve s_z tel que m=m_exp, avec s_xy = 1/√s_z (volume conservé)."""
    def err(s_z):
        if s_z <= 0: return 1e10
        return omega_global(cfg, 1/np.sqrt(s_z), s_z) - m_exp
    try:
        return brentq(err, 0.1, 5.0, xtol=1e-4)
    except:
        return None


print(f"  {'nm':<6} {'config':<22} {'Δ V5':>8} {'s_z (iso-vol)':>14} {'OK':>3}")
print("  " + "-" * 70)
iso_vol_results = []
for nm, cfg, m_exp, b, Q, J, I3 in PARTICULES:
    m_v5 = omega_global(cfg, 1.0, 1.0)
    d_v5 = (m_v5 - m_exp)/m_exp*100
    s_z = find_s_z_iso_vol(cfg, m_exp)
    if s_z is None:
        print(f"  {nm:<6} {'+'.join(cfg):<22} {d_v5:>+7.2f}% {'no_sol':>14}")
        continue
    print(f"  {nm:<6} {'+'.join(cfg):<22} {d_v5:>+7.2f}% {s_z:>14.4f} {'Y':>3}")
    iso_vol_results.append({"nm": nm, "cfg": cfg, "s_z_iso": s_z,
                             "m_exp": m_exp, "d_v5": d_v5})


# ───────── ÉTAPE 4 — Conclusion ─────────
print("\n" + "═"*100)
print("  CONCLUSION")
print("═"*100)

n_resolved = len(results)
n_total = len(PARTICULES)
print(f"\n  • Solutions (s_xy, s_z) globales trouvées : {n_resolved}/{n_total}")
print(f"  • Solutions iso-volume (1 paramètre)       : {len(iso_vol_results)}/{n_total}")

if results:
    print(f"\n  Pattern global :")
    print(f"    σ(s_xy - 1) = {np.std(sxy - 1):.4f}")
    print(f"    σ(s_z - 1)  = {np.std(sz - 1):.4f}")
    if abs(vol.mean() - 1) < 0.05:
        print(f"\n  ★ VOLUME conservé (⟨V⟩ = {vol.mean():.4f}, σ = {vol.std():.4f})")
        print(f"    → suggère une contrainte d'incompressibilité physique")
        print(f"    → réduit le problème à 1 paramètre par particule (s_z)")
    else:
        print(f"\n  Volume NON conservé (⟨V⟩ = {vol.mean():.4f}, σ = {vol.std():.4f})")

print("\n")
