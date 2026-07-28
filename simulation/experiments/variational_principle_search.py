r"""
RECHERCHE d'un principe variationnel prédisant s_z(K, contexte).

Stratégie :
  ÉTAPE 1 — Tabuler s_z requis pour ~12 particules à résidu V5 non-nul.
            Pour chacune, déformer la composante la plus complexe et
            trouver s_z* tel que m_pred = m_exp.

  ÉTAPE 2 — Chercher un pattern dans les s_z* requis vs invariants
            topologiques (b, det, U_κU_τ, ratio, complexité).

  ÉTAPE 3 — Tester plusieurs principes candidats :
    P1. s_z = 1 + α · (U_κU_τ_partners / U_κU_τ_self)
    P2. s_z = 1 + α / (n_components - 1)
    P3. s_z = 1 + α · (max_partner_det / det_self)
    P4. s_z = 1 + α · (cr_self - cr_partners_mean) / cr_self
    P5. s_z = 1 + α · √(b_partners / b_self)
    P6. s_z = 1 + α · (1/n) · Σ √(U_κU_τ_i · U_κU_τ_j) / U_κU_τ_self

  ÉTAPE 4 — Calibrer α sur μ⁻ pour chaque principe, mesurer le résidu
            global sur les autres particules.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import brentq, minimize_scalar
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
A = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); B = 33/20; RHO_0 = 10 - 3/(8*np.pi)

# Précalcul
print("Précalcul…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 50000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)),
               float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def frenet_deformed(K, s_z, n_points=30000):
    """Calcule (U_κ, U_τ) pour K avec scale_z = s_z (scale_xy=1)."""
    curve_fn = _CURVE_REGISTRY[K]
    pts = curve_fn(n_points)
    pts = pts * np.array([1.0, 1.0, s_z])
    dt = 2*np.pi / n_points
    r1 = np.gradient(pts, dt, axis=0)
    r2 = np.gradient(r1, dt, axis=0)
    r3 = np.gradient(r2, dt, axis=0)
    speed = np.linalg.norm(r1, axis=1)
    cross = np.cross(r1, r2)
    kappa = np.linalg.norm(cross, axis=1) / (speed**3 + 1e-12)
    triple = np.einsum('ij,ij->i', cross, r3)
    tau = triple / (np.linalg.norm(cross, axis=1)**2 + 1e-12)
    ds = speed * dt
    return float(np.sum(kappa**2 * ds)), float(np.sum(tau**2 * ds))


def omega_ts_link(components, K_deformed_idx, s_z, U_kt_deformed=None):
    """Ω_ts pour link avec une composante déformée."""
    n = len(components)
    ws = []
    for i, K in enumerate(components):
        w = M_NAIVE[K] * (TOPO[K]["b"] * TOPO[K]["det"] / T_31)**P_EXP
        ws.append(w)
    skel = float(sum(w**A for w in ws)**(1/A))
    G_sum = 0
    for i, K in enumerate(components):
        if i == K_deformed_idx and U_kt_deformed is not None:
            G_sum += U_kt_deformed
        else:
            G_sum += U_KT[K][0]*U_KT[K][1]
    G = G_sum / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(components); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# ───────── ÉTAPE 1 — Catalogue de particules ─────────
# Liste : (nom, config, m_exp). On déforme la composante la plus complexe.
PARTICULES = [
    ("μ⁻",  ["3_1","6_1"],          105.6584),
    ("τ⁻",  ["4_1","5_1","6_1"],   1776.86),
    ("π⁰",  ["3_1","3_1","5_2"],    134.98),
    ("π±",  ["3_1","3_1","5_2"],    139.57),
    ("K⁰",  ["3_1","3_1","6_2"],    497.61),
    ("K±",  ["3_1","3_1","6_2"],    493.68),
    ("η",   ["3_1","3_1","6_1"],    547.86),
    ("φ",   ["3_1","5_1","5_2"],   1019.46),
    ("J/ψ", ["3_1","4_1","6_3"],   3096.9),
    ("Λ_c", ["3_1","3_1","9_1"],   2286.46),
    ("D⁰",  ["3_1","5_1","7_2"],   1864.84),
    ("p",   ["3_1","7_2","7_2"],    938.27),
    ("n",   ["3_1","7_2","7_2"],    939.57),
    ("B±",  ["3_1","6_2","7_7"],   5279.34),
]


def find_s_z(components, m_target, K_def_idx, search=(0.3, 3.0)):
    """Trouve s_z* tel que m_pred = m_exp, en déformant components[K_def_idx]."""
    K_def = components[K_def_idx]
    def err(s):
        Uk_d, Ut_d = frenet_deformed(K_def, s)
        m_p = omega_ts_link(components, K_def_idx, s, Uk_d*Ut_d)
        return m_p - m_target
    try:
        s_opt = brentq(err, *search, xtol=1e-5)
        return s_opt
    except:
        # Pas de solution dans la plage
        return None


# Trouver s_z requis pour chaque particule
print("\n" + "═"*82)
print("  ÉTAPE 1 — s_z requis pour atteindre m_exp")
print("═"*82)
print(f"  {'Particule':<8} {'Configuration':<22} {'m_exp':>9} {'m_V5':>9} "
      f"{'Δ V5':>8} {'K_def':<5} {'s_z_req':>9}")
print("  " + "-" * 80)

data_pts = []
for nm, cfg, m_exp in PARTICULES:
    # Calculer m_V5
    m_v5 = omega_ts_link(cfg, 0, 1.0)  # déformation 0 = canonique
    delta_v5 = (m_v5 - m_exp)/m_exp*100
    # Composante la plus complexe = max cr
    cr_list = [TOPO[K]["cr"] for K in cfg]
    K_def_idx = cr_list.index(max(cr_list))
    K_def = cfg[K_def_idx]
    # Trouver s_z
    s_z = find_s_z(cfg, m_exp, K_def_idx)
    if s_z is None:
        print(f"  {nm:<8} {'+'.join(cfg):<22} {m_exp:>9.2f} {m_v5:>9.2f} "
              f"{delta_v5:>+7.2f}% {K_def:<5} {'pas trouvé':>9}")
        continue
    print(f"  {nm:<8} {'+'.join(cfg):<22} {m_exp:>9.2f} {m_v5:>9.2f} "
          f"{delta_v5:>+7.2f}% {K_def:<5} {s_z:>9.4f}")
    # Calculer invariants pour analyse
    U_self = U_KT[K_def][0] * U_KT[K_def][1]
    U_partners = sum(U_KT[c][0]*U_KT[c][1] for i, c in enumerate(cfg) if i != K_def_idx)
    det_partners_mean = np.mean([TOPO[c]["det"] for i, c in enumerate(cfg) if i != K_def_idx])
    data_pts.append({
        "nm": nm, "K_def": K_def, "K_others": [c for i,c in enumerate(cfg) if i!=K_def_idx],
        "s_z": s_z, "delta_v5": delta_v5,
        "U_self": U_self, "U_partners": U_partners,
        "U_ratio": U_partners / U_self,
        "n_components": len(cfg),
        "det_self": TOPO[K_def]["det"], "det_partners_mean": det_partners_mean,
        "b_self": TOPO[K_def]["b"],
        "b_partners_mean": np.mean([TOPO[c]["b"] for i,c in enumerate(cfg) if i!=K_def_idx]),
        "cr_self": TOPO[K_def]["cr"],
        "cr_partners_sum": sum(TOPO[c]["cr"] for i,c in enumerate(cfg) if i!=K_def_idx),
    })


# ───────── ÉTAPE 2 — Recherche de pattern ─────────
print("\n" + "═"*82)
print("  ÉTAPE 2 — Recherche de pattern (s_z - 1) vs invariants")
print("═"*82)

x_observed = np.array([d["s_z"] for d in data_pts])
y = np.array([d["s_z"] - 1 for d in data_pts])  # déviation
print(f"  N = {len(data_pts)} points")
print(f"\n  s_z - 1 observés : min = {min(y):+.4f}, max = {max(y):+.4f}")
print(f"  Particule la plus déformée : {data_pts[np.argmax(np.abs(y))]['nm']}"
      f"  (s_z = {x_observed[np.argmax(np.abs(y))]:.4f})")

print(f"\n  Corrélations linéaires Pearson (s_z - 1) vs invariants :")
features = ["U_ratio", "n_components", "det_self", "det_partners_mean",
            "b_self", "b_partners_mean", "cr_self", "cr_partners_sum",
            "U_self", "U_partners"]
for f in features:
    v = np.array([d[f] for d in data_pts])
    if np.std(v) > 1e-10:
        r = np.corrcoef(v, y)[0, 1]
        print(f"    {f:<22} r = {r:+.4f}")


# ───────── ÉTAPE 3 — Tester principes candidats ─────────
print("\n" + "═"*82)
print("  ÉTAPE 3 — Calibration de principes variationnels candidats")
print("═"*82)

def evaluate_principle(predict_fn, data_pts):
    """Évalue un principe en calibrant α sur μ⁻ et mesurant RMS global."""
    # Calibrer sur μ⁻
    mu_pt = next(d for d in data_pts if d["nm"] == "μ⁻")
    def err_calib(alpha):
        return predict_fn(mu_pt, alpha) - mu_pt["s_z"]
    try:
        alpha = brentq(err_calib, -100, 100, xtol=1e-6)
    except:
        return None, None
    # Évaluer sur tous
    preds = np.array([predict_fn(d, alpha) for d in data_pts])
    obs = np.array([d["s_z"] for d in data_pts])
    rms_s_z = float(np.sqrt(np.mean((preds - obs)**2)))
    # Conversion en RMS sur m
    rms_m = 0
    for d, pred_s_z in zip(data_pts, preds):
        # Calculer m avec s_z prédit
        cfg = [d["K_def"]] + d["K_others"]
        # Reconstituer cfg dans ordre original — déjà complexe
        # Simplification : utiliser une approximation (s_z prédit donne m_pred)
        # Skip détaillé : utiliser (s_z_pred - s_z_obs)
    return alpha, rms_s_z


candidates = [
    ("P1: 1 + α · (U_partners/U_self)",
     lambda d, a: 1 + a * d["U_ratio"]),
    ("P2: 1 + α / (n_components - 1)",
     lambda d, a: 1 + a / max(d["n_components"]-1, 0.5)),
    ("P3: 1 + α · (det_p_mean / det_self)",
     lambda d, a: 1 + a * d["det_partners_mean"] / d["det_self"]),
    ("P4: 1 + α · (cr_self - cr_p_sum) / cr_self",
     lambda d, a: 1 + a * (d["cr_self"] - d["cr_partners_sum"]) / d["cr_self"]),
    ("P5: 1 + α · √(b_partners/b_self)",
     lambda d, a: 1 + a * np.sqrt(d["b_partners_mean"]/d["b_self"])),
    ("P6: 1 + α · (b_partners·det_self - b_self·det_partners)",
     lambda d, a: 1 + a * (d["b_partners_mean"]*d["det_self"]
                           - d["b_self"]*d["det_partners_mean"])),
    ("P7: 1 + α · (1/U_self)",
     lambda d, a: 1 + a / d["U_self"]),
    ("P8: 1 + α · (cr_partners_sum / cr_self / n_comp)",
     lambda d, a: 1 + a * d["cr_partners_sum"] / d["cr_self"] / d["n_components"]),
    ("P9: 1 + α · (1/n_components²)",
     lambda d, a: 1 + a / d["n_components"]**2),
]

print(f"\n  {'Principe':<55} {'α calib':>10} {'RMS_s_z':>9} {'Max err':>9}")
print("  " + "-" * 88)
results = []
for label, fn in candidates:
    alpha, rms = evaluate_principle(fn, data_pts)
    if alpha is None:
        print(f"  {label:<55} {'no_calib':>10}")
        continue
    preds = np.array([fn(d, alpha) for d in data_pts])
    obs = np.array([d["s_z"] for d in data_pts])
    rms = float(np.sqrt(np.mean((preds-obs)**2)))
    max_err = float(np.max(np.abs(preds-obs)))
    results.append((label, alpha, rms, max_err, preds, obs))
    print(f"  {label:<55} {alpha:>+10.4f} {rms:>9.4f} {max_err:>9.4f}")

# Détail du meilleur principe
results.sort(key=lambda x: x[2])
best = results[0]
print(f"\n  ★ MEILLEUR : {best[0]}")
print(f"     α = {best[1]:.4f}, RMS s_z = {best[2]:.4f}")
print(f"\n  Détail particule par particule :")
print(f"  {'nom':<8} {'s_z observé':>11} {'s_z prédit':>11} {'erreur':>9}")
print("  " + "-" * 45)
for d, p in zip(data_pts, best[4]):
    err = p - d["s_z"]
    flag = "✓" if abs(err) < 0.02 else ("·" if abs(err) < 0.05 else "✗")
    print(f"  {d['nm']:<8} {d['s_z']:>11.4f} {p:>11.4f} {err:>+8.4f}  {flag}")


# ───────── ÉTAPE 4 — Analyse fondamentale ─────────
print("\n" + "═"*82)
print("  ÉTAPE 4 — Diagnostic")
print("═"*82)
print(f"""
  Si un principe variationnel UNIVERSEL existait, on trouverait :
    • un principe avec RMS s_z << 0.01 sur les {len(data_pts)} points
    • forme physiquement justifiable
    • calibré avec 1 paramètre α (et pas 2-3)

  Résultat : meilleur RMS s_z = {best[2]:.4f}
    {'→ excellent (< 0.01)' if best[2] < 0.01 else
     '→ acceptable (< 0.03)' if best[2] < 0.03 else
     '→ pas universel — résidus structurels persistent'}

  Note importante :
    • Le choix de déformer "la composante la plus complexe" est arbitraire.
      Pour μ⁻ on a déformé 6_1; pour le proton on a déformé 7_2.
    • Un principe vraiment universel devrait CHOISIR la composante
      à déformer ET le s_z optimal simultanément.
    • Cela demande une optimisation à plus de variables.
""")
