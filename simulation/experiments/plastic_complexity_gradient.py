r"""
RELAXATION PLASTIQUE — Variante à gradient de complexité

Idée : la force plastique brise la symétrie inter-composantes selon
un critère de complexité topologique. Le composant le plus complexe
(plus de croisements, plus de U_κU_τ) absorbe la déformation
(s > 1, étirement vertical) tandis que les plus simples se compressent.

Configurations symétriques (e.g., proton 3_1+7_2+7_2 avec deux 7_2
identiques) ont des déformations balancées qui préservent
approximativement la masse V5.

Cinq variants à gradient testés :
  G1. F_i = γ·(cr_i - ⟨cr⟩)/⟨cr⟩  -  k·(s_i - 1)
  G2. F_i = γ·(U_κU_τ_i - ⟨U⟩)/⟨U⟩  -  k·(s_i - 1)
  G3. F_i = γ·(b_i - ⟨b⟩)/⟨b⟩  -  k·(s_i - 1)
  G4. F_i = γ·log(cr_i/⟨cr⟩)  -  k·(s_i - 1)
  G5. F_i = γ·(det_i - ⟨det⟩)/⟨det⟩  -  k·(s_i - 1)

Pour chacun, calibration de γ/k sur μ⁻ pour s_6_1 = 1.044.
Évaluation sur 10 particules avec ce γ/k unique.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import brentq
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
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]; T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}

_CACHE = {}
def U_K(K, s_z):
    key = (K, round(s_z, 4))
    if key not in _CACHE:
        pts = _CURVE_REGISTRY[K](15000)
        pts = pts * np.array([1.0, 1.0, s_z])
        dt = 2*np.pi / len(pts)
        r1 = np.gradient(pts, dt, axis=0)
        r2 = np.gradient(r1, dt, axis=0)
        r3 = np.gradient(r2, dt, axis=0)
        speed = np.linalg.norm(r1, axis=1)
        cross = np.cross(r1, r2)
        kappa = np.linalg.norm(cross, axis=1) / (speed**3 + 1e-12)
        triple = np.einsum('ij,ij->i', cross, r3)
        tau = triple / (np.linalg.norm(cross, axis=1)**2 + 1e-12)
        ds_arr = speed * dt
        _CACHE[key] = float(np.sum(kappa**2 * ds_arr)) * float(np.sum(tau**2 * ds_arr))
    return _CACHE[key]


def omega_link(components, s_list):
    n = len(components); ws=[]; Us=[]
    for K, s in zip(components, s_list):
        Us.append(U_K(K, s))
        w = M_NAIVE[K] * (TOPO[K]["b"] * TOPO[K]["det"] / T_31)**P_EXP
        ws.append(w)
    skel = float(sum(w**A_FREN for w in ws)**(1/A_FREN))
    G = sum(Us) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(components); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# ───────── Fonctions de gradient de complexité ─────────
def gradient_cr(components):
    crs = np.array([TOPO[K]["cr"] for K in components], dtype=float)
    return (crs - crs.mean()) / crs.mean()


def gradient_U(components, s_list):
    Us = np.array([U_K(K, s) for K, s in zip(components, s_list)])
    return (Us - Us.mean()) / Us.mean()


def gradient_b(components):
    bs = np.array([TOPO[K]["b"] for K in components], dtype=float)
    return (bs - bs.mean()) / bs.mean()


def gradient_log_cr(components):
    crs = np.array([TOPO[K]["cr"] for K in components], dtype=float)
    return np.log(crs / crs.mean())


def gradient_det(components):
    ds = np.array([TOPO[K]["det"] for K in components], dtype=float)
    return (ds - ds.mean()) / ds.mean()


def force_gradient(s, components, gradient_fn, gamma, k_anchor):
    """Force = γ·gradient + restoration."""
    if gradient_fn == gradient_U:
        grad = gradient_fn(components, s)
    else:
        grad = gradient_fn(components)
    F = gamma * grad - k_anchor * (s - 1)
    return F


def relax(components, gradient_fn, gamma, k_anchor,
          dt=0.01, n_steps=2000, s_init=None):
    n = len(components)
    s = np.ones(n) if s_init is None else np.array(s_init, dtype=float)
    for _ in range(n_steps):
        F = force_gradient(s, components, gradient_fn, gamma, k_anchor)
        s = s + dt * F
        s = np.clip(s, 0.3, 3.0)
        if np.linalg.norm(F) < 1e-8: break
    return s


# ───────── Particules ─────────
PARTICULES = [
    ("μ⁻",  ["3_1","6_1"],          105.6584),
    ("τ⁻",  ["4_1","5_1","6_1"],   1776.86),
    ("π⁰",  ["3_1","3_1","5_2"],    134.98),
    ("η",   ["3_1","3_1","6_1"],    547.86),
    ("K",   ["6_1","6_1","6_1"],    495.6),
    ("φ",   ["3_1","5_1","5_2"],   1019.46),
    ("D⁰",  ["3_1","5_1","7_2"],   1864.84),
    ("p",   ["3_1","7_2","7_2"],    938.27),
    ("n",   ["3_1","7_2","7_2"],    939.57),
    ("Λ_c", ["3_1","3_1","9_1"],   2286.46),
]


def calibrate_mu(gradient_fn, k_anchor=1.0):
    """Calibre γ tel que s_6_1 dans μ⁻ → 1.044."""
    mu_cfg = ["3_1", "6_1"]
    def err(gamma):
        s_eq = relax(mu_cfg, gradient_fn, gamma, k_anchor, n_steps=3000)
        return s_eq[1] - 1.0436  # s_6_1 cible
    try:
        gamma_opt = brentq(err, -2.0, 2.0, xtol=1e-5)
        return gamma_opt
    except Exception as e:
        print(f"     calibration échouée: {e}")
        return None


variants = [
    ("G1 gradient cr",       gradient_cr),
    ("G2 gradient U_κU_τ",   gradient_U),
    ("G3 gradient b",        gradient_b),
    ("G4 gradient log(cr)",  gradient_log_cr),
    ("G5 gradient det",      gradient_det),
]

print("\n" + "═"*100)
print("  TEST des variants à gradient de complexité — calibration sur μ⁻")
print("═"*100)

K_ANCHOR = 1.0
all_results = {}

for label, grad_fn in variants:
    print(f"\n  ─── {label} ───")
    gamma = calibrate_mu(grad_fn, K_ANCHOR)
    if gamma is None:
        print(f"  pas de calibration trouvée")
        continue
    print(f"  γ calibré = {gamma:.4f}  (k_anchor = {K_ANCHOR})")

    print(f"  {'nm':<6} {'config':<22} {'s_attracteur':<28} "
          f"{'m(s*)':>10} {'Δ V5':>7} {'Δ s*':>7}")
    print("  " + "-" * 90)

    res = []
    for nm, cfg, m_exp in PARTICULES:
        s_eq = relax(cfg, grad_fn, gamma, K_ANCHOR, n_steps=3000)
        m_eq = omega_link(cfg, s_eq)
        m_v5 = omega_link(cfg, [1.0]*len(cfg))
        d_v5 = (m_v5 - m_exp)/m_exp*100
        d_eq = (m_eq - m_exp)/m_exp*100
        s_str = ",".join(f"{x:.3f}" for x in s_eq)
        better = "✓" if abs(d_eq) < abs(d_v5) else "✗"
        print(f"  {nm:<6} {'+'.join(cfg):<22} {s_str:<28} "
              f"{m_eq:>10.2f} {d_v5:>+6.2f}% {d_eq:>+6.2f}% {better}")
        res.append({"nm": nm, "s_star": s_eq, "d_v5": d_v5, "d_eq": d_eq,
                    "m_eq": m_eq, "m_exp": m_exp})

    rms_v5 = float(np.sqrt(np.mean([r["d_v5"]**2 for r in res])))
    rms_eq = float(np.sqrt(np.mean([r["d_eq"]**2 for r in res])))
    gain = (rms_v5 - rms_eq) / rms_v5 * 100 if rms_v5 > 0 else 0
    n_better = sum(1 for r in res if abs(r["d_eq"]) < abs(r["d_v5"]))
    print(f"\n    RMS V5 = {rms_v5:.3f}%,  RMS s* = {rms_eq:.3f}%, gain = {gain:+.1f}%")
    print(f"    Améliorations : {n_better}/{len(res)} particules")
    all_results[label] = (gamma, res, rms_eq, gain, n_better)


# ───────── SYNTHÈSE ─────────
print("\n" + "═"*100)
print("  Synthèse — quel variant capture le mieux la physique plastique ?")
print("═"*100)

print(f"\n  {'variant':<25} {'γ':>10} {'RMS s*':>10} {'gain':>8} {'meilleur sur':>15}")
print("  " + "-" * 80)
for label, (gamma, res, rms, gain, n_better) in all_results.items():
    flag = "★★" if gain > 30 else ("★" if gain > 10 else (" " if gain > 0 else " "))
    print(f"  {label:<25} {gamma:>+10.4f} {rms:>9.3f}% {gain:>+7.1f}% "
          f"{n_better:>4}/{len(res)} {flag}")

# Best variant — détail particule par particule
best_label = max(all_results.items(), key=lambda kv: kv[1][3])[0]
print(f"\n  ★ Meilleur variant : {best_label}")
gamma, res, rms, gain, n_better = all_results[best_label]
print(f"     γ = {gamma:.4f}, RMS s* = {rms:.3f}%, gain = {gain:+.1f}%")
print(f"\n  Détail pour μ⁻ (s_6_1 doit converger vers 1.044) :")
for r in res:
    if r["nm"] == "μ⁻":
        print(f"    s_attracteur = {r['s_star']},  m = {r['m_eq']:.3f}")
print(f"\n  Distribution des résidus s* :")
for r in res:
    bar = "█" * min(int(abs(r["d_eq"])*2), 20)
    direction = "+" if r["d_eq"] > 0 else "-"
    print(f"    {r['nm']:<6} {direction}{abs(r['d_eq']):>5.2f}%  {bar}")


# ───────── Diagnostic final ─────────
print("\n" + "═"*100)
print("  DIAGNOSTIC")
print("═"*100)

print(f"""
  Variants testés : 5 (gradients cr, U_κU_τ, b, log(cr), det)
  Calibration : γ ajusté sur μ⁻ pour s_6_1 = 1.044
  k_anchor : {K_ANCHOR} (fixé, restoration vers s=1)

  Question : un gradient unifié réduit-il significativement le RMS
  global tout en respectant la cible μ⁻ ?

  Si OUI (gain > 30%) : principe variationnel à gradient capturé
  Si NON (gain < 10%) : le gradient capté est en partie correct
                        mais pas suffisant

  Configurations symétriques (proton, K = 6_1×3) :
  par construction du gradient (cr_i - ⟨cr⟩), les composantes
  identiques ont gradient nul → s_i = 1 → masse V5 préservée.
""")
