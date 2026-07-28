r"""
SIMULATION DE RELAXATION PLASTIQUE

Cadre conceptuel : la déformation d'un entrelacs après sa formation
est PLASTIQUE (non-élastique). L'état final est un ATTRACTEUR
de la dynamique dissipative, pas un minimum d'énergie réversible.

Dynamique :
    ds_i/dt = -η · ∂E/∂s_i  +  F^drive_i(s)
                ^^^^ dissipation        ^^^^ force de couplage topologique

Test :
  1. Pour chaque link L, partir de l'état canonique s = (1, 1, ..., 1)
  2. Faire évoluer la dynamique avec un schéma Euler
  3. Observer le point d'équilibre s* (attracteur)
  4. Comparer m(L; s*) à m_exp

Variants testés :
  V1. E = Ω_ts(L; s) + coût élastique (s-1)²
       Minimum à s=1 (pas de plasticité)
  V2. E = Ω_ts(L; s) - λ·Σ_{i<j} √(U_κU_τ_i · U_κU_τ_j)
       Couplage attractif inter-composantes
  V3. F_drive_i = γ·(U_κU_τ_others - U_κU_τ_self) (équilibre des contributions)
  V4. Principe d'équipartition : chaque composante porte une fraction
       égale de l'énergie courbure totale.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import brentq
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index
from theory.knot_catalog import _CURVE_REGISTRY

# Constantes TQNT
M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A_FREN = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); BB = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul des invariants topologiques…")
U_KT = {}; U_K_canon = {}; U_T_canon = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 30000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_K_canon[K] = float(np.sum(g["kappa"]**2*ds))
    U_T_canon[K] = float(np.sum(g["tau"]**2*ds))
    U_KT[K] = (U_K_canon[K], U_T_canon[K])
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


# Cache pour les déformations Frenet
_CACHE = {}
def frenet_def(K, s_z, s_xy=1.0):
    key = (K, round(s_xy, 4), round(s_z, 4))
    if key not in _CACHE:
        pts = _CURVE_REGISTRY[K](15000)
        pts = pts * np.array([s_xy, s_xy, s_z])
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
        _CACHE[key] = (float(np.sum(kappa**2 * ds_arr)),
                       float(np.sum(tau**2 * ds_arr)),
                       float(np.sum(ds_arr)))
    return _CACHE[key]


def omega_link_with_def(components, s_z_list):
    """Ω_ts avec un s_z par composante (s_xy = 1 partout)."""
    n = len(components)
    ws = []; Us = []
    for K, s_z in zip(components, s_z_list):
        Uk, Ut, _ = frenet_def(K, s_z)
        Us.append(Uk * Ut)
        w = M_NAIVE[K] * (TOPO[K]["b"] * TOPO[K]["det"] / T_31)**P_EXP
        ws.append(w)
    skel = float(sum(w**A_FREN for w in ws)**(1/A_FREN))
    G = sum(Us) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(components); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# ───────── DYNAMIQUE PLASTIQUE ─────────

def force_V1_elastic_anchor(s, components, k_restore=1.0, gamma_couple=0.05):
    """V1 : Force restauratrice + couplage gravitationnel-like
       F_i = -k(s_i - 1) + γ·Σ_{j≠i} U_κU_τ_j × √(U_κU_τ_i)/normalisation
    """
    n = len(components)
    F = np.zeros(n)
    Us = []
    for i, K in enumerate(components):
        Uk, Ut, _ = frenet_def(K, s[i])
        Us.append(Uk * Ut)
    Us = np.array(Us)
    for i in range(n):
        # Restauration vers canonique
        F[i] = -k_restore * (s[i] - 1)
        # Pull des partenaires : proportionnel à somme des partenaires
        U_others = sum(Us[j] for j in range(n) if j != i)
        F[i] += gamma_couple * U_others / np.sqrt(Us[i] + 1)
    return F


def force_V2_equipartition(s, components, gamma=0.001):
    """V2 : Force d'équipartition des énergies de courbure
       Chaque composante "veut" porter une fraction égale 1/n de
       l'énergie de courbure totale.
       F_i = γ · (⟨U_κU_τ⟩ - U_κU_τ_i)
    """
    n = len(components)
    Us = []
    for K, s_i in zip(components, s):
        Uk, Ut, _ = frenet_def(K, s_i)
        Us.append(Uk * Ut)
    Us = np.array(Us)
    mean_U = np.mean(Us)
    return gamma * (mean_U - Us) / mean_U


def force_V3_chi_density(s, components, gamma=0.0005, k_restore=0.5):
    """V3 : Force du champ χ (CGQ) — chaque composante subit le puits
       gravitationnel quantique des autres.
       F_i = γ · √(U_partenaires / U_self) — modéré
       + restauration faible
    """
    n = len(components)
    F = np.zeros(n)
    Us = []
    for K, s_i in zip(components, s):
        Uk, Ut, _ = frenet_def(K, s_i)
        Us.append(Uk * Ut)
    for i in range(n):
        U_self = Us[i]
        U_others = sum(Us[j] for j in range(n) if j != i)
        F[i] = gamma * np.sqrt(U_others / max(U_self, 1e-6))
        F[i] -= k_restore * (s[i] - 1)
    return F


def force_V4_topological_balance(s, components,
                                  gamma=0.04, k_anchor=0.1):
    """V4 : Force basée sur le déséquilibre topologique.
       Chaque composante tend à équilibrer le rapport
         U_κU_τ_i / (T_i · M_naive_i)
       où T = b·det. Restauration faible.
    """
    n = len(components)
    F = np.zeros(n)
    ratios = []
    for K, s_i in zip(components, s):
        Uk, Ut, _ = frenet_def(K, s_i)
        U_kt = Uk * Ut
        T = TOPO[K]["b"] * TOPO[K]["det"]
        ratio = U_kt / (T * M_NAIVE[K])
        ratios.append(ratio)
    mean_r = np.mean(ratios)
    for i in range(n):
        F[i] = gamma * (mean_r - ratios[i]) / mean_r
        F[i] -= k_anchor * (s[i] - 1)
    return F


def relax_plastic(components, force_fn, dt=0.005, n_steps=4000,
                  s_init=None, force_kwargs=None, verbose=False):
    """Évolue les déformations sous la force jusqu'à équilibre."""
    n = len(components)
    s = np.ones(n) if s_init is None else np.array(s_init)
    kwargs = force_kwargs or {}
    trajectory = [s.copy()]
    energies = [omega_link_with_def(components, s)]
    for step in range(n_steps):
        F = force_fn(s, components, **kwargs)
        s = s + dt * F
        s = np.clip(s, 0.3, 3.0)  # bornes physiques
        trajectory.append(s.copy())
        energies.append(omega_link_with_def(components, s))
        if step % 500 == 0 and verbose:
            print(f"    step {step:4d}: s = {s}, ||F|| = {np.linalg.norm(F):.4e}")
        if np.linalg.norm(F) < 1e-7:
            break
    return np.array(trajectory), np.array(energies)


# ───────── TEST sur particules ─────────
PARTICULES = [
    ("μ⁻",  ["3_1","6_1"],          105.6584, "asymétrique"),
    ("τ⁻",  ["4_1","5_1","6_1"],   1776.86,  "asymétrique"),
    ("π⁰",  ["3_1","3_1","5_2"],    134.98,  "sym en 3_1"),
    ("η",   ["3_1","3_1","6_1"],    547.86,  "sym en 3_1"),
    ("K",   ["6_1","6_1","6_1"],    495.6,   "totalement sym"),
    ("φ",   ["3_1","5_1","5_2"],   1019.46,  "asymétrique"),
    ("D⁰",  ["3_1","5_1","7_2"],   1864.84,  "asymétrique"),
    ("p",   ["3_1","7_2","7_2"],    938.27,  "sym en 7_2"),
]

variants = [
    ("V1 anchor+gravitational", force_V1_elastic_anchor,
     {"k_restore": 1.0, "gamma_couple": 0.005}),
    ("V2 equipartition",        force_V2_equipartition,
     {"gamma": 0.05}),
    ("V3 chi-density CGQ",      force_V3_chi_density,
     {"gamma": 0.001, "k_restore": 0.05}),
    ("V4 topological balance",  force_V4_topological_balance,
     {"gamma": 0.05, "k_anchor": 0.05}),
]

print("\n" + "═"*100)
print("  RELAXATION PLASTIQUE — convergence vers l'attracteur")
print("═"*100)

best_results = {}
for label, force_fn, kwargs in variants:
    print(f"\n  ─── {label} ───")
    print(f"  {'nm':<6} {'config':<22} {'s_attracteur':<30} {'m(s*)':>10} {'Δ V5':>7} {'Δ s*':>7}")
    print("  " + "-" * 90)
    var_results = []
    for nm, cfg, m_exp, sym in PARTICULES:
        traj, ener = relax_plastic(cfg, force_fn,
                                    force_kwargs=kwargs, n_steps=2000)
        s_final = traj[-1]
        m_final = ener[-1]
        m_v5 = omega_link_with_def(cfg, [1.0]*len(cfg))
        d_v5 = (m_v5 - m_exp)/m_exp*100
        d_star = (m_final - m_exp)/m_exp*100
        s_str = ",".join(f"{s:.3f}" for s in s_final)
        print(f"  {nm:<6} {'+'.join(cfg):<22} {s_str:<30} "
              f"{m_final:>10.2f} {d_v5:>+6.2f}% {d_star:>+6.2f}%")
        var_results.append({
            "nm": nm, "s_star": s_final.copy(),
            "m_v5": m_v5, "m_star": m_final, "m_exp": m_exp,
            "d_v5": d_v5, "d_star": d_star,
        })
    # RMS résiduel
    rms = float(np.sqrt(np.mean([r["d_star"]**2 for r in var_results])))
    rms_v5 = float(np.sqrt(np.mean([r["d_v5"]**2 for r in var_results])))
    gain = (rms_v5 - rms) / rms_v5 * 100 if rms_v5 > 0 else 0
    print(f"\n    RMS V5  = {rms_v5:.3f}%")
    print(f"    RMS s*  = {rms:.3f}%")
    print(f"    Gain    = {gain:+.1f}%")
    best_results[label] = (var_results, rms, gain)


# ───────── Comparaison à la cible explicite ─────────
print("\n" + "═"*100)
print("  Comparaison à la cible μ⁻ (s_z requis sur 6_1 = 1.044)")
print("═"*100)
print(f"\n  Pour μ⁻, l'attracteur prédit s_6_1 par chaque variant :")
print(f"  {'variant':<35} {'s_3_1':>8} {'s_6_1':>8} {'Δ s_6_1 vs 1.044':>20}")
for label, _, _ in variants:
    res = best_results[label][0][0]  # μ⁻ est le premier
    s_31, s_61 = res["s_star"]
    err = s_61 - 1.0436
    flag = "★★" if abs(err) < 0.005 else ("★" if abs(err) < 0.02 else "")
    print(f"  {label:<35} {s_31:>8.4f} {s_61:>8.4f} {err:>+19.4f} {flag}")


# ───────── Diagnostic ─────────
print("\n" + "═"*100)
print("  DIAGNOSTIC")
print("═"*100)

best_label = min(best_results.items(), key=lambda kv: kv[1][1])[0]
print(f"\n  Variant le plus performant : {best_label}")
print(f"    RMS = {best_results[best_label][1]:.3f}%")
print(f"    Gain vs V5 = {best_results[best_label][2]:+.1f}%")

print(f"""
  L'idée de relaxation plastique est testée par 4 fonctionnelles
  candidates. Si l'idée était correcte ET si une de ces fonctionnelles
  capturait la bonne dynamique, on devrait observer :

    1. Convergence vers un attracteur stable (pas de divergence) ✓
       (les bornes physiques 0.3 < s < 3.0 garantissent cela)

    2. Attracteur UNIVERSEL : pour chaque particule, un état s*
       reproductible et indépendant des conditions initiales

    3. m(L; s*) très proche de m_exp pour toutes les particules

  Si les RMS résiduels restent ~ même qu'en V5, alors aucune des
  fonctionnelles testées ne capture la vraie dynamique plastique.
  Si une seule réduit le RMS significativement, c'est un candidat
  pour le principe variationnel dissipatif recherché.
""")
