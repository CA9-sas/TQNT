#!/usr/bin/env python3
"""
Fit TQNT V3 — U_EM contextuel (nœuds dans l'entrelacs)
=======================================================

Pour chaque particule = entrelacs de n nœuds:
1. Placer les n courbes composantes en 3D (configuration liée)
2. Optimiser les échelles pour minimiser E_total = Σ U_EM(Kᵢ) + V_interaction
3. Extraire U_EM_contextuel pour chaque composante

Puis: m = α₀ × U_EM_tot_contextuel × (1 + ν×ℓ) × c_J_eff^p
"""
import sys; sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize as sp_minimize, nnls

from theory.jones import jones_polynomial, JONES_KNOT_NAMES
from theory.retraction import retraction_factor
from theory.knot_catalog import knot_curve

# ================================================================
# DONNÉES DE BASE
# ================================================================
CJ = {}
for name in JONES_KNOT_NAMES:
    if name == "0_1":
        continue
    CJ[name] = retraction_factor(jones_polynomial(name))

F_N = {1: 1.00, 2: 0.86, 3: 0.77}
BETA_LK = 0.210; LK_OPT = 3; GEO_MEAN_CJ = 6.15

# ================================================================
# CALCUL U_EM SUR UNE COURBE
# ================================================================
N_PTS = 300  # points par courbe

def U_EM_on_curve(pts):
    """U_EM = ∮[κ²+τ²]ds sur une courbe discrète."""
    n = len(pts)
    dt = 2 * np.pi / (n - 1)
    d1 = np.gradient(pts, dt, axis=0)
    d2 = np.gradient(d1, dt, axis=0)
    d3 = np.gradient(d2, dt, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    cross = np.cross(d1, d2)
    cn = np.maximum(np.linalg.norm(cross, axis=1), 1e-12)
    kappa = cn / speed**3
    tau = np.sum(cross * d3, axis=1) / cn**2
    return float(np.trapezoid((kappa**2 + tau**2) * speed, dx=dt))


def curve_radius(pts):
    """Rayon moyen de la courbe (distance au centre de masse)."""
    c = pts.mean(axis=0)
    return float(np.mean(np.linalg.norm(pts - c, axis=1)))


# Pré-calcul des courbes et propriétés
print("Pré-calcul des courbes de base...")
BASE_CURVES = {}
U_EM0 = {}
R0 = {}  # rayon moyen
for name in CJ:
    pts = knot_curve(name, N_PTS)
    BASE_CURVES[name] = pts
    U_EM0[name] = U_EM_on_curve(pts)
    R0[name] = curve_radius(pts)

print(f"{'Nœud':<8} {'c_J':>5} {'U_EM⁰':>8} {'R₀':>6}")
print("-" * 30)
for k in sorted(U_EM0, key=lambda n: (int(n.split('_')[0]), int(n.split('_')[1]))):
    print(f"{k:<8} {CJ[k]:>5.1f} {U_EM0[k]:>8.1f} {R0[k]:>6.2f}")


# ================================================================
# CALCUL U_EM CONTEXTUEL DANS UN ENTRELACS
# ================================================================
def min_distance_curves(pts_a, pts_b, subsample=20):
    """Distance minimale entre deux courbes (sous-échantillonnées)."""
    step = max(1, len(pts_a) // subsample)
    a = pts_a[::step]
    b = pts_b[::step]
    diffs = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    return float(np.min(dists))


def compute_link_U_EM(knot_names, alpha_int=1.0):
    """Calculer U_EM contextuel pour chaque composante d'un entrelacs.

    Minimise E = Σ U_EM(λᵢ) + α_int × V_interaction
    où λᵢ est l'échelle de chaque composante.

    Retourne: liste des U_EM contextuels.
    """
    n = len(knot_names)

    if n == 1:
        return [U_EM0[knot_names[0]]]

    curves_base = [BASE_CURVES[k].copy() for k in knot_names]
    radii = [R0[k] for k in knot_names]
    u0 = [U_EM0[k] for k in knot_names]

    # Placement initial: centres sur un cercle de rayon R_sep
    R_sep_init = max(radii) * 2

    def energy(params):
        scales = params[:n]
        R_sep = params[n]

        if any(s < 0.05 for s in scales) or R_sep < 0.1:
            return 1e10

        # Self-energy: U_EM = U_EM⁰ / λ
        E_self = sum(u0[i] / scales[i] for i in range(n))

        # Interaction: répulsion entre courbes proches
        E_int = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                angle_i = 2 * np.pi * i / n
                angle_j = 2 * np.pi * j / n
                offset = np.array([
                    R_sep * (np.cos(angle_j) - np.cos(angle_i)),
                    R_sep * (np.sin(angle_j) - np.sin(angle_i)),
                    0.0
                ])
                pts_i = curves_base[i] * scales[i]
                pts_j = curves_base[j] * scales[j] + offset
                d_min = min_distance_curves(pts_i, pts_j)

                # Répulsion douce
                r_excl = 0.3 * (scales[i] + scales[j])
                if d_min < r_excl:
                    E_int += alpha_int * 50 * ((r_excl / d_min) - 1)**2

                # Attraction: les courbes doivent rester liées
                r_link = 3.0 * max(scales[i], scales[j])
                if d_min > r_link:
                    E_int += alpha_int * 10 * ((d_min / r_link) - 1)**2

        return E_self + E_int

    # Optimisation
    x0 = [1.0] * n + [R_sep_init]
    res = sp_minimize(energy, x0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})

    scales_opt = res.x[:n]
    U_EMs = [u0[i] / scales_opt[i] for i in range(n)]
    return U_EMs


# ================================================================
# SPECTRE — 65 particules
# ================================================================
spectrum = [
    ("e⁻",       ["3_1"],                1, 0,  0.511,   False),
    ("μ⁻",       ["3_1","6_1"],          2, 0,  105.66,  False),
    ("τ⁻",       ["3_1","5_1","7_1"],    3, 0,  1776.9,  False),
    ("s",        ["5_2","5_2"],          2, 0,  93.0,    False),
    ("c",        ["7_3","7_7"],          2, 0,  1270.0,  False),
    ("b",        ["4_1","5_2","7_6"],    3, 0,  4180.0,  False),
    ("t",        ["7_4","7_6","8_3"],    3, 0,  172760., False),
    ("π⁰",      ["8_1"],                1, 0,  135.0,   False),
    ("π±",      ["8_1"],                1, 0,  139.6,   False),
    ("K±",      ["4_1","4_1","5_2"],    3, 0,  493.7,   False),
    ("K⁰",      ["4_1","4_1","5_2"],    3, 0,  497.6,   False),
    ("η",       ["3_1","4_1","5_1"],    3, 0,  547.9,   False),
    ("η'",      ["5_2","5_2","6_1"],    3, 0,  957.8,   False),
    ("η_c",     ["4_1","5_2","9_1"],    3, 0,  2983.9,  False),
    ("D⁰",      ["6_2","7_7"],          2, 0,  1864.8,  False),
    ("D±",      ["6_2","7_7"],          2, 0,  1869.7,  False),
    ("D_s",      ["6_3","7_7"],          2, 0,  1968.3,  False),
    ("B±",      ["4_1","10_1"],         2, 0,  5279.3,  False),
    ("B⁰",      ["4_1","10_1"],         2, 0,  5279.7,  False),
    ("B_s",      ["5_2","10_1"],         2, 0,  5366.9,  False),
    ("B_c",      ["7_3","10_1"],         2, 0,  6274.9,  False),
    ("ρ",        ["5_2","7_7"],          2, 1,  775.3,   False),
    ("ω",        ["5_2","7_7"],          2, 1,  782.7,   False),
    ("K*",       ["4_1","4_1","5_2"],    3, 1,  891.7,   False),
    ("φ",        ["3_1","4_1","6_2"],    3, 1,  1019.5,  False),
    ("D*⁰",     ["6_2","7_7"],          2, 1,  2006.9,  False),
    ("D_s*",     ["6_3","7_7"],          2, 1,  2112.2,  False),
    ("J/ψ",     ["4_1","5_2","9_1"],    3, 1,  3096.9,  False),
    ("ψ(2S)",    ["4_1","5_2","9_1"],    3, 2,  3686.1,  False),
    ("B*",       ["4_1","10_1"],         2, 1,  5324.7,  False),
    ("Υ",       ["3_1","7_4","9_1"],    3, 1,  9460.3,  False),
    ("Υ(2S)",   ["3_1","7_4","9_1"],    3, 2,  10023.3, False),
    ("Υ(3S)",   ["3_1","7_4","9_1"],    3, 3,  10355.2, False),
    ("f₀(980)",  ["5_2","5_2","6_1"],    3, 0,  990.0,   False),
    ("a₀(980)",  ["5_2","5_2","6_1"],    3, 0,  980.0,   False),
    ("f₂(1270)", ["5_2","7_7"],          2, 2,  1275.5,  False),
    ("a₁(1260)", ["5_2","7_7"],          2, 1,  1230.0,  False),
    ("p",        ["3_1","7_2","7_2"],    3, 0,  938.3,   False),
    ("n",        ["3_1","7_2","7_2"],    3, 0,  939.6,   True),
    ("Λ",       ["4_1","7_4"],          2, 0,  1115.7,  False),
    ("Σ⁺",     ["6_2","7_2"],          2, 0,  1189.4,  False),
    ("Σ⁰",     ["6_2","7_2"],          2, 0,  1192.6,  False),
    ("Σ⁻",     ["6_2","7_2"],          2, 0,  1197.4,  False),
    ("Ξ⁰",     ["4_1","4_1","7_2"],    3, 0,  1314.9,  False),
    ("Ξ⁻",     ["4_1","4_1","7_2"],    3, 0,  1321.7,  False),
    ("Δ",        ["3_1","5_2","6_2"],    3, 1,  1232.0,  False),
    ("Σ*",      ["5_2","6_2","7_2"],    3, 1,  1383.0,  False),
    ("Ξ*",      ["4_1","5_2","7_2"],    3, 1,  1531.8,  False),
    ("Ω⁻_dec",  ["5_2","5_2","7_2"],    3, 1,  1672.5,  False),
    ("N(1440)",  ["3_1","7_2","7_2"],    3, 1,  1440.0,  False),
    ("N(1520)",  ["3_1","7_2","7_2"],    3, 1,  1515.0,  False),
    ("N(1680)",  ["3_1","7_2","7_2"],    3, 2,  1685.0,  False),
    ("Λ(1520)",  ["4_1","7_4"],          2, 1,  1519.5,  False),
    ("Λ(1600)",  ["4_1","7_4"],          2, 1,  1600.0,  False),
    ("Λ_c",     ["6_3","8_1"],          2, 0,  2286.5,  False),
    ("Σ_c",     ["4_1","5_1","7_1"],    3, 0,  2453.0,  False),
    ("Ξ_c",     ["5_2","6_3","7_2"],    3, 0,  2470.9,  False),
    ("Ω_c",     ["3_1","6_2","6_2"],    3, 0,  2695.2,  False),
    ("Λ_b",     ["5_2","5_2","8_2"],    3, 0,  5619.6,  False),
    ("Σ_b",     ["5_2","7_2","8_2"],    3, 0,  5811.3,  False),
    ("Ξ_b",     ["4_1","5_2","8_2"],    3, 0,  5794.5,  False),
    ("Ω_b",     ["5_2","6_2","8_2"],    3, 0,  6046.1,  False),
    ("W±",      ["4_1","7_3","8_3"],    3, 1,  80377.0, False),
    ("Z⁰",     ["5_1","7_2","8_3"],    3, 1,  91188.0, False),
    ("H",        ["6_1","7_4","8_3"],    3, 0,  125250., False),
]

N_PART = len(spectrum)

# ================================================================
# CALCUL CONTEXTUEL POUR TOUTES LES PARTICULES
# ================================================================
print(f"\n{'='*70}")
print(f"CALCUL U_EM CONTEXTUEL — {N_PART} particules")
print("Chaque entrelacs est optimisé indépendamment...")
print(f"{'='*70}\n")

# Cache: même config → même U_EM contextuel
config_cache = {}

link_results = []
for idx, (name, config, n, ell, m_sm, is_n) in enumerate(spectrum):
    key = tuple(config)
    if key not in config_cache:
        u_ctx = compute_link_U_EM(config, alpha_int=1.0)
        config_cache[key] = u_ctx
        status = "calculé"
    else:
        u_ctx = config_cache[key]
        status = "cache"

    U_tot = sum(u_ctx)
    print(f"  [{idx+1:>2}/{N_PART}] {name:<10} {'+'.join(config):<22} "
          f"U⁰={sum(U_EM0[k] for k in config):>7.1f} → U_ctx={U_tot:>7.1f} "
          f"(×{U_tot/max(sum(U_EM0[k] for k in config),0.01):.2f}) [{status}]")

    link_results.append({
        "name": name, "config": config, "n": n, "ell": ell,
        "m_sm": m_sm, "is_n": is_n,
        "U_ctx": u_ctx, "U_ctx_tot": U_tot,
    })


# ================================================================
# FIT: α₀, p, ν
# ================================================================
print(f"\n{'='*70}")
print("FIT: m = α₀ × U_ctx_tot × (1+ν×ℓ) × c_J_eff^p")
print(f"{'='*70}")

def cost_fit(params):
    alpha0, p, nu = params
    if alpha0 < 0 or p < 1 or nu < -0.5:
        return 1e20
    chi2 = 0
    for r in link_results:
        U_tot = r["U_ctx_tot"] * (1 + nu * r["ell"])
        c_tot = sum(CJ[k] for k in r["config"])
        c_eff = F_N.get(r["n"], 0.77) * c_tot
        m_pred = alpha0 * U_tot * c_eff**p
        if r["is_n"]:
            m_pred += BETA_LK * (2 - LK_OPT)**2 * GEO_MEAN_CJ
        if m_pred > 0 and r["m_sm"] > 0:
            chi2 += (np.log(m_pred) - np.log(r["m_sm"]))**2
    return chi2

# Multi-start
best = None
for x0 in [[8e-4, 3.55, 0.5], [5e-4, 3.8, 0.3], [1e-3, 3.3, 0.8],
            [2e-4, 4.0, 1.0], [1.5e-3, 3.0, 0.2], [6e-4, 3.6, 2.0]]:
    res = sp_minimize(cost_fit, x0, method='Nelder-Mead',
                      options={'maxiter': 500000, 'xatol': 1e-14, 'fatol': 1e-14})
    if best is None or res.fun < best.fun:
        best = res

a0, p, nu = best.x
print(f"\n  α₀ = {a0:.6e}")
print(f"  p  = {p:.4f}")
print(f"  ν  = {nu:.4f}")
print(f"  χ² = {best.fun:.6f}")


# ================================================================
# RÉSULTATS
# ================================================================
print(f"\n{'='*100}")
print(f"RÉSULTATS — {N_PART} PARTICULES")
print(f"{'='*100}")

print(f"\n{'Part.':<10} {'Config':<22} {'ℓ':>1} {'m_SM':>9} {'m_calc':>9} {'Δ':>7} "
      f"{'U⁰_Σ':>7} {'U_ctx':>7} {'×':>5}")
print("-" * 95)

results = []
for r in link_results:
    U_vib = r["U_ctx_tot"] * (1 + nu * r["ell"])
    c_tot = sum(CJ[k] for k in r["config"])
    c_eff = F_N.get(r["n"], 0.77) * c_tot
    m_calc = a0 * U_vib * c_eff**p
    if r["is_n"]:
        m_calc += BETA_LK * (2 - LK_OPT)**2 * GEO_MEAN_CJ
    delta = (m_calc - r["m_sm"]) / r["m_sm"] * 100

    U0_sum = sum(U_EM0[k] for k in r["config"])
    ratio = r["U_ctx_tot"] / U0_sum if U0_sum > 0 else 0

    config_str = "+".join(r["config"])[:20]
    print(f"{r['name']:<10} {config_str:<22} {r['ell']:>1} {r['m_sm']:>9.1f} "
          f"{m_calc:>9.1f} {delta:>+7.2f}% {U0_sum:>7.1f} "
          f"{r['U_ctx_tot']:>7.1f} {ratio:>5.2f}")

    results.append({"name": r["name"], "m_sm": r["m_sm"], "m_calc": m_calc,
                     "delta": delta, "ell": r["ell"], "config": r["config"],
                     "U0_sum": U0_sum, "U_ctx": r["U_ctx_tot"], "ratio": ratio})

deltas = [abs(r["delta"]) for r in results]
print(f"\n{'='*50}")
print(f"Δ moyen  = {np.mean(deltas):.2f}%")
print(f"Δ médian = {np.median(deltas):.2f}%")
print(f"Δ max    = {max(deltas):.1f}% ({results[np.argmax(deltas)]['name']})")
print(f"<5%:  {sum(1 for d in deltas if d<5):>3}/{N_PART}")
print(f"<10%: {sum(1 for d in deltas if d<10):>3}/{N_PART}")
print(f"<20%: {sum(1 for d in deltas if d<20):>3}/{N_PART}")

# 10 pires
print(f"\n10 pires:")
for r in sorted(results, key=lambda r: abs(r["delta"]), reverse=True)[:10]:
    print(f"  {r['name']:<10} Δ={r['delta']:>+7.1f}%  ×={r['ratio']:.2f}")


# ================================================================
# FIGURE
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

ax = axes[0, 0]
for r in results:
    col = "#2E7D32" if abs(r["delta"]) < 5 else "#FF8F00" if abs(r["delta"]) < 20 else "#E53935"
    ax.scatter(r["m_sm"], r["m_calc"], c=col, s=30, zorder=5, edgecolors="white", linewidth=0.3)
ax.plot([0.3, 200000], [0.3, 200000], "k--", linewidth=1, alpha=0.3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("m_SM (MeV)"); ax.set_ylabel("m_TQNT (MeV)")
ax.set_title(f"Fit U_EM contextuel — Δ moy={np.mean(deltas):.1f}%", fontweight="bold")
ax.grid(True, alpha=0.2)

ax = axes[0, 1]
ax.scatter([r["U0_sum"] for r in results], [r["U_ctx"] for r in results],
           c=[r["ell"] for r in results], cmap="viridis", s=30, edgecolors="white")
u_max = max(max(r["U0_sum"] for r in results), max(r["U_ctx"] for r in results)) * 1.1
ax.plot([0, u_max], [0, u_max], "k--", linewidth=1, alpha=0.3)
ax.set_xlabel("Σ U_EM⁰ (isolé)"); ax.set_ylabel("U_EM contextuel (entrelacs)")
ax.set_title("Effet de l'entrelacs sur U_EM", fontweight="bold")
ax.grid(True, alpha=0.2)

ax = axes[1, 0]
ratios = [r["ratio"] for r in results]
n_comps = [len(r["config"]) for r in results]
ax.scatter(n_comps, ratios, c=[abs(r["delta"]) for r in results],
           cmap="RdYlGn_r", s=40, edgecolors="white", vmin=0, vmax=30)
ax.set_xlabel("n (composantes)"); ax.set_ylabel("U_ctx / Σ U_EM⁰")
ax.set_title("Facteur de compression par n", fontweight="bold")
ax.axhline(1, color="k", linewidth=1, linestyle="--", alpha=0.3)
ax.grid(True, alpha=0.2)

ax = axes[1, 1]
ax.hist(deltas, bins=25, range=(0, min(60, max(deltas)*1.1)),
        color="#42A5F5", edgecolor="white")
ax.axvline(np.mean(deltas), color="red", linewidth=2, linestyle="--",
           label=f"Moy: {np.mean(deltas):.1f}%")
ax.axvline(np.median(deltas), color="green", linewidth=2, linestyle="--",
           label=f"Méd: {np.median(deltas):.1f}%")
ax.set_xlabel("|Δ| (%)"); ax.set_ylabel("N")
ax.set_title("Distribution des écarts", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.2)

fig.suptitle(f"TQNT V3 — U_EM contextuel (entrelacs) — {N_PART} particules\n"
             f"α₀={a0:.4e}, p={p:.3f}, ν={nu:.3f}",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fit_tqnt_v3_vibrationnel.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nOK → fit_tqnt_v3_vibrationnel.png")
