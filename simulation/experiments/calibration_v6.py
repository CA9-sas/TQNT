r"""
CALIBRATION TQNT-V6 — test du facteur de desserrage F(L, E).

Hypothèse V6 :
    m_V6(L, E) = Ω_ts(L) · F(L, E)

Avec E = énergie interne de la configuration. Trois proxys testés :
  (a) E = Ω_ts(L)              (l'énergie est la masse elle-même)
  (b) E = c·E_0                 (nb de croisements × échelle nucléonique)
  (c) E = Σ m_naive(K_i)·c_J(K)  (énergie cinétique des composantes)

Quatre formes fonctionnelles testées :
  F_1(x) = (1 - α x)^q          (déclin polynomial — M1 géométrique)
  F_2(x) = exp(-α x)            (déclin exponentiel — M2 déblocage)
  F_3(x) = 1 / (1 + α x)^p      (déclin Lorentzien — M3 rétraction χ)
  F_4(x) = (1 + α x)^{-p}       (forme combinée généralisée)

Méthode : régression des résidus log(m_exp / Ω_ts) en fonction de x.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations_with_replacement
from scipy.optimize import minimize

from theory.abraham import frenet_geometry
from theory.jones import (jones_polynomial, jones_determinant, braid_index,
                          topological_spin)

# ───────── constantes ─────────
M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
E_0 = 1.4  # MeV, échelle nucléonique TQNT
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); B_EXP = 33/20; RHO_0 = 10 - 3/(8*np.pi)

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
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def omega(comps):
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in comps)/P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(comps); Phi_p = 1
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B_EXP))**(ct-1)
    return Phi_n*Phi_p*skel*G**Q_EXP


# ───────── PDG pour calibration ─────────
# (nom, config TQNT, m_exp, fiabilité)
# Configurations trouvées par le scan v2 (formule analytique)
PDG = [
    ("e⁻",    ["3_1"], 0.51099895),
    ("μ⁻",    ["3_1","6_1"], 105.66),
    ("τ⁻",    ["4_1","5_1","6_1"], 1776.86),
    ("π⁰",    ["3_1","3_1","5_2"], 134.98),
    ("π±",    ["3_1","3_1","5_2"], 139.57),
    ("K±",    ["4_1","4_1","4_1"], 493.68),
    ("K⁰",    ["6_1","6_1","6_1"], 497.61),
    ("η",     ["3_1","3_1","6_1"], 547.86),
    ("η'",    ["7_7","7_7"], 957.78),
    ("η_c",   ["6_3","7_7"], 2983.9),
    ("η_c2S", ["6_1","8_2"], 3637.5),
    ("η_b",   ["5_1","7_3","8_1"], 9398.7),
    ("D⁰",    ["3_1","5_1","7_2"], 1864.84),
    ("D±",    ["3_1","5_1","7_2"], 1869.66),
    ("D_s",   ["3_1","6_1","7_1"], 1968.34),
    ("B±",    ["3_1","6_2","7_7"], 5279.34),
    ("B⁰",    ["3_1","6_2","7_7"], 5279.66),
    ("B_s",   ["6_3","7_6"], 5366.92),
    ("B_c",   ["4_1","7_7","9_1"], 6274.9),
    ("ρ",     ["8_1","8_1"], 775.26),
    ("ω",     ["8_1","8_1"], 782.65),
    ("φ",     ["3_1","5_1","5_2"], 1019.46),
    ("K*",    ["6_1","6_2"], 891.66),
    ("J/ψ",   ["3_1","4_1","6_3"], 3096.9),
    ("ψ(2S)", ["6_1","8_2"], 3686.10),
    ("ψ(3770)",["3_1","7_2","7_7"], 3773.7),
    ("ψ(4040)",["6_2","8_1","8_1"], 4039.0),
    ("Υ(1S)", ["6_2","7_1","7_3"], 9460.30),
    ("Υ(2S)", ["6_2","6_2","8_2"], 10023.26),
    ("Υ(3S)", ["7_2","10_1"], 10355.2),
    ("D*0",   ["7_7","9_1"], 2006.85),
    ("D*±",   ["7_7","9_1"], 2010.26),
    ("D_s*",  ["5_2","6_1","7_2"], 2112.2),
    ("B*",    ["3_1","6_2","7_7"], 5324.71),
    ("χ_c0",  ["6_1","7_7","7_7"], 3414.71),
    ("χ_c1",  ["7_2","8_1","8_1"], 3510.67),
    ("χ_c2",  ["6_1","8_2"], 3556.17),
    ("χ_b0",  ["6_1","7_3","7_7"], 9859.44),
    ("χ_b1",  ["6_1","7_3","7_7"], 9892.78),
    ("χ_b2",  ["6_2","6_2","8_2"], 9912.21),
    ("p",     ["3_1","7_2","7_2"], 938.27),
    ("n",     ["3_1","7_2","7_2"], 939.57),
    ("Λ",     ["5_1","8_1"], 1115.68),
    ("Σ⁺",    ["4_1","4_1","5_1"], 1189.37),
    ("Σ⁰",    ["4_1","4_1","5_1"], 1192.64),
    ("Σ⁻",    ["4_1","4_1","5_1"], 1197.45),
    ("Ξ⁰",    ["4_1","5_1","5_2"], 1314.86),
    ("Ξ⁻",    ["4_1","5_1","5_2"], 1321.71),
    ("Λ_c",   ["3_1","3_1","9_1"], 2286.46),
    ("Σ_c",   ["4_1","5_1","7_2"], 2453.97),
    ("Ξ_c⁺",  ["3_1","4_1","8_1"], 2467.94),
    ("Ξ_c⁰",  ["3_1","4_1","8_1"], 2470.9),
    ("Ω_c",   ["4_1","4_1","7_7"], 2695.2),
    ("Ξ_cc",  ["4_1","5_2","7_3"], 3621.6),
    ("Λ_b",   ["6_3","7_4"], 5619.6),
    ("Σ_b",   ["7_3","7_4"], 5811.0),
    ("Ξ_b⁰",  ["7_3","7_4"], 5791.9),
    ("Ξ_b⁻",  ["7_3","7_4"], 5797.0),
    ("W±",    ["7_1","7_2","8_3"], 80369),
    ("Z⁰",    ["7_2","8_3","9_1"], 91188),
    ("H",     ["6_3","8_2","8_3"], 125250),
    ("t",     ["6_3","7_1","7_5"], 172690),
]

# ───────── analyse résidus ─────────
print("\n" + "═" * 78)
print("  ANALYSE DES RÉSIDUS V5 — diagnostic du desserrage P8")
print("═" * 78)

data = []
for nm, cfg, m_exp in PDG:
    m_pred = omega(cfg)
    delta = m_exp / m_pred  # ratio expérience/prédiction
    log_delta = np.log(delta)
    c_total = sum(TOPO[K]["cr"] for K in cfg)
    n = len(cfg)
    sum_naive = sum(M_NAIVE[K] for K in cfg)
    data.append((nm, cfg, m_exp, m_pred, delta, log_delta, c_total, n, sum_naive))

# Stats globales
log_deltas = np.array([d[5] for d in data])
print(f"\n  Résidus log(m_exp/Ω_ts) sur {len(data)} particules :")
print(f"    moyenne   = {log_deltas.mean():+.4f}  (sur-prédiction si <0)")
print(f"    écart-type= {log_deltas.std():.4f}")
print(f"    médian    = {np.median(log_deltas):+.4f}")
print(f"    min       = {log_deltas.min():+.4f}")
print(f"    max       = {log_deltas.max():+.4f}")

# Signe : V5 SURPREDIT (delta<1, log<0) ou SOUSPREDIT (delta>1, log>0) ?
n_over = sum(1 for d in data if d[5] < -0.005)
n_under = sum(1 for d in data if d[5] > 0.005)
n_exact = len(data) - n_over - n_under
print(f"\n  Bilan signe :")
print(f"    V5 SURPREDIT (Ω_ts > m_exp) : {n_over}/{len(data)} ({n_over/len(data)*100:.0f}%)")
print(f"    V5 SOUSPREDIT (Ω_ts < m_exp): {n_under}/{len(data)} ({n_under/len(data)*100:.0f}%)")
print(f"    Quasi exact (|δ|<0.5%)      : {n_exact}/{len(data)} ({n_exact/len(data)*100:.0f}%)")

# ───────── régression V6 sur les écarts ─────────
print("\n" + "─" * 78)
print("  RÉGRESSION V6 : ajustement F(x) sur les résidus")
print("─" * 78)

# Variables prédictives possibles
print("\n  On teste 3 proxys pour l'énergie interne x :")
print("    (a) x = Ω_ts(L)/E_0    (masse en unités E_0)")
print("    (b) x = c (croisements totaux)")
print("    (c) x = (Σ m_naive(K))/E_0  (énergie cinétique des composantes)")

# Si V6 : m_V6 = Ω_ts · F(x), avec F(0)=1 et F décroissant
# Alors m_exp/Ω_ts = F(x), donc résidus négatifs (log_delta < 0) si F < 1.
# Si log_delta tend vers négatif quand x grandit → V6 est correct
# Si log_delta n'a aucune tendance en x → V6 n'apporte rien

print(f"\n  Régression log(m_exp/Ω_ts) ≈ -α·x  pour chaque proxy :")
print(f"  {'proxy':<8} {'α (regression)':>16} {'corrélation':>14} {'pente OK ?':>12}")
print("  " + "-" * 60)

for proxy_name, get_x in [
    ("(a) m/E_0", lambda d: d[3]/E_0),
    ("(b) c    ", lambda d: d[6]),
    ("(c) Σm/E_0", lambda d: d[8]/E_0),
]:
    xs = np.array([get_x(d) for d in data])
    ys = log_deltas
    # Régression linéaire forçant ordonnée à l'origine = 0 (F(0)=1)
    if np.var(xs) > 0:
        alpha = -np.sum(xs*ys)/np.sum(xs*xs)
        r = np.corrcoef(xs, ys)[0,1]
        # Si α > 0, V6 vrai (m diminue quand x augmente)
        verdict = "✓ V6 OK" if alpha > 0 and r < -0.1 else "✗ inattendu"
        print(f"  {proxy_name:<8} {alpha:>16.6e} {r:>14.3f} {verdict:>12}")

# Test analytique : régression des trois mécanismes M1+M2+M3 directement
print("\n" + "─" * 78)
print("  TEST FORME FONCTIONNELLE — quel F(x) s'ajuste le mieux ?")
print("─" * 78)

# On utilise x = Ω_ts/E_0 (proxy a)
xs = np.array([d[3]/E_0 for d in data])

def chi2_F(params, F_func):
    """Coût = somme des résidus² sur log(m_exp/Ω_ts) - log(F(x; params))."""
    pred = np.array([np.log(F_func(x, *params)) for x in xs])
    return np.sum((log_deltas - pred)**2)


def F1_poly(x, alpha, q):
    """F = (1 - α x)^q, troncature à zéro."""
    base = max(1 - alpha*x, 1e-9)
    return base**q


def F2_exp(x, alpha):
    """F = exp(-α x)."""
    return np.exp(-alpha*x)


def F3_lorentz(x, alpha, p):
    """F = (1 + α x)^{-p}."""
    return (1 + alpha*x)**(-p)


def F4_powerlaw(x, alpha):
    """F = 1 / (1 + α x)."""
    return 1 / (1 + alpha*x)


# Pas d'optimisation pour trop de paramètres — explorons quelques valeurs
print(f"\n  Coût χ² (sur log) pour différents F :")
print(f"  {'forme':<24} {'α opt.':>10} {'p/q':>8} {'χ²':>10} {'RMS résidu':>12}")
print("  " + "-" * 68)

best_chi2 = float('inf'); best_form = None
for label, func, params0, bounds in [
    ("F = (1 - αx)^q",      F1_poly,     (0.01, 0.5),   [(1e-6, 1.0), (0.01, 5)]),
    ("F = exp(-αx)",        F2_exp,      (0.01,),        [(1e-6, 1.0)]),
    ("F = (1 + αx)^{-p}",   F3_lorentz,  (0.01, 0.5),   [(1e-6, 1.0), (0.01, 5)]),
    ("F = 1/(1 + αx)",      F4_powerlaw, (0.01,),        [(1e-6, 1.0)]),
]:
    res = minimize(chi2_F, params0, args=(func,), method='L-BFGS-B', bounds=bounds)
    if res.success:
        chi2 = res.fun
        rms = np.sqrt(chi2/len(data))
        params_str = f"{res.x[0]:.4e}"
        p_str = f"{res.x[1]:.3f}" if len(res.x) > 1 else "—"
        print(f"  {label:<24} {params_str:>10} {p_str:>8} {chi2:>10.4f} {rms:>12.4f}")
        if chi2 < best_chi2:
            best_chi2 = chi2; best_form = (label, res.x)

# Affichage des résidus particule par particule
print("\n" + "─" * 78)
print("  RÉSIDUS PARTICULE PAR PARTICULE")
print("─" * 78)
print(f"\n  {'particule':<10} {'cfg':<25} {'m_exp':>9} {'Ω_ts':>9} "
      f"{'δ%':>8} {'x=m/E_0':>10} {'1-F':>10}")
print("  " + "-" * 82)
for nm, cfg, m_exp, m_pred, delta, log_d, c_total, n, sm in sorted(data, key=lambda d: d[3]):
    err_pct = (m_exp/m_pred - 1)*100
    x = m_pred/E_0
    one_minus_F = 1 - delta
    print(f"  {nm:<10} {'+'.join(cfg)[:24]:<25} {m_exp:>9.2f} {m_pred:>9.2f} "
          f"{err_pct:>+7.2f}% {x:>10.1e} {one_minus_F:>+10.4f}")

# Recommandation
print("\n" + "═" * 78)
print("  CONCLUSION DE LA CALIBRATION")
print("═" * 78)
print()
mean_residual = log_deltas.mean()
rms_residual = log_deltas.std()
print(f"  Résidu log moyen : {mean_residual:+.4f}")
print(f"  Résidu log RMS   : {rms_residual:.4f}")
print()
if abs(mean_residual) < 0.005:
    print("  → V5 est NON-BIAISÉ en moyenne. F(x) doit avoir F(0)=1 et corriger")
    print("    surtout aux extrêmes (très haute énergie). Pas de gain global évident.")
elif mean_residual < -0.005:
    print(f"  → V5 SURPREDIT en moyenne ({mean_residual*100:.2f}% trop haut).")
    print(f"    V6 doit appliquer F<1 partout, F→1 à E→0.")
else:
    print(f"  → V5 SOUSPREDIT en moyenne ({mean_residual*100:.2f}% trop bas).")
    print(f"    V6 doit appliquer F>1 partout (effet INVERSE du desserrage attendu).")
print()
if best_form:
    print(f"  Meilleure forme F(x) : {best_form[0]}")
    print(f"  Paramètres optimaux  : {best_form[1]}")
    print(f"  Réduction χ² : {best_chi2:.4f} ({np.sqrt(best_chi2/len(data))*100:.2f}% RMS)")
    print()
    # Comparer au RMS sans correction
    rms_no_correction = np.sqrt(np.sum(log_deltas**2)/len(data))
    gain = (1 - np.sqrt(best_chi2/len(data))/rms_no_correction)*100
    print(f"  Sans correction V6  : RMS = {rms_no_correction*100:.2f}%")
    print(f"  Avec correction V6  : RMS = {np.sqrt(best_chi2/len(data))*100:.2f}%")
    print(f"  Gain                 : {gain:+.1f}% réduction du résidu")
