r"""
TQNT-V6 KONTSEVICH — invariants de Vassiliev pour le spectre de masse.

Idée : la formule Ω_ts est essentiellement la troncature à n=2 de
l'intégrale de Kontsevich (Jones polynomial standard). On l'enrichit
par les coefficients de Vassiliev d'ordre supérieur :

  v_2(K) = -V''(1) / 6      ← Casson invariant (déjà présent via Jones)
  v_3(K) = -V'''(1)/36 + V''(1)/12   ← invariant cubique
  v_4(K) = V''''(1)/240 + ...        ← invariant quartique

Ces invariants sont les coefficients de la SÉRIE DE KONTSEVICH
développée autour de q=1 :

  V(K, q) = 1 + 0 + v_2(K) (q-1)² + v_3(K) (q-1)³ + v_4(K) (q-1)⁴ + …

Pour un entrelacs L = ⊔ K_i, on prend (additivité) :
  v_n(L) = Σ v_n(K_i)

Formule V6 candidate :
  m_V6(L) = Ω_ts(L) · F(v_2, v_3, v_4) où F est à calibrer.

Test : régresser les résidus V5 sur (v_2, v_3, v_4) et voir si on
réduit le RMS de 1.5% (propre) vers <0.5%.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize

from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin

# ───────── précalcul ─────────
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
               "det": jones_determinant(pj), "s": topological_spin(K),
               "jones": pj}
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


# ═════════════════════════════════════════════════════════════════════
#  CALCUL DES INVARIANTS DE VASSILIEV via Jones polynomial
# ═════════════════════════════════════════════════════════════════════

def jones_derivatives_at_1(poly: dict[int, float], order: int = 4) -> list[float]:
    """Dérivées de V(K)(t) à t=1, jusqu'à l'ordre demandé.

    V(K)(t) = Σ_p c_p · t^p
    V^(k)(t) = Σ_p c_p · p · (p-1) · ... · (p-k+1) · t^(p-k)
    V^(k)(1) = Σ_p c_p · p · (p-1) · ... · (p-k+1)
    """
    derivs = []
    for k in range(order + 1):
        val = 0.0
        for p, c in poly.items():
            # facteur descendant p(p-1)...(p-k+1)
            fact = 1.0
            for i in range(k):
                fact *= (p - i)
            val += c * fact
        derivs.append(val)
    return derivs


def vassiliev_invariants(K: str) -> dict[str, float]:
    """Calcule v_2, v_3, v_4 pour le nœud K à partir de V(K)(t).

    Conventions :
      v_2 = -V''(1) / 6
      v_3 = -V'''(1)/36 - V''(1)/12  (Bar-Natan)
      v_4 ≈ V''''(1)/720 + v_2²/2
    """
    poly = TOPO[K]["jones"]
    d = jones_derivatives_at_1(poly, 4)
    V0, V1, V2, V3, V4 = d
    v2 = -V2 / 6.0
    v3 = -V3 / 36.0 - V2 / 12.0
    v4 = V4 / 720.0 + v2 * v2 / 2.0
    return {"V_at_1": V0, "v2": v2, "v3": v3, "v4": v4}


# Précalculer pour les 19 nœuds
VAS = {K: vassiliev_invariants(K) for K in KNOTS}

print("\n  Invariants de Vassiliev sur les 19 nœuds :")
print(f"  {'nœud':<8} {'V(1)':>8} {'v_2 (Casson)':>13} {'v_3':>10} {'v_4':>10}")
print("  " + "-" * 55)
for K in KNOTS:
    v = VAS[K]
    print(f"  {K:<8} {v['V_at_1']:>8.2f} {v['v2']:>13.3f} {v['v3']:>10.3f} {v['v4']:>10.3f}")


def vassiliev_link(comps: list[str]) -> dict[str, float]:
    """v_n pour un entrelacs L = ⊔ K_i (additif au premier ordre)."""
    n2 = sum(VAS[K]["v2"] for K in comps)
    n3 = sum(VAS[K]["v3"] for K in comps)
    n4 = sum(VAS[K]["v4"] for K in comps)
    return {"v2": n2, "v3": n3, "v4": n4}


# ═════════════════════════════════════════════════════════════════════
#  PDG avec configurations identifiées
# ═════════════════════════════════════════════════════════════════════
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
    ("Σ_c(2455)",["4_1","5_1","7_2"],  2453.97),
    ("Ξ_c⁺",     ["3_1","4_1","8_1"],  2467.94),
    ("Ξ_c⁰",     ["3_1","4_1","8_1"],  2470.9),
    ("Ω_c",      ["4_1","4_1","7_7"],  2695.2),
    ("Λ_b",      ["6_3","7_4"],        5619.6),
    ("W±",       ["7_1","7_2","8_3"],  80369),
    ("Z⁰",       ["7_2","8_3","9_1"],  91188),
    ("H",        ["6_3","8_2","8_3"],  125250),
    ("t",        ["6_3","7_1","7_5"],  172690),
]

# ═════════════════════════════════════════════════════════════════════
#  ANALYSE
# ═════════════════════════════════════════════════════════════════════
print("\n" + "═"*82)
print("  TQNT-V6 KONTSEVICH — régression des résidus V5 sur v_n")
print("═"*82)

data = []
for nm, cfg, m_exp in PDG:
    m_pred = omega(cfg)
    res_log = np.log(m_exp / m_pred)
    res_pct = (m_exp - m_pred) / m_exp * 100
    vas = vassiliev_link(cfg)
    data.append((nm, cfg, m_exp, m_pred, res_log, res_pct,
                 vas["v2"], vas["v3"], vas["v4"]))

# Statistiques globales
res_logs = np.array([d[4] for d in data])
print(f"\n  Sur {len(data)} particules :")
print(f"    RMS log(m_exp/Ω_ts) = {np.sqrt(np.mean(res_logs**2)):.4f}")
print(f"    RMS résidu en %     = {np.sqrt(np.mean(np.array([d[5] for d in data])**2)):.3f}%")

# Corrélations avec v_2, v_3, v_4
v2s = np.array([d[6] for d in data])
v3s = np.array([d[7] for d in data])
v4s = np.array([d[8] for d in data])
print(f"\n  Corrélations résidu log vs invariants Vassiliev :")
for name, x in [("v_2", v2s), ("v_3", v3s), ("v_4", v4s)]:
    if np.var(x) > 0:
        r = np.corrcoef(res_logs, x)[0, 1]
        print(f"    log(m_exp/Ω_ts) ~ {name} : r = {r:+.3f}")

# ═════════════════════════════════════════════════════════════════════
#  RÉGRESSION MULTIVARIÉE : log(résidu) = β_2 v_2 + β_3 v_3 + β_4 v_4
# ═════════════════════════════════════════════════════════════════════
print("\n  ─── RÉGRESSION MULTIVARIÉE ───")

# Régression linéaire sans intercept
X = np.column_stack([v2s, v3s, v4s])
y = res_logs
betas, *_ = np.linalg.lstsq(X, y, rcond=None)
y_pred = X @ betas
res_corrected = y - y_pred
print(f"\n  Coefficients optimaux :")
print(f"    β_2 (Casson)    = {betas[0]:+.6f}")
print(f"    β_3 (cubique)   = {betas[1]:+.6f}")
print(f"    β_4 (quartique) = {betas[2]:+.6f}")

rms_before = np.sqrt(np.mean(y**2))
rms_after = np.sqrt(np.mean(res_corrected**2))
print(f"\n  RMS log résidus :")
print(f"    avant correction Vassiliev : {rms_before:.4f}")
print(f"    après correction Vassiliev : {rms_after:.4f}")
print(f"    Réduction : {(1 - rms_after/rms_before)*100:.1f}%")

# Convertir en pourcentages
rms_pct_before = np.sqrt(np.mean(np.array([d[5] for d in data])**2))
rms_pct_after = rms_pct_before * (rms_after/rms_before)
print(f"\n  Équivalent en pourcentage :")
print(f"    RMS V5 brut            : {rms_pct_before:.3f}%")
print(f"    RMS V5 + Kontsevich n≤4: {rms_pct_after:.3f}%")

# ═════════════════════════════════════════════════════════════════════
#  RÉGRESSIONS PARTIELLES (v_2 seul, v_2+v_3, v_2+v_3+v_4)
# ═════════════════════════════════════════════════════════════════════
print("\n  ─── RÉGRESSIONS PARTIELLES ───")
print(f"  {'modèle':<28} {'RMS log':>10} {'RMS %':>10} {'gain':>8}")
print("  " + "-" * 60)
for label, X_partial in [
    ("Ω_ts seul (V5)", np.zeros((len(data), 0))),
    ("Ω_ts + v_2", v2s[:, None]),
    ("Ω_ts + v_2 + v_3", np.column_stack([v2s, v3s])),
    ("Ω_ts + v_2 + v_3 + v_4", X),
]:
    if X_partial.shape[1] == 0:
        rms_log = np.sqrt(np.mean(y**2))
    else:
        b, *_ = np.linalg.lstsq(X_partial, y, rcond=None)
        rms_log = np.sqrt(np.mean((y - X_partial @ b)**2))
    rms_p = rms_log / np.sqrt(np.mean(y**2)) * rms_pct_before
    gain = (1 - rms_log/np.sqrt(np.mean(y**2)))*100
    print(f"  {label:<28} {rms_log:>10.5f} {rms_p:>9.3f}% {gain:>+7.1f}%")

# ═════════════════════════════════════════════════════════════════════
#  AFFICHAGE DES PARTICULES "CORRIGÉES"
# ═════════════════════════════════════════════════════════════════════
print("\n  ─── EFFET PARTICULE PAR PARTICULE (correction v_n=2,3,4) ───")
print(f"  {'particule':<10} {'m_exp':>9} {'Ω_ts':>9} {'Δ V5':>8} {'Δ V6':>8} {'gain':>8}")
print("  " + "-" * 66)
for i, (nm, cfg, m_exp, m_pred, _, res_pct, _, _, _) in enumerate(data):
    res_log_corr = res_corrected[i]
    res_pct_corr = (np.exp(res_log_corr) - 1) * 100
    gain = abs(res_pct) - abs(res_pct_corr)
    print(f"  {nm:<10} {m_exp:>9.2f} {m_pred:>9.2f} {res_pct:>+7.2f}% {res_pct_corr:>+7.2f}% {gain:>+7.2f}")

# ═════════════════════════════════════════════════════════════════════
#  CONCLUSION
# ═════════════════════════════════════════════════════════════════════
print("\n" + "═"*82)
print("  CONCLUSION")
print("═"*82)
print(f"""
  Hypothèse Kontsevich V6 : m_V6 = Ω_ts · exp(β_2 v_2 + β_3 v_3 + β_4 v_4)

  Résultats numériques sur {len(data)} particules :
    • RMS V5 brut             : {rms_pct_before:.3f}%
    • RMS V5 + ordre 2-4 Vass.: {rms_pct_after:.3f}%
    • Gain                    : {(rms_pct_before - rms_pct_after)/rms_pct_before*100:.1f}%

  Coefficients calibrés :
    β_2 = {betas[0]:+.5f}  (Casson, ordre 2)
    β_3 = {betas[1]:+.5f}  (cubique, ordre 3 — NOUVEAU pour TQNT)
    β_4 = {betas[2]:+.5f}  (quartique, ordre 4)

  Interprétation :
    Si gain >> 0 : les invariants de Vassiliev d'ordre >=3 contiennent
    de l'information que Ω_ts ne capture pas. La voie Kontsevich est
    une extension naturelle de la TQNT-V5.

    Si gain ≈ 0 : les v_n>=2 sont déjà encodés dans Ω_ts via det(K) et
    autres invariants topologiques utilisés. Pas de gain immédiat.
""")
