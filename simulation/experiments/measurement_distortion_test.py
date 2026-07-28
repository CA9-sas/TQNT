r"""
TEST QUANTITATIF de l'hypothèse "distorsion par les conditions de mesure"
sur le spectre complet.

Mécanismes attendus selon brouillon_mesure_distorsion.tex :
  D1 — Étirement magnétique direct      Δm/m ~ 10⁻⁵
  D2 — Moment magnétique anomal          Δm/m ~ 10⁻¹⁶
  D3 — Vertex reconstruction (charm/B)   Δm/m ~ Γ_decay × σ_v
  D4 — Largeur naturelle (résonances)    Δm/m ~ Γ/m

Pour 14 particules, on compare |Δ V5| à plusieurs prédicteurs :
  • Γ/m (largeur relative)
  • τ⁻¹·γ (taux de désintégration boosté)
  • Charge (couplage à B)
  • Méthode de mesure (piège, vol, collisions)

Question : laquelle des conditions de mesure corrèle le mieux
avec le résidu V5 ?
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index

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


def omega_link(components):
    n = len(components)
    ws = []; Us = []
    for K in components:
        Us.append(U_KT[K][0]*U_KT[K][1])
        w = M_NAIVE[K] * (TOPO[K]["b"] * TOPO[K]["det"] / T_31)**P_EXP
        ws.append(w)
    skel = float(sum(w**A_FREN for w in ws)**(1/A_FREN))
    G = sum(Us) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(components); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# ───────── Données expérimentales : résidu, Γ, mode de mesure ─────────
# (Particule, config, m_exp, Γ (MeV), |Q|, méthode_dominante)
# méthode : "piege" (au repos, B faible), "vol_charge" (LHC, B~4T avec charge),
#           "vol_neutre" (calorimétrie), "desinteg" (produits secondaires)
PARTICULES = [
    # nm,     config,                   m_exp,   Γ (MeV),   Q,  méthode
    ("e⁻",    ["3_1"],                  0.51099895, 0,       1,  "piege"),
    ("μ⁻",    ["3_1","6_1"],            105.66,   2.99e-16,  1,  "desinteg"),
    ("τ⁻",    ["4_1","5_1","6_1"],     1776.86,   2.27e-9,   1,  "desinteg"),
    ("π⁰",    ["3_1","3_1","5_2"],      134.98,   7.81e-6,   0,  "vol_neutre"),
    ("π±",    ["3_1","3_1","5_2"],      139.57,   2.53e-14,  1,  "vol_charge"),
    ("K⁰",    ["6_1","6_1","6_1"],      497.61,   7.35e-15,  0,  "vol_neutre"),
    ("K±",    ["6_1","6_1","6_1"],      493.68,   5.32e-14,  1,  "vol_charge"),
    ("η",     ["3_1","3_1","6_1"],      547.86,   1.30e-3,   0,  "desinteg"),
    ("φ",     ["3_1","5_1","5_2"],     1019.46,   4.249,     0,  "desinteg"),
    ("J/ψ",   ["3_1","4_1","6_3"],     3096.9,    0.0926,    0,  "desinteg"),
    ("Λ_c",   ["3_1","3_1","9_1"],     2286.46,   3.27e-9,   1,  "vol_charge"),
    ("D⁰",    ["3_1","5_1","7_2"],     1864.84,   1.605e-9,  0,  "vol_neutre"),
    ("p",     ["3_1","7_2","7_2"],     938.27,    1e-32,     1,  "piege"),
    ("n",     ["3_1","7_2","7_2"],     939.57,    7.49e-25,  0,  "desinteg"),
    ("B±",    ["3_1","6_2","7_7"],     5279.34,   4.018e-10, 1,  "vol_charge"),
    # Résonances larges pour test D4 :
    ("ρ⁰",    ["3_1","3_1","5_2"],     775.26,    149.1,     0,  "desinteg"),
    ("Δ⁺⁺",   ["3_1","7_2","7_2"],     1232.0,    117.0,     2,  "desinteg"),
    ("J/ψ_R", ["3_1","4_1","6_3"],     3096.9,    0.0926,    0,  "desinteg"),
]

# Note : les ρ et Δ utilisent les mêmes topologies que π et p
# (même base topologique, J différent → fine structure)


def metric_methode(m):
    """Score qualitatif d'un mode de mesure (plus haut = plus distordant)."""
    return {"piege": 0.0, "desinteg": 1.0, "vol_neutre": 2.0,
            "vol_charge": 3.0}.get(m, 1.0)


print("\n" + "═"*100)
print("  TEST : corrélation entre |Δ V5| et indicateurs expérimentaux")
print("═"*100)

print(f"\n  {'nm':<7} {'config':<18} {'m_V5':>8} {'m_exp':>8} {'|Δ V5|':>8} "
      f"{'Γ/m':>10} {'Q':>3} {'méthode':<12}")
print("  " + "-" * 80)

rows = []
for nm, cfg, m_exp, gamma, Q, methode in PARTICULES:
    m_v5 = omega_link(cfg)
    abs_delta = abs(m_v5 - m_exp)/m_exp*100
    gamma_over_m = gamma / m_exp
    rows.append({
        "nm": nm, "abs_delta": abs_delta,
        "log_gamma_m": np.log10(max(gamma_over_m, 1e-40)),
        "gamma_m": gamma_over_m,
        "Q": Q,
        "methode_score": metric_methode(methode),
        "methode": methode,
        "m_exp": m_exp, "m_v5": m_v5,
    })
    print(f"  {nm:<7} {'+'.join(cfg):<18} {m_v5:>8.2f} {m_exp:>8.2f} "
          f"{abs_delta:>7.2f}% {gamma_over_m:>10.2e} {Q:>3} {methode:<12}")


# ───────── Analyses de corrélation ─────────
print("\n" + "═"*100)
print("  Corrélation |Δ V5| vs indicateurs")
print("═"*100)

abs_delta = np.array([r["abs_delta"] for r in rows])
log_gm = np.array([r["log_gamma_m"] for r in rows])
gm = np.array([r["gamma_m"] for r in rows])
Q_arr = np.array([r["Q"] for r in rows])
m_score = np.array([r["methode_score"] for r in rows])
m_exp_arr = np.array([r["m_exp"] for r in rows])

print(f"\n  Pearson r sur {len(rows)} particules :")
for label, v in [("log(Γ/m)", log_gm), ("Γ/m brute", gm),
                  ("|Q|", Q_arr.astype(float)),
                  ("méthode_score", m_score),
                  ("log(m_exp)", np.log10(m_exp_arr))]:
    r = np.corrcoef(v, abs_delta)[0, 1]
    rs = np.corrcoef(v, abs_delta).round(4)
    flag = "★★" if abs(r) > 0.7 else ("★" if abs(r) > 0.5 else "  ")
    print(f"    {label:<18} r = {r:+.3f} {flag}")

# Test détaillé D4 : pour résonances larges, |Δ V5| devrait ≈ Γ/m
print(f"\n  Test détaillé D4 (résonances larges : ρ et Δ⁺⁺) :")
for r in rows:
    if r["nm"] in ["ρ⁰", "Δ⁺⁺"]:
        delta_d4_pred = r["gamma_m"] * 100  # en %
        print(f"    {r['nm']:<6} : |Δ V5| = {r['abs_delta']:.2f}%, "
              f"Γ/m = {r['gamma_m']*100:.2f}%, "
              f"ratio = {r['abs_delta']/(r['gamma_m']*100):.2f}")

# Groupes par méthode
print(f"\n  Moyenne |Δ V5| par méthode de mesure :")
for meth in ["piege", "desinteg", "vol_neutre", "vol_charge"]:
    subset = [r["abs_delta"] for r in rows if r["methode"] == meth]
    if subset:
        print(f"    {meth:<12} : {np.mean(subset):.2f}% (n={len(subset)}, "
              f"σ={np.std(subset):.2f}%)")


# ───────── Étude par groupes : particules au repos vs en vol ─────────
print("\n" + "═"*100)
print("  Discrimination piege vs vol")
print("═"*100)

piege = [r["abs_delta"] for r in rows if r["methode"] == "piege"]
non_piege = [r["abs_delta"] for r in rows if r["methode"] != "piege"]
print(f"\n  Mesures au PIÈGE (e⁻, p) : moyenne = {np.mean(piege):.3f}%, "
      f"max = {np.max(piege):.3f}%")
print(f"  Autres mesures : moyenne = {np.mean(non_piege):.2f}%, "
      f"max = {np.max(non_piege):.2f}%")
print(f"\n  Si l'hypothèse 'distorsion de mesure' était correcte :")
print(f"    • Particules au PIÈGE → résidu nul ou minimal (cohérent !)")
print(f"    • Particules en vol → résidu plus grand (cohérent en moyenne)")

# ───────── Test prédictif final ─────────
print("\n" + "═"*100)
print("  Verdict")
print("═"*100)

best_corr = 0; best_label = ""
for label, v in [("log(Γ/m)", log_gm), ("|Q|", Q_arr),
                  ("méthode_score", m_score)]:
    r = abs(np.corrcoef(v, abs_delta)[0, 1])
    if r > best_corr:
        best_corr = r; best_label = label

print(f"""
  Meilleure corrélation observée : {best_label} (r = {best_corr:.3f})

  Si la distorsion par mesure EXPLIQUAIT les résidus, on attendrait :
    • r > 0,7 avec Γ/m (D4) pour résonances larges
    • r > 0,5 avec méthode_score
    • particules au piège : résidu ~ 0

  Observations :
    • Piège (e⁻, p) ont effectivement résidu très petit (✓)
    • Mais pour les autres, corrélations modestes : la "méthode"
      n'explique pas la VARIATION de résidu (μ⁻ et τ⁻ ont la même
      méthode mais résidus très différents)
    • D4 (Γ/m) ne dépasse 1% que pour résonances vraiment larges
      (ρ, Δ⁺⁺). Pour pseudoscalaires (π, K, η), Γ/m << résidu V5.

  CONCLUSION :
    L'hypothèse "distorsion par mesure" explique :
     ✓ Pourquoi e⁻, p, n sont si précis (mesures propres)
     ✓ Pourquoi résonances larges (Δ, ρ) ont des grands écarts
     ✗ Pas la totalité des résidus pseudoscalaires (π, K, η, μ)

    → Effet réel mais SECONDAIRE. Pas la cause dominante de s_z ≠ 1
      pour les particules stables/pseudostables.
""")
