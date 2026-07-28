r"""
DÉRIVATION DE Ω_ts À BOUCLE FINIE — approche phénoménologique.

Le développement en boucles de Chern-Simons donne :
  m_K = m_K^(0)  +  (1/(k+2)) · m_K^(1)  +  (1/(k+2))^2 · m_K^(2)  +  ...

où m_K^(0) = Ω_ts strict, et les corrections m_K^(n) sont des fonctionnels
d'invariants du nœud (signature, hyperbolic volume, Casson-Gordon, ...).

À ce stade, on identifie EMPIRIQUEMENT la structure du terme NLO en
testant divers ansätze sur les résidus Ω_ts strict.

Ansätze testés :
  A1 : m = Ω · (1 + α/Ω)              [correction additive constante]
  A2 : m = Ω · (1 + α/Ω^p)             [puissance fittée]
  A3 : m = Ω · (1 + α/Σ U_κU_τ)        [inverse de l'énergie de courbure]
  A4 : m = Ω · (1 + α·class_factor/Ω)  [dépendance par classe]
  A5 : m = Ω · (1 + α/(k+2)·f(K))      [forme CS 1-loop]

avec k = 8 (niveau SU(2) de Ω_ts) ou k = 135 (niveau α_EM).
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize_scalar, minimize
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
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def omega_strict(components):
    n = len(components); ws=[]; Us=[]
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


# ───────── Données : assignations optimales ─────────
PARTICULES = [
    # (nm, cfg, m_exp, classe, 2J)
    ("μ⁻",   ["3_1","6_1"],          105.66,   "lepton", 1),
    ("τ⁻",   ["5_1","5_1","6_1"],   1776.86,   "lepton", 1),
    ("π̄",    ["3_1","3_1","3_1"],   137.27,    "pseudo", 0),
    ("η",    ["3_1","5_2","6_1"],   547.86,    "pseudo", 0),
    ("K̄",    ["6_1","6_1","6_1"],   495.65,    "pseudo", 0),
    ("D̄",    ["3_1","4_1","6_2"],  1867.0,     "pseudo", 0),
    ("J/ψ",  ["3_1","4_1","6_3"],  3096.9,     "vector", 2),
    ("p",    ["3_1","7_2","7_2"],   938.27,    "baryon", 1),
    ("n",    ["3_1","7_2","7_2"],   939.57,    "baryon", 1),
    ("Λ_c",  ["5_1","5_1","7_2"],  2286.46,    "baryon", 1),
    ("Σ_c",  ["4_1","5_1","7_1"],  2453.0,     "baryon", 1),
    ("Λ",    ["5_1","5_1","5_1"],  1115.68,    "baryon", 1),
]


# Précalcul des Ω et invariants
print("\n  Particules et résidus Ω_ts strict :")
print(f"  {'nm':<6} {'config':<22} {'cls':<8} {'2J':>3} {'Ω_ts':>9} "
      f"{'m_exp':>9} {'Δm MeV':>9} {'Δ %':>8}")
print("  " + "-" * 90)

data = []
for nm, cfg, m_exp, cls, two_J in PARTICULES:
    omega = omega_strict(cfg)
    sum_U = sum(U_KT[K][0]*U_KT[K][1] for K in cfg)
    sum_cr = sum(TOPO[K]["cr"] for K in cfg)
    Dm = m_exp - omega
    d_pct = Dm/m_exp*100
    print(f"  {nm:<6} {'+'.join(cfg):<22} {cls:<8} {two_J:>3} "
          f"{omega:>9.2f} {m_exp:>9.2f} {Dm:>+8.2f} {d_pct:>+7.2f}%")
    data.append({"nm": nm, "cfg": cfg, "omega": omega, "m_exp": m_exp,
                  "Dm": Dm, "d_pct": d_pct, "cls": cls, "two_J": two_J,
                  "sum_U": sum_U, "sum_cr": sum_cr})


# ───────── Ansatz A2 : m_exp = Ω · (1 + α/Ω^p) ─────────
print("\n" + "═"*100)
print("  Ansatz A2 : m = Ω · (1 + α·Ω^(-p))")
print("═"*100)


def fit_A2(data):
    """Fit α et p tels que m_pred = Ω·(1 + α/Ω^p) minimise RMS."""
    def chi2(params):
        alpha, p = params
        total = 0
        for d in data:
            corr = alpha / d["omega"]**p
            m_p = d["omega"] * (1 + corr)
            total += ((m_p - d["m_exp"])/d["m_exp"])**2
        return total
    res = minimize(chi2, [10.0, 0.5], method='Nelder-Mead')
    return res.x, np.sqrt(res.fun/len(data))*100


(alpha, p_exp), rms_A2 = fit_A2(data)
print(f"  α = {alpha:.4f}, p = {p_exp:.4f}")
print(f"  RMS post-A2 : {rms_A2:.3f}%  (vs RMS Ω_ts strict avant correction)")

# Détail
print(f"\n  {'nm':<6} {'Ω_ts':>9} {'corr_A2':>10} {'m_A2':>9} "
      f"{'Δ avant':>9} {'Δ après':>9}")
for d in data:
    corr = alpha / d["omega"]**p_exp
    m_A2 = d["omega"] * (1 + corr)
    d_after = (m_A2 - d["m_exp"])/d["m_exp"]*100
    print(f"  {d['nm']:<6} {d['omega']:>9.2f} {corr:>10.4f} {m_A2:>9.2f} "
          f"{d['d_pct']:>+8.2f}% {d_after:>+8.2f}%")


# ───────── Ansatz A4 : correction par classe ─────────
print("\n" + "═"*100)
print("  Ansatz A4 : m = Ω · (1 + α_classe / Ω)")
print("═"*100)


def fit_A4(data):
    """Fit α par classe : α_pseudo, α_lepton, α_baryon, α_vector."""
    classes = sorted(set(d["cls"] for d in data))
    n_p = len(classes)
    def chi2(params):
        cls_alpha = dict(zip(classes, params))
        total = 0
        for d in data:
            corr = cls_alpha[d["cls"]] / d["omega"]
            m_p = d["omega"] * (1 + corr)
            total += ((m_p - d["m_exp"])/d["m_exp"])**2
        return total
    res = minimize(chi2, [0.0]*n_p, method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-10})
    return dict(zip(classes, res.x)), np.sqrt(res.fun/len(data))*100


alpha_cls, rms_A4 = fit_A4(data)
print(f"  α par classe :")
for cls, a in alpha_cls.items():
    print(f"    {cls:<8} : α = {a:+.4f}  (en MeV après mult par Ω/Ω = MeV)")
print(f"  RMS post-A4 : {rms_A4:.3f}%")

print(f"\n  {'nm':<6} {'cls':<8} {'Ω_ts':>9} {'α/Ω':>8} {'m_A4':>9} "
      f"{'Δ avant':>9} {'Δ après':>9}")
for d in data:
    corr = alpha_cls[d["cls"]] / d["omega"]
    m_A4 = d["omega"] * (1 + corr)
    d_after = (m_A4 - d["m_exp"])/d["m_exp"]*100
    print(f"  {d['nm']:<6} {d['cls']:<8} {d['omega']:>9.2f} {corr:>+8.4f} "
          f"{m_A4:>9.2f} {d['d_pct']:>+8.2f}% {d_after:>+8.2f}%")


# ───────── Ansatz A5 : forme CS 1-loop par knot ─────────
print("\n" + "═"*100)
print("  Ansatz A5 : m = Ω · (1 + Σ_K c_1(K)·count_K / (k+2))")
print("═"*100)


def fit_A5(data, k_level=8):
    """Fit c_1(K) pour chaque type de nœud."""
    # Identifier les knots présents
    all_knots = set()
    for d in data:
        all_knots.update(d["cfg"])
    knots_list = sorted(all_knots)

    def chi2(params):
        c1 = dict(zip(knots_list, params))
        total = 0
        for d in data:
            counts = Counter(d["cfg"])
            corr = sum(c1[K] * c for K, c in counts.items()) / (k_level + 2)
            m_p = d["omega"] * (1 + corr)
            total += ((m_p - d["m_exp"])/d["m_exp"])**2
        return total

    n_p = len(knots_list)
    res = minimize(chi2, [0.0]*n_p, method='Nelder-Mead',
                   options={'maxiter': 5000, 'fatol': 1e-12})
    return dict(zip(knots_list, res.x)), np.sqrt(res.fun/len(data))*100


for k_test in [8, 135]:
    print(f"\n  Niveau k = {k_test} (Ω → SU(2)_{k_test})")
    c1_K, rms_A5 = fit_A5(data, k_test)
    print(f"  c_1(K) ajusté par nœud :")
    for K, c in c1_K.items():
        print(f"    c_1({K:<5}) = {c:+.5f}")
    print(f"  RMS post-A5 : {rms_A5:.3f}%")


# ───────── Synthèse ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE")
print("═"*100)

rms_initial = float(np.sqrt(np.mean(np.array([d["d_pct"]**2 for d in data]))))
print(f"""
  RMS Ω_ts strict initial : {rms_initial:.3f}%

  Comparaison des ansätze :
    A2 (m = Ω·(1 + α/Ω^p))      : RMS = {rms_A2:.3f}%
    A4 (correction par classe)  : RMS = {rms_A4:.3f}%
    A5 (CS 1-loop par knot)     : RMS = {rms_A5:.3f}%

  Interprétation :
    Si A5 < 0.5% → la structure CS 1-loop par knot est correcte
    Si A4 ~ 0.5-1% → la correction par classe (pseudo/baryon/...) marche
    Si A2 ~ A4 ~ A5 ~ niveau du résidu → simple inversion de la masse

  Le meilleur ansatz révèle la structure manquante dans Ω_ts.
""")
