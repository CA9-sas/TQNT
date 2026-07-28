r"""
Validation étendue de m_c et m_b sur tout le secteur lourd PDG.

Pour chaque hadron contenant c ou b, on extrait m_eff via :
  m_eff(q) = [m_TQNT(H) - Σ_{q'≠q} N_{q'} · m_eff(q')_const] / N_q

avec les masses constituantes déjà obtenues :
  m_u = 311 MeV,  m_d = 315 MeV,  m_s = 513 MeV

Test :
  - Charme : D mesons, charmonium (η_c, J/ψ, ψ', χ_c), baryons Λ_c, Σ_c, Ξ_c, Ω_c
  - Beauté : B mesons (B_s, B_c), bottomonium (η_b, Υ, χ_b), baryons Λ_b, Ξ_b, Ω_b

On distingue les états fondamentaux (1S) des états excités (2S, 1P, 2P) pour
détecter l'énergie d'excitation interne dans la masse extraite.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


# ─── Constantes ───
M_E = 0.51099895
F_KAPPA = 0.8682
M_C = M_E * F_KAPPA / 3
M_L = M_E * (1 - F_KAPPA) / 3
A_F = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); BB = 33/20; RHO_0 = 10 - 3/(8*np.pi)

KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3",
         "7_1","7_2","7_3","7_4","7_5","7_6","7_7",
         "8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}

print("Précalcul Frenet+Jones…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 30000)
    ds = g["speed"]*2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)), float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj),
               "U": U_KT[K][0]*U_KT[K][1]}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def omega_ts(cfg):
    n = len(cfg)
    ws = [M_NAIVE[K] * (TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in cfg]
    skel = float(sum(w**A_F for w in ws)**(1/A_F))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in cfg) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(cfg); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2:
            Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# ─── Masses constituantes TQNT (obtenues précédemment) ───
M_CONST = {"u": 311.0, "d": 315.0, "s": 513.0}


# ─── Hadrons avec c (catalogue V5) ───
# (nom, config, q-content = (n_u, n_d, n_s, n_c, n_b), état, PDG)
HADRONS_CHARM = [
    # D mésons (Q = ±1)
    ("D⁰",    ["3_1","5_1","7_2"], (1,0,0,1,0), "1S", 1864.84),  # cū
    ("D±",    ["3_1","5_1","7_2"], (0,1,0,1,0), "1S", 1869.66),  # cd̄
    ("D_s±",  ["3_1","6_1","7_1"], (0,0,1,1,0), "1S", 1968.34),  # cs̄
    ("D*(2007)⁰", ["7_7","9_1"],   (1,0,0,1,0), "1S vec", 2006.85),
    ("D*(2010)±", ["7_7","9_1"],   (0,1,0,1,0), "1S vec", 2010.26),
    # Charmonium (cc̄)
    ("η_c(1S)",   ["6_3","7_7"],   (0,0,0,2,0), "1S",  2983.90),
    ("J/ψ",       ["3_1","4_1","6_3"], (0,0,0,2,0), "1S vec", 3096.90),
    ("η_c(2S)",   ["6_1","8_2"],   (0,0,0,2,0), "2S",  3637.50),
    ("ψ(2S)",     ["6_1","8_2"],   (0,0,0,2,0), "2S vec", 3686.10),
    ("χ_c0(1P)",  ["6_1","7_7","7_7"], (0,0,0,2,0), "1P", 3414.71),
    ("χ_c1(1P)",  ["7_2","8_1","8_1"], (0,0,0,2,0), "1P", 3510.67),
    ("χ_c2(1P)",  ["6_1","8_2"],   (0,0,0,2,0), "1P", 3556.17),
    # Charm baryons
    ("Λ_c⁺",  ["3_1","3_1","9_1"], (1,1,0,1,0), "1S", 2286.46),  # udc
    ("Σ_c⁺⁺", ["4_1","5_1","7_2"], (2,0,0,1,0), "1S", 2454.0),    # uuc
    ("Ξ_c⁺",  ["4_1","5_1","7_2"], (1,0,1,1,0), "1S", 2467.71),  # usc
    ("Ω_c⁰",  ["4_1","4_1","7_7"], (0,0,2,1,0), "1S", 2695.20),  # ssc
]

# ─── Hadrons avec b ───
HADRONS_BEAUTY = [
    # B mésons
    ("B_s⁰",  ["6_3","7_6"],         (0,0,1,0,1), "1S",  5366.92),  # sb̄
    ("B_c±",  ["4_1","7_7","9_1"],   (0,0,0,1,1), "1S",  6274.9),   # cb̄
    # Bottomonium (bb̄)
    ("η_b(1S)",   ["5_1","7_3","8_1"],   (0,0,0,0,2), "1S", 9398.7),
    ("Υ(1S)",     ["6_2","7_1","7_3"],   (0,0,0,0,2), "1S vec", 9460.30),
    ("Υ(2S)",     ["6_2","6_2","8_2"],   (0,0,0,0,2), "2S vec", 10023.26),
    ("χ_b0(1P)",  ["6_1","7_3","7_7"],   (0,0,0,0,2), "1P", 9859.44),
    ("χ_b1(1P)",  ["6_1","7_3","7_7"],   (0,0,0,0,2), "1P", 9892.78),
    ("χ_b2(1P)",  ["6_2","6_2","8_2"],   (0,0,0,0,2), "1P", 9912.21),
    # Bottom baryons
    ("Λ_b⁰",  ["6_3","7_4"],         (1,1,0,0,1), "1S", 5619.60),
    ("Ξ_b⁰",  ["7_3","7_4"],         (1,0,1,0,1), "1S", 5791.90),
    ("Ω_b⁻",  ["7_7","8_2"],         (0,0,2,0,1), "1S", 6046.0),
]


def extract_m_eff(target_q, q_content, cfg, M_const_dict):
    """Extrait m_eff(target_q) à partir de m_TQNT(cfg) en soustrayant
    les contributions constituantes des autres quarks.

    q_content peut être un dict {q:n} ou un tuple (n_u, n_d, n_s, n_c, n_b).
    """
    if isinstance(q_content, dict):
        qd = q_content
    else:
        qd = dict(zip("udscb", q_content))
    n_q = qd[target_q]
    if n_q == 0: return None
    m_H = omega_ts(cfg)
    other_mass = 0
    for q_other, ng in qd.items():
        if q_other == target_q: continue
        if q_other in M_const_dict:
            other_mass += ng * M_const_dict[q_other]
    return (m_H - other_mass) / n_q


# Q-tuple convention: (n_u, n_d, n_s, n_c, n_b)
def q_dict(qt):
    return dict(zip("udscb", qt))


# ════════════════════════════════════════════════════════════
print("\n" + "═"*86)
print(" TEST CHARME — m_eff(c) sur 16 hadrons")
print("═"*86)
print(f"\n  Soustraction des constituants TQNT : m_u={M_CONST['u']}, m_d={M_CONST['d']}, m_s={M_CONST['s']}")
print()
print(f"  {'Hadron':<14} {'État':<6} {'PDG':>9} {'TQNT':>9}  q-content     {'m_eff(c)':>10}")
print("─"*86)

m_eff_c_1S = []  # ground states seulement
m_eff_c_all = []
for name, cfg, qt, state, m_pdg in HADRONS_CHARM:
    qd = q_dict(qt)
    m_T = omega_ts(cfg)
    m_eff_TQNT = extract_m_eff("c", qd, cfg, M_CONST)
    sig = "".join(f"{q}{n}" if n else "" for q, n in qd.items())
    flag = " ← 1S" if state == "1S" else ""
    print(f"  {name:<14} {state:<6} {m_pdg:>9.2f} {m_T:>9.2f}  {sig:<14} {m_eff_TQNT:>10.2f}{flag}")
    m_eff_c_all.append((name, state, m_eff_TQNT))
    if state == "1S":
        m_eff_c_1S.append(m_eff_TQNT)

# Statistiques
mc_1S_avg = float(np.mean(m_eff_c_1S))
mc_1S_std = float(np.std(m_eff_c_1S))
print()
print(f"  Moyenne 1S (ground states uniquement) : m_eff(c) = {mc_1S_avg:.1f} ± {mc_1S_std:.1f} MeV")
print(f"    Nombre de hadrons 1S : {len(m_eff_c_1S)}")
print(f"  PDG MS-bar @ μ=2 GeV : m_c = 1270 MeV  → ratio {mc_1S_avg/1270:.2f}")
print(f"  PDG constituant     : m_c ~ 1550 MeV  → ratio {mc_1S_avg/1550:.2f}")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*86)
print(" TEST BEAUTÉ — m_eff(b) sur 11 hadrons")
print("═"*86)
# Pour B_c, il faut m_eff(c) ; utilisons la valeur 1S moyenne obtenue
M_CONST_ext = {**M_CONST, "c": mc_1S_avg}

print(f"\n  Soustraction des constituants TQNT : m_u={M_CONST['u']}, m_d={M_CONST['d']}, m_s={M_CONST['s']}, m_c={mc_1S_avg:.0f}")
print()
print(f"  {'Hadron':<14} {'État':<6} {'PDG':>9} {'TQNT':>9}  q-content     {'m_eff(b)':>10}")
print("─"*86)

m_eff_b_1S = []
m_eff_b_all = []
for name, cfg, qt, state, m_pdg in HADRONS_BEAUTY:
    qd = q_dict(qt)
    m_T = omega_ts(cfg)
    m_eff_TQNT = extract_m_eff("b", qd, cfg, M_CONST_ext)
    sig = "".join(f"{q}{n}" if n else "" for q, n in qd.items())
    flag = " ← 1S" if state == "1S" else ""
    print(f"  {name:<14} {state:<6} {m_pdg:>9.2f} {m_T:>9.2f}  {sig:<14} {m_eff_TQNT:>10.2f}{flag}")
    m_eff_b_all.append((name, state, m_eff_TQNT))
    if state == "1S":
        m_eff_b_1S.append(m_eff_TQNT)

mb_1S_avg = float(np.mean(m_eff_b_1S))
mb_1S_std = float(np.std(m_eff_b_1S))
print()
print(f"  Moyenne 1S : m_eff(b) = {mb_1S_avg:.1f} ± {mb_1S_std:.1f} MeV")
print(f"    Nombre de hadrons 1S : {len(m_eff_b_1S)}")
print(f"  PDG MS-bar @ μ=2 GeV : m_b = 4180 MeV  → ratio {mb_1S_avg/4180:.2f}")
print(f"  PDG constituant     : m_b ~ 4730 MeV  → ratio {mb_1S_avg/4730:.2f}")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*86)
print(" SYNTHÈSE FINALE — masses constituantes TQNT sur 5 saveurs")
print("═"*86)
print()
print(f"  {'Saveur':<8} {'m^TQNT (MeV)':<15} {'m^const PDG':<14} {'Ratio':<8} {'Nb hadrons 1S':<15}")
print("─"*68)
print(f"  {'u':<8} {'~311':<15} {'~336':<14} {0.93:<8.2f} {'p, n':<15}")
print(f"  {'d':<8} {'~315':<15} {'~340':<14} {0.93:<8.2f} {'p, n':<15}")
print(f"  {'s':<8} {'~513':<15} {'~540':<14} {0.95:<8.2f} {'Λ, Σ, Ω':<15}")
print(f"  {'c':<8} {f'{mc_1S_avg:.0f} ± {mc_1S_std:.0f}':<15} {'~1550':<14} "
      f"{mc_1S_avg/1550:<8.2f} {f'{len(m_eff_c_1S)}':<15}")
print(f"  {'b':<8} {f'{mb_1S_avg:.0f} ± {mb_1S_std:.0f}':<15} {'~4730':<14} "
      f"{mb_1S_avg/4730:<8.2f} {f'{len(m_eff_b_1S)}':<15}")

# RMS sur toutes les saveurs
saveur_data = [
    ("u", 311, 336),
    ("d", 315, 340),
    ("s", 513, 540),
    ("c", mc_1S_avg, 1550),
    ("b", mb_1S_avg, 4730),
]
import math
log_resid = [math.log10(t/p) for _, t, p in saveur_data]
rms = float(np.sqrt(np.mean(np.array(log_resid)**2)))
print()
print(f"  RMS log_10 sur 5 saveurs : {rms:.4f} décade (×{10**rms:.3f} d'écart moyen)")
print(f"  Écart moyen absolu      : {np.mean([abs(t-p)/p for _,t,p in saveur_data])*100:.1f} %")

# Sub-tests : excités vs fondamentaux
print()
print("  ─── Analyse fondamentaux vs excités ───")
print()
print(f"  Charme :")
states_charm = {}
for name, state, m_eff in m_eff_c_all:
    states_charm.setdefault(state, []).append(m_eff)
for st, ms in sorted(states_charm.items()):
    if ms:
        print(f"    {st:<8} : moyenne {np.mean(ms):>7.1f} MeV, n={len(ms)}")
print(f"  Beauté :")
states_beauty = {}
for name, state, m_eff in m_eff_b_all:
    states_beauty.setdefault(state, []).append(m_eff)
for st, ms in sorted(states_beauty.items()):
    if ms:
        print(f"    {st:<8} : moyenne {np.mean(ms):>7.1f} MeV, n={len(ms)}")

print()
print("  Note : les excités (2S, 1P) ont une m_eff systématiquement plus")
print("  élevée — c'est l'énergie d'excitation interne (radial/orbital) qui")
print("  s'ajoute. Le fondamental 1S est la mesure pure de m_q.")
