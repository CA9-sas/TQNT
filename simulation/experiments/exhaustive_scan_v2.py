r"""
SCAN COMBINATOIRE EXHAUSTIF — règle de spin enrichie (Clebsch-Gordan)

Au lieu de la règle P7bis qui n'autorise que J ∈ {1/2, 1, s_max}, on
applique la règle de couplage des moments cinétiques :

    Pour des spins {s_1, ..., s_n}, l'entrelacs admet tout J tel que
        J ∈ {P, P+1, P+2, ..., S_max}
        où S_max = Σ s_i  et  P = S_max mod 1
    (i.e. J ≥ P, J ≤ S_max, et J - S_max ∈ ℤ)

C'est la règle quantique standard : le couplage de plusieurs moments
cinétiques donne tous les J entre |J_max - 2·j_min| et J_max par pas
de 1 ; pour des spins arbitraires, J va de la parité au max.

PHYSIQUE :
- J = J_min = parité  → singulet (anti-aligné, état lié pseudoscalaire/scalaire)
- J = S_max           → multiplet étiré (état totalement symétrique)
- intermédiaires      → orbitales avec L > 0

Pour chaque particule PDG, on cherche la meilleure config (m, J, Q)
parmi les 1349 entrelacs ≤ 3 nœuds.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations_with_replacement

from theory.abraham import frenet_geometry
from theory.jones import (
    jones_polynomial, jones_determinant, braid_index,
    topological_spin, topological_charge, is_half_integer_spin,
)

M_E = 0.51099895
F_KAPPA = 0.8682
M_C = M_E * F_KAPPA / 3
M_L = M_E * (1 - F_KAPPA) / 3

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
N_LOBES = {
    "3_1": 3, "4_1": 2, "5_1": 5, "5_2": 2, "6_1": 2,
    "6_2": 3, "6_3": 3, "7_1": 7, "7_2": 2, "7_3": 3,
    "7_4": 2, "7_5": 3, "7_6": 2, "7_7": 3, "8_1": 2,
    "8_2": 3, "8_3": 3, "9_1": 9, "10_1": 2,
}

A = np.sqrt(7 / (8 * np.pi))
P_EXP = 3 / 4
Q_EXP = 3 / 4
X_EXP = -np.sqrt(6)
B_EXP = 33 / 20
RHO_0 = 10 - 3 / (8 * np.pi)

print("Précalcul Frenet…")
U_KT = {}
TOPO = {}
for K in KNOTS:
    geo = frenet_geometry(K, 50000)
    kappa = geo["kappa"]; tau = geo["tau"]; speed = geo["speed"]
    dt = 2 * np.pi / len(kappa)
    ds = speed * dt
    U_KT[K] = (
        float(np.sum(kappa ** 2 * ds)),
        float(np.sum(tau ** 2 * ds)),
    )
    pj = jones_polynomial(K)
    TOPO[K] = {
        "cr":  int(K.split("_")[0]),
        "n_l": N_LOBES[K],
        "b":   braid_index(K),
        "det": jones_determinant(pj),
        "s":   topological_spin(K),
        "q":   topological_charge(K),
        "half": is_half_integer_spin(K),
    }
P_31 = U_KT["3_1"][0] * U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"] * TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"] * M_C + TOPO[K]["n_l"] * M_L for K in KNOTS}


def operator_mass(comps):
    n = len(comps)
    ws = []
    for K in comps:
        T = TOPO[K]["b"] * TOPO[K]["det"] / T_31
        ws.append(M_NAIVE[K] * T ** P_EXP)
    skeleton = float(sum(w ** A for w in ws) ** (1 / A))
    G_comp = sum(U_KT[K][0] * U_KT[K][1] for K in comps) / P_31
    Phi_n = ((n + 1) / 2) ** (-X_EXP) if n > 0 else 1.0
    c = Counter(comps)
    Phi_pair = 1.0
    for K, count in c.items():
        if count >= 2:
            rho_K = RHO_0 * TOPO[K]["cr"] ** (-B_EXP)
            Phi_pair *= rho_K ** (count - 1)
    return Phi_n * Phi_pair * skeleton * G_comp ** Q_EXP


def allowed_J_set(comps):
    """Ensemble des J accessibles par couplage des spins des composantes.

    Règle Clebsch-Gordan :
      J ∈ {P, P+1, ..., S_max} où P = S_max mod 1

    Retourne un ensemble de valeurs J accessibles (multiples de 1/2).
    """
    S_max = sum(TOPO[K]["s"] for K in comps)
    P_parity = S_max - int(S_max)  # 0 si entier, 0.5 si demi-entier
    n_steps = int(S_max - P_parity) + 1
    return {P_parity + i for i in range(n_steps)}


def link_J_label(comps, J_target):
    """Étiquette de la position de J dans la chaîne CG.

    'singulet' = état fondamental (J = parité)
    'stretched' = état max-aligné (J = S_max)
    'L=n' = n-ième niveau orbital
    """
    S_max = sum(TOPO[K]["s"] for K in comps)
    P_parity = S_max - int(S_max)
    level = int(round(J_target - P_parity))
    if level == 0:
        return "singulet" if J_target == 0 else "fondamental"
    if abs(J_target - S_max) < 1e-6:
        return "stretched"
    return f"L={level}"


def link_charge_topo(comps):
    return sum(TOPO[K]["q"] for K in comps)


# ═══════ Catalogue PDG (122 particules, identique au scan v1) ═══════
PDG = [
    ("e⁻",         0.51099895,  0.5, -1,   "lepton"),
    ("μ⁻",         105.6583755, 0.5, -1,   "lepton"),
    ("τ⁻",         1776.86,     0.5, -1,   "lepton"),
    ("ν_e",        2.2e-6,      0.5,  0,   "neutrino"),
    ("ν_μ",        0.17,        0.5,  0,   "neutrino"),
    ("ν_τ",        15.5,        0.5,  0,   "neutrino"),
    ("γ",          0.0,         1.0,  0,   "boson-jauge"),
    ("W±",         80369.0,     1.0,  1,   "EW"),
    ("Z⁰",         91188.0,     1.0,  0,   "EW"),
    ("H",          125250.0,    0.0,  0,   "EW"),
    ("t",          173100.0,    0.5,  2/3, "EW"),
    ("π⁰",         134.9768,    0.0,  0,   "méson-PS"),
    ("π±",         139.57039,   0.0,  1,   "méson-PS"),
    ("K±",         493.677,     0.0,  1,   "méson-PS"),
    ("K⁰",         497.611,     0.0,  0,   "méson-PS"),
    ("η",          547.862,     0.0,  0,   "méson-PS"),
    ("η'",         957.78,      0.0,  0,   "méson-PS"),
    ("η_c(1S)",    2983.9,      0.0,  0,   "charmonium"),
    ("η_c(2S)",    3637.5,      0.0,  0,   "charmonium"),
    ("η_b(1S)",    9398.7,      0.0,  0,   "bottomonium"),
    ("D⁰",         1864.84,     0.0,  0,   "charmé"),
    ("D±",         1869.66,     0.0,  1,   "charmé"),
    ("D_s±",       1968.34,     0.0,  1,   "charmé"),
    ("B±",         5279.34,     0.0,  1,   "beau"),
    ("B⁰",         5279.66,     0.0,  0,   "beau"),
    ("B_s⁰",       5366.92,     0.0,  0,   "beau"),
    ("B_c±",       6274.9,      0.0,  1,   "beau"),
    ("ρ",          775.26,      1.0,  1,   "méson-V"),
    ("ω",          782.65,      1.0,  0,   "méson-V"),
    ("φ",          1019.461,    1.0,  0,   "méson-V"),
    ("K*(892)",    891.66,      1.0,  1,   "méson-V"),
    ("J/ψ",        3096.9,      1.0,  0,   "charmonium"),
    ("ψ(2S)",      3686.10,     1.0,  0,   "charmonium"),
    ("ψ(3770)",    3773.7,      1.0,  0,   "charmonium"),
    ("ψ(4040)",    4039.0,      1.0,  0,   "charmonium"),
    ("Υ(1S)",      9460.30,     1.0,  0,   "bottomonium"),
    ("Υ(2S)",      10023.26,    1.0,  0,   "bottomonium"),
    ("Υ(3S)",      10355.2,     1.0,  0,   "bottomonium"),
    ("D*(2007)⁰",  2006.85,     1.0,  0,   "charmé-V"),
    ("D*(2010)±",  2010.26,     1.0,  1,   "charmé-V"),
    ("D_s*",       2112.2,      1.0,  1,   "charmé-V"),
    ("B*",         5324.71,     1.0,  0,   "beau-V"),
    ("a₀(980)",    980.0,       0.0,  1,   "scalaire"),
    ("f₀(500)",    500.0,       0.0,  0,   "scalaire"),
    ("f₀(980)",    990.0,       0.0,  0,   "scalaire"),
    ("f₀(1370)",   1370.0,      0.0,  0,   "scalaire"),
    ("a₁(1260)",   1230.0,      1.0,  1,   "axial"),
    ("b₁(1235)",   1229.5,      1.0,  1,   "axial"),
    ("K₁(1270)",   1272.0,      1.0,  1,   "axial"),
    ("K*(1430)",   1425.6,      2.0,  1,   "méson-T"),
    ("a₂(1320)",   1318.0,      2.0,  1,   "méson-T"),
    ("f₂(1270)",   1275.5,      2.0,  0,   "méson-T"),
    ("χ_c0(1P)",   3414.71,     0.0,  0,   "charmonium"),
    ("χ_c1(1P)",   3510.67,     1.0,  0,   "charmonium"),
    ("χ_c2(1P)",   3556.17,     2.0,  0,   "charmonium"),
    ("χ_b0(1P)",   9859.44,     0.0,  0,   "bottomonium"),
    ("χ_b1(1P)",   9892.78,     1.0,  0,   "bottomonium"),
    ("χ_b2(1P)",   9912.21,     2.0,  0,   "bottomonium"),
    ("p",          938.27208,   0.5,  1,   "nucléon"),
    ("n",          939.56542,   0.5,  0,   "nucléon"),
    ("Λ",          1115.683,    0.5,  0,   "hypéron"),
    ("Σ⁺",         1189.37,     0.5,  1,   "hypéron"),
    ("Σ⁰",         1192.642,    0.5,  0,   "hypéron"),
    ("Σ⁻",         1197.449,    0.5, -1,   "hypéron"),
    ("Ξ⁰",         1314.86,     0.5,  0,   "hypéron"),
    ("Ξ⁻",         1321.71,     0.5, -1,   "hypéron"),
    ("Δ⁺⁺",        1232.0,      1.5,  2,   "Delta"),
    ("Δ⁺",         1232.0,      1.5,  1,   "Delta"),
    ("Δ⁰",         1232.0,      1.5,  0,   "Delta"),
    ("Δ⁻",         1232.0,      1.5, -1,   "Delta"),
    ("Σ*⁺",        1382.83,     1.5,  1,   "Σ*"),
    ("Σ*⁰",        1383.7,      1.5,  0,   "Σ*"),
    ("Σ*⁻",        1387.2,      1.5, -1,   "Σ*"),
    ("Ξ*⁰",        1531.8,      1.5,  0,   "Ξ*"),
    ("Ξ*⁻",        1535.0,      1.5, -1,   "Ξ*"),
    ("Ω⁻",         1672.45,     1.5, -1,   "Ω"),
    ("N(1440)",    1440.0,      0.5,  1,   "N*"),
    ("N(1535)",    1535.0,      0.5,  1,   "N*"),
    ("N(1650)",    1655.0,      0.5,  1,   "N*"),
    ("N(1710)",    1710.0,      0.5,  1,   "N*"),
    ("Δ(1600)",    1600.0,      1.5,  1,   "Δ*"),
    ("Δ(1620)",    1620.0,      0.5,  1,   "Δ*"),
    ("Δ(1700)",    1700.0,      1.5,  1,   "Δ*"),
    ("Δ(1905)",    1880.0,      2.5,  1,   "Δ*"),
    ("Λ_c⁺",       2286.46,     0.5,  1,   "charmé-B"),
    ("Σ_c(2455)",  2453.97,     0.5,  1,   "charmé-B"),
    ("Σ_c(2520)",  2518.0,      1.5,  1,   "charmé-B"),
    ("Ξ_c⁺",       2467.94,     0.5,  1,   "charmé-B"),
    ("Ξ_c⁰",       2470.9,      0.5,  0,   "charmé-B"),
    ("Ω_c⁰",       2695.2,      0.5,  0,   "charmé-B"),
    ("Ξ_cc⁺⁺",     3621.6,      0.5,  2,   "double-c"),
    ("Ω_cc⁰",      3700.0,      0.5,  0,   "double-c"),
    ("Λ_b⁰",       5619.6,      0.5,  0,   "beau-B"),
    ("Σ_b±",       5811.0,      0.5,  1,   "beau-B"),
    ("Σ_b*±",      5832.5,      1.5,  1,   "beau-B"),
    ("Ξ_b⁰",       5791.9,      0.5,  0,   "beau-B"),
    ("Ξ_b⁻",       5797.0,      0.5, -1,   "beau-B"),
    ("Ω_b⁻",       6045.2,      0.5, -1,   "beau-B"),
    ("Λ_b(6072)",  6072.0,      0.5,  0,   "Λ_b*"),
    ("Σ_b(6097)",  6097.0,      1.5,  1,   "Σ_b*"),
    ("Ξ_b(6227)",  6227.0,      0.5, -1,   "Ξ_b*"),
    ("Ω_b(6316)",  6316.0,      0.5, -1,   "Ω_b*"),
    ("X(3872)",    3871.69,     1.0,  0,   "exotique-X"),
    ("X(3915)",    3918.4,      0.0,  0,   "exotique-X"),
    ("X(4140)",    4146.5,      1.0,  0,   "exotique-X"),
    ("X(4274)",    4274.0,      1.0,  0,   "exotique-X"),
    ("X(6900)",    6900.0,      0.0,  0,   "tetra-c"),
    ("Y(4260)",    4222.0,      1.0,  0,   "exotique-Y"),
    ("Y(4360)",    4374.0,      1.0,  0,   "exotique-Y"),
    ("Y(4660)",    4630.0,      1.0,  0,   "exotique-Y"),
    ("Z_c(3900)",  3886.6,      1.0,  1,   "exotique-Zc"),
    ("Z_c(4020)",  4024.1,      1.0,  1,   "exotique-Zc"),
    ("Z_c(4430)",  4478.0,      1.0,  1,   "exotique-Zc"),
    ("Z_b(10610)", 10607.2,     1.0,  1,   "exotique-Zb"),
    ("Z_b(10650)", 10652.2,     1.0,  1,   "exotique-Zb"),
    ("P_c(4312)",  4311.9,      0.5,  1,   "pentaquark"),
    ("P_c(4380)",  4380.0,      1.5,  1,   "pentaquark"),
    ("P_c(4450)",  4449.8,      2.5,  1,   "pentaquark"),
    ("P_c(4337)",  4337.0,      0.5,  1,   "pentaquark"),
    ("P_cs(4459)", 4458.8,      0.5,  0,   "pentaquark"),
    ("T_cc⁺",      3875.0,      1.0,  1,   "tetraquark"),
    ("T_ccs(2941)", 2941.0,     0.0,  0,   "tetraquark"),
]


def enumerate_all_configs(max_components=3):
    configs = []
    for n in range(1, max_components + 1):
        for combo in combinations_with_replacement(KNOTS, n):
            configs.append(list(combo))
    return configs


print("Enumération…")
ALL_CONFIGS = enumerate_all_configs(max_components=3)
print(f"  → {len(ALL_CONFIGS)} configs")

print("Précalcul (m, J_set, Q_topo, S_max)…")
CONFIG_DATA = []
for cfg in ALL_CONFIGS:
    m = operator_mass(cfg)
    Js = allowed_J_set(cfg)
    Q_topo = link_charge_topo(cfg)
    S_max = sum(TOPO[K]["s"] for K in cfg)
    CONFIG_DATA.append((cfg, m, Js, Q_topo, S_max))


def find_best_match(m_exp, J_exp, Q_exp, tol_m=0.05, lk_range=(-5, 5)):
    """Cherche la meilleure config TQNT-CG (J doit être dans J_set)."""
    best = None
    best_score = float("inf")
    for cfg, m_pred, Js, Q_topo, S_max in CONFIG_DATA:
        if J_exp not in Js:
            continue
        residue = Q_exp - Q_topo
        lk = round(residue)
        if lk < lk_range[0] or lk > lk_range[1]:
            continue
        if abs(residue - lk) > 0.34:
            continue
        err = abs(m_pred - m_exp) / max(m_exp, 1e-3)
        if err > tol_m:
            continue
        score = err + 0.001 * abs(residue - lk)
        if score < best_score:
            best_score = score
            best = (cfg, m_pred, Js, Q_topo, lk, err, S_max)
    return best


print(f"\nScan PDG ({len(PDG)} particules) …")
print("=" * 130)
print(f"  {'particule':<14} {'m_exp':>10} {'J':>4} {'Q':>5} | "
      f"{'config':<22} {'m_pred':>10} {'Js':>20} {'S_max':>5} {'Q_topo':>7} "
      f"{'Lk':>4} {'Δm%':>8} {'niv':>10} {'flag':>4}")
print("-" * 130)

found = {"★★": 0, "★": 0, "✓": 0, "—": 0}
by_sector = {}
unmatched = []
all_results = []

for nm, me, J, Q, sector in PDG:
    by_sector.setdefault(sector, [0, 0])
    by_sector[sector][1] += 1
    best = find_best_match(me, J, Q, tol_m=0.20)
    if best is None:
        flag = "—"
        unmatched.append((nm, me, J, Q, sector))
        found["—"] += 1
        all_results.append((nm, me, J, Q, sector, None))
        Js_disp = "—"
        S_max_disp = "—"
        print(f"  {nm:<14} {me:>10.2f} {J:>4} {Q:>+5.2f} | "
              f"{'(aucun match valide)':<60} {flag}")
    else:
        cfg, mp, Js, Qt, lk, err, S_max = best
        if err < 0.005: flag = "★★"
        elif err < 0.05: flag = "★"
        else: flag = "✓"
        found[flag] += 1
        by_sector[sector][0] += 1
        Js_disp = "{" + ", ".join(f"{j:.1f}" for j in sorted(Js)) + "}"
        if len(Js_disp) > 20: Js_disp = Js_disp[:17] + "…}"
        level = link_J_label(cfg, J)
        all_results.append((nm, me, J, Q, sector,
                            (cfg, mp, Js, Qt, lk, err, S_max, level)))
        print(f"  {nm:<14} {me:>10.2f} {J:>4} {Q:>+5.2f} | "
              f"{'+'.join(cfg):<22} {mp:>10.2f} {Js_disp:>20} {S_max:>5.1f} "
              f"{Qt:>+7.2f} {lk:>+4d} {err*100:>+7.2f} {level:>10} {flag}")

print("=" * 130)
total = len(PDG)
matched = found["★★"] + found["★"] + found["✓"]
print(f"\n─── BILAN GLOBAL ───")
print(f"  ★★ (<0.5%, J ∈ J_set, Q valide) : {found['★★']}/{total}")
print(f"  ★  (<5%, …)                      : {found['★']}/{total}")
print(f"  ✓  (<20%, …)                     : {found['✓']}/{total}")
print(f"  —  (aucun match)                 : {found['—']}/{total}")
print(f"\n  Matchs valides : {matched}/{total} ({matched/total*100:.0f}%)")
print(f"  ── COMPARAISON ──")
print(f"     v1 (P7bis stricte)  : 69/122 (57%)")
print(f"     v2 (CG enrichi)     : {matched}/{total} ({matched/total*100:.0f}%)")
print(f"     gain               : +{matched - 69}")

print(f"\n─── PAR SECTEUR ───")
for sect, (m_, t_) in sorted(by_sector.items(), key=lambda x: -x[1][1]):
    bar = "█" * int(m_ / t_ * 20) + "·" * (20 - int(m_ / t_ * 20))
    print(f"  {sect:<16} {bar}  {m_:>3}/{t_:<3}  ({m_/t_*100:>4.0f}%)")

if unmatched:
    print(f"\n─── PARTICULES SANS MATCH VALIDE ({len(unmatched)}) ───")
    for nm, me, J, Q, sect in unmatched:
        print(f"    {nm:<14} m={me:>10.2f}  J={J}  Q={Q:+.2f}  ({sect})")

# Statistiques de la formule de masse SEULE (sur les matchs valides)
print(f"\n─── STATISTIQUES DE LA FORMULE DE MASSE (matchs valides) ───")
errs = [abs(r[5][5]) for r in all_results if r[5] is not None]
if errs:
    arr = np.array(errs) * 100
    print(f"    Δm/m moyen     : {arr.mean():.2f}%")
    print(f"    Δm/m médian    : {np.median(arr):.2f}%")
    print(f"    Δm/m max       : {arr.max():.2f}%")
    print(f"    n              : {len(errs)}")
    print(f"    <0.5%          : {(arr<0.5).sum()}/{len(errs)} ({(arr<0.5).sum()/len(errs)*100:.0f}%)")
    print(f"    <2%            : {(arr<2).sum()}/{len(errs)} ({(arr<2).sum()/len(errs)*100:.0f}%)")
    print(f"    <5%            : {(arr<5).sum()}/{len(errs)} ({(arr<5).sum()/len(errs)*100:.0f}%)")

# Analyse par niveau CG
print(f"\n─── DISTRIBUTION PAR NIVEAU DE COUPLAGE ───")
levels = Counter()
for r in all_results:
    if r[5] is not None:
        levels[r[5][7]] += 1
for lvl, count in sorted(levels.items()):
    print(f"    {lvl:<12} {count:>3}")
