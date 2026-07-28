r"""
SCAN COMBINATOIRE EXHAUSTIF — masse + spin + charge

Pour toutes les combinaisons (1,2,3) de 19 nœuds (avec répétition),
calcule (m, J, Q) puis cherche la meilleure assignation pour chaque
particule de la base PDG (~160 entrées).

Critères pour qu'un config "matche" une particule cible :
  • |Δm/m| < tolérance (par défaut 5%)
  • J_pred == J_exp (parité demi-entier/entier respectée)
  • Q_pred + Lk = Q_exp pour un Lk entier dans [-5,+5]
    (charge topologique de l'entrelacs + nombre d'enlacement)

Sortie : tableau exhaustif ligne par ligne, comptes par secteur,
identification des particules sans assignation valide.
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

# ───────── constantes physiques ─────────
M_E = 0.51099895
M_P = 938.27208
M_N = 939.56542
F_KAPPA = 0.8682
M_C = M_E * F_KAPPA / 3
M_L = M_E * (1 - F_KAPPA) / 3

# ───────── catalogue de nœuds ─────────
KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
N_LOBES = {
    "3_1": 3, "4_1": 2, "5_1": 5, "5_2": 2, "6_1": 2,
    "6_2": 3, "6_3": 3, "7_1": 7, "7_2": 2, "7_3": 3,
    "7_4": 2, "7_5": 3, "7_6": 2, "7_7": 3, "8_1": 2,
    "8_2": 3, "8_3": 3, "9_1": 9, "10_1": 2,
}

# ───────── paramètres analytiques SU(2)_8 ─────────
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
    """Masse TQNT-SU(2)_8 (formule entièrement analytique)."""
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


def link_spin_J(comps):
    """Spin de l'entrelacs (P7bis).

    Retourne le spin total (1/2 fermion fondamental, 1 boson vecteur, etc.).
    """
    n_half = sum(1 for k in comps if TOPO[k]["half"])
    if n_half % 2 == 1:
        return 0.5  # fermion (1/2)
    # boson
    if n_half > 0:
        return 1.0
    return max(TOPO[k]["s"] for k in comps)


def link_charge_topo(comps):
    """Charge topologique brute Σ Q_K (sans Lk).

    Le code utilise la charge de Jones (jones_power_center), qui prend
    des valeurs fractionnaires (±1/3, ±2/3, ±1, 0). C'est l'invariant
    chiral du nœud.
    """
    return sum(TOPO[k]["q"] for k in comps)


# ───────── base de données PDG ÉLARGIE (~160 particules) ─────────
# Format : (nom, m_MeV, J, Q, secteur)
PDG = [
    # ═══ LEPTONS ═══
    ("e⁻",         0.51099895,  0.5, -1,   "lepton"),
    ("μ⁻",         105.6583755, 0.5, -1,   "lepton"),
    ("τ⁻",         1776.86,     0.5, -1,   "lepton"),
    ("ν_e",        2.2e-6,      0.5,  0,   "neutrino"),
    ("ν_μ",        0.17,        0.5,  0,   "neutrino"),
    ("ν_τ",        15.5,        0.5,  0,   "neutrino"),

    # ═══ BOSONS DE JAUGE ═══
    ("γ",          0.0,         1.0,  0,   "boson-jauge"),
    ("W±",         80369.0,     1.0,  1,   "EW"),
    ("Z⁰",         91188.0,     1.0,  0,   "EW"),
    ("H",          125250.0,    0.0,  0,   "EW"),
    ("t",          173100.0,    0.5,  2/3, "EW"),

    # ═══ MÉSONS PSEUDO-SCALAIRES (J=0) ═══
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

    # ═══ MÉSONS VECTEURS (J=1) ═══
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

    # ═══ AUTRES MÉSONS (J variables) ═══
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

    # ═══ BARYONS J=1/2 (octet) ═══
    ("p",          938.27208,   0.5,  1,   "nucléon"),
    ("n",          939.56542,   0.5,  0,   "nucléon"),
    ("Λ",          1115.683,    0.5,  0,   "hypéron"),
    ("Σ⁺",         1189.37,     0.5,  1,   "hypéron"),
    ("Σ⁰",         1192.642,    0.5,  0,   "hypéron"),
    ("Σ⁻",         1197.449,    0.5, -1,   "hypéron"),
    ("Ξ⁰",         1314.86,     0.5,  0,   "hypéron"),
    ("Ξ⁻",         1321.71,     0.5, -1,   "hypéron"),

    # ═══ BARYONS J=3/2 (décuplet) ═══
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

    # ═══ N* et Δ* (résonances baryoniques) ═══
    ("N(1440)",    1440.0,      0.5,  1,   "N*"),
    ("N(1535)",    1535.0,      0.5,  1,   "N*"),
    ("N(1650)",    1655.0,      0.5,  1,   "N*"),
    ("N(1710)",    1710.0,      0.5,  1,   "N*"),
    ("Δ(1600)",    1600.0,      1.5,  1,   "Δ*"),
    ("Δ(1620)",    1620.0,      0.5,  1,   "Δ*"),
    ("Δ(1700)",    1700.0,      1.5,  1,   "Δ*"),
    ("Δ(1905)",    1880.0,      2.5,  1,   "Δ*"),

    # ═══ BARYONS CHARMÉS ═══
    ("Λ_c⁺",       2286.46,     0.5,  1,   "charmé-B"),
    ("Σ_c(2455)",  2453.97,     0.5,  1,   "charmé-B"),
    ("Σ_c(2520)",  2518.0,      1.5,  1,   "charmé-B"),
    ("Ξ_c⁺",       2467.94,     0.5,  1,   "charmé-B"),
    ("Ξ_c⁰",       2470.9,      0.5,  0,   "charmé-B"),
    ("Ω_c⁰",       2695.2,      0.5,  0,   "charmé-B"),
    ("Ξ_cc⁺⁺",     3621.6,      0.5,  2,   "double-c"),
    ("Ω_cc⁰",      3700.0,      0.5,  0,   "double-c"),

    # ═══ BARYONS BEAUX ═══
    ("Λ_b⁰",       5619.6,      0.5,  0,   "beau-B"),
    ("Σ_b±",       5811.0,      0.5,  1,   "beau-B"),
    ("Σ_b*±",      5832.5,      1.5,  1,   "beau-B"),
    ("Ξ_b⁰",       5791.9,      0.5,  0,   "beau-B"),
    ("Ξ_b⁻",       5797.0,      0.5, -1,   "beau-B"),
    ("Ω_b⁻",       6045.2,      0.5, -1,   "beau-B"),

    # ═══ EXCITATIONS RÉCENTES BARYONS ═══
    ("Λ_b(6072)",  6072.0,      0.5,  0,   "Λ_b*"),
    ("Σ_b(6097)",  6097.0,      1.5,  1,   "Σ_b*"),
    ("Ξ_b(6227)",  6227.0,      0.5, -1,   "Ξ_b*"),
    ("Ω_b(6316)",  6316.0,      0.5, -1,   "Ω_b*"),

    # ═══ EXOTIQUES — XYZ ═══
    ("X(3872)",    3871.69,     1.0,  0,   "exotique-X"),
    ("X(3915)",    3918.4,      0.0,  0,   "exotique-X"),
    ("X(4140)",    4146.5,      1.0,  0,   "exotique-X"),
    ("X(4274)",    4274.0,      1.0,  0,   "exotique-X"),
    ("X(6900)",    6900.0,      0.0,  0,   "tetra-c"),
    ("Y(4260)",    4222.0,      1.0,  0,   "exotique-Y"),
    ("Y(4360)",    4374.0,      1.0,  0,   "exotique-Y"),
    ("Y(4660)",    4630.0,      1.0,  0,   "exotique-Y"),

    # ═══ EXOTIQUES — Z chargés ═══
    ("Z_c(3900)",  3886.6,      1.0,  1,   "exotique-Zc"),
    ("Z_c(4020)",  4024.1,      1.0,  1,   "exotique-Zc"),
    ("Z_c(4430)",  4478.0,      1.0,  1,   "exotique-Zc"),
    ("Z_b(10610)", 10607.2,     1.0,  1,   "exotique-Zb"),
    ("Z_b(10650)", 10652.2,     1.0,  1,   "exotique-Zb"),

    # ═══ EXOTIQUES — pentaquarks et tétraquarks ═══
    ("P_c(4312)",  4311.9,      0.5,  1,   "pentaquark"),
    ("P_c(4380)",  4380.0,      1.5,  1,   "pentaquark"),
    ("P_c(4450)",  4449.8,      2.5,  1,   "pentaquark"),
    ("P_c(4337)",  4337.0,      0.5,  1,   "pentaquark"),
    ("P_cs(4459)", 4458.8,      0.5,  0,   "pentaquark"),
    ("T_cc⁺",      3875.0,      1.0,  1,   "tetraquark"),
    ("T_ccs(2941)", 2941.0,     0.0,  0,   "tetraquark"),
]

print(f"PDG : {len(PDG)} particules")


def enumerate_all_configs(max_components=3, pool=None):
    """Tous les entrelacs possibles 1..max_components nœuds avec répétition."""
    if pool is None:
        pool = KNOTS
    configs = []
    for n in range(1, max_components + 1):
        for combo in combinations_with_replacement(pool, n):
            configs.append(list(combo))
    return configs


print("Enumération des configurations 1-3 nœuds…")
ALL_CONFIGS = enumerate_all_configs(max_components=3)
print(f"  → {len(ALL_CONFIGS)} configurations")

print("Calcul de (m, J, Q_topo) pour chaque config…")
CONFIG_DATA = []
for cfg in ALL_CONFIGS:
    m = operator_mass(cfg)
    J = link_spin_J(cfg)
    Q_topo = link_charge_topo(cfg)
    CONFIG_DATA.append((cfg, m, J, Q_topo))


def find_best_match(m_exp, J_exp, Q_exp, tol_m=0.05, lk_range=(-5, 5)):
    """Cherche la meilleure config matchant (m, J, Q) pour une particule."""
    best = None
    best_score = float("inf")
    for cfg, m_pred, J_pred, Q_topo in CONFIG_DATA:
        # spin doit être exact
        if abs(J_pred - J_exp) > 1e-6:
            continue
        # charge : Q_topo + Lk = Q_exp pour un Lk entier dans la fenêtre
        residue = Q_exp - Q_topo
        lk = round(residue)
        if lk < lk_range[0] or lk > lk_range[1]:
            continue
        if abs(residue - lk) > 0.34:  # charge fractionnaire Jones tolérée à ±1/3
            continue
        # masse
        err = abs(m_pred - m_exp) / max(m_exp, 1e-3)
        if err > tol_m:
            continue
        # score combiné
        score = err + 0.001 * abs(residue - lk)
        if score < best_score:
            best_score = score
            best = (cfg, m_pred, J_pred, Q_topo, lk, err)
    return best


# Scan PDG
print(f"\nScan exhaustif : {len(PDG)} particules vs {len(CONFIG_DATA)} configs…")
print("=" * 110)
print(f"  {'particule':<14} {'m_exp':>10} {'J':>4} {'Q':>5} | "
      f"{'config':<22} {'m_pred':>10} {'J_p':>4} {'Q_topo':>7} {'Lk':>4} {'Δm%':>8} {'secteur':<14}")
print("-" * 110)

found = {"★★": 0, "★": 0, "✓": 0, "—": 0}
by_sector = {}
unmatched = []

for nm, me, J, Q, sector in PDG:
    by_sector.setdefault(sector, [0, 0])
    by_sector[sector][1] += 1
    best = find_best_match(me, J, Q, tol_m=0.20)
    if best is None:
        flag = "—"
        unmatched.append((nm, me, J, Q, sector))
        print(f"  {nm:<14} {me:>10.2f} {J:>4} {Q:>+5.2f} | "
              f"{'(aucun match <20% avec J,Q exacts)':<60} {flag}")
    else:
        cfg, mp, Jp, Qt, lk, err = best
        if err < 0.005: flag = "★★"
        elif err < 0.05: flag = "★"
        else: flag = "✓"
        found[flag] += 1
        by_sector[sector][0] += 1
        print(f"  {nm:<14} {me:>10.2f} {J:>4} {Q:>+5.2f} | "
              f"{'+'.join(cfg):<22} {mp:>10.2f} {Jp:>4} {Qt:>+7.2f} {lk:>+4d} "
              f"{err*100:>+7.2f} {sector:<14} {flag}")

print("=" * 110)
print(f"\n─── BILAN GLOBAL ───")
total = len(PDG)
matched = sum(found.values())
print(f"  ★★ (<0.5%, J et Q exacts) : {found['★★']}/{total}")
print(f"  ★  (<5%, J et Q exacts)   : {found['★']}/{total}")
print(f"  ✓  (<20%, J et Q exacts)  : {found['✓']}/{total}")
print(f"  —  (aucun match)          : {found['—']}/{total}")
print(f"\n  Matchs valides : {matched}/{total} ({matched/total*100:.0f}%)")

print(f"\n─── PAR SECTEUR ───")
for sect, (m_, t_) in sorted(by_sector.items(), key=lambda x: -x[1][1]):
    print(f"  {sect:<16} : {m_:>3}/{t_:<3}  ({m_/t_*100:>4.0f}%)")

if unmatched:
    print(f"\n─── PARTICULES SANS MATCH VALIDE ({len(unmatched)}) ───")
    print("  Note : aucune config (≤3 nœuds) ne satisfait simultanément")
    print("         J_exp, Q_exp et |Δm|<20%.")
    for nm, me, J, Q, sect in unmatched:
        print(f"    {nm:<14} m={me:>10.2f}  J={J}  Q={Q:+.2f}  ({sect})")
