r"""
EXPORT EXCEL — catalogue complet des nœuds et entrelacs TQNT.

Pour chaque configuration (1, 2 ou 3 nœuds parmi 19), on calcule :
  • masse Ω_ts (V5)
  • spectre de spin (P7quater Clebsch-Gordan)
  • charge topologique Q_topo
  • durée de vie (si match PDG)

Et on liste TOUTES les particules PDG candidates (m, J, Q compatibles).

Codes couleurs PDG (fiabilité expérimentale) :
  🟢 VERT       : mesure très précise (<0.1%)         — leptons, p, n, π, K, mésons stables
  🟡 JAUNE      : mesure précise (<1%)                — hadrons octet/décuplet, charmonium 1S
  🟠 ORANGE     : résonance large ou récente          — résonances Δ, N*, X(3872), exotiques 2010s
  🔴 ROUGE      : préliminaire / état exotique récent — X(6900), Tccs, Pcs, Zc/Zb tout récents

Sortie : docs/TQNT_catalog_complete.xlsx
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations_with_replacement

from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from theory.abraham import frenet_geometry
from theory.jones import (
    jones_polynomial, jones_determinant, braid_index,
    topological_spin, topological_charge,
)

# ───────── constantes ─────────
M_E = 0.51099895
F_KAPPA = 0.8682
M_C = M_E * F_KAPPA / 3
M_L = M_E * (1 - F_KAPPA) / 3
HBAR_C = 197.3269788  # MeV·fm

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
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
               "n_l": N_LOBES[K],
               "b": braid_index(K),
               "det": jones_determinant(pj),
               "s": topological_spin(K),
               "q": topological_charge(K)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def omega(comps):
    """Masse Ω_ts (V5)."""
    n = len(comps)
    ws = [M_NAIVE[K] * (TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in comps)/P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2:
            Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B_EXP)) ** (ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


def J_set(comps):
    """P7quater Clebsch-Gordan."""
    s_max = sum(TOPO[K]["s"] for K in comps)
    parity = s_max - int(s_max)
    n_steps = int(round(s_max - parity)) + 1
    return [parity + i for i in range(n_steps)]


def Q_topo(comps):
    return sum(TOPO[K]["q"] for K in comps)


# ───────── base PDG avec FIABILITÉ ─────────
# Tier : 1=VERT, 2=JAUNE, 3=ORANGE, 4=ROUGE
PDG = [
    # ═══ Leptons (1=très précis) ═══
    ("e⁻",         0.51099895,  0.5, -1,    1, "0.5 ppt PDG, Δ extrêmement précis"),
    ("μ⁻",         105.6583755, 0.5, -1,    1, "ppm PDG"),
    ("τ⁻",         1776.86,     0.5, -1,    1, "0.07% PDG"),
    ("ν_e",        2.2e-6,      0.5,  0,    2, "limite < 0.8 eV"),
    ("ν_μ",        0.17,        0.5,  0,    2, "limite estimée"),
    ("ν_τ",        15.5,        0.5,  0,    2, "limite estimée"),

    # ═══ Bosons de jauge (1=très précis) ═══
    ("γ",          0.0,         1.0,  0,    1, "exact"),
    ("W±",         80369.0,     1.0,  1,    1, "0.01% LEP+LHC"),
    ("Z⁰",         91187.6,     1.0,  0,    1, "0.003% LEP"),
    ("H",          125250.0,    0.0,  0,    1, "0.13% ATLAS+CMS combiné"),
    ("t",          172690.0,    0.5,  2/3,  1, "0.18% LHC combiné"),

    # ═══ Mésons PS (1=très précis) ═══
    ("π⁰",         134.9768,    0.0,  0,    1, "0.0004% PDG"),
    ("π±",         139.57039,   0.0,  1,    1, "ppm PDG"),
    ("K±",         493.677,     0.0,  1,    1, "0.003% PDG"),
    ("K⁰",         497.611,     0.0,  0,    1, "0.003% PDG"),
    ("η",          547.862,     0.0,  0,    1, "0.003% PDG"),
    ("η'",         957.78,      0.0,  0,    1, "0.006% PDG"),

    # ═══ Charmonium 1S/2S (1=très précis) ═══
    ("η_c(1S)",    2983.9,      0.0,  0,    1, "0.01% PDG"),
    ("η_c(2S)",    3637.5,      0.0,  0,    2, "0.3% PDG"),
    ("J/ψ",        3096.9,      1.0,  0,    1, "0.0003% PDG"),
    ("ψ(2S)",      3686.10,     1.0,  0,    1, "0.0003% PDG"),
    ("ψ(3770)",    3773.7,      1.0,  0,    2, "0.1% PDG"),
    ("ψ(4040)",    4039.0,      1.0,  0,    2, "0.2% PDG"),
    ("χ_c0(1P)",   3414.71,     0.0,  0,    1, "0.01% PDG"),
    ("χ_c1(1P)",   3510.67,     1.0,  0,    1, "0.002% PDG"),
    ("χ_c2(1P)",   3556.17,     2.0,  0,    1, "0.002% PDG"),

    # ═══ Bottomonium (1=très précis) ═══
    ("η_b(1S)",    9398.7,      0.0,  0,    2, "0.02% PDG"),
    ("Υ(1S)",      9460.30,     1.0,  0,    1, "0.003% PDG"),
    ("Υ(2S)",      10023.26,    1.0,  0,    1, "0.003% PDG"),
    ("Υ(3S)",      10355.2,     1.0,  0,    1, "0.005% PDG"),
    ("χ_b0(1P)",   9859.44,     0.0,  0,    1, "0.05% PDG"),
    ("χ_b1(1P)",   9892.78,     1.0,  0,    1, "0.04% PDG"),
    ("χ_b2(1P)",   9912.21,     2.0,  0,    1, "0.03% PDG"),

    # ═══ D/B mésons (1) ═══
    ("D⁰",         1864.84,     0.0,  0,    1, "0.003% PDG"),
    ("D±",         1869.66,     0.0,  1,    1, "0.003% PDG"),
    ("D_s±",       1968.34,     0.0,  1,    1, "0.003% PDG"),
    ("D*(2007)⁰",  2006.85,     1.0,  0,    1, "0.004% PDG"),
    ("D*(2010)±",  2010.26,     1.0,  1,    1, "0.005% PDG"),
    ("D_s*",       2112.2,      1.0,  1,    1, "0.02% PDG"),
    ("B±",         5279.34,     0.0,  1,    1, "0.002% PDG"),
    ("B⁰",         5279.66,     0.0,  0,    1, "0.002% PDG"),
    ("B_s⁰",       5366.92,     0.0,  0,    1, "0.003% PDG"),
    ("B_c±",       6274.9,      0.0,  1,    1, "0.07% PDG"),
    ("B*",         5324.71,     1.0,  0,    1, "0.004% PDG"),

    # ═══ Vecteurs légers (1) ═══
    ("ρ",          775.26,      1.0,  1,    1, "0.03% PDG"),
    ("ω",          782.65,      1.0,  0,    1, "0.02% PDG"),
    ("φ",          1019.461,    1.0,  0,    1, "0.002% PDG"),
    ("K*(892)",    891.66,      1.0,  1,    1, "0.03% PDG"),

    # ═══ Mésons tenseur / axial / scalaire (2-3) ═══
    ("a₀(980)",    980.0,       0.0,  1,    3, "résonance large, 2% PDG"),
    ("f₀(500)",    500.0,       0.0,  0,    3, "très large, ±100 MeV"),
    ("f₀(980)",    990.0,       0.0,  0,    2, "1% PDG"),
    ("f₀(1370)",   1370.0,      0.0,  0,    3, "5% PDG, très large"),
    ("a₁(1260)",   1230.0,      1.0,  1,    3, "4% PDG, large"),
    ("b₁(1235)",   1229.5,      1.0,  1,    2, "0.3% PDG"),
    ("K₁(1270)",   1272.0,      1.0,  1,    2, "0.5% PDG"),
    ("K*(1430)",   1425.6,      2.0,  1,    2, "0.1% PDG"),
    ("a₂(1320)",   1318.0,      2.0,  1,    2, "0.1% PDG"),
    ("f₂(1270)",   1275.5,      2.0,  0,    2, "0.06% PDG"),

    # ═══ Octet baryonique J=1/2 (1) ═══
    ("p",          938.27208,   0.5,  1,    1, "ppb PDG, fondamental"),
    ("n",          939.56542,   0.5,  0,    1, "ppb PDG"),
    ("Λ",          1115.683,    0.5,  0,    1, "0.0005% PDG"),
    ("Σ⁺",         1189.37,     0.5,  1,    1, "0.006% PDG"),
    ("Σ⁰",         1192.642,    0.5,  0,    1, "0.002% PDG"),
    ("Σ⁻",         1197.449,    0.5, -1,    1, "0.0006% PDG"),
    ("Ξ⁰",         1314.86,     0.5,  0,    1, "0.015% PDG"),
    ("Ξ⁻",         1321.71,     0.5, -1,    1, "0.0006% PDG"),

    # ═══ Décuplet baryonique J=3/2 (2-3) ═══
    ("Δ⁺⁺",        1232.0,      1.5,  2,    3, "résonance ±1 MeV, large"),
    ("Δ⁺",         1232.0,      1.5,  1,    3, "résonance large"),
    ("Δ⁰",         1232.0,      1.5,  0,    3, "résonance large"),
    ("Δ⁻",         1232.0,      1.5, -1,    3, "résonance large"),
    ("Σ*⁺",        1382.83,     1.5,  1,    2, "0.04% PDG"),
    ("Σ*⁰",        1383.7,      1.5,  0,    2, "0.04% PDG"),
    ("Σ*⁻",        1387.2,      1.5, -1,    2, "0.05% PDG"),
    ("Ξ*⁰",        1531.8,      1.5,  0,    2, "0.03% PDG"),
    ("Ξ*⁻",        1535.0,      1.5, -1,    2, "0.04% PDG"),
    ("Ω⁻",         1672.45,     1.5, -1,    1, "0.002% PDG"),

    # ═══ N* et Δ* (2-3) ═══
    ("N(1440)",    1440.0,      0.5,  1,    3, "résonance large (>200 MeV)"),
    ("N(1535)",    1535.0,      0.5,  1,    3, "résonance large"),
    ("N(1650)",    1655.0,      0.5,  1,    3, "résonance large"),
    ("N(1710)",    1710.0,      0.5,  1,    3, "résonance large"),
    ("Δ(1600)",    1600.0,      1.5,  1,    3, "résonance large"),
    ("Δ(1620)",    1620.0,      0.5,  1,    3, "résonance large"),
    ("Δ(1700)",    1700.0,      1.5,  1,    3, "résonance large"),
    ("Δ(1905)",    1880.0,      2.5,  1,    3, "résonance large"),

    # ═══ Charmé/beau baryons (1-2) ═══
    ("Λ_c⁺",       2286.46,     0.5,  1,    1, "0.006% PDG"),
    ("Σ_c(2455)",  2453.97,     0.5,  1,    1, "0.005% PDG"),
    ("Σ_c(2520)",  2518.0,      1.5,  1,    2, "0.2% PDG"),
    ("Ξ_c⁺",       2467.94,     0.5,  1,    1, "0.01% PDG"),
    ("Ξ_c⁰",       2470.9,      0.5,  0,    1, "0.01% PDG"),
    ("Ω_c⁰",       2695.2,      0.5,  0,    2, "0.06% PDG"),
    ("Ξ_cc⁺⁺",     3621.6,      0.5,  2,    2, "0.1% LHCb"),
    ("Ω_cc⁰",      3700.0,      0.5,  0,    3, "prédit, pas mesuré"),
    ("Λ_b⁰",       5619.6,      0.5,  0,    1, "0.004% LHCb"),
    ("Σ_b±",       5811.0,      0.5,  1,    2, "0.1% PDG"),
    ("Σ_b*±",      5832.5,      1.5,  1,    2, "0.1% PDG"),
    ("Ξ_b⁰",       5791.9,      0.5,  0,    2, "0.06% PDG"),
    ("Ξ_b⁻",       5797.0,      0.5, -1,    1, "0.01% PDG"),
    ("Ω_b⁻",       6045.2,      0.5, -1,    2, "0.1% PDG"),

    # ═══ Baryons lourds récents (3) ═══
    ("Λ_b(6072)",  6072.0,      0.5,  0,    3, "LHCb 2020, ±2 MeV"),
    ("Σ_b(6097)",  6097.0,      1.5,  1,    3, "LHCb 2019"),
    ("Ξ_b(6227)",  6227.0,      0.5, -1,    3, "LHCb 2018"),
    ("Ω_b(6316)",  6316.0,      0.5, -1,    3, "LHCb 2020"),

    # ═══ Exotiques XYZ (3-4) ═══
    ("X(3872)",    3871.69,     1.0,  0,    2, "0.005% PDG, bien établi depuis Belle 2003"),
    ("X(3915)",    3918.4,      0.0,  0,    3, "résonance ±1.9 MeV, nature débattue"),
    ("X(4140)",    4146.5,      1.0,  0,    3, "résonance large CDF 2008, confirmé"),
    ("X(4274)",    4274.0,      1.0,  0,    3, "résonance large, CMS+LHCb"),
    ("X(6900)",    6900.0,      0.0,  0,    4, "LHCb 2020, tetra-charm, nature débattue"),
    ("Y(4260)",    4222.0,      1.0,  0,    3, "BaBar 2005, parfois rebaptisé ψ(4230)"),
    ("Y(4360)",    4374.0,      1.0,  0,    3, "Belle 2007"),
    ("Y(4660)",    4630.0,      1.0,  0,    3, "Belle 2008, ±6 MeV"),
    ("Z_c(3900)",  3886.6,      1.0,  1,    3, "BESIII 2013, confirmé"),
    ("Z_c(4020)",  4024.1,      1.0,  1,    3, "BESIII 2013, confirmé"),
    ("Z_c(4430)",  4478.0,      1.0,  1,    3, "Belle 2008, controversé puis confirmé LHCb"),
    ("Z_b(10610)", 10607.2,     1.0,  1,    3, "Belle 2011"),
    ("Z_b(10650)", 10652.2,     1.0,  1,    3, "Belle 2011"),
    ("P_c(4312)",  4311.9,      0.5,  1,    4, "LHCb 2019, pentaquark"),
    ("P_c(4380)",  4380.0,      1.5,  1,    4, "LHCb 2015, large"),
    ("P_c(4450)",  4449.8,      2.5,  1,    4, "LHCb 2015"),
    ("P_c(4337)",  4337.0,      0.5,  1,    4, "LHCb 2022, récent"),
    ("P_cs(4459)", 4458.8,      0.5,  0,    4, "LHCb 2020, pentaquark + strangeness"),
    ("T_cc⁺",      3875.0,      1.0,  1,    4, "LHCb 2021, double-c tetraquark"),
    ("T_ccs(2941)",2941.0,      0.0,  0,    4, "LHCb 2024, très récent"),
]

# Durées de vie connues (depuis lifetime_weak_nocalib + lifetime_strong/ew + PDG)
TAU_PDG = {
    "e⁻": "stable", "μ⁻": "2.20e-6", "τ⁻": "2.90e-13",
    "n": "878.4", "p": "stable", "π⁰": "8.43e-17", "π±": "2.60e-8",
    "K±": "1.24e-8", "K⁰": "8.95e-11", "η": "5.02e-19", "η'": "3.39e-21",
    "η_c(1S)": "5.1e-23", "J/ψ": "7.11e-21", "ψ(2S)": "1.04e-21",
    "ψ(3770)": "1.6e-23", "Υ(1S)": "1.22e-20", "χ_c0(1P)": "9.65e-23",
    "χ_c1(1P)": "1.36e-21", "χ_c2(1P)": "3.18e-22",
    "D⁰": "4.10e-13", "D±": "1.03e-12", "D_s±": "5.04e-13",
    "B±": "1.64e-12", "B⁰": "1.52e-12", "B_s⁰": "1.51e-12", "B_c±": "5.10e-13",
    "ρ": "4.41e-24", "ω": "7.75e-23", "φ": "1.55e-22", "K*(892)": "1.30e-23",
    "Λ": "2.62e-10", "Σ⁺": "8.02e-11", "Σ⁰": "7.40e-20", "Σ⁻": "1.48e-10",
    "Ξ⁰": "2.90e-10", "Ξ⁻": "1.64e-10", "Ω⁻": "8.21e-11",
    "W±": "3.16e-25", "Z⁰": "2.64e-25", "H": "1.60e-22", "t": "5.00e-25",
    "X(3872)": "9.4e-23", "Λ_c⁺": "2.03e-13", "Λ_b⁰": "1.47e-12",
}


def lifetime_TQNT(comps, J, Q):
    """Estime la durée de vie TQNT selon la nature de l'entrelacs.

    Renvoie une chaîne (avec unité) ou '—' si non applicable.
    """
    n = len(comps)
    # Cas particuliers reconnus
    if comps == ["3_1"] and J == 0.5 and abs(Q + 1) < 0.5:
        return "stable (électron)"
    # Nucléon
    if Counter(comps) == Counter(["3_1","7_2","7_2"]):
        if abs(Q - 1) < 0.5:
            return "stable (proton)"
        if abs(Q) < 0.5:
            return "880.3 s (n, Fermi-Sirlin)"
    return "—"  # non-évalué (nécessite canal spécifique)


# Couleurs PDG
TIER_COLORS = {
    1: "C6EFCE",  # vert pâle  — très précis
    2: "FFEB9C",  # jaune pâle — précis
    3: "FFC7CE",  # rose       — résonance/exotique 2010s
    4: "F4B4B4",  # rouge pâle — très récent / préliminaire
}
TIER_LABELS = {1: "🟢 très précis", 2: "🟡 précis", 3: "🟠 résonance/exotique", 4: "🔴 récent/préliminaire"}


def all_configs(max_n=3):
    configs = []
    for n in range(1, max_n+1):
        for combo in combinations_with_replacement(KNOTS, n):
            configs.append(list(combo))
    return configs


def find_pdg_matches(m_pred, J_set_pred, Q_topo_pred, tol_m=0.05, lk_range=(-5,5)):
    """Liste les PDG candidats pour ce config (mass tolérance, J ∈ J_set, Q ajustable par Lk)."""
    matches = []
    for nm, me, J, Q, tier, note in PDG:
        if J not in J_set_pred:
            continue
        residue = Q - Q_topo_pred
        lk = round(residue)
        if lk < lk_range[0] or lk > lk_range[1] or abs(residue - lk) > 0.34:
            continue
        err = (m_pred - me) / max(me, 1e-3)
        if abs(err) > tol_m:
            continue
        matches.append((nm, me, J, Q, tier, note, lk, err*100))
    matches.sort(key=lambda x: abs(x[7]))
    return matches


# ═══════════════ GÉNÉRATION EXCEL ═══════════════
print("Enumération des configs (1-3 nœuds)…")
ALL = all_configs(3)
print(f"  → {len(ALL)} configs")

wb = Workbook()
ws = wb.active
ws.title = "Catalogue TQNT V5"

# Bordures et styles
thin = Side(border_style="thin", color="888888")
border = Border(top=thin, bottom=thin, left=thin, right=thin)
header_fill = PatternFill("solid", fgColor="305496")
header_font = Font(bold=True, color="FFFFFF", size=11)
sub_header_fill = PatternFill("solid", fgColor="D9E1F2")
sub_header_font = Font(bold=True, size=10)

# ─── En-tête ───
ws["A1"] = "CATALOGUE COMPLET TQNT-V5 — nœuds, entrelacs, prédictions et particules PDG candidates"
ws["A1"].font = Font(bold=True, size=14, color="305496")
ws.merge_cells("A1:P1")
ws["A2"] = ("Pour chaque configuration topologique (1 à 3 nœuds) : nombre de croisements, "
            "masse Ω_ts (V5), spectre de spin Clebsch-Gordan, charge topologique, durée de vie TQNT, "
            "puis particules PDG candidates avec leur niveau de fiabilité expérimentale.")
ws["A2"].font = Font(italic=True, size=9)
ws.merge_cells("A2:P2")

# Légende fiabilité
ws["A4"] = "FIABILITÉ PDG :"
ws["A4"].font = Font(bold=True, size=10)
for i, (tier, label) in enumerate(TIER_LABELS.items()):
    c = ws.cell(row=4, column=2+i, value=label)
    c.fill = PatternFill("solid", fgColor=TIER_COLORS[tier])
    c.font = Font(bold=True, size=9)
    c.alignment = Alignment(horizontal="center")
    c.border = border

# Légende sélection
ws["A5"] = "CONFIG TQNT :"
ws["A5"].font = Font(bold=True, size=10)
c = ws.cell(row=5, column=2, value="🔵 sélectionnée comme particule (≥1 match PDG)")
c.fill = PatternFill("solid", fgColor="D6E9FF")
c.font = Font(bold=True, size=9)
c.alignment = Alignment(horizontal="center")
c.border = border
ws.merge_cells("B5:C5")
c2 = ws.cell(row=5, column=4, value="⚫ candidat matière noire (c>18, Q=0, P18)")
c2.fill = PatternFill("solid", fgColor="A6A6A6")
c2.font = Font(bold=True, size=9, color="FFFFFF")
c2.alignment = Alignment(horizontal="center")
c2.border = border
ws.merge_cells("D5:F5")
c3 = ws.cell(row=5, column=7, value="⚪ pas de particule identifiée")
c3.fill = PatternFill("solid", fgColor="F2F2F2")
c3.font = Font(bold=True, size=9)
c3.alignment = Alignment(horizontal="center")
c3.border = border
ws.merge_cells("G5:H5")

# ─── En-têtes colonnes ───
headers = ["#", "Config TQNT", "n", "c (croisements)", "m_Ω (MeV)", "Spin J_set", "Q_topo",
           "τ TQNT", "Lk", "Particule PDG", "m_exp (MeV)", "J_exp", "Q_exp",
           "Δm (%)", "τ_exp", "Note fiabilité"]
ROW0 = 6
for col, h in enumerate(headers, 1):
    c = ws.cell(row=ROW0, column=col, value=h)
    c.fill = header_fill; c.font = header_font
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = border

# Largeurs
widths = [4, 18, 4, 8, 11, 24, 9, 18, 6, 16, 12, 7, 7, 9, 14, 38]
for i, w in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(i)].width = w
ws.row_dimensions[ROW0].height = 36

# ─── Lignes ───
print("Calcul des configs et matches PDG…")
row = ROW0 + 1
config_idx = 0
n_with_match = 0
n_total = 0

light_grey = PatternFill("solid", fgColor="F2F2F2")
white = PatternFill("solid", fgColor="FFFFFF")
# Bleu clair pour les configs sélectionnées comme particules (≥1 match PDG)
light_blue = PatternFill("solid", fgColor="D6E9FF")
# Gris matière noire (P18) : c > 18 AND Q=0 possible AND aucun match PDG
dark_matter_grey = PatternFill("solid", fgColor="A6A6A6")
n_dark_matter = 0


def is_dark_matter_candidate(c_total, Q_topo_val):
    """Critère P18 (V5, chap. 21) : nœud complexe ET intrinsèquement neutre.

    Conditions :
      1. c > 18  : complexité topologique au-delà du seuil nucléonique
      2. |Q_topo| < 0.5 : neutralité INTRINSÈQUE (sans gymnastique de Lk)
         (les particules connues ont besoin de Lk = ±3 ou ±4 pour leur
          charge ; la matière noire doit être neutre dans sa configuration
          naturelle, donc Q_topo ≈ 0 avec Lk = 0)
    """
    if c_total <= 18:
        return False
    # Neutralité intrinsèque : Q_topo doit être proche de 0 sans Lk
    return abs(Q_topo_val) < 0.5


for cfg in ALL:
    config_idx += 1
    n_total += 1
    m = omega(cfg)
    Js = J_set(cfg)
    Qt = Q_topo(cfg)
    matches = find_pdg_matches(m, Js, Qt)
    n_comp = len(cfg)
    # Nombre total de croisements = somme des croisements de chaque nœud
    c_total = sum(TOPO[K]["cr"] for K in cfg)
    cfg_str = " + ".join(cfg)
    Js_str = ", ".join(f"{j:g}" for j in sorted(Js))
    tau_str = lifetime_TQNT(cfg, Js[0], Qt)
    # Candidat matière noire (P18) : c > 18, Q=0 possible, pas de PDG match
    is_dm = (not matches) and is_dark_matter_candidate(c_total, Qt)
    if is_dm:
        n_dark_matter += 1
    # Une ligne par PDG match (au moins une ligne par config)
    if matches:
        # Configuration SÉLECTIONNÉE comme particule → fond bleu clair sur les cols TQNT
        row_fill = light_blue
        n_with_match += 1
        for m_idx, (nm, me, Jx, Qx, tier, note, lk, dm) in enumerate(matches):
            data = [config_idx if m_idx == 0 else "",
                    cfg_str if m_idx == 0 else "",
                    n_comp if m_idx == 0 else "",
                    c_total if m_idx == 0 else "",
                    f"{m:.2f}" if m_idx == 0 else "",
                    Js_str if m_idx == 0 else "",
                    f"{Qt:+.2f}" if m_idx == 0 else "",
                    tau_str if m_idx == 0 else "",
                    f"{lk:+d}",
                    nm,
                    f"{me:.4g}",
                    f"{Jx:g}",
                    f"{Qx:+g}",
                    f"{dm:+.2f}",
                    TAU_PDG.get(nm, ""),
                    note]
            for col, v in enumerate(data, 1):
                c = ws.cell(row=row, column=col, value=v)
                c.border = border
                c.alignment = Alignment(horizontal="center" if col != 16 else "left",
                                        vertical="center", wrap_text=(col == 16))
                # Couleur PDG sur cols 10-16 (cellules de la particule, tier de fiabilité)
                if col >= 10:
                    c.fill = PatternFill("solid", fgColor=TIER_COLORS[tier])
                else:
                    # Cols 1-9 (config TQNT) : bleu clair = SÉLECTIONNÉE comme particule
                    c.fill = row_fill
            row += 1
    else:
        # Pas de match PDG : matière noire (P18) si c>18 et Q=0 possible, sinon neutre
        if is_dm:
            row_fill = dark_matter_grey
            pdg_label = "★ candidat matière noire (P18, c>18, Q=0)"
        else:
            row_fill = light_grey if config_idx % 2 == 0 else white
            pdg_label = "(aucune particule PDG)"
        data = [config_idx, cfg_str, n_comp, c_total, f"{m:.2f}", Js_str, f"{Qt:+.2f}",
                tau_str, "", pdg_label, "", "", "", "", "", ""]
        for col, v in enumerate(data, 1):
            c = ws.cell(row=row, column=col, value=v)
            c.border = border
            c.alignment = Alignment(horizontal="center" if col != 10 else "left",
                                    vertical="center")
            c.fill = row_fill
            if is_dm and col == 10:
                c.font = Font(bold=True, color="FFFFFF")
        row += 1

# ─── Statistiques ───
ws.cell(row=row+1, column=1, value="STATISTIQUES").font = Font(bold=True, size=11)
ws.cell(row=row+2, column=1, value=f"Configurations totales : {n_total}")
ws.cell(row=row+3, column=1, value=f"Configurations avec ≥1 match PDG : {n_with_match} ({n_with_match/n_total*100:.0f}%)")
ws.cell(row=row+4, column=1, value=f"Candidats matière noire (P18 : c>18, Q=0, sans PDG match) : {n_dark_matter}")
ws.cell(row=row+5, column=1, value=f"Particules PDG cataloguées : {len(PDG)}")
ws.cell(row=row+6, column=1, value=f"Formule masse : Ω_ts (V5, paramètres analytiques SU(2)_8)")
ws.cell(row=row+7, column=1, value=f"Règle spin   : P7quater (Clebsch-Gordan)")
ws.cell(row=row+8, column=1, value=f"Règle charge : Q = Σχ_Jones + Lk ∈ ℤ")

# Geler en-tête
ws.freeze_panes = ws[f"A{ROW0+1}"]

# Sauvegarder
output_path = "/home/euk9/projets/physique_nouvelle/docs/TQNT_catalog_complete.xlsx"
wb.save(output_path)
print(f"\n✓ Fichier Excel généré : {output_path}")
print(f"  • {n_total} configurations topologiques")
print(f"  • {n_with_match} avec ≥1 candidat PDG ({n_with_match/n_total*100:.0f}%)")
print(f"  • {n_dark_matter} candidats matière noire (P18 : c>18, Q=0, sans PDG)")
print(f"  • {len(PDG)} particules PDG cataloguées")
print(f"  • Codes couleurs PDG (4 tiers de fiabilité)")
