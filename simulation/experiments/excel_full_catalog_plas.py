r"""
EXPORT EXCEL — catalogue complet avec PLASTICITÉ (P23) intégrée.

Version étendue de excel_full_catalog.py qui ajoute :
  • masse Ω_ts_plast = Ω_ts × (1 + correction P23)
  • correction P23 = σ(L) · γ(L) · ε(n) où :
      σ(L) = +1 si pseudoscalaire (J=0), -1 sinon (anomalie U(1)_A)
      γ(L) = C_plast / Σ U_κU_τ(L) (amplitude inverse énergie)
      ε(n) = 0 si n=1 (symétrie radiale), 1 sinon
  • C_plast calibré sur μ⁻ (3_1+6_1) : -2.92

Sortie : docs/TQNT_catalog_plas_complete.xlsx

Particules réassignées en fonction de la nouvelle formule.
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

# Constante de plasticité P23 (calibrée sur μ⁻)
C_PLAST = -2.92  # MeV équivalent dans Σ U
# μ⁻ = 3_1+6_1 : Σ U ≈ 125.34, sigma = -1 (fermion)
# correction = (-1)(-2.92)/125.34 = +0.0233 ≈ +2.33% sur 103.26 = +2.4 MeV → cible 105.66 ✓

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
    g = frenet_geometry(K, 30000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)),
               float(np.sum(g["tau"]**2*ds)))
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


def omega_v5(comps):
    """Ω_ts strict (V5, sans plasticité)."""
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


def plasticity_correction(comps, J_options):
    """Correction P23 σ(L) · γ(L) · ε(n).
       Renvoie un facteur multiplicatif (1 + delta)."""
    n = len(comps)
    if n < 2:
        return 1.0, 0.0  # symétrie radiale → pas de plasticité
    # σ(L) = +1 si pseudoscalaire (J=0 dans J_options), -1 sinon
    is_pseudo = (0.0 in J_options)
    sigma = +1 if is_pseudo else -1
    # γ(L) = C_plast / Σ U_κU_τ
    sum_U = sum(U_KT[K][0]*U_KT[K][1] for K in comps)
    gamma = C_PLAST / sum_U
    delta = sigma * gamma
    return 1.0 + delta, delta * 100  # facteur, % de correction


def omega_with_plasticity(comps, J_options):
    """Ω_ts × (1 + correction P23)."""
    m_v5 = omega_v5(comps)
    factor, _ = plasticity_correction(comps, J_options)
    return m_v5 * factor


def J_set(comps):
    """P7quater Clebsch-Gordan."""
    s_max = sum(TOPO[K]["s"] for K in comps)
    parity = s_max - int(s_max)
    n_steps = int(round(s_max - parity)) + 1
    return [parity + i for i in range(n_steps)]


def Q_topo(comps):
    return sum(TOPO[K]["q"] for K in comps)


# ───────── base PDG (réutilisée depuis excel_full_catalog) ─────────
PDG = [
    # Leptons
    ("e⁻", 0.51099895, 0.5, -1, 1, "0.5 ppt PDG"),
    ("μ⁻", 105.6583755, 0.5, -1, 1, "ppm PDG"),
    ("τ⁻", 1776.86, 0.5, -1, 1, "0.07% PDG"),
    # Bosons
    ("γ", 0.0, 1.0, 0, 1, "exact"),
    ("W±", 80369.0, 1.0, 1, 1, "0.01% LEP+LHC"),
    ("Z⁰", 91187.6, 1.0, 0, 1, "0.003% LEP"),
    ("H", 125250.0, 0.0, 0, 1, "0.13% ATLAS+CMS"),
    ("t", 172690.0, 0.5, 2/3, 1, "0.18% LHC"),
    # Pseudoscalars
    ("π⁰", 134.9768, 0.0, 0, 1, "0.0004% PDG"),
    ("π±", 139.57039, 0.0, 1, 1, "ppm PDG"),
    ("K±", 493.677, 0.0, 1, 1, "0.003% PDG"),
    ("K⁰", 497.611, 0.0, 0, 1, "0.003% PDG"),
    ("η", 547.862, 0.0, 0, 1, "0.003% PDG"),
    ("η'", 957.78, 0.0, 0, 1, "0.006% PDG"),
    # Charmonium
    ("η_c(1S)", 2983.9, 0.0, 0, 1, "0.01% PDG"),
    ("η_c(2S)", 3637.5, 0.0, 0, 2, "0.3% PDG"),
    ("J/ψ", 3096.9, 1.0, 0, 1, "0.0003% PDG"),
    ("ψ(2S)", 3686.10, 1.0, 0, 1, "0.0003% PDG"),
    ("χ_c0(1P)", 3414.71, 0.0, 0, 1, "0.01% PDG"),
    ("χ_c1(1P)", 3510.67, 1.0, 0, 1, "0.002% PDG"),
    ("χ_c2(1P)", 3556.17, 2.0, 0, 1, "0.002% PDG"),
    # Bottomonium
    ("η_b(1S)", 9398.7, 0.0, 0, 2, "0.02% PDG"),
    ("Υ(1S)", 9460.30, 1.0, 0, 1, "0.003% PDG"),
    ("Υ(2S)", 10023.26, 1.0, 0, 1, "0.003% PDG"),
    ("χ_b0(1P)", 9859.44, 0.0, 0, 1, "0.05% PDG"),
    ("χ_b1(1P)", 9892.78, 1.0, 0, 1, "0.04% PDG"),
    ("χ_b2(1P)", 9912.21, 2.0, 0, 1, "0.03% PDG"),
    # Charmed mesons
    ("D⁰", 1864.84, 0.0, 0, 1, "0.003% PDG"),
    ("D±", 1869.66, 0.0, 1, 1, "0.003% PDG"),
    ("D_s±", 1968.34, 0.0, 1, 1, "0.003% PDG"),
    ("D*(2007)⁰", 2006.85, 1.0, 0, 1, "0.004% PDG"),
    ("D*(2010)±", 2010.26, 1.0, 1, 1, "0.005% PDG"),
    # Bottom mesons
    ("B±", 5279.34, 0.0, 1, 1, "0.002% PDG"),
    ("B⁰", 5279.66, 0.0, 0, 1, "0.002% PDG"),
    ("B_s⁰", 5366.92, 0.0, 0, 1, "0.003% PDG"),
    ("B_c±", 6274.9, 0.0, 1, 1, "0.07% PDG"),
    # Light vector mesons
    ("ρ", 775.26, 1.0, 1, 1, "0.03% PDG"),
    ("ω", 782.65, 1.0, 0, 1, "0.02% PDG"),
    ("φ", 1019.461, 1.0, 0, 1, "0.002% PDG"),
    ("K*(892)", 891.66, 1.0, 1, 1, "0.03% PDG"),
    # Resonances
    ("f₀(980)", 990.0, 0.0, 0, 2, "1% PDG"),
    ("a₂(1320)", 1318.0, 2.0, 1, 2, "0.1% PDG"),
    ("f₂(1270)", 1275.5, 2.0, 0, 2, "0.06% PDG"),
    # Octet baryons
    ("p", 938.27208, 0.5, 1, 1, "ppb PDG"),
    ("n", 939.56542, 0.5, 0, 1, "ppb PDG"),
    ("Λ", 1115.683, 0.5, 0, 1, "0.0005% PDG"),
    ("Σ⁺", 1189.37, 0.5, 1, 1, "0.006% PDG"),
    ("Σ⁰", 1192.642, 0.5, 0, 1, "0.002% PDG"),
    ("Σ⁻", 1197.449, 0.5, -1, 1, "0.0006% PDG"),
    ("Ξ⁰", 1314.86, 0.5, 0, 1, "0.001% PDG"),
    ("Ξ⁻", 1321.71, 0.5, -1, 1, "0.001% PDG"),
    # Decuplet
    ("Δ", 1232.0, 1.5, 1, 2, "1% PDG"),
    ("Σ*⁺", 1382.83, 1.5, 1, 2, "0.1% PDG"),
    ("Ξ*⁰", 1531.80, 1.5, 0, 2, "0.05% PDG"),
    ("Ω⁻", 1672.45, 1.5, -1, 1, "0.002% PDG"),
    # Charm baryons
    ("Λ_c⁺", 2286.46, 0.5, 1, 1, "0.01% PDG"),
    ("Σ_c⁺⁺", 2454.0, 0.5, 2, 2, "0.05% PDG"),
    ("Ξ_c⁺", 2467.71, 0.5, 1, 1, "0.02% PDG"),
    ("Ω_c⁰", 2695.2, 0.5, 0, 1, "0.05% PDG"),
    # Bottom baryons
    ("Λ_b⁰", 5619.6, 0.5, 0, 1, "0.01% PDG"),
    ("Ξ_b⁰", 5791.9, 0.5, 0, 2, "0.05% PDG"),
    ("Ω_b⁻", 6046.0, 0.5, -1, 2, "0.4% PDG"),
    # Exotic
    ("X(3872)", 3871.65, 1.0, 0, 3, "0.01% LHCb"),
    ("Z_c(3900)", 3887.1, 1.0, 1, 3, "0.07% BESIII"),
    ("X(6900)", 6900.0, 0.0, 0, 4, "très récent"),
    ("P_c(4380)", 4380.0, 1.5, 1, 3, "LHCb 2015"),
    ("P_c(4450)", 4449.8, 2.5, 1, 3, "LHCb 2015"),
]


TIER_COLORS = {
    1: "C6EFCE", 2: "FFEB9C", 3: "FFC7CE", 4: "F4B4B4",
}
TIER_LABELS = {1: "🟢 très précis", 2: "🟡 précis",
                3: "🟠 résonance/exotique", 4: "🔴 récent/préliminaire"}


def all_configs(max_n=3):
    configs = []
    for n in range(1, max_n+1):
        for combo in combinations_with_replacement(KNOTS, n):
            configs.append(list(combo))
    return configs


def find_pdg_matches_plas(m_v5, m_plas, J_set_pred, Q_topo_pred,
                          tol_m=0.05, lk_range=(-5,5)):
    """PDG candidats avec masse plastique."""
    matches = []
    for nm, me, J, Q, tier, note in PDG:
        if J not in J_set_pred:
            continue
        residue = Q - Q_topo_pred
        lk = round(residue)
        if lk < lk_range[0] or lk > lk_range[1] or abs(residue - lk) > 0.34:
            continue
        # Test sur masse plastique (plus précise)
        err_plas = (m_plas - me) / max(me, 1e-3)
        if abs(err_plas) > tol_m:
            continue
        err_v5 = (m_v5 - me) / max(me, 1e-3)
        matches.append((nm, me, J, Q, tier, note, lk,
                         err_v5*100, err_plas*100))
    matches.sort(key=lambda x: abs(x[8]))  # tri par erreur plastique
    return matches


# ═══════════════ GÉNÉRATION ═══════════════
print("Enumération des configs…")
ALL = all_configs(3)
print(f"  → {len(ALL)} configs")

print("Calcul des configs avec plasticité…")
results = []
n_dark = 0
for cfg in ALL:
    n = len(cfg)
    m_v5 = omega_v5(cfg)
    js = J_set(cfg)
    q_top = Q_topo(cfg)
    factor, delta_pct = plasticity_correction(cfg, js)
    m_plas = m_v5 * factor
    matches = find_pdg_matches_plas(m_v5, m_plas, js, q_top)
    # Dark matter : c>18, Q=0, sans match
    cr_tot = sum(TOPO[K]["cr"] for K in cfg)
    is_dark = (cr_tot > 18 and abs(q_top) < 0.5 and len(matches) == 0)
    if is_dark: n_dark += 1
    results.append({
        "cfg": cfg, "n": n, "m_v5": m_v5, "m_plas": m_plas,
        "delta_pct": delta_pct, "J_set": js, "Q_topo": q_top,
        "matches": matches, "is_dark": is_dark,
        "cr_tot": cr_tot,
    })

n_with_match = sum(1 for r in results if r["matches"])
print(f"  → {n_with_match} configs avec PDG candidat ({n_with_match/len(results)*100:.0f}%)")
print(f"  → {n_dark} candidats matière noire")


# ═══════════════ ÉCRITURE EXCEL ═══════════════
wb = Workbook()
ws = wb.active
ws.title = "Catalogue TQNT plast"

# En-têtes
headers = ["Config", "n", "c (croisements)", "Ω_ts V5 (MeV)",
           "Correction P23 (%)", "Ω_ts plast (MeV)",
           "J spectre", "Q_topo", "Type",
           "PDG candidat(s)", "Δ V5 (%)", "Δ plast (%)", "Lk", "Note"]

for col, h in enumerate(headers, 1):
    c = ws.cell(row=1, column=col, value=h)
    c.font = Font(bold=True, color="FFFFFF")
    c.fill = PatternFill("solid", fgColor="4472C4")
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

# Couleurs
COLOR_PDG_MATCH = "ADD8E6"  # bleu clair
COLOR_DARK = "BFBFBF"        # gris
COLOR_PLAS_POS = "C6EFCE"    # vert plus (correction positive)
COLOR_PLAS_NEG = "FFC7CE"    # rose moins (correction négative)

# Lignes
row = 2
for r in sorted(results, key=lambda x: x["m_v5"]):
    cfg_str = " + ".join(r["cfg"])
    matches_str = "; ".join(m[0] for m in r["matches"][:3])
    if len(r["matches"]) > 3:
        matches_str += f" (+{len(r['matches'])-3} more)"
    delta_v5_str = "" if not r["matches"] else f"{r['matches'][0][7]:+.2f}"
    delta_plas_str = "" if not r["matches"] else f"{r['matches'][0][8]:+.2f}"
    lk_str = "" if not r["matches"] else str(r["matches"][0][6])
    note_str = "" if not r["matches"] else r["matches"][0][5]

    type_str = ""
    if r["is_dark"]: type_str = "🌑 matière noire (P18)"
    elif r["matches"]: type_str = TIER_LABELS.get(r["matches"][0][4], "")

    J_set_str = "{" + ",".join(f"{j:g}" for j in r["J_set"]) + "}"

    cells = [cfg_str, r["n"], r["cr_tot"],
             round(r["m_v5"], 3),
             round(r["delta_pct"], 3),
             round(r["m_plas"], 3),
             J_set_str, round(r["Q_topo"], 3),
             type_str, matches_str,
             delta_v5_str, delta_plas_str, lk_str, note_str]

    for col, val in enumerate(cells, 1):
        c = ws.cell(row=row, column=col, value=val)
        c.alignment = Alignment(horizontal="center", vertical="center")

    # Coloration ligne
    if r["is_dark"]:
        for col in range(1, len(headers)+1):
            ws.cell(row=row, column=col).fill = PatternFill("solid", fgColor=COLOR_DARK)
    elif r["matches"]:
        tier = r["matches"][0][4]
        color = TIER_COLORS.get(tier, "FFFFFF")
        for col in range(1, len(headers)+1):
            ws.cell(row=row, column=col).fill = PatternFill("solid", fgColor=color)

    # Coloration spéciale colonne correction P23
    if abs(r["delta_pct"]) > 0.01:
        col_corr = 5
        c = ws.cell(row=row, column=col_corr)
        if r["delta_pct"] > 0:
            c.fill = PatternFill("solid", fgColor=COLOR_PLAS_POS)
        else:
            c.fill = PatternFill("solid", fgColor=COLOR_PLAS_NEG)
        c.font = Font(bold=True)

    row += 1

# Largeurs
widths = [22, 4, 10, 14, 14, 14, 16, 10, 22, 30, 10, 10, 8, 24]
for col, w in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(col)].width = w

# Gel de la ligne d'en-tête
ws.freeze_panes = "A2"


# ───────── Deuxième feuille : Résumé / Reassignations ─────────
ws2 = wb.create_sheet("Réassignations PDG")
ws2.append(["Particule PDG", "m_exp (MeV)", "Tier",
            "V5 : meilleure config", "V5 : m_pred", "V5 : Δ %",
            "PLAST : meilleure config", "PLAST : m_pred", "PLAST : Δ %",
            "Δ V5 → PLAST"])
for c in ws2[1]:
    c.font = Font(bold=True, color="FFFFFF")
    c.fill = PatternFill("solid", fgColor="4472C4")
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

# Pour chaque PDG, trouver la meilleure config sous V5 et sous PLAST
for nm, me, J, Q, tier, note in PDG:
    best_v5 = (None, float('inf'), None)
    best_plas = (None, float('inf'), None)
    for r in results:
        if J not in r["J_set"]:
            continue
        residue = Q - r["Q_topo"]
        lk = round(residue)
        if abs(residue - lk) > 0.34:
            continue
        err_v5 = abs(r["m_v5"] - me) / max(me, 1e-3)
        err_plas = abs(r["m_plas"] - me) / max(me, 1e-3)
        if err_v5 < best_v5[1]:
            best_v5 = (r["cfg"], err_v5, r["m_v5"])
        if err_plas < best_plas[1]:
            best_plas = (r["cfg"], err_plas, r["m_plas"])

    cfg_v5 = " + ".join(best_v5[0]) if best_v5[0] else "—"
    cfg_plas = " + ".join(best_plas[0]) if best_plas[0] else "—"
    delta_v5 = (best_v5[2] - me) / max(me, 1e-3) * 100 if best_v5[2] else 0
    delta_plas = (best_plas[2] - me) / max(me, 1e-3) * 100 if best_plas[2] else 0
    changed = (best_v5[0] != best_plas[0])

    row_data = [nm, me, TIER_LABELS.get(tier, ""),
                cfg_v5, round(best_v5[2] or 0, 3), f"{delta_v5:+.2f}",
                cfg_plas, round(best_plas[2] or 0, 3), f"{delta_plas:+.2f}",
                "OUI" if changed else "—"]
    ws2.append(row_data)
    if changed:
        for col in range(1, len(row_data)+1):
            ws2.cell(row=ws2.max_row, column=col).fill = PatternFill("solid", fgColor="FFEB9C")

widths2 = [14, 12, 22, 20, 14, 10, 22, 14, 10, 16]
for col, w in enumerate(widths2, 1):
    ws2.column_dimensions[get_column_letter(col)].width = w
ws2.freeze_panes = "A2"

# Compter réassignations
n_reassigned = sum(1 for nm, me, J, Q, tier, note in PDG
                    if True)  # à recalculer
n_reassigned = 0
for nm, me, J, Q, tier, note in PDG:
    best_v5 = None; best_plas = None
    err_v5_min = float('inf'); err_plas_min = float('inf')
    for r in results:
        if J not in r["J_set"]: continue
        residue = Q - r["Q_topo"]
        lk = round(residue)
        if abs(residue - lk) > 0.34: continue
        if abs(r["m_v5"] - me) / max(me, 1e-3) < err_v5_min:
            err_v5_min = abs(r["m_v5"] - me) / max(me, 1e-3)
            best_v5 = r["cfg"]
        if abs(r["m_plas"] - me) / max(me, 1e-3) < err_plas_min:
            err_plas_min = abs(r["m_plas"] - me) / max(me, 1e-3)
            best_plas = r["cfg"]
    if best_v5 != best_plas:
        n_reassigned += 1


# ───────── Troisième feuille : explication ─────────
ws3 = wb.create_sheet("Documentation")
docs = [
    ["TQNT-V5 + Plasticité (P23) — Catalogue complet"],
    [""],
    ["Formule de masse avec plasticité :"],
    ["  m_plast(L) = Ω_ts(L) × (1 + correction P23)"],
    [""],
    ["Correction P23 :"],
    ["  σ(L) = +1 si pseudoscalaire (J=0), -1 sinon (anomalie U(1)_A)"],
    ["  γ(L) = C_plast / Σ U_κU_τ(L)"],
    ["  ε(n) = 0 si n=1 (symétrie radiale), 1 sinon"],
    ["  correction = σ · γ · ε"],
    [""],
    [f"  C_plast calibré sur μ⁻ : {C_PLAST}"],
    [""],
    ["Statistiques :"],
    [f"  Total configs topologiques : {len(results)}"],
    [f"  Configs avec PDG candidat : {n_with_match} ({n_with_match/len(results)*100:.0f}%)"],
    [f"  Candidats matière noire (P18) : {n_dark}"],
    [f"  PDG particules cataloguées : {len(PDG)}"],
    [f"  Réassignations dues à plasticité : {n_reassigned}"],
    [""],
    ["Codes couleurs :"],
    ["  🟢 vert pâle (C6EFCE) : très précis (<0.1%)"],
    ["  🟡 jaune pâle (FFEB9C) : précis (<1%)"],
    ["  🟠 rose (FFC7CE) : résonance/exotique"],
    ["  🔴 rouge pâle (F4B4B4) : récent/préliminaire"],
    ["  ⬜ gris (BFBFBF) : candidat matière noire (P18)"],
    [""],
    ["Coloration spéciale colonne 'Correction P23' :"],
    ["  vert : correction positive (m_plast > m_V5)"],
    ["  rose : correction négative (m_plast < m_V5)"],
]
for i, line in enumerate(docs, 1):
    c = ws3.cell(row=i, column=1, value=line[0] if line else "")
    if i == 1: c.font = Font(bold=True, size=14)
    elif "Formule" in line[0] or "Correction P23" in line[0] or "Statistiques" in line[0] or "Codes" in line[0] or "Coloration" in line[0]:
        c.font = Font(bold=True)
ws3.column_dimensions["A"].width = 60


output_path = "/home/euk9/projets/physique_nouvelle/docs/TQNT_catalog_plas_complete.xlsx"
wb.save(output_path)

print(f"\n✓ Fichier Excel généré : {output_path}")
print(f"  • {len(results)} configurations topologiques")
print(f"  • {n_with_match} avec ≥1 candidat PDG ({n_with_match/len(results)*100:.0f}%)")
print(f"  • {n_dark} candidats matière noire (P18)")
print(f"  • {len(PDG)} particules PDG cataloguées")
print(f"  • {n_reassigned} réassignations dues à plasticité P23")
print(f"  • 3 onglets : Catalogue, Réassignations, Documentation")
