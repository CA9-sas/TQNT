r"""
EXPORT EXCEL — spectre V6 avec corrections QM standards.

Pour chaque particule PDG : montre côte à côte
  • Ω_ts (V5) — masse brute topologique
  • m_V6      — masse avec corrections QM (mixing, fine structure, EM)
  • Δm V5     — écart V5
  • Δm V6     — écart V6
  • Gain      — réduction de l'écart

Codes couleurs :
  ★★ vert     : V6 très précis (<0.5%)
  ★  vert clair: V6 précis (<1%)
  ✓  jaune    : V6 bon (<5%)
  ⚠  orange   : V6 marginal (5-20%)
  ✗  rouge    : V6 mauvais (>20%)

Type de correction (col dédiée) :
  🟦 mixing η-η'
  🟪 fine structure (SO)
  🟨 EM + isospin
  ⬜ aucun (TQNT pure)

Sortie : docs/TQNT_spectrum_V6.xlsx
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

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
               "det": jones_determinant(pj), "s": topological_spin(K)}
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


# ───────── PARAMÈTRES V6 calibrés (résultat de qm_refinements.py) ─────────
PARAMS_V6 = {
    "M12_PS_sq":  128887,    # M_12 ≈ 359 MeV (η-η' mixing)
    "alpha_SO_c":  3.11,     # MeV/⟨LS⟩ (charmonium spin-orbite)
    "alpha_SO_b": 21.40,     # MeV/⟨LS⟩ (bottomonium spin-orbite)
    "k_EM":       -7.74,     # MeV/Q² (Coulomb self-energy)
    "k_iso":      +2.26,     # MeV/I_3 (isospin splitting)
}


def m_V6(cfg, classe, J=None, Q=None, I_3=None):
    """Masse V6 avec corrections QM applicables."""
    # η, η' : mixing 2×2
    if classe == "eta":
        m_8 = omega(["3_1","3_1","6_1"]); m_0 = omega(["7_7","7_7"])
        h = (m_0**2 - m_8**2)/2
        d = np.sqrt(h**2 + PARAMS_V6["M12_PS_sq"]**2)
        return np.sqrt(max((m_8**2+m_0**2)/2 - d, 0))
    if classe == "etap":
        m_8 = omega(["3_1","3_1","6_1"]); m_0 = omega(["7_7","7_7"])
        h = (m_0**2 - m_8**2)/2
        d = np.sqrt(h**2 + PARAMS_V6["M12_PS_sq"]**2)
        return np.sqrt((m_8**2+m_0**2)/2 + d)
    m = omega(cfg)
    if classe == "chi_c" and J is not None:
        LS = 0.5*(J*(J+1) - 1*2 - 1*2)
        m += PARAMS_V6["alpha_SO_c"] * LS
    elif classe == "chi_b" and J is not None:
        LS = 0.5*(J*(J+1) - 1*2 - 1*2)
        m += PARAMS_V6["alpha_SO_b"] * LS
    if classe in ("ps_em","D_em","B_em","Sigma_em","Xi_em"):
        if Q is not None: m += PARAMS_V6["k_EM"] * Q*Q
        if I_3 is not None: m += PARAMS_V6["k_iso"] * I_3
    return m


# ───────── PDG ÉTENDU avec classes V6 et fiabilité ─────────
# (nom, config, m_exp, J, Q, I_3, classe_V6, tier_fiabilité, note)
# tier : 1=très précis, 2=précis, 3=résonance/exotique, 4=très récent
PDG = [
    # Leptons
    ("e⁻",        ["3_1"],            0.51099895,  0.5, -1, None, "lepton", 1, "0.5 ppt PDG"),
    ("μ⁻",        ["3_1","6_1"],      105.66,      0.5, -1, None, "lepton", 1, "ppm PDG"),
    ("τ⁻",        ["4_1","5_1","6_1"],1776.86,     0.5, -1, None, "lepton", 1, "0.07% PDG"),
    # Bosons jauge
    ("W±",        ["7_1","7_2","8_3"],80369.0,     1,  1, None, "EW", 1, "0.01% LEP+LHC"),
    ("Z⁰",        ["7_2","8_3","9_1"],91187.6,     1,  0, None, "EW", 1, "0.003% LEP"),
    ("H",         ["6_3","8_2","8_3"],125250.0,    0,  0, None, "EW", 1, "0.13% ATLAS+CMS"),
    ("t",         ["6_3","7_1","7_5"],172690.0,    0.5,2/3,None, "EW", 1, "0.18% LHC"),
    # Mésons PS (π : EM ; η,η' : mixing)
    ("π⁰",        ["3_1","3_1","5_2"],134.98,      0,  0,  0,   "ps_em", 1, "0.0004% PDG"),
    ("π±",        ["3_1","3_1","5_2"],139.57,      0, +1, +1,   "ps_em", 1, "ppm PDG"),
    ("K±",        ["4_1","4_1","4_1"],493.68,      0, +1, None, "other", 1, "0.003% PDG"),
    ("K⁰",        ["6_1","6_1","6_1"],497.61,      0,  0, None, "other", 1, "0.003% PDG"),
    ("η",         ["3_1","3_1","6_1"],547.86,      0,  0, None, "eta", 1, "0.003% PDG, MIXING"),
    ("η'",        ["7_7","7_7"],      957.78,      0,  0, None, "etap", 1, "0.006% PDG, MIXING"),
    # Vecteurs
    ("ρ",         ["8_1","8_1"],      775.26,      1, +1, None, "other", 2, "résonance Γ=147 MeV"),
    ("ω",         ["8_1","8_1"],      782.65,      1,  0, None, "other", 1, "0.02% PDG"),
    ("φ",         ["3_1","5_1","5_2"],1019.46,     1,  0, None, "other", 1, "0.002% PDG (lacune s)"),
    ("K*(892)",   ["6_1","6_2"],      891.66,      1, +1, None, "other", 2, "résonance Γ=51 MeV"),
    # Charmonium
    ("η_c(1S)",   ["6_3","7_7"],      2983.9,      0,  0, None, "other", 1, "0.01% PDG"),
    ("η_c(2S)",   ["6_1","8_2"],      3637.5,      0,  0, None, "other", 2, "0.3% PDG"),
    ("J/ψ",       ["3_1","4_1","6_3"],3096.9,      1,  0, None, "other", 1, "0.0003% PDG"),
    ("ψ(2S)",     ["6_1","8_2"],      3686.1,      1,  0, None, "other", 1, "0.0003% PDG"),
    ("ψ(3770)",   ["3_1","7_2","7_7"],3773.7,      1,  0, None, "other", 2, "résonance large"),
    ("χ_c0",      ["6_1","7_7","7_7"],3414.71,     0,  0, None, "chi_c", 1, "FINE STRUCT"),
    ("χ_c1",      ["7_2","8_1","8_1"],3510.67,     1,  0, None, "chi_c", 1, "FINE STRUCT"),
    ("χ_c2",      ["6_1","8_2"],      3556.17,     2,  0, None, "chi_c", 1, "FINE STRUCT"),
    # Bottomonium
    ("η_b(1S)",   ["5_1","7_3","8_1"],9398.7,      0,  0, None, "other", 2, "0.02% PDG"),
    ("Υ(1S)",     ["6_2","7_1","7_3"],9460.30,     1,  0, None, "other", 1, "0.003% PDG"),
    ("Υ(2S)",     ["6_2","6_2","8_2"],10023.26,    1,  0, None, "other", 1, "0.003% PDG"),
    ("Υ(3S)",     ["7_2","10_1"],     10355.2,     1,  0, None, "other", 1, "0.005% PDG"),
    ("χ_b0",      ["6_1","7_3","7_7"],9859.44,     0,  0, None, "chi_b", 1, "FINE STRUCT"),
    ("χ_b1",      ["6_1","7_3","7_7"],9892.78,     1,  0, None, "chi_b", 1, "FINE STRUCT"),
    ("χ_b2",      ["6_2","6_2","8_2"],9912.21,     2,  0, None, "chi_b", 1, "FINE STRUCT"),
    # D/B mésons
    ("D⁰",        ["3_1","5_1","7_2"],1864.84,     0,  0, None, "D_em", 1, "0.003% PDG EM"),
    ("D±",        ["3_1","5_1","7_2"],1869.66,     0, +1, None, "D_em", 1, "0.003% PDG EM"),
    ("D_s",       ["3_1","6_1","7_1"],1968.34,     0, +1, None, "other", 1, "0.003% PDG"),
    ("B±",        ["3_1","6_2","7_7"],5279.34,     0, +1, None, "B_em", 1, "0.002% PDG EM"),
    ("B⁰",        ["3_1","6_2","7_7"],5279.66,     0,  0, None, "B_em", 1, "0.002% PDG EM"),
    ("B_s",       ["6_3","7_6"],      5366.92,     0,  0, None, "other", 1, "0.003% PDG"),
    ("B_c",       ["4_1","7_7","9_1"],6274.9,      0, +1, None, "other", 1, "0.07% PDG"),
    # Octet baryons
    ("p",         ["3_1","7_2","7_2"],938.27,      0.5,+1, None, "nucleon", 1, "ppb PDG, β tension"),
    ("n",         ["3_1","7_2","7_2"],939.57,      0.5, 0, None, "nucleon", 1, "ppb PDG, β tension"),
    ("Λ",         ["5_1","8_1"],      1115.68,     0.5, 0, None, "other", 1, "0.0005% PDG"),
    ("Σ⁺",        ["4_1","4_1","5_1"],1189.37,     0.5,+1,+1,   "Sigma_em", 1, "0.006% PDG EM"),
    ("Σ⁰",        ["4_1","4_1","5_1"],1192.64,     0.5, 0, 0,   "Sigma_em", 1, "0.002% PDG EM"),
    ("Σ⁻",        ["4_1","4_1","5_1"],1197.45,     0.5,-1,-1,   "Sigma_em", 1, "0.0006% PDG EM"),
    ("Ξ⁰",        ["4_1","5_1","5_2"],1314.86,     0.5, 0,+0.5, "Xi_em", 1, "0.015% PDG EM"),
    ("Ξ⁻",        ["4_1","5_1","5_2"],1321.71,     0.5,-1,-0.5, "Xi_em", 1, "0.0006% PDG EM"),
    # Décuplet
    ("Δ(1232)",   ["7_2","9_1"],      1232.0,      1.5,+1, None, "other", 3, "Γ=120 MeV"),
    # Charmé/beau baryons
    ("Λ_c",       ["3_1","3_1","9_1"],2286.46,     0.5,+1, None, "other", 1, "0.006% PDG"),
    ("Σ_c(2455)", ["4_1","5_1","7_2"],2453.97,     0.5,+1, None, "other", 1, "0.005% PDG"),
    ("Ξ_c⁺",      ["3_1","4_1","8_1"],2467.94,     0.5,+1, None, "other", 1, "0.01% PDG"),
    ("Ξ_c⁰",      ["3_1","4_1","8_1"],2470.9,      0.5, 0, None, "other", 1, "0.01% PDG"),
    ("Ω_c",       ["4_1","4_1","7_7"],2695.2,      0.5, 0, None, "other", 2, "0.06% PDG"),
    ("Λ_b",       ["6_3","7_4"],      5619.6,      0.5, 0, None, "other", 1, "0.004% LHCb"),
    ("Σ_b",       ["7_3","7_4"],      5811.0,      0.5,+1, None, "other", 2, "0.1% PDG"),
    ("Ξ_b⁰",      ["7_3","7_4"],      5791.9,      0.5, 0, None, "other", 2, "0.06% PDG"),
    ("Ξ_b⁻",      ["7_3","7_4"],      5797.0,      0.5,-1, None, "other", 1, "0.01% PDG"),
]

# ───────── styles ─────────
COLORS = {
    "header":      "305496",
    "subheader":   "9DC3E6",
    "v6_excellent":"00B050",  # vert vif <0.5%
    "v6_good":     "92D050",  # vert clair <1%
    "v6_ok":       "FFC000",  # jaune <5%
    "v6_warn":     "ED7D31",  # orange 5-20%
    "v6_bad":      "C00000",  # rouge >20%
    "corr_mixing": "B4C7E7",  # bleu clair (η-η')
    "corr_so":     "D9C6E7",  # violet clair (fine structure)
    "corr_em":     "FFF2CC",  # jaune pâle (EM)
    "corr_none":   "F2F2F2",  # gris clair (aucun)
    "tier_1":      "C6EFCE",  # vert pâle (très précis)
    "tier_2":      "FFEB9C",  # jaune pâle (précis)
    "tier_3":      "FFC7CE",  # rose (résonance)
    "tier_4":      "F4B4B4",  # rouge clair (préliminaire)
}


def v6_color(abs_err_pct):
    if abs_err_pct < 0.5: return COLORS["v6_excellent"]
    if abs_err_pct < 1:   return COLORS["v6_good"]
    if abs_err_pct < 5:   return COLORS["v6_ok"]
    if abs_err_pct < 20:  return COLORS["v6_warn"]
    return COLORS["v6_bad"]


def corr_label_color(classe):
    if classe in ("eta","etap"):
        return ("🟦 mixing η-η'", COLORS["corr_mixing"])
    if classe in ("chi_c","chi_b"):
        return ("🟪 fine structure", COLORS["corr_so"])
    if classe in ("ps_em","D_em","B_em","Sigma_em","Xi_em"):
        return ("🟨 EM + isospin", COLORS["corr_em"])
    if classe == "nucleon":
        return ("🟧 β tension V5", COLORS["corr_em"])  # tension déjà en V5
    return ("⬜ Ω_ts pur", COLORS["corr_none"])


# ───────── construction du fichier Excel ─────────
print("\nConstruction du fichier Excel V6…")
wb = Workbook()
ws = wb.active
ws.title = "Spectre TQNT-V6"

# En-tête principal
ws["A1"] = "SPECTRE TQNT-V6 : V5 + corrections QM standards (mixing η-η', fine structure, EM)"
ws["A1"].font = Font(bold=True, size=14, color=COLORS["header"])
ws.merge_cells("A1:N1")
ws["A2"] = ("Pour chaque particule : Ω_ts(V5) + corrections QM applicables, comparaison "
            "côte à côte avec l'expérience. Codes couleurs pour précision V6 et type de correction.")
ws["A2"].font = Font(italic=True, size=9)
ws.merge_cells("A2:N2")

# Légendes
ws["A4"] = "PRÉCISION V6 :"; ws["A4"].font = Font(bold=True, size=10)
for i, (label, color_key) in enumerate([
    ("★★ <0.5%", "v6_excellent"),
    ("★ <1%", "v6_good"),
    ("✓ <5%", "v6_ok"),
    ("⚠ 5-20%", "v6_warn"),
    ("✗ >20%", "v6_bad"),
]):
    c = ws.cell(row=4, column=2+i, value=label)
    c.fill = PatternFill("solid", fgColor=COLORS[color_key])
    c.font = Font(bold=True, size=9, color="FFFFFF" if i==4 else "000000")
    c.alignment = Alignment(horizontal="center")

ws["A5"] = "TYPE CORRECTION :"; ws["A5"].font = Font(bold=True, size=10)
for i, (label, color_key) in enumerate([
    ("🟦 mixing η-η'", "corr_mixing"),
    ("🟪 fine structure", "corr_so"),
    ("🟨 EM + isospin", "corr_em"),
    ("⬜ Ω_ts pur", "corr_none"),
]):
    c = ws.cell(row=5, column=2+i, value=label)
    c.fill = PatternFill("solid", fgColor=COLORS[color_key])
    c.font = Font(bold=True, size=9)
    c.alignment = Alignment(horizontal="center")

# Paramètres V6 calibrés
ws["A7"] = "PARAMÈTRES V6 (calibrés localement) :"
ws["A7"].font = Font(bold=True, size=10, color=COLORS["header"])
params_text = [
    ("M_12 (η-η' mixing)",   f"{np.sqrt(PARAMS_V6['M12_PS_sq']):.0f} MeV", "anomalie U(1)_A"),
    ("α_SO_c (charmonium)",  f"{PARAMS_V6['alpha_SO_c']:+.2f} MeV/⟨LS⟩", "spin-orbite c"),
    ("α_SO_b (bottomonium)", f"{PARAMS_V6['alpha_SO_b']:+.2f} MeV/⟨LS⟩", "spin-orbite b"),
    ("k_EM (Coulomb)",       f"{PARAMS_V6['k_EM']:+.2f} MeV/Q²",         "self-énergie"),
    ("k_iso (isospin)",      f"{PARAMS_V6['k_iso']:+.2f} MeV/I_3",       "différence d-u"),
]
for i, (name, val, desc) in enumerate(params_text):
    ws.cell(row=8+i, column=1, value=name).font = Font(bold=True, size=9)
    ws.cell(row=8+i, column=2, value=val).font = Font(size=9)
    ws.cell(row=8+i, column=3, value=desc).font = Font(italic=True, size=9)

# En-têtes colonnes
ROW0 = 15
headers = ["Particule", "Config TQNT", "Classe V6", "m_exp (MeV)",
           "Ω_ts V5 (MeV)", "Δ V5 %", "m V6 (MeV)", "Δ V6 %",
           "Gain Δ", "J", "Q", "Correction", "Tier PDG", "Note"]
thin = Side(border_style="thin", color="888888")
border = Border(top=thin, bottom=thin, left=thin, right=thin)
for col, h in enumerate(headers, 1):
    c = ws.cell(row=ROW0, column=col, value=h)
    c.fill = PatternFill("solid", fgColor=COLORS["header"])
    c.font = Font(bold=True, color="FFFFFF", size=10)
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = border

widths = [12, 22, 12, 11, 12, 8, 12, 8, 8, 5, 5, 17, 7, 35]
for i, w in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(i)].width = w
ws.row_dimensions[ROW0].height = 34

# Lignes
print(f"  Traitement de {len(PDG)} particules…")
n_v6_excellent = n_v6_good = n_v6_ok = n_v6_warn = n_v6_bad = 0
gains = []
for i, (nm, cfg, m_exp, J, Q, I_3, classe, tier, note) in enumerate(PDG):
    row = ROW0 + 1 + i
    # Calcul Ω_ts V5
    if classe == "eta":   m_v5 = omega(["3_1","3_1","6_1"])
    elif classe == "etap": m_v5 = omega(["7_7","7_7"])
    else:                 m_v5 = omega(cfg)
    m_v6_val = m_V6(cfg, classe, J, Q, I_3)
    d_v5 = (m_exp - m_v5)/m_exp*100
    d_v6 = (m_exp - m_v6_val)/m_exp*100
    gain = abs(d_v5) - abs(d_v6)
    gains.append(gain)
    abs_v6 = abs(d_v6)
    if abs_v6 < 0.5: n_v6_excellent += 1
    elif abs_v6 < 1: n_v6_good += 1
    elif abs_v6 < 5: n_v6_ok += 1
    elif abs_v6 < 20: n_v6_warn += 1
    else: n_v6_bad += 1

    corr_label, corr_color = corr_label_color(classe)
    tier_color = COLORS[f"tier_{tier}"]
    v6_col = v6_color(abs_v6)

    data = [
        nm,
        " + ".join(cfg),
        classe,
        f"{m_exp:.4g}",
        f"{m_v5:.2f}",
        f"{d_v5:+.2f}",
        f"{m_v6_val:.2f}",
        f"{d_v6:+.2f}",
        f"{gain:+.2f}",
        f"{J:g}",
        f"{Q:+g}" if isinstance(Q,(int,float)) else str(Q),
        corr_label,
        tier,
        note,
    ]
    fills = [
        None, None, corr_color, None, None,  # particule, cfg, classe, m_exp, V5
        None,                                  # Δ V5
        None,                                  # m_v6
        PatternFill("solid", fgColor=v6_col),  # Δ V6 (couleur précision)
        PatternFill("solid", fgColor="C6EFCE" if gain > 0.05 else
                                     ("F2F2F2" if abs(gain) < 0.05 else "FFC7CE")),
        None, None,                            # J, Q
        PatternFill("solid", fgColor=corr_color),
        PatternFill("solid", fgColor=tier_color),
        None,
    ]
    for col, (v, fill) in enumerate(zip(data, fills), 1):
        c = ws.cell(row=row, column=col, value=v)
        c.border = border
        c.alignment = Alignment(horizontal="center" if col != 14 else "left",
                                vertical="center", wrap_text=(col == 14))
        if isinstance(fill, PatternFill):
            c.fill = fill
            # Texte blanc sur fond rouge
            if fill.fgColor and "C00000" in str(fill.fgColor.rgb):
                c.font = Font(color="FFFFFF", bold=True)

# Statistiques
print(f"  Statistiques V6 :")
print(f"    ★★ <0.5%  : {n_v6_excellent}")
print(f"    ★  <1%    : {n_v6_good}")
print(f"    ✓  <5%    : {n_v6_ok}")
print(f"    ⚠  5-20%  : {n_v6_warn}")
print(f"    ✗  >20%   : {n_v6_bad}")

# Bloc statistiques en bas
stat_row = ROW0 + len(PDG) + 3
ws.cell(row=stat_row, column=1, value="STATISTIQUES V6").font = Font(bold=True, size=11, color=COLORS["header"])
ws.cell(row=stat_row+1, column=1, value=f"Particules cataloguées : {len(PDG)}")
ws.cell(row=stat_row+2, column=1, value=f"★★ V6 <0.5%  : {n_v6_excellent} ({n_v6_excellent/len(PDG)*100:.0f}%)")
ws.cell(row=stat_row+3, column=1, value=f"★  V6 <1%    : {n_v6_good} ({n_v6_good/len(PDG)*100:.0f}%)")
ws.cell(row=stat_row+4, column=1, value=f"✓  V6 <5%    : {n_v6_ok} ({n_v6_ok/len(PDG)*100:.0f}%)")
ws.cell(row=stat_row+5, column=1, value=f"⚠  V6 5-20%  : {n_v6_warn}")
ws.cell(row=stat_row+6, column=1, value=f"✗  V6 >20%   : {n_v6_bad}")
mean_gain = float(np.mean(gains))
positive_gains = sum(1 for g in gains if g > 0.01)
ws.cell(row=stat_row+8, column=1, value=f"Gain moyen V5→V6     : {mean_gain:+.3f}%").font = Font(bold=True)
ws.cell(row=stat_row+9, column=1, value=f"Particules améliorées : {positive_gains}/{len(PDG)}")

# Geler en-tête
ws.freeze_panes = ws[f"A{ROW0+1}"]

# Sauvegarder
output_path = "/home/euk9/projets/physique_nouvelle/docs/TQNT_spectrum_V6.xlsx"
wb.save(output_path)
print(f"\n✓ Fichier Excel V6 généré : {output_path}")
print(f"  • {len(PDG)} particules")
print(f"  • V5 + 5 corrections QM standards (1 mixing, 2 fine structure, 2 EM)")
print(f"  • Codes couleurs : précision V6 + type de correction + tier PDG")
