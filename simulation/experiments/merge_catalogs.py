r"""
Fusion des deux catalogues TQNT-V5 pour diffusion publique (v2).

Combine :
  - docs/TQNT_catalog_complete.xlsx       (V5 strict, Ω_ts canonique)
  - docs/TQNT_catalog_plas_complete.xlsx  (V5 + correction P23 plasticité)

Produit :
  docs/TQNT_catalog_V5_diffusion.xlsx

v2 ajoute :
  • c_max corrigé (30, pas 18)
  • Lk neutralisant pour CHAQUE config (intégration entière qui minimise |Q_phys|)
  • Q_phys = Q_topo (Jones) + Lk (neutralisation par enlacement)
  • Nouvelle colonne "Q_phys_min" = charge atteignable la plus proche de 0
  • Onglet matière noire RÉALISABLE = configs où Q peut être = 0 avec Lk physique
"""
from __future__ import annotations
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from theory.jones import topological_charge

SRC_V5   = "/home/euk9/projets/physique_nouvelle/docs/TQNT_catalog_complete.xlsx"
SRC_PLAS = "/home/euk9/projets/physique_nouvelle/docs/TQNT_catalog_plas_complete.xlsx"
DST      = "/home/euk9/projets/physique_nouvelle/docs/TQNT_catalog_V5_diffusion.xlsx"

# ─── Styles ───
TITLE = Font(bold=True, size=14, color="FFFFFF")
HDR   = Font(bold=True, size=11, color="FFFFFF")
SUBHDR = Font(bold=True, size=11)
BODY  = Font(size=10)
NOTE  = Font(size=9, italic=True, color="555555")
FILL_TITLE = PatternFill("solid", fgColor="1F4E78")
FILL_HDR   = PatternFill("solid", fgColor="2E75B6")
FILL_SUB   = PatternFill("solid", fgColor="DDEBF7")
FILL_SEL   = PatternFill("solid", fgColor="E2EFDA")
FILL_DM    = PatternFill("solid", fgColor="EDEDED")
FILL_DM_R  = PatternFill("solid", fgColor="D9E1F2")  # bleu pâle pour DM réalisable
FILL_RA    = PatternFill("solid", fgColor="FFF2CC")
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT   = Alignment(horizontal="left",   vertical="center", wrap_text=True)
THIN   = Side(style="thin", color="999999")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


# ─── Charges intrinsèques par nœud (Jones-derived) ───
KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
Q_KNOT = {K: topological_charge(K) for K in KNOTS}
print("Charges intrinsèques (Q_topo par nœud, Jones-derived) :")
for K, q in sorted(Q_KNOT.items(), key=lambda x: abs(x[1])):
    print(f"  {K:6s} : Q = {q:+.3f}")


def parse_config(cfg_str):
    """Parse 'K_1 + K_2 + ...' en liste de nœuds."""
    return [c.strip() for c in str(cfg_str).split('+')]


def lk_neutral(Q_intr):
    """Entier Lk qui minimise |Q_intr + Lk|."""
    return -round(Q_intr)


def q_phys_min(Q_intr):
    """Charge physique minimale atteignable par Lk entier."""
    return Q_intr + lk_neutral(Q_intr)


# ─── Lecture des sources ───
print("\nLecture des sources…")
wb_v5   = openpyxl.load_workbook(SRC_V5,   data_only=True)
wb_plas = openpyxl.load_workbook(SRC_PLAS, data_only=True)

v5_data = {}
ws = wb_v5["Catalogue TQNT V5"]
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i < 6: continue
    if row[1] is None: continue
    cfg = row[1]
    v5_data[cfg] = {
        "n": row[2], "c": row[3], "m_omega": row[4],
        "J_set": row[5], "Q_topo": row[6], "tau": row[7], "Lk": row[8],
        "pdg": row[9], "m_exp": row[10], "J_exp": row[11], "Q_exp": row[12],
        "delta_v5": row[13],
    }

plas_data = {}
ws = wb_plas["Catalogue TQNT plast"]
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i < 1: continue
    if row[0] is None: continue
    plas_data[row[0]] = {
        "m_v5": row[3], "corr_P23": row[4], "m_plast": row[5],
        "J_set": row[6], "Q_topo": row[7], "type": row[8], "pdg": row[9],
        "delta_v5": row[10], "delta_plas": row[11], "Lk": row[12], "note": row[13],
    }

print(f"  V5 strict      : {len(v5_data)} configs")
print(f"  V5 plasticité  : {len(plas_data)} configs")

# Calcul Q_phys_min + Lk_neutral pour chaque config
for cfg, data in v5_data.items():
    comps = parse_config(cfg)
    try:
        Q_intr = sum(Q_KNOT[K] for K in comps)
    except KeyError:
        Q_intr = None
    if Q_intr is None:
        data["Q_intrinsic"] = None
        data["Lk_neutral"]  = None
        data["Q_phys_min"]  = None
    else:
        data["Q_intrinsic"] = Q_intr
        data["Lk_neutral"]  = lk_neutral(Q_intr)
        data["Q_phys_min"]  = q_phys_min(Q_intr)


# ─── Construction du fichier unifié ───
print("\nConstruction du fichier unifié v2…")
wb = openpyxl.Workbook()
wb.remove(wb.active)

# ─── Sheet 1 sera remplie en fin ───
ws_intro = wb.create_sheet("1. Présentation")


# ─── Sheet 2 : Catalogue unifié ───
print("  → Catalogue unifié")
ws = wb.create_sheet("2. Catalogue unifié")

ws.cell(row=1, column=1, value="CATALOGUE COMPLET TQNT-V5 (strict, plasticité et neutralisation Lk)").font = TITLE
ws.cell(row=1, column=1).fill = FILL_TITLE
ws.cell(row=1, column=1).alignment = CENTER
ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=18)
ws.row_dimensions[1].height = 26

HEADERS = [
    "#", "Config TQNT", "n", "c", "J spectre",
    "Q_topo (Jones)", "Lk neutralisant", "Q_phys min", "Lk PDG",
    "m(Ω_ts) V5 (MeV)", "P23 correction (%)", "m(Ω_ts) plast (MeV)",
    "Type", "Particule PDG", "m_exp (MeV)", "Δ V5 (%)", "Δ plast (%)", "Note",
]
for c, h in enumerate(HEADERS, start=1):
    cell = ws.cell(row=2, column=c, value=h)
    cell.font = HDR; cell.fill = FILL_HDR; cell.alignment = CENTER; cell.border = BORDER
ws.row_dimensions[2].height = 38

widths = [5, 22, 4, 4, 14, 11, 11, 10, 7, 18, 12, 18,
          11, 16, 11, 9, 9, 30]
for i, w in enumerate(widths, start=1):
    ws.column_dimensions[get_column_letter(i)].width = w


def _f(x, d=None):
    try: return float(x)
    except (TypeError, ValueError): return d


def _mass_key(k):
    v = _f(v5_data[k]["m_omega"])
    return v if v is not None else 1e30


sorted_cfgs = sorted(v5_data.keys(), key=_mass_key)

row = 3
n_pdg = 0
n_dm_strict = 0       # Q_topo brut près de 0 (sans Lk gymnastics)
n_dm_realisable = 0   # |Q_phys_min| < 0.05 (neutralisable par Lk entier)
n_dm_strict_high_c = 0
n_dm_real_high_c = 0
c_max = 0

for cfg in sorted_cfgs:
    v5 = v5_data[cfg]
    pl = plas_data.get(cfg, {})

    pdg = v5["pdg"]
    has_pdg = pdg and "aucune" not in str(pdg)

    n_val = _f(v5["n"])
    c_val = _f(v5["c"])
    m_val = _f(v5["m_omega"])
    if c_val: c_max = max(c_max, int(c_val))

    Q_intr = v5["Q_intrinsic"]
    Q_min  = v5["Q_phys_min"]
    Lk_n   = v5["Lk_neutral"]

    is_dm_strict = (not has_pdg) and m_val and m_val > 1e3 \
                   and n_val and n_val >= 2 \
                   and Q_intr is not None and abs(Q_intr) < 0.1
    is_dm_real = (not has_pdg) and m_val and m_val > 1e3 \
                 and n_val and n_val >= 2 \
                 and Q_min is not None and abs(Q_min) < 0.05 \
                 and Lk_n is not None and abs(Lk_n) <= max(5, c_val or 5)

    if has_pdg: fill, marker = FILL_SEL, ""
    elif is_dm_strict:
        fill, marker = FILL_DM, "⚫ MN stricte"
        n_dm_strict += 1
        if c_val and c_val >= 18: n_dm_strict_high_c += 1
    elif is_dm_real:
        fill, marker = FILL_DM_R, "⚫ MN réalisable (Lk)"
        n_dm_realisable += 1
        if c_val and c_val >= 18: n_dm_real_high_c += 1
    else:
        fill, marker = None, "⚪ libre"

    if has_pdg: n_pdg += 1

    type_str = pl.get("type", "—")
    corr     = pl.get("corr_P23", "—")
    m_plast  = pl.get("m_plast", v5["m_omega"])
    d_plast  = pl.get("delta_plas", "—")

    values = [
        row - 2, cfg, v5["n"], v5["c"], v5["J_set"],
        round(Q_intr, 3) if Q_intr is not None else "—",
        Lk_n if Lk_n is not None else "—",
        round(Q_min, 3) if Q_min is not None else "—",
        v5["Lk"] if has_pdg else "—",
        v5["m_omega"], corr, m_plast,
        type_str, pdg if has_pdg else marker,
        v5["m_exp"] if has_pdg else "—",
        v5["delta_v5"] if has_pdg else "—",
        d_plast if has_pdg else "—",
        pl.get("note", "—") if pl else "—",
    ]
    for c, v in enumerate(values, start=1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.font = BODY
        if fill is not None: cell.fill = fill
        cell.alignment = CENTER if c != 18 else LEFT
        cell.border = BORDER
    row += 1

ws.freeze_panes = "A3"

print(f"     {n_pdg} PDG, {n_dm_strict} MN stricte ({n_dm_strict_high_c} à c≥18), "
      f"{n_dm_realisable} MN réalisable ({n_dm_real_high_c} à c≥18)")
print(f"     c_max effectif : {c_max}")


# ─── Sheet 3 : Particules PDG ───
print("  → Particules PDG")
ws = wb.create_sheet("3. Particules PDG")
ws.cell(row=1, column=1, value="PARTICULES PDG ASSIGNÉES — V5 strict & plasticité").font = TITLE
ws.cell(row=1, column=1).fill = FILL_TITLE; ws.cell(row=1, column=1).alignment = CENTER
ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=10)
ws.row_dimensions[1].height = 26

HDRS = ["Particule", "m_exp (MeV)", "Tier", "Config V5", "m V5", "Δ V5 %",
        "Config plast", "m plast", "Δ plast %", "Évolution"]
for c, h in enumerate(HDRS, start=1):
    cell = ws.cell(row=2, column=c, value=h)
    cell.font = HDR; cell.fill = FILL_HDR; cell.alignment = CENTER; cell.border = BORDER
ws.row_dimensions[2].height = 30
for i, w in enumerate([14, 14, 8, 22, 12, 10, 22, 12, 10, 14], start=1):
    ws.column_dimensions[get_column_letter(i)].width = w

ws_src = wb_plas["Réassignations PDG"]
row = 3
for i, src_row in enumerate(ws_src.iter_rows(values_only=True)):
    if i < 1: continue
    for c, v in enumerate(src_row[:10], start=1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.font = BODY; cell.alignment = CENTER; cell.border = BORDER
    row += 1
ws.freeze_panes = "A3"


# ─── Sheet 4 : Réassignations P23 ───
print("  → Réassignations P23")
ws = wb.create_sheet("4. Réassignations P23")
ws.cell(row=1, column=1, value="PARTICULES RÉASSIGNÉES PAR LA PLASTICITÉ P23").font = TITLE
ws.cell(row=1, column=1).fill = FILL_TITLE; ws.cell(row=1, column=1).alignment = CENTER
ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=8)
ws.row_dimensions[1].height = 26

HDRS = ["Particule", "m_exp (MeV)", "Config V5 strict", "Δ V5 %",
        "Config V5 plast", "Δ plast %", "Δ amélioration (%)", "Commentaire"]
for c, h in enumerate(HDRS, start=1):
    cell = ws.cell(row=2, column=c, value=h)
    cell.font = HDR; cell.fill = FILL_HDR; cell.alignment = CENTER; cell.border = BORDER
ws.row_dimensions[2].height = 30
for i, w in enumerate([14, 14, 22, 10, 22, 10, 14, 40], start=1):
    ws.column_dimensions[get_column_letter(i)].width = w

ws_src = wb_plas["Réassignations PDG"]
row = 3
n_reassigned = 0
for i, src_row in enumerate(ws_src.iter_rows(values_only=True)):
    if i < 1: continue
    if not src_row[0]: continue
    cfg_v5 = src_row[3]; cfg_plas = src_row[6]
    if cfg_v5 != cfg_plas:
        n_reassigned += 1
        try:
            d_v5  = abs(float(src_row[5]))
            d_pla = abs(float(src_row[8]))
            delta = round(d_v5 - d_pla, 2)
        except (TypeError, ValueError):
            delta = "—"
        cells = [src_row[0], src_row[1], cfg_v5, src_row[5],
                 cfg_plas, src_row[8], delta,
                 "Plasticité change l'assignation optimale"]
        for c, v in enumerate(cells, start=1):
            cell = ws.cell(row=row, column=c, value=v)
            cell.font = BODY; cell.fill = FILL_RA; cell.alignment = CENTER; cell.border = BORDER
        row += 1
ws.freeze_panes = "A3"


# ─── Sheet 5 : Matière noire ───
print("  → Matière noire (stricte + réalisable)")
ws = wb.create_sheet("5. Matière noire")
ws.cell(row=1, column=1, value="CANDIDATS MATIÈRE NOIRE — stricts ET réalisables par Lk").font = TITLE
ws.cell(row=1, column=1).fill = FILL_TITLE; ws.cell(row=1, column=1).alignment = CENTER
ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=9)
ws.row_dimensions[1].height = 26

# Bloc explicatif
expl = [
    ("Deux niveaux de candidats matière noire :", SUBHDR, FILL_SUB, LEFT),
    ("⚫ STRICT     : Q_topo (Jones) intrinsèquement ≈ 0 sans aucun Lk — "
     "configurations d'amphichiraux pur (4_1, 6_3, 8_3).", BODY, None, LEFT),
    ("⚫ RÉALISABLE : Q_phys = Q_topo + Lk peut atteindre 0 par un Lk entier physiquement réaliste "
     "(|Lk| ≤ max(5, c)). Inclut TOUTES les configs où Q_topo est entier modulo Lk.", BODY, None, LEFT),
    ("Critères communs : m > 1 GeV, n ≥ 2, sans correspondance PDG dans 5%.", BODY, None, LEFT),
    ("", None, None, None),
]
for i, (text, font, fill, align) in enumerate(expl, start=2):
    cell = ws.cell(row=i, column=1, value=text)
    if font: cell.font = font
    if fill: cell.fill = fill
    if align: cell.alignment = align
    ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=9)

HDRS = ["#", "Niveau", "Config TQNT", "n", "c",
        "m (MeV)", "Q_topo", "Lk neutr.", "Q_phys"]
hdr_row = 2 + len(expl) + 1
for c, h in enumerate(HDRS, start=1):
    cell = ws.cell(row=hdr_row, column=c, value=h)
    cell.font = HDR; cell.fill = FILL_HDR; cell.alignment = CENTER; cell.border = BORDER
ws.row_dimensions[hdr_row].height = 28
for i, w in enumerate([5, 22, 26, 4, 4, 14, 11, 10, 10], start=1):
    ws.column_dimensions[get_column_letter(i)].width = w

# 5a : DM stricte d'abord, triée par masse
row = hdr_row + 1
n_strict_listed = 0
for cfg in sorted_cfgs:
    v5 = v5_data[cfg]
    pdg = v5["pdg"]
    if pdg and "aucune" not in str(pdg): continue
    n_val = _f(v5["n"]); c_val = _f(v5["c"]); m_val = _f(v5["m_omega"])
    Q_intr = v5["Q_intrinsic"]
    if not (n_val and n_val >= 2 and m_val and m_val > 1e3): continue
    if Q_intr is None or abs(Q_intr) >= 0.1: continue
    n_strict_listed += 1
    cells = [n_strict_listed, "⚫ STRICT", cfg, int(n_val), int(c_val),
             round(m_val, 1), round(Q_intr, 3), 0, round(v5["Q_phys_min"], 3)]
    for c, v in enumerate(cells, start=1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.font = BODY; cell.fill = FILL_DM
        cell.alignment = CENTER; cell.border = BORDER
    row += 1

# Séparateur
ws.cell(row=row, column=1, value=f"— {n_strict_listed} candidats STRICTS ; suite : RÉALISABLES —").font = SUBHDR
ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=9)
ws.cell(row=row, column=1).alignment = CENTER
ws.cell(row=row, column=1).fill = FILL_SUB
row += 1

# 5b : DM réalisable
n_real_listed = 0
for cfg in sorted_cfgs:
    v5 = v5_data[cfg]
    pdg = v5["pdg"]
    if pdg and "aucune" not in str(pdg): continue
    n_val = _f(v5["n"]); c_val = _f(v5["c"]); m_val = _f(v5["m_omega"])
    Q_intr = v5["Q_intrinsic"]; Q_min = v5["Q_phys_min"]; Lk_n = v5["Lk_neutral"]
    if not (n_val and n_val >= 2 and m_val and m_val > 1e3): continue
    if Q_intr is None or abs(Q_intr) < 0.1: continue   # déjà compté en strict
    if Q_min is None or abs(Q_min) >= 0.05: continue
    if Lk_n is None or abs(Lk_n) > max(5, c_val): continue
    n_real_listed += 1
    cells = [n_strict_listed + n_real_listed, "⚫ RÉALISABLE", cfg,
             int(n_val), int(c_val), round(m_val, 1),
             round(Q_intr, 3), Lk_n, round(Q_min, 3)]
    for c, v in enumerate(cells, start=1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.font = BODY; cell.fill = FILL_DM_R
        cell.alignment = CENTER; cell.border = BORDER
    row += 1

ws.freeze_panes = f"A{hdr_row + 1}"
print(f"     {n_strict_listed} STRICTS + {n_real_listed} RÉALISABLES = {n_strict_listed+n_real_listed} candidats")


# ─── Sheet 6 : Postulats ───
print("  → Postulats P21-P23")
ws = wb.create_sheet("6. Postulats P21-P23")
ws.column_dimensions['A'].width = 100

content = [
    ("POSTULATS P21-P23 DE LA TQNT-V5", TITLE, FILL_TITLE, CENTER),
    ("", None, None, None),
    ("P21 — FORMULE DE MASSE UNIVERSELLE", HDR, FILL_HDR, LEFT),
    ("Pour tout entrelacs L = {K_1, …, K_n} :", BODY, None, LEFT),
    ("    m(L) = α₀ · Φ_n(L) · Φ_p(L) · ‖μ · τ^q‖_a · ‖υ‖₁^q", SUBHDR, FILL_SUB, LEFT),
    ("où :", BODY, None, LEFT),
    ("   α₀ = 0.51099895 MeV   (calibration : 3_1 ↦ électron)", BODY, None, LEFT),
    ("   Φ_n(L) = ((n+1)/2)^(√6)", BODY, None, LEFT),
    ("   Φ_p(L) = Π_K (ρ₀ · c_K^(−β))^(multiplicité−1) avec ρ₀ = 10−3/(8π), β = 33/20", BODY, None, LEFT),
    ("   μ_K = M_naïf(K)/m_e ;  τ_K = (b_K · det V_K(−1)) / T_{3_1} ;  υ_K = U_κ(K)·U_τ(K)/P_{3_1}", BODY, None, LEFT),
    ("   a = √(7/(8π)), q = 3/4", BODY, None, LEFT),
    ("", None, None, None),
    ("CHARGE PHYSIQUE — Q_phys = Q_topo + Lk", HDR, FILL_HDR, LEFT),
    ("Q_topo(L) = Σ_K Q(K)   où Q(K) = (p_min + p_max)/span(V_K)  (Jones-derived, fractionnaire)", BODY, None, LEFT),
    ("Q(4_1) = Q(6_3) = Q(8_3) = 0   (nœuds amphichiraux)", BODY, None, LEFT),
    ("Q(3_1) = +1.667, Q(5_1) = +1.800, Q(5_2) = +1.400  (asymétriques)", BODY, None, LEFT),
    ("Lk ∈ ℤ : nombre d'enlacement total, ajusté par compression/torsion physique.", BODY, None, LEFT),
    ("Q_phys peut être rendu = 0 ssi Q_topo est entier modulo Lk.", BODY, None, LEFT),
    ("", None, None, None),
    ("P22 — NUCLÉATION AUX FRONTIÈRES", HDR, FILL_HDR, LEFT),
    ("    Γ(r) = Γ₀ · exp(η · ξ),  ξ = |∇Ω²| · λ_C / Ω²", SUBHDR, FILL_SUB, LEFT),
    ("Trois régimes : ξ≪1 → Λ ;  ξ~1 → Schwinger ;  ξ≫1 → Hawking.", BODY, None, LEFT),
    ("", None, None, None),
    ("P23 — PLASTICITÉ TOPOLOGIQUE (nouveau en V5)", HDR, FILL_HDR, LEFT),
    ("    m_mesurée(L) = m(L) · [1 + σ(L) · γ(L) · ε(n)]", SUBHDR, FILL_SUB, LEFT),
    ("σ = +1 pseudoscalaire, −1 sinon ;  γ = C_plast / Σ U_κU_τ ;  ε(n) = 0 si n = 1.", BODY, None, LEFT),
    ("C_plast = −2.92 MeV  (calibration μ⁻).", BODY, None, LEFT),
    ("", None, None, None),
    ("PARAMÈTRES GLOBAUX", HDR, FILL_HDR, LEFT),
    ("  10 paramètres : α₀, γ, p, f(2), f(3), β, α, E₀, η, C_plast", BODY, None, LEFT),
    ("  0 paramètre par particule.", BODY, None, LEFT),
]
for i, (text, font, fill, align) in enumerate(content, start=1):
    cell = ws.cell(row=i, column=1, value=text)
    if font:  cell.font = font
    if fill:  cell.fill = fill
    if align: cell.alignment = align
ws.row_dimensions[1].height = 26


# ─── Sheet 7 : Méthodologie ───
print("  → Méthodologie")
ws = wb.create_sheet("7. Méthodologie")
ws.column_dimensions['A'].width = 35
ws.column_dimensions['B'].width = 75

ws.cell(row=1, column=1, value="LECTURE DES COLONNES DU CATALOGUE").font = TITLE
ws.cell(row=1, column=1).fill = FILL_TITLE; ws.cell(row=1, column=1).alignment = CENTER
ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=2)
ws.row_dimensions[1].height = 26

ws.cell(row=2, column=1, value="Colonne").font = HDR
ws.cell(row=2, column=1).fill = FILL_HDR; ws.cell(row=2, column=1).alignment = CENTER
ws.cell(row=2, column=2, value="Signification").font = HDR
ws.cell(row=2, column=2).fill = FILL_HDR; ws.cell(row=2, column=2).alignment = CENTER

definitions = [
    ("Config TQNT", "Entrelacs « K_1 + … + K_n » (notation Rolfsen)."),
    ("n", "Nombre de composantes."),
    ("c", "Nombre total de croisements (de 3 à 30 dans le catalogue)."),
    ("J spectre", "Spins permis dérivés des braid index."),
    ("Q_topo (Jones)", "Charge intrinsèque Σ Q(K) où Q(K) = (p_min+p_max)/span(V_K), AVANT enlacement."),
    ("Lk neutralisant", "Entier Lk qui minimise |Q_topo + Lk| — Lk de compression à appliquer pour neutralité."),
    ("Q_phys min", "Charge physique résiduelle minimale = Q_topo + Lk_neutralisant. Entière si neutralisable."),
    ("Lk PDG", "Lk effectivement utilisé par l'assignation PDG (proton: +3, neutron: +2, …) ou '—'."),
    ("m(Ω_ts) V5", "Masse prédite par P21 sans plasticité (canonique)."),
    ("P23 correction (%)", "Correction multiplicative de plasticité (σ·γ·ε)."),
    ("m(Ω_ts) plast", "Masse post-altération expérimentale."),
    ("Δ V5 %", "(Ω_ts strict − m_exp) / m_exp en %."),
    ("Δ plast %", "(Ω_ts plast − m_exp) / m_exp en %."),
    ("", ""),
    ("LECTURE RAPIDE", None),
    ("• Vert clair", "Configuration assignée à une particule PDG."),
    ("• Gris", "Candidat matière noire STRICT (Q_topo intrinsèque ≈ 0)."),
    ("• Bleu pâle", "Candidat matière noire RÉALISABLE (Q_phys = 0 atteint par Lk physique)."),
    ("• Jaune (onglet 4)", "Particule dont la plasticité change l'attribution."),
]
for i, (col, defn) in enumerate(definitions, start=3):
    if defn is None:
        cell = ws.cell(row=i, column=1, value=col)
        cell.font = HDR; cell.fill = FILL_HDR; cell.alignment = CENTER
        ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=2)
    else:
        ws.cell(row=i, column=1, value=col).font = SUBHDR
        ws.cell(row=i, column=1).alignment = LEFT
        ws.cell(row=i, column=2, value=defn).font = BODY
        ws.cell(row=i, column=2).alignment = LEFT
    ws.row_dimensions[i].height = 24


# ─── Sheet 1 (rempli en fin de script) ───
print("  → Présentation")
n_libres = len(v5_data) - n_pdg - n_dm_strict - n_dm_realisable
ws_intro.column_dimensions['A'].width = 95

content = [
    ("CATALOGUE TQNT-V5 — Diffusion", TITLE, FILL_TITLE, CENTER),
    ("Théorie Quantique des Nœuds Topologiques (V5, juin 2026)", SUBHDR, FILL_SUB, CENTER),
    ("", None, None, None),
    ("OBJET DU CATALOGUE", HDR, FILL_HDR, LEFT),
    (f"Énumération exhaustive de {len(v5_data)} configurations topologiques (3 ≤ c ≤ {c_max} croisements), "
     "avec masses prédites par la formule universelle Ω_ts et identification des particules PDG candidates.",
     BODY, None, LEFT),
    ("", None, None, None),
    ("FORMULE DE MASSE (postulat P21)", HDR, FILL_HDR, LEFT),
    ("  m(L) = α₀ · Φ_n(L) · Φ_p(L) · ‖μ · τ^(3/4)‖_a · ‖υ‖₁^(3/4)", SUBHDR, FILL_SUB, LEFT),
    ("avec α₀ = 0.511 MeV calibré sur l'électron, et a = √(7/8π), q = 3/4, etc. dérivés "
     "analytiquement du flot 1-boucle de Chern-Simons à SU(2)_8 — aucun paramètre par particule.", BODY, None, LEFT),
    ("", None, None, None),
    ("CORRECTION DE PLASTICITÉ (postulat P23, nouveau en V5)", HDR, FILL_HDR, LEFT),
    ("  m_mesurée(L) = m(L) · [1 + σ(L) · γ(L) · ε(n)]", SUBHDR, FILL_SUB, LEFT),
    ("σ(L) = +1 si L est pseudoscalaire, −1 sinon ; γ(L) = C_plast / Σ U_κU_τ ; ε(n) = 0 si n = 1.",
     BODY, None, LEFT),
    ("C_plast = −2.92 MeV (calibré sur le résidu de μ⁻ à 2.27%).", BODY, None, LEFT),
    ("", None, None, None),
    ("CHARGE PHYSIQUE (deux niveaux)", HDR, FILL_HDR, LEFT),
    ("• Q_topo (Jones) = Σ_K Q(K) — charge fractionnaire intrinsèque, indépendante de l'enlacement.", BODY, None, LEFT),
    ("• Q_phys = Q_topo + Lk — charge physique mesurée après enlacement.", BODY, None, LEFT),
    ("Un entrelacs est NEUTRALISABLE si ∃ Lk ∈ ℤ tel que Q_topo + Lk = 0, soit Q_topo entier.", BODY, None, LEFT),
    ("", None, None, None),
    ("LÉGENDE DES TYPES", HDR, FILL_HDR, LEFT),
    ("🟢🟡🟠🔴 = fiabilité de la mesure PDG", BODY, None, LEFT),
    ("⚫ STRICT     = matière noire à Q_topo intrinsèquement ≈ 0 (amphichiraux purs)", BODY, None, LEFT),
    ("⚫ RÉALISABLE = matière noire où Q_phys = 0 est atteignable par Lk entier physique", BODY, None, LEFT),
    ("⚪ libre      = configuration sans identification PDG actuelle (prédiction libre)", BODY, None, LEFT),
    ("", None, None, None),
    ("INVENTAIRE", HDR, FILL_HDR, LEFT),
    (f"  Configurations totales            : {len(v5_data)}", BODY, None, LEFT),
    (f"  c_max effectif du catalogue       : {c_max}", BODY, None, LEFT),
    (f"  Configurations PDG-assignées      : {n_pdg}  (≈ 122 particules PDG distinctes, multi-isomérie)",
     BODY, None, LEFT),
    (f"  Matière noire STRICTE (Q_topo≈0)  : {n_dm_strict}  (dont {n_dm_strict_high_c} à c ≥ 18)", BODY, None, LEFT),
    (f"  Matière noire RÉALISABLE par Lk   : {n_dm_realisable}  (dont {n_dm_real_high_c} à c ≥ 18)",
     BODY, None, LEFT),
    (f"  Prédictions libres (Q non neutr.) : {n_libres}", BODY, None, LEFT),
    (f"  Réassignations P23 vs V5 strict   : {n_reassigned}", BODY, None, LEFT),
    ("", None, None, None),
    ("VALIDATION STATISTIQUE", HDR, FILL_HDR, LEFT),
    ("  Ω_ts strict, N=95 PDG          : 0.767% RMS (validation croisée 5-fold)", BODY, None, LEFT),
    ("   + Kernel Ridge sur 5 invariants topologiques : 0.670%", BODY, None, LEFT),
    ("   + ξ_BS (signature de plasticité Biot-Savart) : 0.658%", BODY, None, LEFT),
    ("", None, None, None),
    ("ONGLETS", HDR, FILL_HDR, LEFT),
    ("  1. Présentation        (ce document)", BODY, None, LEFT),
    (f"  2. Catalogue unifié    — {len(v5_data)} configs avec strict, plasticité et neutralisation Lk",
     BODY, None, LEFT),
    ("  3. Particules PDG      — synthèse des assignations validées", BODY, None, LEFT),
    ("  4. Réassignations P23  — particules dont l'attribution change avec la plasticité",
     BODY, None, LEFT),
    (f"  5. Matière noire       — {n_dm_strict} stricts + {n_dm_realisable} réalisables",
     BODY, None, LEFT),
    ("  6. Postulats P21-P23   — formulation analytique", BODY, None, LEFT),
    ("  7. Méthodologie        — lecture des colonnes", BODY, None, LEFT),
    ("", None, None, None),
    ("RÉFÉRENCE", HDR, FILL_HDR, LEFT),
    ("Pour la dérivation complète des postulats, voir TQNT_detaillee_V5_FR.tex (230 pages).",
     NOTE, None, LEFT),
]
for i, (text, font, fill, align) in enumerate(content, start=1):
    cell = ws_intro.cell(row=i, column=1, value=text)
    if font:  cell.font = font
    if fill:  cell.fill = fill
    if align: cell.alignment = align
ws_intro.row_dimensions[1].height = 28

# Réordonner les onglets
wb.move_sheet("1. Présentation", offset=-len(wb.sheetnames) + 1)

# ─── Sauvegarde ───
print(f"\nSauvegarde → {DST}")
wb.save(DST)
import os
print(f"  Taille : {os.path.getsize(DST) // 1024} ko")
print(f"  Onglets : {len(wb.sheetnames)}")
for s in wb.sheetnames: print(f"    • {s}")
print("\n✓ Catalogue de diffusion v2 généré.")
