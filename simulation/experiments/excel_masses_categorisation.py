r"""
EXPORT EXCEL — Masses des nœuds/entrelacs, rapprochement aux particules,
catégorisation TQNT (taxonomie de la structure des composites, §B.10/B.11 V6).

Source de masse UNIQUE et cohérente : omega_unified.mass_unified (1 paramètre
calibré x, secteur mono-nœud dérivé). Ceci évite le mélange des ≥3
implémentations de masse du dépôt (omega_ts_lookup, omega_ts_axiomatic,
excel_spectrum omega X=-sqrt6).

Onglets :
  1. Nœuds simples     — masse + invariants + rôle taxonomique
  2. Spectre & catégo. — particule ↔ assignation ↔ catégorie TQNT ↔ PDG
  3. Vérification V6    — divergences ch15 (catalogue) vs §B.10 sous omega_unified
  4. Matière noire      — candidats isolés c>18 (rappel)

Sortie : docs/TQNT_masses_categorisation.xlsx
"""
from __future__ import annotations
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from simulation.experiments.omega_unified import mass_unified
from theory.omega_ts import _axiom_data

# ───────────────────────── styles ─────────────────────────
HDR = PatternFill("solid", fgColor="1F3864")
HDRF = Font(bold=True, color="FFFFFF", size=10)
CAT = PatternFill("solid", fgColor="D6DCE4")
CATF = Font(bold=True, color="1F3864")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CEN = Alignment(horizontal="center", vertical="center")
LEF = Alignment(horizontal="left", vertical="center")

def dev_fill(pct):
    a = abs(pct)
    if a < 1:  return PatternFill("solid", fgColor="C6EFCE")
    if a < 5:  return PatternFill("solid", fgColor="FFEB9C")
    if a < 20: return PatternFill("solid", fgColor="FFD966")
    return PatternFill("solid", fgColor="F8CBAD")

# ───────────────────────── catalogue de nœuds ─────────────────────────
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
AMPHI = {"4_1","6_3","8_3"}

def role(K):
    d = _axiom_data(K); c = d["cr"]
    if K == "3_1": return "cœur lepton (3₁)"
    if K == "5_2": return "cœur baryon décuplet (5₂)"
    if K == "7_2": return "cœur baryon octet (7₂)"
    if K == "8_3": return "cœur boson EW (8₃)"
    if K in AMPHI: return "amphichiral (auto-conjugué)"
    return "composant méson / partenaire"

# ───────────────────── particules : assignation + catégorie ─────────────────────
# cat : catégorie taxonomique ; sub : sous-catégorie ; cfg : assignation §B.10/catalogue ;
# ch15 : assignation du tableau ch15 (si différente) ; pdg : masse PDG (MeV)
SANS_NOEUD = [
    ("γ (photon)", "Sans nœud (c=0)", "quantum de jauge χ", "—", None, 0.0),
    ("ν (neutrinos)", "Sans nœud (c=0)", "onde χ auto-rotative", "—", None, 6.6e-5),
]
LEPTONS = [
    ("e⁻", "Lepton (cœur 3₁)", "mode Z₃ k=0", "3_1", None, 0.511),
    ("μ⁻", "Lepton (cœur 3₁)", "mode Z₃ k=1", "3_1+6_1", None, 105.66),
    ("τ⁻", "Lepton (cœur 3₁)", "mode Z₃ k=2", "3_1+5_1+7_1", None, 1776.9),
]
# Mésons — assignation §B.10 (cfg) vs ch15 (ch15)
MESONS = [
    ("π⁰", "Méson diagonal (K+K̄)", "pseudoscalaire 0⁻⁺", "8_1", "8_1", 134.98),
    ("η",  "Méson diagonal (K+K̄)", "pseudoscalaire 0⁻⁺", "3_1+4_1+5_1", "3_1+4_1+5_1", 547.9),
    ("η′", "Méson diagonal (K+K̄)", "pseudoscalaire 0⁻⁺ (masse ← χ_top, Witten-Veneziano)", "5_2+5_2+6_1", "5_2+5_2+6_1", 957.8),
    ("ρ⁰", "Méson diagonal (K+K̄)", "vecteur 1⁻⁻ (bouclé, large)", "6_2+6_2", "5_2+7_7", 775.3),
    ("ω",  "Méson diagonal (K+K̄)", "vecteur 1⁻⁻ (torsadé, étroit)", "8_1+8_1", "5_2+7_7", 782.7),
    ("φ",  "Méson diagonal (K+K̄)", "vecteur 1⁻⁻ (ss̄)", "7_5+7_5", "3_1+4_1+6_2", 1019.5),
    ("J/ψ","Méson diagonal (K+K̄)", "vecteur 1⁻⁻ (cc̄, tour S+P 1.5%)", "7_4+7_4", "4_1+5_2+9_1", 3096.9),
    ("Υ",  "Méson diagonal (K+K̄)", "vecteur 1⁻⁻ (bb̄, triplet auto-conj.)", "7_6+7_6+4_1", "3_1+7_4+9_1", 9460.3),
    ("K*", "Méson diagonal-ish (vecteur)", "vecteur 1⁻⁻ (tour Salpeter)", "5_2+7_5", None, 891.7),
    ("π±", "Méson hors-diagonal (K+K̄′)", "pseudoscalaire 0⁻⁺ (Goldstone)", "8_1", "8_1", 139.57),
    ("K±", "Méson hors-diagonal (K+K̄′)", "pseudoscalaire 0⁻⁺ (Goldstone, ne fitte pas Ω)", "4_1+4_1+5_2", "4_1+4_1+5_2", 493.7),
    ("D", "Méson hors-diagonal (K+K̄′)", "pseudoscalaire (saveuré, SOUS-DÉTERMINÉ)", "3_1+5_1+7_2", None, 1864.8),
    ("B", "Méson hors-diagonal (K+K̄′)", "pseudoscalaire (saveuré, SOUS-DÉTERMINÉ)", "8_1+8_2", None, 5279.3),
]
BARYONS_OCT = [
    ("p", "Baryon octet (cœur 7₂)", "spin ½, Lk=+3", "3_1+7_2+7_2", None, 938.27),
    ("n", "Baryon octet (cœur 7₂)", "spin ½, Lk=+2", "3_1+7_2+7_2", None, 939.57),
    ("Λ", "Baryon octet (cœur 7₂)", "spin ½ (re-dérivé §rectif)", "3_1+5_2+7_2", "4_1+7_4", 1115.7),
    ("Σ", "Baryon octet (cœur 7₂)", "spin ½ (re-dérivé §rectif)", "5_2+7_2+7_2", "6_2+7_2", 1193.2),
    ("Ξ", "Baryon octet (cœur 7₂)", "spin ½ (re-dérivé, cœur 7₂ respecté)", "3_1+3_1+7_2", "4_1+4_1+7_2", 1318.3),
]
BARYONS_DEC = [
    ("Δ",  "Baryon décuplet (cœur 5₂)", "spin 3/2 (re-dérivé)", "5_2+6_3", None, 1232.0),
    ("Σ*", "Baryon décuplet (cœur 5₂)", "spin 3/2 (re-dérivé)", "5_1+5_1+5_2", None, 1384.6),
    ("Ξ*", "Baryon décuplet (cœur 5₂)", "spin 3/2 (re-dérivé)", "4_1+5_2+7_2", None, 1533.4),
    ("Ω⁻", "Baryon décuplet — MUR", "spin 3/2 ; apex stable ΔS=1 (physique) mais assignation NON résolue : nœud hors-cat. RÉFUTÉ (9-10 crois. dépassent ×5)", "aucun (mur réel)", None, 1672.5),
]
EW = [
    ("W",   "Boson EW (cœur 8₃)", "spin 1, f=1", "7_1+7_2+8_3", None, 80377.0),
    ("Z",   "Boson EW (cœur 8₃)", "spin 1, f=1", "7_2+8_3+9_1", None, 91188.0),
    ("H",   "Boson EW (cœur 8₃)", "spin 0, f=1", "7_1+8_3+10_1", None, 125250.0),
    ("top", "Boson EW (cœur 8₃)", "spin ½, f=1", "8_2+8_3+10_1", None, 172570.0),
]
ALL = SANS_NOEUD + LEPTONS + MESONS + BARYONS_OCT + BARYONS_DEC + EW

def omega(cfg):
    if cfg in ("—", None) or "nœud" in cfg or "onde" in cfg:
        return None
    try: return mass_unified(cfg)
    except Exception: return None

# ───────────────────────── construction du classeur ─────────────────────────
wb = Workbook()

# --- Onglet 1 : Nœuds simples ---
ws = wb.active; ws.title = "Nœuds simples"
h1 = ["Nœud","cr","n_ℓ","b","det","U_κ","U_τ","amphichiral","masse Ω (MeV)","rôle taxonomique"]
for j,t in enumerate(h1,1):
    c=ws.cell(1,j,t); c.fill=HDR; c.font=HDRF; c.alignment=CEN; c.border=BOX
for i,K in enumerate(KNOTS,2):
    d=_axiom_data(K); m=mass_unified(K)
    row=[K.replace("_","_"), d["cr"], d["nl"], d["b"], d["det"],
         round(d["Uk"],2), round(d["Ut"],2), "oui" if K in AMPHI else "—",
         round(m,2), role(K)]
    for j,v in enumerate(row,1):
        c=ws.cell(i,j,v); c.border=BOX; c.alignment=LEF if j==10 else CEN
for j,w in enumerate([7,5,5,5,6,8,8,11,13,30],1):
    ws.column_dimensions[get_column_letter(j)].width=w
ws.freeze_panes="A2"

# particules dont l'assignation est DÉRIVÉE cette session (cohérente sous Ω_unifié) ;
# les autres sont des assignations catalogue ch15 (calibrées avec une AUTRE formule)
DERIVED = {"e⁻","ρ⁰","ω","φ","J/ψ","Υ","K*","W","Z","H","top","p","n",
           "Λ","Σ","Ξ","Δ","Σ*","Ξ*"}
RESIST = {"Ω⁻"}
GOLD = {"π⁰","π±","η","η′","K±"}
UNDER = {"D","B"}
def source(name):
    if name in DERIVED: return "re-dérivé Ω-cohérent"
    if name in RESIST:  return "apex stable, nœud protégé hors catalogue"
    if name in GOLD:    return "Goldstone chiral (hors Ω)"
    if name in UNDER:   return "sous-déterminé (multi-cand.)"
    return "catalogue (à revoir)"

# --- Onglet 2 : Spectre & catégorisation ---
ws2 = wb.create_sheet("Spectre & catégorisation")
h2 = ["Particule","Catégorie TQNT","Sous-catégorie / indice","Assignation",
      "masse Ω (MeV)","masse PDG (MeV)","Δ %","Source assignation","Assign. ch15 (si ≠)","Δ ch15 % (sous Ω)"]
for j,t in enumerate(h2,1):
    c=ws2.cell(1,j,t); c.fill=HDR; c.font=HDRF; c.alignment=CEN; c.border=BOX
r=2; cur=None
for name,cat,sub,cfg,ch15,pdg in ALL:
    if cat!=cur:
        cur=cat
        cc=ws2.cell(r,1,cat); cc.fill=CAT; cc.font=CATF
        ws2.merge_cells(start_row=r,start_column=1,end_row=r,end_column=10)
        for j in range(1,11): ws2.cell(r,j).border=BOX
        r+=1
    m=omega(cfg)
    dv=100*(m-pdg)/pdg if (m and pdg) else None
    src = source(name) if m else "—"
    row=[name,cat,sub,cfg,
         round(m,2) if m else "—", round(pdg,3) if pdg else "—",
         round(dv,1) if dv is not None else "—", src, "", ""]
    if ch15 and ch15!=cfg:
        m15=omega(ch15); d15=100*(m15-pdg)/pdg if (m15 and pdg) else None
        row[8]=ch15
        row[9]=round(d15,1) if d15 is not None else "—"
    for j,v in enumerate(row,1):
        c=ws2.cell(r,j,v); c.border=BOX; c.alignment=LEF if j in (1,2,3,4,8,9) else CEN
    if dv is not None: ws2.cell(r,7).fill=dev_fill(dv)
    if row[9] not in ("","—") and isinstance(row[9],(int,float)): ws2.cell(r,10).fill=dev_fill(row[9])
    r+=1
for j,w in enumerate([11,26,34,16,13,14,7,26,17,14],1):
    ws2.column_dimensions[get_column_letter(j)].width=w
ws2.freeze_panes="A2"

# --- Onglet 3 : Vérification V6 (divergences mésons) ---
ws3 = wb.create_sheet("Vérification V6")
note=("VÉRIFICATION — Sous la formule UNIQUE omega_unified, les assignations mésoniques "
      "du tableau ch15 (calibrées avec une autre formule) sont incohérentes ; "
      "celles de §B.10/B.11 (cette session) sont cohérentes. Rectification recommandée : "
      "aligner le tableau ch15 sur §B.10 (ou documenter les deux formules).")
ws3.cell(1,1,note).alignment=Alignment(wrap_text=True,vertical="top")
ws3.merge_cells("A1:G3")
h3=["Particule","PDG","Assign. ch15","Ω(ch15)","Δ ch15 %","Assign. §B.10","Δ §B.10 %"]
for j,t in enumerate(h3,1):
    c=ws3.cell(5,j,t); c.fill=HDR; c.font=HDRF; c.alignment=CEN; c.border=BOX
r=6
for name,cat,sub,cfg,ch15,pdg in MESONS:
    if not ch15 or ch15==cfg: continue
    m15=omega(ch15); mB=omega(cfg)
    d15=100*(m15-pdg)/pdg if m15 else None
    dB=100*(mB-pdg)/pdg if mB else None
    row=[name,round(pdg,1),ch15,round(m15,1) if m15 else "—",
         round(d15,1) if d15 is not None else "—",
         cfg,round(dB,1) if dB is not None else "—"]
    for j,v in enumerate(row,1):
        c=ws3.cell(r,j,v); c.border=BOX; c.alignment=CEN
    if d15 is not None: ws3.cell(r,5).fill=dev_fill(d15)
    if dB is not None: ws3.cell(r,7).fill=dev_fill(dB)
    r+=1
for j,w in enumerate([10,8,16,10,10,16,10],1):
    ws3.column_dimensions[get_column_letter(j)].width=w

# --- Onglet 4 : Matière noire (rappel taxonomie) ---
ws4 = wb.create_sheet("Matière noire (c>18)")
ws4.cell(1,1,"Secteur sombre : réservoir dominant = nœuds simples ISOLÉS c>18 "
             "(T(2,q≥19), T(3,q≥10)), stables (barrière CS k=135, τ>10³⁴⁷ s), "
             "invisibles (Vassiliev σ~α^(c-18)). Entrelacs sombres = secondaires "
             "(6 amphichiraux stricts) ou instables.").alignment=Alignment(wrap_text=True,vertical="top")
ws4.merge_cells("A1:E4")
h4=["Candidat (6 stricts)","type","masse (GeV)","catégorie","stabilité"]
for j,t in enumerate(h4,1):
    c=ws4.cell(6,j,t); c.fill=HDR; c.font=HDRF; c.alignment=CEN; c.border=BOX
DM=[("6₃+6₃","entrelacs amphichiral",1.55,"MN secondaire","stable"),
    ("6₃+6₃+6₃","entrelacs amphichiral",4.70,"MN secondaire","stable"),
    ("4₁+8₃","entrelacs amphichiral",16.01,"MN secondaire","stable"),
    ("8₃+8₃","entrelacs amphichiral",17.24,"MN secondaire","stable"),
    ("6₃+8₃","entrelacs amphichiral",26.48,"MN secondaire","stable"),
    ("4₁+4₁+8₃","entrelacs amphichiral",51.78,"MN secondaire","stable")]
for i,rowd in enumerate(DM,7):
    for j,v in enumerate(rowd,1):
        c=ws4.cell(i,j,v); c.border=BOX; c.alignment=CEN
for j,w in enumerate([18,22,12,16,10],1):
    ws4.column_dimensions[get_column_letter(j)].width=w

out="docs/TQNT_masses_categorisation.xlsx"
wb.save(out)
print("Écrit :", out)
print("\n=== VÉRIFICATION : mésons ch15 vs §B.10 sous omega_unified ===")
for name,cat,sub,cfg,ch15,pdg in MESONS:
    if ch15 and ch15!=cfg:
        m15=omega(ch15); mB=omega(cfg)
        print(f"  {name:4s} PDG {pdg:7.1f} | ch15 {ch15:12s} Ω={m15:7.0f} ({100*(m15-pdg)/pdg:+5.0f}%) "
              f"| §B.10 {cfg:10s} Ω={mB:7.0f} ({100*(mB-pdg)/pdg:+4.0f}%)")
