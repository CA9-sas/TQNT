"""Catalogue TQNT-V6 : masses PAR SECTEUR, catégorisation canonique (16/09/2026).

Remplace docs/TQNT_catalog_V6_complete.xlsx (construit le 809d14d sur les masses
Ω_ts V5 — barème archivé) par un catalogue où chaque configuration reçoit :
  - sa CATÉGORIE V6 (taxonomie chap:v6-composites + théorème du désert) ;
  - la masse de la LOI DE SON SECTEUR (m = √σ_χ·𝒮), jamais Ω_ts :
      base           : 𝒮 = 0 (γ, ν) ;
      lepton         : 3₁ modes Z₃ (Koide) ;
      nœud libre n=1 : M_topo = m_ℓ·ℓ_ACPR(K)/ℓ(3₁) (état libre, §noeuds-libres) ;
      éremnion       : nœud premier ISOLÉ c ≥ 23, loi géométrique (réservoir) ;
      strict sombre  : 6 entrelacs amphichiraux Q=0 exact (estimation ΣM_topo) ;
      réalisable     : Q_phys = 0 via Lk entier — transitoire/contextuel P22 ;
      composite assigné : catalogue σ_χ (GMOR / Regge octet-décuplet / constituant /
                          Cornell / EW-VEV) ou moteur aligné (spectre étendu) ;
      désert         : n ≥ 2 sans protection — cascade P15 ~1e-18 s, PAS de masse
                       liée ; on donne la borne déconfinée ΣM_topo et l'ordre de la
                       fonctionnelle maîtresse (√((Σm)² + πσ_χ n_½)).
  - la durée de vie de sa classe (stable / Fermi-Sargent=PDG / dénouage P15 /
    cascade P15 / barrière CS).
Correction majeure : l'ancienne table étiquetait « DM : stable τ>1e347 » 139
entrelacs n ≥ 2 à Q_phys = 0. Le réservoir est fait de nœuds premiers ISOLÉS
(n = 1, c ≥ 23) — absents de la combinatoire ≤ 10₁ ; seuls les 6 stricts sont une
sous-population stable secondaire ; les 133 autres sont des réalisables.
Sortie : docs/TQNT_catalog_V6_complete.xlsx (+ résumé console).
"""
import sys, math, csv, re, collections
sys.path.insert(0, '.')
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from theory.ropelength_acpr import ROPELENGTH_ACPR, ML

SQS = 418.47                      # MeV, √σ_χ
PI_SIGMA = math.pi * SQS ** 2     # MeV², demi-tour du ruban
HBAR = 6.582119569e-22            # MeV s
V_EW = 246220.0                   # MeV
L31 = ROPELENGTH_ACPR["3_1"]
ROPE_HIST = {"10_1": 2 * 42.33}   # table historique (L/D → L/R) — borne non ACPR
AMPHI = {"4_1", "6_3", "8_3"}
TORIQUE = {"3_1", "5_1", "7_1", "9_1"}
U_DENOUAGE = {"4_1": 1, "5_1": 2, "5_2": 1, "6_1": 1, "6_2": 1, "6_3": 1, "7_1": 3,
              "7_2": 1, "7_3": 2, "7_4": 2, "7_5": 2, "7_6": 1, "7_7": 1, "8_1": 1,
              "8_2": 2, "8_3": 2, "9_1": 4, "10_1": 1}
KOIDE = {"e⁻": 0.511, "μ⁻": 105.66, "τ⁻": 1776.9}

from theory.jones import braid_index
def role_noeud(K):
    """Rôle taxonomique V6 du nœud premier (chap:v6-composites)."""
    if K == "3_1": return "cœur LEPTON (modes Z₃ e/μ/τ) ; cœur fermionique du nucléon"
    if K == "5_2": return "cœur BARYON décuplet (nœud étrange, m_s)"
    if K == "7_2": return "compagnon du nucléon (paire miroir 7₂+7̄₂) ; cœur BARYON octet"
    if K == "8_3": return "cœur ÉLECTROFAIBLE (commun aux 4 asthénions) ; amphichiral, brique sombre stricte"
    if K in AMPHI: return "amphichiral (auto-conjugué) — brique sombre stricte"
    if K in TORIQUE: return "torique T(2,q) — composante d'habillage / EW ; loi d'état libre ouverte"
    return "stoikhion (composant de méson / compagnon)"

def structure(cs):
    """Structure de l'entrelacs : cœurs présents, paire miroir, contenu."""
    tags = []
    cnt = collections.Counter(cs)
    if any(v >= 2 for v in cnt.values()): tags.append("paire K+K (diagonale possible K+K̄)")
    for core, nom in (("3_1", "cœur 3₁"), ("5_2", "cœur 5₂"), ("7_2", "cœur 7₂"), ("8_3", "cœur 8₃")):
        if core in cs: tags.append(nom)
    if all(k in AMPHI for k in cs): tags.append("tout amphichiral")
    if all(k in TORIQUE for k in cs): tags.append("tout torique")
    b = [braid_index(k) for k in cs]
    nferm = sum(1 for x in b if x % 2 == 0)
    tags.append(f"tresses {'+'.join(map(str, b))} → {nferm} fermion(s)")
    return " ; ".join(tags)

def rope(K):
    if K in ROPELENGTH_ACPR: return ROPELENGTH_ACPR[K], "ACPR"
    return ROPE_HIST[K], "historique"

def m_topo(K):
    r, src = rope(K); return ML * r / L31, src

def comps(cfg): return [c.strip() for c in cfg.split('+')]
def key(cfg): return "+".join(sorted(comps(cfg)))
def cross(K): return int(K.split('_')[0])

def tau_denouage(K):
    m, _ = m_topo(K)
    return U_DENOUAGE[K] * 384 * math.pi ** 3 * V_EW ** 4 * HBAR / m ** 5

# --- réservoir : loi géométrique des toriques T(2,q) (ACPR + point mesuré T(2,23)) ---
tor = [(3, 32.7448), (5, 47.2016), (7, 61.4109), (9, 75.7252), (23, 180.42)]
n_ = len(tor); sx = sum(q for q, _ in tor); sy = sum(l for _, l in tor)
sxx = sum(q * q for q, _ in tor); sxy = sum(q * l for q, l in tor)
SLOPE = (n_ * sxy - sx * sy) / (n_ * sxx - sx * sx); INTER = (sy - SLOPE * sx) / n_
def rope_T2(q): return INTER + SLOPE * q

# --- catalogue σ_χ (canonique 07/07) ---
CSV = collections.defaultdict(list)
with open('docs/catalogue_sigma_chi.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        cfg = r['config_spin_charge']
        if not re.match(r'^[0-9]', cfg): continue
        CSV[key(cfg)].append(r)
SECTEUR = {"koide": "LEPTON (Koide)", "GMOR": "GOLDSTONE (GMOR)", "GMO": "GOLDSTONE (GMO)",
           "anomalie_U1A": "GOLDSTONE (anomalie U(1)_A)", "octet": "BARYON octet (Regge)",
           "decuplet": "BARYON décuplet (Regge)", "EW_VEV": "ÉLECTROFAIBLE (VEV)"}
def secteur(meth):
    if meth in SECTEUR: return SECTEUR[meth]
    if meth.startswith("Regge+Delta") : return "BARYON étrange (Regge+Δ_s)"
    if meth.startswith("Regge"): return "MÉSON vecteur (Regge)"
    if "Cornell" in meth: return "LOURD quarkonium (Cornell)"
    if "constituant" in meth: return "LOURD (stoikhion + torsion)"
    return meth

# --- lecture de la table 1539 canonique ---
ws = openpyxl.load_workbook('docs/TQNT_catalog_complete.xlsx', read_only=True, data_only=True)['Entrelacs complets (1539)']
rows = []
for i, r in enumerate(ws.iter_rows(values_only=True)):
    if i < 2 or r[0] is None: continue
    rows.append(r)

def f(x):
    try: return float(str(x).replace(',', '.'))
    except Exception: return None

REAL117 = {}
for i, r in enumerate(openpyxl.load_workbook('docs/TQNT_catalog_complete.xlsx', read_only=True, data_only=True)['117 candidats DM'].iter_rows(values_only=True)):
    if i < 2 or r[0] is None or not isinstance(r[0], (int, float)): continue
    REAL117[key(r[2])] = (r[1], r[10])   # niveau, sous-population
STRICTS = {key(c) for c in ["4_1+8_3", "6_3+6_3", "6_3+8_3", "8_3+8_3", "4_1+4_1+8_3", "6_3+6_3+6_3"]}

out = []
stats = collections.Counter()
for r in rows:
    num, cfg, n, c, J, Qt, Lkn, Qp, LkP, _sum, mV6, cls, tau, pdg, mexp, note = r
    cs = comps(cfg)
    msum = sum(m_topo(k)[0] for k in cs)
    src = "ACPR" if all(m_topo(k)[1] == "ACPR" for k in cs) else "ACPR + 10₁ historique"
    k = key(cfg)
    entry = dict(structure=structure(cs) if n > 1 else role_noeud(cs[0]), roles=" | ".join(role_noeud(k).split(" ;")[0] for k in cs) if n > 1 else "", num=num, cfg=cfg, n=n, c=c, J=J, Qt=Qt, Lkn=Lkn, Qp=Qp, LkP=LkP, msum=round(msum, 1),
                 src=src, pdg=pdg, mexp=mexp, note=note or "")
    if n == 1:
        K = cs[0]
        if K == "3_1":
            entry.update(cat="LEPTON", sect="LEPTON (Koide, modes Z₃ du trèfle)", m=0.511,
                         loi="√m_k = μ(1+√2cos(2πk/3+δ)) ; e 0,511 / μ 105,66 / τ 1776,9", statut="dérivé (≤0,005 %)",
                         mfunc=None, tcls="stable (plancher leptonique)", tau="stable", pdg="e⁻ (μ⁻, τ⁻ = modes k=1,2)")
        else:
            m, s = m_topo(K)
            tor_note = " ; torique : échelle seule (loi d'état libre ouverte)" if K in TORIQUE else ""
            entry.update(cat="NŒUD LIBRE (n=1)", sect="État libre — cœur réalisable P22" + (" ; amphichiral (brique sombre stricte)" if K in AMPHI else ""),
                         m=round(m, 1), loi=f"M_topo = m_ℓ·ℓ(K)/ℓ(3₁), ℓ {s}" + tor_note,
                         statut="géométrique (loi du plateau)" if s == "ACPR" else "géométrique (ℓ historique, borne)",
                         mfunc=None, tcls="dénouage faible P15 (u=%d)" % U_DENOUAGE[K], tau=f"{tau_denouage(K):.1e}", pdg="(aucune — produit par paires seulement)")
    elif k in CSV:
        recs = CSV[k]
        names = ", ".join(x['particule'] for x in recs)
        m_th = f(recs[0]['m_theo_MeV']); meth = recs[0]['methode']
        entry.update(cat="COMPOSITE ASSIGNÉ", sect=secteur(meth), m=m_th, loi=f"m = √σ_χ·𝒮 — {meth} (catalogue σ_χ 07/07)",
                     statut="loi de secteur", mfunc=None,
                     tcls="Fermi/Sargent (Q=Δm, g_topo=1) — structure dérivée, τ = PDG" if names.split(',')[0].strip() not in ("p", "n") else "stable (p : plancher baryonique ; n : Fermi Q=0,78 MeV)",
                     tau=tau if tau else "PDG", pdg=names, mexp=", ".join(x['m_exp_MeV'] for x in recs))
    elif k in STRICTS:
        entry.update(cat="SOMBRE STRICT", sect="Entrelacs amphichiral, Q_topo = 0 exact — sous-population stable secondaire",
                     m=round(msum, 1), loi="ΣM_topo (loi de secteur sombre non établie — estimation)", statut="estimation",
                     mfunc=None, tcls="stable (Q=0 exact, protégé) ; raie γ fine si annihilation", tau="stable", pdg="—")
    elif k in REAL117 or (f(Qp) is not None and abs(f(Qp)) < 1e-9):
        sp = REAL117.get(k, ("hors table 117", "Q_phys=0 exact"))[1]
        entry.update(cat="RÉALISABLE (Q_phys=0 via Lk)", sect=f"Neutralisé par enlacement — transitoire / contextuel P22 (table des 117 : {sp})",
                     m=None, loi="pas de masse liée (n ≥ 2 non protégé)", statut="—",
                     mfunc=round(math.sqrt(msum ** 2 + PI_SIGMA * (1 if n == 3 else 0)), 1),
                     tcls="cascade P15 (~1e-18 s, théorème du désert)", tau="~1e-18", pdg="—",
                     note=((note + " ; ") if note else "") + (f"assign. historique V5 : {pdg} (m V6 {mV6}) — supplantée : Q_phys=0 → réalisable (table des 117)" if pdg else ""))
    elif mV6 not in (None, "") and pdg not in (None, ""):
        entry.update(cat="COMPOSITE ASSIGNÉ (spectre étendu)", sect="Moteur aligné 21-22/07 (correspondance ; assignation historique, multiple)", m=f(mV6),
                     loi="fonctionnelle maîtresse / bandes (moteur 224)", statut="loi de secteur (assign. historique V5)",
                     mfunc=None, tcls="Fermi/Sargent — τ = PDG", tau=tau if tau else "PDG")
    else:
        entry.update(cat="DÉSERT (transitoire)", sect="n ≥ 2 sans protection", m=None,
                     loi="pas de masse liée — cascade P15", statut="—",
                     mfunc=round(math.sqrt(msum ** 2 + PI_SIGMA * (1 if n == 3 else 0)), 1),
                     tcls="cascade P15 (~1e-18 s, théorème du désert)", tau="~1e-18", pdg="—")
    stats[entry["cat"]] += 1
    out.append(entry)

# --- éremnions : réservoir n=1, c ≥ 23 (absent de la combinatoire ≤ 10₁) ---
erem = []
for q in range(23, 32, 2):
    l = rope_T2(q); m = ML * l / L31
    erem.append(dict(cfg=f"T(2,{q})", c=q, m=round(m), l=round(l, 1)))

# ============================ EXCEL ============================
wb = openpyxl.Workbook()
H = Font(bold=True, color="FFFFFF"); HF = PatternFill("solid", fgColor="1F3864")
FILL = {"LEPTON": "DDEBF7", "NŒUD LIBRE (n=1)": "E2EFDA", "COMPOSITE ASSIGNÉ": "FFF2CC",
        "COMPOSITE ASSIGNÉ (spectre étendu)": "FFF9E6", "SOMBRE STRICT": "D9D2E9",
        "RÉALISABLE (Q_phys=0 via Lk)": "EDEDED", "DÉSERT (transitoire)": "F8F8F8"}
hdr_n = ["#", "Nœud", "c", "ℓ (L/R)", "source ℓ", "M_topo = m libre (MeV)", "indice de tresse b", "parité (b pair = fermion)",
         "amphichiral", "torique", "u (dénouage)", "τ libre (s)", "Rôle taxonomique V6", "Catégorie V6", "Loi de masse", "Statut", "Particule", "Note"]
w = wb.active; w.title = "Nœuds premiers (n=1)"
w.append(hdr_n)
for cell in w[1]: cell.font = H; cell.fill = HF; cell.alignment = Alignment(wrap_text=True, vertical="center")
for e in sorted([e for e in out if e["n"] == 1], key=lambda e: (e["c"], e["cfg"])):
    K = e["cfg"]; r_, src_ = rope(K); b = braid_index(K)
    w.append([e["num"], K, e["c"], round(r_, 4), src_, e["m"], b, "fermion" if b % 2 == 0 else "boson",
              "oui" if K in AMPHI else "non", "oui" if K in TORIQUE else "non", U_DENOUAGE.get(K, "—"), e["tau"],
              role_noeud(K), e["cat"], e["loi"], e["statut"], e["pdg"], e["note"]])
    for cell in w[w.max_row]: cell.fill = PatternFill("solid", fgColor=FILL.get(e["cat"], "FFFFFF"))
w.append([]); w.append(["Réservoir sombre (n=1, c ≥ 23) : voir l'onglet Éremnions — hors combinatoire ≤ 10₁."])
for col, wd in zip("ABCDEFGHIJKLMNOPQR", [5, 8, 4, 10, 11, 14, 8, 12, 8, 8, 8, 10, 60, 22, 46, 26, 30, 20]): w.column_dimensions[col].width = wd
w.freeze_panes = "C2"

hdr_e = ["#", "Config", "n", "c", "Catégorie V6", "Secteur / rôle", "Structure (cœurs, paire miroir, tresses)", "Rôles des composantes",
         "J spectre", "Q_topo", "Lk neutr.", "Q_phys", "m V6 (MeV) — loi de secteur", "Loi de masse", "Statut",
         "ΣM_topo déconfiné (MeV)", "source ℓ", "m fonctionnelle (ordre, non canonique)", "E_liaison = ΣM_topo − m (MeV)",
         "τ classe V6", "τ V6 (s)", "Particule (canon σ_χ)", "m_exp (MeV)", "Note"]
ORDER = ["COMPOSITE ASSIGNÉ", "COMPOSITE ASSIGNÉ (spectre étendu)", "SOMBRE STRICT", "RÉALISABLE (Q_phys=0 via Lk)", "DÉSERT (transitoire)"]
for nn, title in ((2, "Entrelacs n=2 (paires)"), (3, "Entrelacs n=3 (triplets)")):
    w = wb.create_sheet(title)
    w.append(hdr_e)
    for cell in w[1]: cell.font = H; cell.fill = HF; cell.alignment = Alignment(wrap_text=True, vertical="center")
    for e in sorted([e for e in out if e["n"] == nn], key=lambda e: (ORDER.index(e["cat"]) if e["cat"] in ORDER else 9, e["c"], e["cfg"])):
        El = round(e["msum"] - e["m"], 1) if e.get("m") else None
        w.append([e["num"], e["cfg"], e["n"], e["c"], e["cat"], e["sect"], e["structure"], e["roles"], e["J"], e["Qt"], e["Lkn"], e["Qp"],
                  e.get("m"), e["loi"], e["statut"], e["msum"], e["src"], e.get("mfunc"), El,
                  e["tcls"], e["tau"], e["pdg"], e.get("mexp"), e["note"]])
        fill = FILL.get(e["cat"])
        if fill:
            for cell in w[w.max_row]: cell.fill = PatternFill("solid", fgColor=fill)
    for col, wd in zip("ABCDEFGHIJKLMNOPQRSTUVWX", [5, 20, 3, 3, 30, 46, 44, 44, 14, 7, 7, 7, 14, 50, 26, 12, 12, 14, 12, 40, 10, 26, 12, 30]):
        w.column_dimensions[col].width = wd
    w.freeze_panes = "C2"

w2 = wb.create_sheet("Éremnions (réservoir)")
w2.append(["Le réservoir sombre = nœuds premiers ISOLÉS c ≥ 23 (T(2,q≥23), T(3,q≥12)) — absents de la combinatoire ≤ 10₁ de la table principale"])
w2.append(["Nœud", "c", "ℓ (L/R, loi toriques ACPR + T(2,23) mesuré)", "M (MeV) = m_ℓ·ℓ/ℓ(3₁)", "Q", "σ_eq (cm²)", "τ", "Statut"])
for cell in w2[2]: cell.font = H; cell.fill = HF
for e in erem:
    w2.append([e["cfg"], e["c"], e["l"], e["m"], "0 (neutralisé par framing)", "~1e-48 (bord c=23) ; α_pol = R_K³",
               "> 1e347 s (barrière CS k=135)", "réservoir dominant (bord c=23-25 : 34/22/15 %)"])
w2.append([]); w2.append([f"pente géométrique : {SLOPE:.3f} L/R par croisement = {ML*SLOPE/L31:.1f} MeV par croisement (pas ≈ T_c) ; T(3,q≥12) : même loi par croisement, ropelengths non tabulées"])
w2.append(["Aucun éremnion, ni aucune transition du secteur sombre, n'a d'échelle d'énergie au keV : masses au GeV, pas ~70 MeV par croisement, hyperfin πσ_χ au GeV²."])
for col, wd in zip("ABCDEFGH", [10, 4, 30, 22, 26, 34, 28, 40]): w2.column_dimensions[col].width = wd

w3 = wb.create_sheet("Résumé & corrections")
w3.append(["Catalogue V6 par secteurs — 16/09/2026 — remplace la version 809d14d (masses Ω_ts V5, canal « plus proche en dessous »)"])
w3.append([]); w3.append(["Catégorie", "Nombre", "Masse", "Durée de vie", "n=1", "n=2", "n=3"])
for cell in w3[3]: cell.font = H; cell.fill = HF
desc = {"LEPTON": ("Koide (3 modes Z₃ du 3₁)", "stable / Fermi"),
        "NŒUD LIBRE (n=1)": ("M_topo géométrique ACPR (398-801 MeV)", "dénouage P15 3e-10 – 3e-9 s"),
        "COMPOSITE ASSIGNÉ": ("catalogue σ_χ (loi de secteur)", "Fermi/Sargent = PDG"),
        "COMPOSITE ASSIGNÉ (spectre étendu)": ("moteur aligné 224", "PDG"),
        "SOMBRE STRICT": ("ΣM_topo 1,07-1,64 GeV (estimation)", "stable (Q=0 exact)"),
        "RÉALISABLE (Q_phys=0 via Lk)": ("aucune masse liée", "cascade P15 ~1e-18 s"),
        "DÉSERT (transitoire)": ("aucune masse liée (borne ΣM_topo ≤ 2,4 GeV)", "cascade P15 ~1e-18 s")}
w3.cell(row=3, column=5, value="n=1").font = H; w3.cell(row=3, column=6, value="n=2").font = H; w3.cell(row=3, column=7, value="n=3").font = H
for cat, nb in stats.most_common():
    w3.append([cat, nb, *desc.get(cat, ("", "")), sum(1 for e in out if e["cat"] == cat and e["n"] == 1),
               sum(1 for e in out if e["cat"] == cat and e["n"] == 2), sum(1 for e in out if e["cat"] == cat and e["n"] == 3)])
w3.append(["ÉREMNIONS (hors table, n=1 c≥23)", "réservoir", "1,7-2,2 GeV géométrique", "> 1e347 s"])
w3.append([])
for line in [
    "CORRECTIONS par rapport à l'ancienne table :",
    "1. Les Q_désint de 30-300 keV et les τ de 1e7-1e12 s étaient des artefacts Ω_ts : sous ACPR, 10 canaux sur 14 sont fermés (Q<0), aucun keV ne survit.",
    "2. « DM : stable τ>1e347 » était attribué à 139 entrelacs n≥2 à Q_phys=0. Le réservoir = nœuds premiers ISOLÉS c≥23 (théorème d'instabilité des entrelacs noirs) ; seuls les 6 stricts amphichiraux restent stables (sous-population secondaire) ; les 133 autres sont des réalisables transitoires.",
    "3. Aucune loi de secteur V6 ne donne de masse liée à un entrelacs non protégé : la colonne m V6 est vide pour le désert (borne déconfinée ΣM_topo ≤ 3×801 MeV — rien ne peuple [12, 80] GeV, théorème du désert renforcé).",
    "4. Les composites assignés portent la masse du catalogue σ_χ (07/07, précédence maximale) ; les états du spectre étendu, celle du moteur aligné. Aucune masse Ω_ts.",
    "5. Les entrelacs à Q_phys=0 (139) sont classés réalisables (table des 117 : transitoires/contextuels P22) AVANT toute correspondance historique du spectre étendu ; l'assignation V5 supplantée est conservée en note.",
    "6. Le 10₁ n'a pas de ropelength ACPR (valeur historique 84,66 L/R → 801 MeV, borne) : les 231 configurations qui le contiennent sont marquées.",
]:
    w3.append([line])
w3.column_dimensions["A"].width = 44; w3.column_dimensions["B"].width = 10; w3.column_dimensions["C"].width = 44; w3.column_dimensions["D"].width = 30

wb.move_sheet("Résumé & corrections", offset=-(len(wb.sheetnames) - 1))
wb.save('docs/TQNT_catalog_V6_complete.xlsx')

print("=== catégories ===")
for cat, nb in stats.most_common(): print(f"  {nb:5d}  {cat}")
print("\n=== nœuds libres n=1 ===")
for e in out:
    if e["cat"] == "NŒUD LIBRE (n=1)": print(f"  {e['cfg']:>5}  M = {e['m']:6.1f} MeV  τ = {e['tau']} s   {e['sect'][:60]}")
print("\n=== éremnions (loi toriques : pente %.3f L/R/croisement = %.1f MeV/croisement) ===" % (SLOPE, ML * SLOPE / L31))
for e in erem: print(f"  {e['cfg']:>8}  ℓ = {e['l']:6.1f}  M = {e['m']} MeV")
print("\n=== composites assignés (catalogue σ_χ) ===")
for e in out:
    if e["cat"] == "COMPOSITE ASSIGNÉ": print(f"  {e['cfg']:>18}  {e['pdg']:<22} m = {e['m']:>9}  {e['sect']}")
mx = max(e["msum"] for e in out)
print(f"\nΣM_topo max sur les 1539 : {mx:.0f} MeV ; fonctionnelle max : {max(e['mfunc'] or 0 for e in out):.0f} MeV — rien dans [12, 80] GeV")
print("stricts :", [e["cfg"] for e in out if e["cat"] == "SOMBRE STRICT"])
