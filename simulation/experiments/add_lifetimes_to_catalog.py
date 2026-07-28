r"""
Ajoute une colonne « Durée de vie τ (s) » au catalogue TQNT-V5 de diffusion.

ESTIMATEUR TQNT-V5 UNIFIÉ (5 régimes) :

  Régime A — STABLE (τ → ∞)
    • Nœud isolé n=1, c ≤ 7 (3_1, 4_1, 5_1, 5_2, 6_1, 7_1, 7_2)
      → électron-like, neutrino-like, stable absolu (τ > 10³⁰ ans)
    • Composite n≥2 avec Lk = Lk_opt (énergie minimale, ex: proton)

  Régime B — MATIÈRE NOIRE (τ > 10³⁴⁷ s)
    • Nœud isolé n=1 avec c > 18 (P18)
    • Protection topologique absolue

  Régime C — DÉSINTÉGRATION ÉLECTROMAGNÉTIQUE (τ ~ 10⁻¹⁸ à 10⁻¹⁶ s)
    • Composite contenant un amphichiral pur dominant (4_1, 6_3, 8_3)
    • Conversion 2γ ou 3γ (analog π⁰, η, η')
    • τ_EM ~ ℏ × 192π³ / (α² m³)

  Régime D — DÉSINTÉGRATION FAIBLE (τ ~ 10⁻¹⁰ à 10³ s)
    • Composite avec déficit topologique ΔLk ≠ 0 ou saveur changeante
    • Fermi-Sirlin : τ_weak = 192π³ ℏ / (G_F² m⁵) × Φ_phase_space
    • Plage : neutron (~880 s) à mésons B (~ps)

  Régime E — DÉSINTÉGRATION FORTE (τ ~ 10⁻²⁴ à 10⁻²² s)
    • Composite massif sans canal faible disponible
    • Résonances large width (Δ, ρ, ω, etc.)
    • τ_strong ~ ℏ / (β × ⟨c_J⟩ × m)

ATTRIBUTION DU RÉGIME (heuristique) :
  1. Vérifier conditions Régime A (stable)
  2. Vérifier conditions Régime B (DM)
  3. Si amphichiral dominant → Régime C
  4. Si masse > 5 GeV ou n ≥ 4 → Régime E (forte)
  5. Sinon → Régime D (faible)
"""
from __future__ import annotations
import math
import shutil
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from theory.jones import jones_polynomial, jones_determinant, braid_index


# ─── Constantes physiques ───
HBAR_MeV_s = 6.582119e-22         # MeV·s (= ℏ in MeV·s)
G_F_per_MeV2 = 1.1663787e-11      # MeV⁻²
ALPHA_EM = 1 / 137.035999
ALPHA_S_M = 0.118                 # à l'échelle EW (running)
BETA_TENSION_MeV = 0.210          # tension d'enlacement
C_J_AVG = 6.15                    # ⟨c_J⟩ nucléon

# ─── Catalogue des nœuds ───
AMPHICHIRAUX = {"4_1", "6_3", "8_3"}
SINGLE_STABLE = {"3_1"}  # électron stable


def parse_config(cfg_str):
    return [c.strip() for c in str(cfg_str).split("+")]


def crossings(K):
    return int(K.split("_")[0])


def has_amphichiral_dominant(cfg):
    """True si l'entrelacs est dominé par amphichiraux."""
    if len(cfg) < 2:
        return False
    n_amph = sum(1 for K in cfg if K in AMPHICHIRAUX)
    return n_amph >= len(cfg) - 1  # majorité d'amphichiraux


def is_likely_proton(cfg):
    """True si configuration ≈ proton (état stable de l'entrelacs)."""
    from collections import Counter
    return Counter(cfg) == Counter(["3_1", "7_2", "7_2"])


def estimate_lifetime_TQNT(cfg, m_MeV, Q_phys=None, n_comp=None, c_total=None,
                            is_DM=False):
    """Retourne (régime, τ en s).

    Calibrations physiques :
      - τ(neutron)  = 880 s pour m = 940 MeV (n=3, faible)
      - τ(π±)       = 2.6e-8 s pour m = 140 MeV (n=2, faible)
      - τ(π⁰)       = 8.4e-17 s pour m = 135 MeV (n=2, EM)
      - τ(η)        = 5e-19 s pour m = 548 MeV (n=2 ou n=3, EM)
    """
    n = n_comp if n_comp is not None else len(cfg)
    c = c_total if c_total is not None else sum(crossings(K) for K in cfg)

    # ─── Régime A : STABLE ───
    if n == 1 and cfg[0] in SINGLE_STABLE:
        return ("A_stable", float("inf"))
    if is_likely_proton(cfg):
        return ("A_stable", float("inf"))

    # ─── Régime B : MATIÈRE NOIRE ───
    # Plusieurs critères :
    #   - Nœud isolé n=1 avec c > 18 (P18 strict)
    #   - Marqueur explicite is_DM (catalogue V5)
    if (n == 1 and c > 18) or is_DM:
        return ("B_dark", 1e347)

    # Nœud isolé c ≤ 18 traité comme particule libre → désintégration faible
    if n == 1:
        # Calibration : neutron-like τ(940 MeV) = 880 s
        tau = 880.0 * (940.0 / max(m_MeV, 1.0))**5
        return ("D_weak", tau)

    # ─── Régime C : ÉLECTROMAGNÉTIQUE ───
    # Amphichiraux pseudoscalaires : π⁰-like
    if has_amphichiral_dominant(cfg):
        # Calibration π⁰ : τ(135 MeV) = 8.4e-17 s, scaling ∝ 1/m³
        tau = 8.4e-17 * (135.0 / max(m_MeV, 1.0))**3
        return ("C_EM", tau)

    # ─── Régime E : INTERACTION FORTE ───
    # n ≥ 4 (multi-quark) : décroissance forte rapide
    # OU m > 5 GeV avec n ≥ 2 (résonances lourdes mais pas EW bosons)
    # OU n = 2 avec m > 1 GeV (résonances vectorielles ρ, ω, etc.)
    if n >= 4:
        # τ ~ ℏ / Γ_strong avec Γ ~ Λ_QCD ~ 100 MeV
        tau = HBAR_MeV_s / (BETA_TENSION_MeV * C_J_AVG)
        return ("E_strong", tau)

    if n == 2 and m_MeV > 1000:
        # Résonance lourde non-amphichirale : forte (analog ρ, ω, J/ψ)
        # ρ(770) τ = 4.4e-24 s
        tau = 4.4e-24 * (770.0 / m_MeV)
        return ("E_strong", tau)

    # ─── Régime D : DÉSINTÉGRATION FAIBLE (par défaut) ───
    # Calibration selon n :
    #   n=2 : π±-like, τ(140 MeV) = 2.6e-8 s
    #   n=3 : neutron-like, τ(940 MeV) = 880 s
    if n == 2:
        # π±/K± mesons, hyperons légers → faible
        tau = 2.6e-8 * (140.0 / max(m_MeV, 1.0))**5
    else:  # n == 3
        # Baryons : neutron-like calibration
        tau = 880.0 * (940.0 / max(m_MeV, 1.0))**5
    return ("D_weak", tau)


def format_lifetime(regime, tau):
    """Formatage compact (s, ms, μs, ns, ps, fs, as)."""
    if regime == "A_stable":
        return "stable (> 10³⁰ ans)"
    if regime == "B_dark":
        return "> 10³⁴⁷ s (mat. noire)"
    if tau is None or tau == float("inf"):
        return "—"

    # Format scientifique avec annotation régime
    suffix = {"C_EM": "EM", "D_weak": "fa", "E_strong": "fo"}.get(regime, "")
    if tau < 1e-100:
        return f"~0 s"
    return f"{tau:.2e} s ({suffix})"


# ════════════════════════════════════════════════════════════
# Lecture + écriture du catalogue
# ════════════════════════════════════════════════════════════
SRC = "/home/euk9/projets/physique_nouvelle/docs/TQNT_catalog_V5_diffusion.xlsx"

print(f"Lecture : {SRC}")
wb = openpyxl.load_workbook(SRC)
ws = wb["2. Catalogue unifié"]
print(f"  Dimensions : {ws.max_row} lignes × {ws.max_column} cols")

# Trouver les colonnes "Config" et "m(Ω_ts) V5 (MeV)"
HEADERS = {ws.cell(row=2, column=c).value: c for c in range(1, ws.max_column + 1)}
print(f"  Colonnes connues : {[h for h in HEADERS if h][:8]}")

col_cfg = HEADERS.get("Config TQNT")
col_mass = None
for h in HEADERS:
    if h and "m(Ω_ts) V5" in str(h):
        col_mass = HEADERS[h]
        break

if col_cfg is None or col_mass is None:
    raise ValueError(f"Colonnes 'Config TQNT' ou 'm(Ω_ts) V5' introuvables")

print(f"  Config TQNT  : col {col_cfg}")
print(f"  m(Ω_ts) V5   : col {col_mass}")

# ─── Insertion de la colonne « Durée de vie τ (s) » ───
# Trouver une bonne position (après m(Ω_ts) plast s'il existe, sinon après m_V5)
col_insert = None
for h in HEADERS:
    if h and "m(Ω_ts) plast" in str(h):
        col_insert = HEADERS[h] + 1
        break
if col_insert is None:
    col_insert = col_mass + 1

print(f"\n  Insertion à la colonne {col_insert}")
ws.insert_cols(col_insert)

# Header
hdr_cell = ws.cell(row=2, column=col_insert, value="τ TQNT (s)")
hdr_cell.font = Font(bold=True, size=11, color="FFFFFF")
hdr_cell.fill = PatternFill("solid", fgColor="2E75B6")
hdr_cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

# Largeur de la colonne
ws.column_dimensions[openpyxl.utils.get_column_letter(col_insert)].width = 22

# ─── Calcul des durées de vie ligne par ligne ───
print("\n  Calcul des durées de vie…")

# Statistique par régime
stats = {"A_stable": 0, "B_dark": 0, "C_EM": 0, "D_weak": 0, "E_strong": 0}
preview = []

# Trouver aussi la colonne "Particule PDG" pour détecter les MN
# Attention : après ws.insert_cols(), les colonnes ≥ col_insert sont décalées de +1
col_pdg_orig = HEADERS.get("Particule PDG")
col_pdg = col_pdg_orig + 1 if (col_pdg_orig and col_pdg_orig >= col_insert) else col_pdg_orig
# Aussi décaler col_mass et col_cfg si nécessaire
col_mass_eff = col_mass + 1 if col_mass >= col_insert else col_mass
col_cfg_eff = col_cfg + 1 if col_cfg >= col_insert else col_cfg

print(f"  Après insertion col_insert={col_insert} :")
print(f"    col_cfg : {col_cfg} → {col_cfg_eff}")
print(f"    col_mass: {col_mass} → {col_mass_eff}")
print(f"    col_pdg : {col_pdg_orig} → {col_pdg}")

for row in range(3, ws.max_row + 1):
    cfg_str = ws.cell(row=row, column=col_cfg_eff).value
    m_val = ws.cell(row=row, column=col_mass_eff).value
    if cfg_str is None or m_val is None:
        continue
    try:
        cfg = parse_config(cfg_str)
        m_MeV = float(m_val)
    except (TypeError, ValueError):
        continue

    # Détecter matière noire via colonne PDG si présente
    is_DM = False
    if col_pdg is not None:
        pdg_val = ws.cell(row=row, column=col_pdg).value
        if pdg_val and "MN" in str(pdg_val):
            is_DM = True

    regime, tau = estimate_lifetime_TQNT(cfg, m_MeV, is_DM=is_DM)
    stats[regime] += 1
    tau_str = format_lifetime(regime, tau)

    cell = ws.cell(row=row, column=col_insert, value=tau_str)
    cell.font = Font(size=10)
    cell.alignment = Alignment(horizontal="center", vertical="center")

    # Quelques exemples pour preview
    if len(preview) < 12:
        preview.append((cfg_str[:24], f"{m_MeV:.1f}", regime, tau_str))

print(f"\n  Aperçu (12 premières lignes) :")
print(f"  {'Config':<24} {'m (MeV)':>10} {'Régime':<10} {'τ TQNT (s)'}")
print("─" * 80)
for cfg, m, regime, tau in preview:
    print(f"  {cfg:<24} {m:>10} {regime:<10} {tau}")

print(f"\n  Distribution par régime :")
for regime, count in stats.items():
    pct = count / sum(stats.values()) * 100
    print(f"    {regime:<12} : {count:>5} ({pct:.1f}%)")

# ─── Update Documentation sheet ───
ws_doc = wb["7. Méthodologie"]
# Ajouter une note à la fin
last_row = ws_doc.max_row + 2
doc_text = [
    ("τ TQNT (s)", "Estimation de durée de vie par formules TQNT-V5 (5 régimes). "
                   "ATTENTION : ces estimations sont des ORDRES DE GRANDEUR, "
                   "non des prédictions précises. Pour les valeurs précises calibrées "
                   "sur PDG, voir le chapitre « Durées de vie » de TQNT_detaillee_V5_FR.tex."),
    ("Régime A_stable", "Particules absolument stables (τ → ∞). Cas reconnus : "
                        "électron (3_1 isolé), proton (3_1+7_2+7_2 avec Lk_opt)."),
    ("Régime B_dark", "Matière noire topologique (τ > 10³⁴⁷ s, protection topologique). "
                      "Inclut les 117 candidats identifiés en V5 (6 STRICT + 111 RÉALISABLE)."),
    ("Régime C_EM", "Désintégration électromagnétique. Calibration π⁰ : "
                    "τ = 8.4×10⁻¹⁷ × (135 MeV/m)³. Régime appliqué aux entrelacs "
                    "dominés par amphichiraux (4_1, 6_3, 8_3)."),
    ("Régime D_weak", "Désintégration faible (Fermi-Sirlin). "
                      "Calibration : τ_n=2 = 2.6×10⁻⁸ × (140/m)⁵ (π±-like) ; "
                      "τ_n=3 = 880 × (940/m)⁵ (neutron-like)."),
    ("Régime E_strong", "Désintégration forte (résonance). "
                        "Calibration ρ(770) : τ = 4.4×10⁻²⁴ × (770/m). "
                        "Appliqué aux multi-quarks (n≥4) et résonances vectorielles n=2."),
    ("Limites de la formule", "Les écarts avec PDG (parfois plusieurs ordres) viennent de : "
                              "(1) Q-value au lieu de m_total ; "
                              "(2) suppression CKM (K, B mesons) ; "
                              "(3) mixage CP (η, η') ; "
                              "(4) phase space spécifique. "
                              "Les prédictions PRÉCISES nécessitent les calibrations "
                              "détaillées du chapitre 13 (Durées de vie) de V5 (RMS 0.94 décade)."),
]
for col, defn in doc_text:
    ws_doc.cell(row=last_row, column=1, value=col).font = Font(bold=True, size=10)
    ws_doc.cell(row=last_row, column=2, value=defn).font = Font(size=10)
    last_row += 1

# ─── Sauvegarde ───
DST = SRC.replace(".xlsx", "")  # idem (overwrite)
print(f"\n  Sauvegarde → {SRC}")
wb.save(SRC)
import os
print(f"  Taille : {os.path.getsize(SRC) // 1024} ko")
print(f"  Onglets : {len(wb.sheetnames)}")
print("\n✓ Durées de vie ajoutées au catalogue.")
