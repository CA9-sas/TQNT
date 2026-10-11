"""Limites sur les perles de matière noire (Classe A/E) — enveloppe f_max(M).

POST-12e RÉFUTATION (V7) : la machinerie nominale des perles (M_typ =
2,03e-3 Msun, f_A = 5 %) est exclue ×5 par OGLE-IV 20 ans (Mróz+ 2024)
et son flux de raies ×13 par notre line search Fermi-LAT 18 ans
(s < 0,076, joint). Ce script DÉFINIT ce qui reste permis : l'enveloppe
combinée f_max(M) sur tout l'axe de masse, et les bornes STRUCTURELLES
du domaine des perles sous les masses canoniques (ACPR 2026-07-22).

Canaux (95 %) :
  1. OGLE-IV 20 ans LMC (Mróz+ 2024, arXiv:2403.02386) : f < 1 % sur
     [1,8e-4, 6,3] Msun ; f < 10 % sur [1,3e-5, 860] Msun (abstract).
  1bis. OGLE haute cadence Nuages de Magellan (Mróz+ 2024b, ApJL 976,
     L19 ; ajouté 04/08 soir) : f < 1 % sur [1,4e-8, 0,013] Msun —
     REMPLACE EROS (8 %) et le 10 % d'OGLE-IV sur toute la bande
     médiane ; un seul candidat (OGLE-LMC-20, long), aucun événement
     court.
  2. EROS-2/MACHO (Tisserand+ 2007) : f < 8 % sur [1e-6, 1] Msun
     (pont, désormais supplanté par 1bis ; gardé pour mémoire).
  3. Subaru HSC M31 : digitisation PROPRE (2026-08-04) de la courbe
     benchmark de Smyth+ 2020 (PRD 101, 063005, Fig. « Updated
     Constraints », extraction pixel calibrée sur les ticks —
     docs/references/hsc_smyth2020_digitized.csv) ; variante déclarée :
     Croon+ 2020 R90=0 (compilation PBHbounds). La réserve ±0,5 dex du
     04/08 matin est LEVÉE. Le benchmark Smyth (tailles stellaires
     réalistes) est plus CONSERVATIF que la variante aux deux bords :
     fenêtre [5,3e-11, 5,6e-6] Msun, min f = 0,61 % à 1,2e-9.
  4. Raies γ (contenu Classe E des perles) : renormalisation du verrou
     s < 0,076 (18 ans, |b|>30°) — recette du 15/07 à l'identique
     (gaz TOV de briques m = 1,221/2 GeV, Δc = 22, L = 2Γ) ;
     flux ∝ (f/M)·L(M) → f_γ(M) = f_nom·s_lim·(M/M_typ)·L_typ/L(M).
     Ce canal suppose le contenu E nominal : l'enveloppe est donnée
     AVEC et SANS lui.

Bornes structurelles (référence ACPR) :
  - Chandrasekhar topologique M_Ch = M_Pl³/m² : bulk = réservoir
    (bord c = 23-25, m = 1,7-1,9 GeV dominant) → M_Ch ≈ 0,45-0,57 Msun.
    AU-DESSUS : pas de perles (effondrement) — les limites y sont sans objet.
  - Validité lentille ponctuelle : r_perle(M) vs rayon d'Einstein
    (LMC 49,6 kpc pour OGLE, M31 770 kpc pour HSC, x = 0,5) ;
    r_perle ≳ R_E → le canal microlentille se vide (lentille étendue).
  - Burn-out amphi : t_dep = (N/2)/Γ vs t_Hubble (vérifié non atteint).

Structure interne : polytrope de Lane-Emden n = 3/2 ANALYTIQUE (gaz de
Fermi NR), recoupé avec tov_pearl APRÈS CORRECTION DU BUG DE BRACKET
(2026-08-04) ; v_F plafonnée à c, régime v_F > 0,5c étiqueté indicatif
(la courbe γ ne fait que se renforcer).

AUDIT (2026-08-04, ce script) — LE BRAS γ DE LA 12e RÉFUTATION TOMBE :
la bissection TOV avait un bracket de densité INDÉPENDANT de la masse
cible → toute cible < 6,4e-3 Msun renvoyait l'étoile-plancher de
6,45e-3 Msun (sur-dense). Le flux de raies nominal du 15/07 (Γ =
2,529e24 Hz à M_typ) était celui de cette étoile-artefact : Γ vrai =
3,657e22 Hz, INFLATION ×69,2. La mesure 18 ans (UL de flux) reste
valide ; sa traduction devient s_vrai < 0,076×69,2 = 5,26 → le flux
nominal VRAI est ×5,3 SOUS la limite : pas d'exclusion γ au nominal.
La 12e réfutation reste enregistrée par son bras OGLE (×5, indépendant
de la structure interne). Corollaire favorable : les 6 raies fines de
la levée de F3 descendent ×69 aussi (encore plus sous les limites 2015).

CONFRONTATION Powell+ 2025 (Nature Astronomy 9, 1714 — l'objet d'un
million de masses solaires de B1938+666, gravitational imaging VLBI ;
compagnon Vegetti+ : point 2-5e5 + enveloppe) : M ~ 1e6 Msun est
×2e6 AU-DESSUS du plafond structurel des perles (M_Ch = 0,50) — cet
objet NE PEUT PAS être une perle ; sa lecture canonique TQNT est la
Classe F (graine primordiale 1e5-3e5 habillée, brouillon_b1938_classeF,
f_F pincé 1e-5..1e-4 par B1938+GD-1). Les deux populations compactes
TQNT sont disjointes de ~6 ordres de grandeur ; la détection Powell
peuple la seconde et ne mord pas sur les limites perles.

Sorties : table des fenêtres survivantes + output/limites_perles_dm.png.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.dm_pearl_internal import (GEV_KG, M_SUN_KG, MPL_KG, T_HUBBLE_S,
                                      HBAR, C_LIGHT, G_NEWTON,
                                      SIGMA_QED, ALPHA_EM,
                                      entanglement_rate)

# ---------------------------------------------------------------- référence
M_TYP = 2.03e-3          # Msun — nominal V5.19 (référence de renormalisation)
F_NOM = 0.05             # fraction DM nominale Classe A (V5.19)
S_LIM = 0.076            # verrou joint 18 ans (fermi_hl_line_search)
DC = 22                  # Δc dérivé (Vassiliev, V5.21)
M_BRIQUE = 1.221 / 2.0   # GeV — recette 15/07 (état 6_3+8_3)
M_RESERVOIR = 1.8        # GeV — bord du réservoir c=23-25 (1,7-1,9 dominant)
M_RESERVOIR_BAND = (1.7, 1.9)
M_STRICT_ACPR = [1.07, 1.10, 1.22, 1.35, 1.47, 1.64]   # GeV, 22/07

D_LMC_M = 49.6e3 * 3.0857e16      # m
D_M31_M = 770.0e3 * 3.0857e16     # m

# ---------------------------------------------------------------- polytrope n=3/2
def _lane_emden_n32():
    """Intègre Lane-Emden n=3/2 ; renvoie (xi1, omega=-xi²θ', I3=∫θ³ξ²dξ)."""
    dx = 1e-4
    xi, th, dth = dx, 1.0 - dx * dx / 6.0, -dx / 3.0
    I3 = 0.0
    while th > 0:
        d2 = -th ** 1.5 - 2.0 * dth / xi
        I3 += th ** 3 * xi ** 2 * dx
        th += dth * dx
        dth += d2 * dx
        xi += dx
    return xi, -xi ** 2 * dth, I3

XI1, OMEGA32, I3_LE = _lane_emden_n32()   # attendu ~3.654, ~2.714

def pearl_poly(M_msun: float, m_knot_GeV: float) -> dict:
    """Perle = polytrope NR n=3/2 analytique (gaz de Fermi dégénéré)."""
    m = m_knot_GeV * GEV_KG
    M = M_msun * M_SUN_KG
    K = (HBAR ** 2) * (3.0 * math.pi ** 2) ** (2.0 / 3.0) / (5.0 * m ** (8.0 / 3.0))
    # M = 4π ω a³ ρ_c, a² = 2,5 K ρ_c^(-1/3)/(4πG)  →  ρ_c = [M/(4πω)]²(4πG/2,5K)³
    rho_c = (M / (4.0 * math.pi * OMEGA32)) ** 2 * (4.0 * math.pi * G_NEWTON / (2.5 * K)) ** 3
    a = math.sqrt(2.5 * K * rho_c ** (-1.0 / 3.0) / (4.0 * math.pi * G_NEWTON))
    R = XI1 * a
    n_c = rho_c / m
    p_F = HBAR * (3.0 * math.pi ** 2 * n_c) ** (1.0 / 3.0)
    v_F = min(p_F / m, C_LIGHT)
    sigma = SIGMA_QED * ALPHA_EM ** DC
    # Γ = ½ σ v_rel ∫n² dV ; ∫n²dV = (ρ_c/m)² 4π a³ I3
    Gamma = 0.5 * sigma * (math.sqrt(2.0) * v_F) * (rho_c / m) ** 2 * 4.0 * math.pi * a ** 3 * I3_LE
    return {"R_m": R, "rho_c": rho_c, "n_c": n_c, "v_F": v_F,
            "Gamma_Hz": Gamma, "N_knots": M / m}

# ---------------------------------------------------------------- validation vs TOV
print(__doc__)
print(f"Lane-Emden n=3/2 : xi1 = {XI1:.4f} (attendu 3,6538), "
      f"omega = {OMEGA32:.4f} (attendu 2,7141), I3 = {I3_LE:.4f}")
print("\n--- validation polytrope vs TOV (m_brique, TOV corrigé 04/08) :")
for M in (1e-6, 1e-3, 1e-2, 1e-1):
    tov = entanglement_rate(M, M_BRIQUE, DC)
    pol = pearl_poly(M, M_BRIQUE)
    print(f"  M = {M:.0e} : R {pol['R_m']:.3e} vs {tov['r_pearl_m']:.3e} m "
          f"({pol['R_m']/tov['r_pearl_m']-1:+.1%}) ; "
          f"Γ {pol['Gamma_Hz']:.3e} vs {tov['Gamma_total_pearl_Hz']:.3e} Hz "
          f"({pol['Gamma_Hz']/tov['Gamma_total_pearl_Hz']-1:+.1%})")

# ---------------------------------------------------------------- contrôles ×5 / ×13
print("\n--- contrôles (12e réfutation) :")
# OGLE : couverture de la log-normale (M0=2,03e-3, σ=0,3 dex) par [1,8e-4, 6,3]
from math import erf, sqrt
cov = 0.5 * (1 + erf((math.log10(6.3) - math.log10(M_TYP)) / 0.3 / sqrt(2))) \
    - 0.5 * (1 + erf((math.log10(1.8e-4) - math.log10(M_TYP)) / 0.3 / sqrt(2)))
print(f"  OGLE  : couverture fonction de masse = {cov:.1%} → exclusion "
      f"×{F_NOM*cov/0.01:.1f} (attendu ×5,0)")
GAMMA_OLD_1507 = 2.529e24     # Hz — valeur de l'étoile-plancher utilisée le 15/07
GAMMA_TRUE_TYP = entanglement_rate(M_TYP, M_BRIQUE, DC)["Gamma_total_pearl_Hz"]
INFL = GAMMA_OLD_1507 / GAMMA_TRUE_TYP
S_TRUE = S_LIM * INFL
print(f"  γ 18a : ancien verdict ×13,2 CONTRÔLÉ (s<{S_LIM}, unités gonflées) ; "
      f"AUDIT : Γ_typ vrai = {GAMMA_TRUE_TYP:.3e} Hz vs {GAMMA_OLD_1507:.3e} "
      f"utilisé → inflation ×{INFL:.1f} → s_vrai < {S_TRUE:.2f} : "
      f"le nominal est ×{S_TRUE:.1f} SOUS la limite (bras γ NON contraignant)")

# ---------------------------------------------------------------- bornes structurelles
print("\n--- bornes structurelles (référence ACPR) :")
mch = lambda m_gev: MPL_KG ** 3 / (m_gev * GEV_KG) ** 2 / M_SUN_KG
M_CH = mch(M_RESERVOIR)
print(f"  M_Ch(bulk réservoir 1,8 GeV) = {M_CH:.3f} Msun "
      f"[{mch(M_RESERVOIR_BAND[1]):.3f}, {mch(M_RESERVOIR_BAND[0]):.3f}] ; "
      f"stricts ACPR : {mch(max(M_STRICT_ACPR)):.2f} (1,64) à {mch(min(M_STRICT_ACPR)):.2f} (1,07) "
      f"(l'ancien 0,68 = 1,55 GeV Ω_ts, caduc)")

def r_einstein(M_msun, D_source_m, x=0.5):
    return math.sqrt(4.0 * G_NEWTON * M_msun * M_SUN_KG / C_LIGHT ** 2
                     * D_source_m * x * (1.0 - x))

def m_fluffy(m_knot_GeV, D_source_m):
    """Croisement r_perle(M) = R_E(M) : en dessous, lentille ÉTENDUE (canal vidé)."""
    lo, hi = 1e-16, 1.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if pearl_poly(mid, m_knot_GeV)["R_m"] > r_einstein(mid, D_source_m):
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)

MF_HSC = m_fluffy(M_RESERVOIR, D_M31_M)
MF_HSC_BRIQUE = m_fluffy(M_BRIQUE, D_M31_M)
MF_OGLE = m_fluffy(M_RESERVOIR, D_LMC_M)
print(f"  lentille étendue (r_perle > R_E) : HSC en dessous de "
      f"{MF_HSC:.1e} Msun (bulk 1,8 GeV) / {MF_HSC_BRIQUE:.1e} (variante briques "
      f"{M_BRIQUE:.2f} GeV) ; OGLE en dessous de {MF_OGLE:.1e} Msun")
p_typ = pearl_poly(M_TYP, M_RESERVOIR)
print(f"  au nominal : r_perle = {p_typ['R_m']/1e3:.1f} km (bulk 1,8 GeV), "
      f"N = {p_typ['N_knots']:.2e} nœuds")

# burn-out amphi (garde) : t_dep = (N/2)/Γ, recette m_brique
tdep_typ = (entanglement_rate(M_TYP, M_BRIQUE, DC)["N_knots"] / 2.0) \
    / entanglement_rate(M_TYP, M_BRIQUE, DC)["Gamma_total_pearl_Hz"]
tdep_mch = tdep_typ * (M_TYP / M_CH) ** (8.0 / 3.0)
print(f"  burn-out : t_dep(M_typ) = {tdep_typ:.1e} s, t_dep(M_Ch) = {tdep_mch:.1e} s "
      f"vs t_H = {T_HUBBLE_S:.1e} s → jamais atteint dans le domaine structurel")

# ---------------------------------------------------------------- canaux f_max(M)
GRID = np.logspace(-13, 1, 561)

def f_ogle(M):
    if 1.8e-4 <= M <= 6.3:
        return 0.01
    if 1.3e-5 <= M <= 860.0:
        return 0.10
    return 1.0

def f_eros(M):
    return 0.08 if 1e-6 <= M <= 1.0 else 1.0

def f_ogle_hc(M):   # Mróz+ 2024b, ApJL 976 L19 (haute cadence MC)
    return 0.01 if 1.4e-8 <= M <= 0.013 else 1.0

# HSC M31 — digitisation propre Smyth+2020 benchmark (primaire) ;
# variante Croon+2020 R90=0 (PBHbounds) pour l'enveloppe systématique
_HSC_SMYTH = np.loadtxt(ROOT / 'docs/references/hsc_smyth2020_digitized.csv',
                        delimiter=',', skiprows=4)
_HSC_CROON = np.loadtxt(ROOT / 'docs/references/hsc_croon2020_R90_pbhbounds.txt')

def _f_table(M, tab):
    lm = math.log10(M)
    lms, lfs = np.log10(tab[:, 0]), np.log10(np.minimum(tab[:, 1], 1.0))
    if lm <= lms[0] or lm >= lms[-1]:
        return 1.0
    return min(10 ** float(np.interp(lm, lms, lfs)), 1.0)

def f_hsc(M):
    if M < MF_HSC:          # lentille étendue : canal vidé
        return 1.0
    return _f_table(M, _HSC_SMYTH)

def _edge_f1(tab, side):
    """Masse où la table croise f=1 (bord bas/haut de la fenêtre)."""
    m, f = tab[:, 0], np.minimum(tab[:, 1], 1.0)
    inside = np.where(f < 1.0)[0]
    return m[inside[0]] if side == 'lo' else m[inside[-1]]

HSC_LO, HSC_HI = _edge_f1(_HSC_SMYTH, 'lo'), _edge_f1(_HSC_SMYTH, 'hi')
HSC_LO_CROON = _edge_f1(_HSC_CROON, 'lo')

# γ : le verrou 18 ans est un UL de FLUX ; en unités de la machinerie VRAIE
# (TOV corrigé), s_vrai = S_LIM×INFL ; Γ ∝ M^{11/3} (NR exact) → f_γ ∝ M^{-8/3}
GAMMA_TYP = pearl_poly(M_TYP, M_BRIQUE)["Gamma_Hz"]

def f_gamma(M):
    g = pearl_poly(M, M_BRIQUE)["Gamma_Hz"]
    return F_NOM * S_TRUE * (M / M_TYP) * (GAMMA_TYP / g)

def f_gamma_old(M):   # l'artefact (pour le tracé d'audit)
    g = pearl_poly(M, M_BRIQUE)["Gamma_Hz"]
    return F_NOM * S_LIM * (M / M_TYP) * (GAMMA_TYP / g)

f_micro = np.array([min(f_ogle(M), f_ogle_hc(M), f_eros(M), f_hsc(M)) for M in GRID])
f_gam = np.array([min(f_gamma(M), 1.0) for M in GRID])
f_all = np.minimum(f_micro, f_gam)

# ---------------------------------------------------------------- fenêtres définies
print("\n--- LIMITES DÉFINIES (95 %, contenu E nominal ; sans E : colonne micro) :")
edges = [1e-13, MF_HSC, 10 ** -11.3, 1e-10, 1e-8, 1e-6, 1.3e-5, 1.8e-4,
         None, M_CH]  # None = croisement γ/OGLE, calculé
Mx_gamma = M_TYP * (F_NOM * S_TRUE / 0.01) ** (3.0 / 8.0)   # f_γ = 1 % (OGLE)
edges[8] = Mx_gamma
print(f"  (croisement γ/OGLE-1 % : M = {Mx_gamma:.1e} Msun ; "
      f"f_γ(M_Ch) = {f_gamma(M_CH):.1e})")
f_hsc_min = _HSC_SMYTH[np.argmin(_HSC_SMYTH[:, 1]), 1]
M_hsc_min = _HSC_SMYTH[np.argmin(_HSC_SMYTH[:, 1]), 0]
print(f"  (fenêtre HSC Smyth : [{HSC_LO:.1e}, {HSC_HI:.1e}], min f = "
      f"{f_hsc_min:.2%} à {M_hsc_min:.1e} ; variante Croon R90=0 : bord bas "
      f"{HSC_LO_CROON:.0e} — la fenêtre ouverte dépend de la digitisation retenue)")
rows = [
    (f"M < {HSC_LO:.1e}", "AUCUNE contrainte (sous la fenêtre HSC benchmark ; "
     f"lentille étendue de toute façon sous {MF_HSC:.0e})", "—",
     f"FENÊTRE OUVERTE (variante Croon : {HSC_LO_CROON:.0e})"),
    (f"{HSC_LO:.1e} – 1,4e-8", f"HSC M31 (Smyth benchmark) : "
     f"min {f_hsc_min:.1%} à {M_hsc_min:.0e}", "0,6-100 %", "digitisé propre"),
    (f"1,4e-8 – {Mx_gamma:.1e}", "OGLE haute cadence MC + OGLE-IV : 1 % "
     "(∧ HSC ~0,6-1 % vers 1e-9)", "1 %", "Mróz+ 2024b"),
    (f"{Mx_gamma:.1e} – M_Ch={M_CH:.2f}", "raies γ (corrigées) : 1 % → "
     f"{f_gamma(M_CH):.0e} (∝ M^-8/3)", "micro seul : 1 %", "si contenu E"),
    (f"M > M_Ch = {M_CH:.2f}", "PAS DE PERLES (Chandrasekhar topologique, "
     "bulk 1,8 GeV)", "—", "STRUCTUREL"),
]
for a, b, c, d in rows:
    print(f"  {a:<24} | {b:<55} | {d}")

M_POWELL = 1.0e6   # Msun — Powell+ 2025, Nat. Astron. 9, 1714 (B1938+666)
print(f"\n  CONFRONTATION Powell+ 2025 (B1938+666, ~1e6 Msun, VLBI) : "
      f"×{M_POWELL/M_CH:.0e} au-dessus de M_Ch → PAS une perle ; lecture "
      f"canonique = Classe F (graine 1e5-3e5 + enveloppe) — populations "
      f"disjointes de 6 ordres, aucune tension avec les limites perles")
print(f"\n  Nominal V5.19 (2,03e-3, 5 %) : f_max micro = {f_ogle(M_TYP):.0%} "
      f"→ EXCLU ×{F_NOM/f_ogle(M_TYP):.0f} (OGLE, tient) ; "
      f"f_max γ vrai = {F_NOM*S_TRUE:.1%} > 5 % → le bras γ NE CONTRAINT PLUS "
      f"le nominal (12e réfutation = OGLE seul)")
print(f"  f_A max (fonction de masse V5.19 log-normale) : 1 % (OGLE) — "
      f"le γ ne pince qu'au-dessus de {Mx_gamma:.1e} Msun")

# ---------------------------------------------------------------- tracé
fig, ax = plt.subplots(figsize=(9.5, 6.2))
ax.loglog(GRID, f_micro, color="#1f4e79", lw=1.8,
          label="microlentille (HSC Smyth + OGLE haute cadence MC + OGLE-IV), 95 %")
f_croon = np.array([min(_f_table(M, _HSC_CROON), f_ogle(M), f_eros(M)) for M in GRID])
ax.loglog(GRID, f_croon, color="#1f4e79", lw=0.9, ls=":", alpha=0.7,
          label="variante HSC Croon+20 R90=0 (PBHbounds)")
ax.loglog(GRID, f_gam, color="#b91c1c", lw=1.6, ls="--",
          label="raies γ Fermi 18 ans (contenu E nominal, TOV corrigé)")
f_gam_old = np.array([min(f_gamma_old(M), 1.0) for M in GRID])
ax.loglog(GRID, f_gam_old, color="#9ca3af", lw=1.0, ls=":",
          label="canal γ AVANT correction (étoile-plancher, artefact)")
ax.loglog(GRID, f_all, color="black", lw=2.6, alpha=0.65,
          label="enveloppe combinée")
ax.axvline(M_CH, color="#6b21a8", lw=1.6, ls="-.")
ax.text(M_CH * 1.15, 3e-7, "M$_{Ch}$ (bulk 1,8 GeV)\npas de perles au-delà",
        color="#6b21a8", fontsize=8.5)
ax.axvspan(GRID[0], MF_HSC, color="#f59e0b", alpha=0.12)
ax.text(math.sqrt(GRID[0] * MF_HSC), 2e-6, "lentille étendue\n(r$_p$ > R$_E$)",
        color="#92400e", fontsize=8.5, ha="center")
ax.plot([M_TYP], [F_NOM], marker="*", ms=16, color="#b91c1c", zorder=5)
ax.annotate("nominal V5.19 (exclu ×5, OGLE)", (M_TYP, F_NOM),
            textcoords="offset points", xytext=(10, 8), fontsize=9,
            color="#b91c1c")
ax.axvspan(M_CH, GRID[-1], color="#6b21a8", alpha=0.08)
ax.annotate("objet B1938 (Powell+ 25) : 10$^6$ M$_\\odot$ →\ndomaine Classe F, hors perles",
            xy=(GRID[-1], 0.5), xytext=(-8, 0), textcoords="offset points",
            ha="right", fontsize=8, color="#6b21a8")
ax.set_xlim(GRID[0], GRID[-1]); ax.set_ylim(1e-9, 2)
ax.set_xlabel("M$_{perle}$ [M$_\\odot$]")
ax.set_ylabel("f$_{max}$ = Ω$_{perles}$/Ω$_{DM}$ (95 %)")
ax.set_title("Perles de matière noire : limites combinées et domaine structurel "
             "(post-12e réfutation)")
ax.grid(alpha=0.25, which="both")
ax.legend(loc="lower left", fontsize=9)
out = ROOT / "output" / "limites_perles_dm.png"
out.parent.mkdir(exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=150)
print(f"\nfigure → {out}")
