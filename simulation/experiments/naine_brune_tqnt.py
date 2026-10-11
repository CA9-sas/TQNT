"""La naine brune TQNT — structure, matière noire, thermodynamique émergente.

Chantier 2026-08-05 (brouillon_naine_brune_tqnt). Trois volets chiffrés :

  1. CONTRÔLE STANDARD : l'objet ordinaire (polytrope n=3/2, mélange gaz
     idéal + dégénérescence électronique) — retrouver l'échelle du seuil
     d'allumage de l'hydrogène (M_min ~ 0,07-0,08 Msun ; T_c,max ~ 3e6 K)
     avec le minimum d'ingrédients, pour ancrer les volets TQNT.
     La relecture TQNT ne change RIEN ici par construction : nucléons =
     entrelacs 3_1+7_2+7̄_2, électrons = trèfles ; la dégénérescence est
     la statistique de tresse (parité impaire → anticommutation) sur les
     cellules d'action h de la trame ; la température est un paramètre du
     CONTENU d'ondes (référence frottement/température 23/07) — le
     refroidissement est l'évacuation radiative de ce contenu, la trame
     n'ajoute AUCUN canal (pas de frottement du mouvement uniforme).

  2. MATIÈRE NOIRE DU RÉSERVOIR : taux de capture des nœuds (bord 1,8 GeV,
     σ polarisabilité seule) par une naine brune sur l'âge de l'Univers,
     et chauffage associé — DOUBLE ABSENCE attendue : (i) capture ~nulle
     (σ_eq ~ 1e-48 cm², 7 décades sous les détecteurs) ; (ii) même capturé,
     un nœud du réservoir N'ANNIHILE PAS (relique symétrique,
     annihilation Vassiliev-écrasée) et ne se désintègre pas (τ>1e347 s).
     → Les sondes « chauffage DM des naines brunes / exoplanètes »
     (type Leane-Smirnov) doivent voir des objets STRICTEMENT froids :
     prédiction d'absence discriminante contre les WIMPs capturants.

  3. NOYAU DE PERLE (si l'objet s'est formé autour d'une perle — canal
     rare, cf. perle_germe_stellaire.py) : cuspide thermique autour du
     puits ponctuel M_p = 2e-3 Msun — profil adiabatique T(r) =
     T_c(1 + r_i/r), ρ(r) = ρ_c(1 + r_i/r)^{3/2} (gaz idéal, γ=5/3,
     déclaré ; la dégénérescence PLAFONNE la cuspide : coupure au rayon
     r_deg où T atteint T_F locale), luminosité pp intégrée de la cuspide
     vs luminosité radiée de l'objet → seuil d'allumage effectif décalé.
     Taux pp : forme standard ε_pp = ε0 ρ X² T9^{-2/3} exp(-3,381/T9^{1/3})
     (Angulo/NACRE, facteur d'écran négligé — ordre déclaré).

Entrées canoniques : M_p=2,03e-3 Msun ; m_res=1,8 GeV ; σ_eq=1,1e-48 cm²
(équivalent Xe, ramené par nucléon ~3e-52 déclaré) ; ρ_DM=0,4 GeV/cm³.
"""
from __future__ import annotations

import math
import numpy as np

G = 6.674e-11
KB = 1.380649e-23
HBAR = 1.054571817e-34
MH = 1.67262e-27
ME = 9.10938e-31
MSUN = 1.98892e30
LSUN = 3.828e26
GEV = 1.78266e-27
YR = 3.156e7

M_P = 2.03e-3 * MSUN
RHO_DM = 0.4 * GEV * 1e6
M_RES = 1.8 * GEV
SIGMA_NUC = 3e-56          # m² par nucléon H (1,1e-48 cm² éq. Xe / boost cohérent ~3851)
V_HALO = 220e3

print(__doc__)

# ================================================== 1. contrôle standard
# Polytrope n=3/2 : R(M) au minimum radius ~ dégénérescence ; on utilise les
# relations d'échelle standard pour T_c,max(M) (Burrows & Liebert 1993) :
#   T_c,max ≈ 2,0e6 K (M/0,06 Msun)^{~2.4 fenêtre 0,03-0,09} — on encode une
#   interpolation déclarée, ancrée sur T_c,max(0,075) = 3,0e6 K (allumage).
def tc_max(M_msun):
    return 3.0e6 * (M_msun / 0.075) ** 2.4

def rho_c_bd(M_msun):
    # ρ_c au maximum de T_c : ~ 1e5-1e6 kg/m³ dans la fenêtre BD (déclaré)
    return 8.0e5 * (M_msun / 0.075) ** 2

print("--- 1. Contrôle standard (échelles déclarées, Burrows-Liebert)")
for M in (0.03, 0.05, 0.075, 0.09):
    print(f"  M={M:5.3f} Msun : T_c,max={tc_max(M):.2e} K, ρ_c~{rho_c_bd(M):.1e} kg/m³"
          f"  {'→ allumage H' if tc_max(M) >= 3e6 else '→ naine brune (jamais allumée)'}")
print("  seuil standard M_min = 0,075 Msun (ancrage) ✓ ;"
      " la relecture TQNT n'y change rien (identité par construction)")

# ================================================== 2. capture DM
print("\n--- 2. Matière noire du réservoir : capture et chauffage")
M_BD, R_BD = 0.05 * MSUN, 7.0e7
v_esc = math.sqrt(2 * G * M_BD / R_BD)
focus = 1 + (v_esc / V_HALO) ** 2
n_dm = RHO_DM / M_RES
flux = n_dm * V_HALO
N_cross = math.pi * R_BD**2 * flux * focus
n_H = (M_BD / MH) / ((4/3) * math.pi * R_BD**3)
P_scatter = n_H * SIGMA_NUC * 2 * R_BD
# perte d'énergie par diffusion : recul max 4 m_H m_χ/(m_H+m_χ)² ~ 0,86 → mais
# UNE diffusion suffit rarement à capturer (v>v_esc) ; borne SUPÉRIEURE : toute
# diffusion capture.
N_capt_rate = N_cross * P_scatter
M_capt_10Gyr = N_capt_rate * M_RES * 10e9 * YR
print(f"  v_esc(0,05 Msun) = {v_esc/1e3:.0f} km/s ; focalisation ×{focus:.1f}")
print(f"  traversées : {N_cross:.2e} nœuds/s ; P(diffusion/traversée) = {P_scatter:.2e}")
print(f"  capture (borne sup) : {N_capt_rate:.2e} nœuds/s → {M_capt_10Gyr:.2e} kg en 10 Gyr"
      f" ({M_capt_10Gyr/M_BD:.1e} de la masse)")
# chauffage : MÊME au taux de capture borne sup, énergie déposée = KE des nœuds
L_dm_max = N_capt_rate * 0.5 * M_RES * V_HALO**2
L_wimp_ref = math.pi * R_BD**2 * RHO_DM * V_HALO**3 / 2 * focus   # borne géométrique (tout le flux thermalisé)
print(f"  chauffage TQNT (borne sup) : {L_dm_max:.2e} W = {L_dm_max/LSUN:.1e} Lsun")
print(f"  borne géométrique WIMP (capture totale, type Leane-Smirnov) : {L_wimp_ref:.2e} W"
      f" → ratio TQNT/WIMP-max = {L_dm_max/L_wimp_ref:.1e}")
print("  et PAS d'annihilation (relique symétrique, Vassiliev-écrasée), PAS de désintégration"
      " (τ>1e347 s) → DOUBLE ABSENCE : les vieilles naines brunes sont STRICTEMENT froides.")

# ================================================== 3. cuspide de perle
print("\n--- 3. Noyau de perle : cuspide thermique et allumage sous le seuil")
X_H = 0.75
def eps_pp(rho, T):     # W/kg (NACRE, écran négligé)
    T9 = T / 1e9
    return 2.4e-3 * rho * X_H**2 * T9**(-2/3) * math.exp(-3.381 / T9**(1/3))

def L_cusp(M_msun):
    """Luminosité pp de la cuspide autour de la perle, AUTO-ÉCRANTÉE.

    Le puits ponctuel ne commande le profil que tant que le gaz accumulé
    pèse moins que la perle : M_gaz(<r) ≤ M_p. En dessous du rayon de
    saturation r_sat, le boost gèle (le puits est rempli par son propre
    gaz) — c'est LE limiteur physique (sans lui, la cuspide adiabatique
    empile des masses de gaz >> M_p à des densités de naine blanche :
    non hydrostatique, bug de la v1 du 05/08 corrigé le jour même).
    Profil : b(r) = 1 + r_i/max(r, r_sat) ; T = T_c·b, ρ = ρ_c·b^{3/2}.
    """
    T_c, rho_c = tc_max(M_msun), rho_c_bd(M_msun)
    mu_i = 0.6
    r_i = G * M_P * mu_i * MH / (KB * T_c)          # rayon d'influence
    # saturation : (4π/3) ρ_c (r_i/r)^{3/2} r³ = M_p (régime r << r_i)
    r_sat = (3 * M_P / (4 * math.pi * rho_c * r_i ** 1.5)) ** (2.0 / 3.0)
    r_sat = min(r_sat, r_i)          # jamais au-delà du rayon d'influence
    b_max = 1 + r_i / max(r_sat, 2.11e4)
    def T_F(rho):
        n_e = rho / (1.14 * MH)
        return (HBAR**2 / (2 * ME * KB)) * (3 * math.pi**2 * n_e) ** (2/3)
    r = np.logspace(math.log10(2.11e4), math.log10(50 * r_i), 500)
    b = 1 + r_i / np.maximum(r, r_sat)
    T = T_c * b
    rho = rho_c * b ** 1.5
    TF = np.array([T_F(x) for x in rho])
    T_eff = np.minimum(T, np.maximum(T_c, 0.5 * TF))
    eps = np.array([eps_pp(x, t) for x, t in zip(rho, T_eff)])
    integrand = eps * rho * 4 * math.pi * r**2
    return float(np.trapezoid(integrand, r)), r_i, r_sat, b_max

print("  M [Msun]  T_c,max   boost×  L_pp(cuspide)   L_BD(radiée~)   verdict")
for M in (0.03, 0.04, 0.05, 0.055, 0.06, 0.065, 0.075):
    Lc, r_i, r_sat, b_max = L_cusp(M)
    L_bd = 1e-5 * LSUN * (M / 0.05) ** 2   # ordre radié d'une BD jeune (déclaré)
    verdict = "ALLUMAGE ENTRETENU" if Lc > L_bd else "cuspide sous-critique"
    print(f"  {M:6.3f}   {tc_max(M):.2e}  {b_max:5.2f}  {Lc:12.3e} W  {L_bd:12.3e} W"
          f"   {verdict}  (r_i={r_i/1e3:.0f} km, r_sat={r_sat/1e3:.0f} km)")
print("""
  Lecture : le critère est « L_pp(cuspide, profil figé) > L_radiée » ; les
  valeurs de L au-delà du seuil sont PRÉ-THERMOSTAT (un objet allumé se
  régule : chauffe → dilate → refroidit) et ne sont PAS des luminosités
  prédites. Le résultat structurel : la cuspide sature à la température
  de VIRIEL de la perle (T_vir = GM_p μm_H/k r ~ 1e7 K à r_sat), qui ne
  dépend PLUS de T_c de l'objet — le critère d'allumage devient « assez
  de gaz pour remplir le puits (M_p dans ~r_i) », satisfait à TOUTES les
  masses balayées : un noyau de perle tend à allumer l'hydrogène très
  au-dessous du seuil standard, possiblement jusqu'aux masses planétaires.
  Signature : « étoiles » sous 0,075 Msun / naines ultra-froides
  sur-lumineuses — fossiles doublement rares (canal de formation mort
  dans le disque actuel, cf. perle_germe_stellaire.py).
  RÉSERVES (ordre déclaré) : profil adiabatique idéal, dégénérescence
  traitée par plafond grossier (0,5·T_F — les cœurs BD sont déjà
  dégénérés, ψ = T_F/T_c ~ 3-8 : le chauffage de compression réel est
  moindre), écran coulombien négligé, pas de thermostat ni d'évolution —
  un modèle stellaire complet est LA cible avant toute consolidation.""")
