"""Naine brune à noyau de perle — gravothermique à potentiel mixte + ε_prim.

Les deux verrous nommés au §3ter (2026-08-05 soir) :

  A. GRAVOTHERMIQUE À POTENTIEL MIXTE. L'objet est ENTIÈREMENT convectif
     (hypothèse du modèle) : la chaleur déposée dans la cuspide est
     redistribuée par convection en τ_conv ≪ τ_thermique — la stabilité
     est donc GLOBALE : équilibre L_nuc(s;M) = L_surf(s;M), stable ssi
       d(L_nuc − L_surf)/ds < 0  à M exactement fixé.
     Le test naïf du §3ter échouait par levier de masse (fenêtres ±6-10 %) :
     ici la masse est appariée à ±0,5 % (bissection serrée, mémoïsée) —
     c'est LE correctif. L_surf : Hayashi (T_eff ≈ 2800 K constant,
     déclaré) → L_surf ∝ R²(s), pente mesurée sur les structures.
     VALIDATION : le contrôle 0,09 Msun sans perle doit sortir STABLE.
     + contrôle LOCAL : τ_conv(cuspide) vs τ_chauffe(cuspide) — si
     τ_conv ≪ τ_chauffe, le flash local (mécanisme dégénéré) est
     court-circuité par le couplage convectif et l'analyse globale vaut.

  B. ε_prim : l'efficacité du canal primordial co-mobile, décomposée en
     facteurs déclarés :
       ε_prim = P(perle dans le cœur en effondrement) × P(liée | dedans)
                × f_lowmass(objet final de faible masse)
     avec un mini-halo canonique (z=20, M_h=1e6 Msun, centre NFW ×c_boost,
     cœur Pop III r_core~1 pc, M_core~1000 Msun) et la vitesse de
     streaming baryons-DM v_bc(z) = 30·(1+z)/1090 km/s. Le troisième
     facteur (IMF Pop III) reste OUVERT — le résultat est donné en
     N_MW = g(paramètres) × f_lowmass, bornes déclarées.
"""
from __future__ import annotations

import math
import numpy as np

from simulation.experiments.naine_brune_modele_complet import (
    structure, eos, t_adiabat, LT, MSUN, LSUN, M_P, G)

KB = 1.380649e-23
MH = 1.67262192e-27
SIGMA_SB = 5.670374e-8
PC = 3.0857e16

print(__doc__)

# =========================================================== A. gravothermique
_MEMO = {}
def run(lr0, s, pearl):
    lt0 = t_adiabat(lr0, s)
    if lt0 >= LT[-1] - 0.02 or lt0 <= LT[0] + 0.02:
        return None
    key = (round(lr0, 4), round(s, 1), pearl)
    if key not in _MEMO:
        _MEMO[key] = structure(10**lr0, 10**lt0, pearl)
    return _MEMO[key]

def M_exact(s, M_target, pearl, tol=0.01):
    """Appariement de masse SERRÉ : scan dense puis bissection (±0,5 %)."""
    lo_hi = (5.0, 8.6) if pearl else (3.8, 7.0)
    grid = []
    for lr0 in np.linspace(*lo_hi, 11):
        out = run(lr0, s, pearl)
        if out:
            grid.append((lr0, out[0], out[2], out[1]))
    for k in range(len(grid) - 1):
        m1, m2 = grid[k][1], grid[k + 1][1]
        if (m1 - M_target) * (m2 - M_target) <= 0:
            lo, hi = grid[k][0], grid[k + 1][0]
            f_lo = m1 - M_target
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                out = run(mid, s, pearl)
                if out is None:
                    return None
                if (out[0] - M_target) * f_lo > 0:
                    lo = mid
                else:
                    hi = mid
            out = run(0.5 * (lo + hi), s, pearl)
            if out and abs(out[0] - M_target) / M_target < tol:
                return out    # (M, R, L)
    return None

def gravothermal(Mt_msun, pearl, label):
    Mt = Mt_msun * MSUN
    # gamme d'entropies : celles où la masse cible existe
    _s = sorted({round(eos(10**lr, 10**lt)[1], -2)
                 for lr in np.linspace(4.0, 6.8, 7)
                 for lt in np.linspace(5.2, 6.8, 7)})
    s_all = sorted({_s[int(q * (len(_s) - 1))] for q in np.linspace(0, 1, 12)})
    pts = []
    for s in s_all:
        out = M_exact(s, Mt, pearl)
        if out:
            pts.append((s, out[2], out[1]))     # (s, L_nuc, R)
    pts.sort()
    print(f"\n  [{label}] M = {Mt_msun} Msun : {len(pts)} structures à ±0,5 %")
    if len(pts) < 4:
        print("    couverture insuffisante — pas de verdict")
        return
    s_a = np.array([p[0] for p in pts])
    L_a = np.array([max(p[1], 1e-30) for p in pts])
    R_a = np.array([p[2] for p in pts])
    L_surf = 4 * math.pi * R_a**2 * SIGMA_SB * 2800.0**4
    # équilibre L_nuc = L_surf ?
    diff = np.log(L_a) - np.log(L_surf)
    for k in range(len(pts) - 1):
        if diff[k] * diff[k + 1] <= 0:
            dLn = (math.log(L_a[k+1]) - math.log(L_a[k])) / (s_a[k+1] - s_a[k])
            dLs = (math.log(L_surf[k+1]) - math.log(L_surf[k])) / (s_a[k+1] - s_a[k])
            stable = dLn < dLs
            print(f"    équilibre L_nuc=L_surf à s≈{0.5*(s_a[k]+s_a[k+1]):.0f} ; "
                  f"dlnL_nuc/ds={dLn:.2e}, dlnL_surf/ds={dLs:.2e} → "
                  f"{'THERMOSTAT STABLE' if stable else 'INSTABLE (flash)'}")
            return stable
    # pas de croisement : rapporter les pentes
    dLn = np.polyfit(s_a, np.log(L_a), 1)[0]
    dLs = np.polyfit(s_a, np.log(L_surf), 1)[0]
    side = "L_nuc>L_surf partout" if diff.min() > 0 else "L_nuc<L_surf partout"
    print(f"    {side} ; pentes globales dlnL_nuc/ds={dLn:.2e}, "
          f"dlnL_surf/ds={dLs:.2e} → "
          f"{'converge vers équilibre STABLE' if dLn < dLs else 'ATTENTION : pente nucléaire > pente de surface'}")
    return dLn < dLs

print("===== A. GRAVOTHERMIQUE (critère global, masse à ±0,5 %) =====")
ok_ctrl = gravothermal(0.090, False, "CONTRÔLE sans perle — attendu STABLE")
ok_p05 = gravothermal(0.050, True, "perle")
ok_p03 = gravothermal(0.030, True, "perle")

# LECTURE DES VERDICTS (ajoutée après le run du 05/08) : le contrôle 0,09
# (empiriquement stable) sort « instable » sous ce critère → la photosphère à
# T_eff FIGÉ (Hayashi) ne porte pas la réponse de refroidissement T_eff(s) :
# le critère ABSOLU n'est pas validable sans vraie atmosphère (cible).
# CE QUI TIENT : l'ARGUMENT DIFFÉRENTIEL — dlnL_nuc/ds(perle 0,05/0,03) =
# 4,4e-4/1,8e-4 < 8,1e-4 (contrôle) : à réponse de photosphère identique,
# la stabilité observée du bas de séquence principale IMPLIQUE celle des
# étoiles-perle (pentes nucléaires plus douces). Conditionnel déclaré.
print("\n  [lecture] contrôle stable connu sort « instable » → critère absolu"
      " NON VALIDÉ (photosphère T_eff figé) ; ARGUMENT DIFFÉRENTIEL : pentes"
      " nucléaires perle (0,4-0,5×) plus douces que le contrôle empiriquement"
      " stable → au moins aussi stables, conditionnel à la photosphère commune.")

# contrôle local : couplage convectif dans la cuspide (0,05 Msun perle)
print("\n  [contrôle local] couplage convectif de la cuspide (0,05 Msun, perle)")
# ordres : L_cusp ~ 1e22 W (près du seuil, thermostaté) ; r_i ~ 1e7 m ;
# ρ_cusp ~ 1e6 kg/m³ ; v_conv (MLT) ~ (F/ρ)^(1/3) = (L/4πr²ρ)^(1/3)
L_c, r_i, rho_c = 1e22, 1.0e7, 1e6
v_conv = (L_c / (4 * math.pi * r_i**2 * rho_c)) ** (1.0 / 3.0)
tau_conv = r_i / v_conv
cP_T = 2.5 * KB * 3e6 / (0.6 * MH)          # ~c_P·T par kg (gaz ionisé, ordre)
tau_heat = (rho_c * (4/3) * math.pi * r_i**3 * cP_T) / L_c
print(f"    v_conv(MLT) ~ {v_conv:.1f} m/s ; τ_conv ~ {tau_conv/3.156e7:.1f} an ;"
      f" τ_chauffe ~ {tau_heat/3.156e7:.1e} ans → rapport {tau_heat/tau_conv:.1e}")
print("    τ_conv ≪ τ_chauffe : la convection évacue avant tout emballement local"
      " — le flash dégénéré local est court-circuité, l'analyse GLOBALE vaut.")

# =========================================================== B. ε_prim
print("\n===== B. ε_prim : le canal primordial co-mobile, décomposé =====")
Z = 20.0
V_BC = 30e3 * (1 + Z) / 1091                 # streaming baryon-DM (m/s)
M_HALO = 1e6 * MSUN
OM_M, RHO_CRIT0 = 0.31, 8.6e-27
rho_m = OM_M * RHO_CRIT0 * (1 + Z) ** 3
rho_vir = 200 * rho_m
R_vir = (3 * M_HALO / (4 * math.pi * rho_vir)) ** (1 / 3)
v_vir = math.sqrt(G * M_HALO / R_vir)
print(f"  mini-halo canonique : M_h = 1e6 Msun à z = {Z:.0f} → R_vir = {R_vir/PC:.0f} pc,"
      f" v_vir = {v_vir/1e3:.1f} km/s ; v_bc(z) = {V_BC/1e3:.2f} km/s (co-mobile ✓)")

C_BOOST = 300.0                              # centre NFW + contraction (déclaré, ±1 dex)
rho_dm_center = C_BOOST * rho_vir
F_PEARL = 0.01
n_pearl = F_PEARL * rho_dm_center / M_P
R_CORE, M_CORE = 1.0 * PC, 1000 * MSUN       # cœur Pop III (déclaré)
V_core = (4 / 3) * math.pi * R_CORE**3
P_inside = n_pearl * V_core                  # perles attendues dans le cœur
v_esc_core = math.sqrt(2 * G * M_CORE / R_CORE)
# fraction liée : maxwellienne v_vir (3D), P(v < v_esc_core)
x = v_esc_core / (v_vir / math.sqrt(3))      # v_esc / sigma_1d
P_bound = math.erf(x / math.sqrt(2)) - math.sqrt(2 / math.pi) * x * math.exp(-x**2 / 2)
print(f"  centre : ρ_DM×{C_BOOST:.0f} → n_perle = {n_pearl:.2e} m⁻³ ;"
      f" cœur (1 pc, 1000 Msun) : ⟨N_perles dedans⟩ = {P_inside:.2f}")
print(f"  v_esc(cœur) = {v_esc_core/1e3:.1f} km/s vs σ_1d = {v_vir/math.sqrt(3)/1e3:.1f}"
      f" → P(liée) = {P_bound:.2f}")

N_pearls_per_halo = F_PEARL * M_HALO / M_P
eps_per_halo = P_inside * P_bound / N_pearls_per_halo
print(f"  perles par mini-halo : {N_pearls_per_halo:.1e} →"
      f" ε_prim(structure) = {eps_per_halo:.1e} × f_lowmass")

N_PEARLS_MW = F_PEARL * 1e12 * MSUN / M_P
N_MINIHALO_MW = 1e6                          # progéniteurs Pop III de la MW (déclaré ±1 dex)
N_star_pearl = N_MINIHALO_MW * P_inside * P_bound
print(f"\n  ASSEMBLAGE MW ({N_MINIHALO_MW:.0e} mini-halos, déclaré ±1 dex) :")
print(f"  objets nés avec perle ~ {N_star_pearl:.1e} × f_lowmass")
print(f"  → ε_prim(global MW) = {N_star_pearl/N_PEARLS_MW:.1e} × f_lowmass"
      f"  (cohérence ≤ 0,24 ✓ par {0.24/(N_star_pearl/N_PEARLS_MW):.0e})")
print(f"""
  LE FACTEUR OUVERT : f_lowmass — la fraction des captures qui finissent
  dans un objet de FAIBLE masse (0,02-0,075 Msun). L'IMF Pop III est
  top-heavy : la voie principale (perle dans un cœur massif) finit
  AVALÉE par une étoile massive → perle léguée au reste compact
  (corollaire : des perles dans les trous noirs stellaires primordiaux,
  sans signature). f_lowmass vient de la fragmentation des disques
  Pop III (mini-fragments éjectés) : littérature 0,01-0,3 par cœur,
  NON tranché → fourchette honnête :
    N_MW(étoiles-perle) ~ {N_star_pearl:.0e} × [0,01-0,3] = {N_star_pearl*0.01:.0e} à {N_star_pearl*0.3:.0e} objets
  — toutes encore allumées (t_burn ~ 1e14 ans), toutes sous la masse
  limite : le recensement Gaia+WISE des naines « trop chaudes pour leur
  masse » teste directement cette fourchette.
  Réserves : c_boost ±1 dex, N_minihalos ±1 dex, (R,M)_cœur déclarés —
  la fourchette porte ±2 dex ; f=1 % est un PLAFOND (limites 04/08).""")
