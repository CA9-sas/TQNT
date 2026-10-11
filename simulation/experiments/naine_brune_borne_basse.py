"""Naine brune à noyau de perle — borne basse, stabilité thermique, population.

Complète naine_brune_modele_complet.py (les trois ouverts nommés au §3bis) :

  A. BORNE BASSE : à masse totale FIXÉE (échelle 0,005 → 0,065 Msun), on
     apparie la densité de bord interne lr0 par balayage+interpolation pour
     chaque entropie s, et on prend max_s L_nuc → allumé si ≥ L_crit(M)
     (même calibration différentielle : L_crit ∝ M^2,2 ancré au contrôle).
  B. STABILITÉ THERMIQUE : à M fixé, courbe L_nuc(s) à masse constante ;
     l'équilibre L_nuc = L_crit est un THERMOSTAT si dL_nuc/ds < 0 au
     croisement (chauffage → s monte → si L_nuc décroît, ça se régule ;
     s'il croît : flash/instable). VALIDATION du critère sur le contrôle
     0,09 Msun (bas de séquence principale : doit sortir STABLE).
  C. POPULATION : budgets disponibles — non baryonique (nombre de perles
     au plafond f=1 %) vs baryonique (gaz+étoiles MW à ~0,05 Msun par
     objet) ; le facteur limitant et la fourchette honnête via
     l'efficacité de germination ε (canal disque : mesuré quasi nul ;
     canal primordial co-mobile : ouvert, borne à poser).
"""
from __future__ import annotations

import math
import numpy as np

from simulation.experiments.naine_brune_modele_complet import (
    structure, eos, t_adiabat, LT, MSUN, LSUN, M_P)

# ================================================== outils masse fixée
def M_of(s, lr0):
    lt0 = t_adiabat(lr0, s)
    if lt0 >= LT[-1] - 0.02:
        return None
    M, R, L = _memo_structure(10**lr0, 10**lt0, True)
    return M, L

def match_mass(s, M_target, lr_lo=5.2, lr_hi=8.6, n=6):
    """Trouve lr0 tel que M_tot ≈ M_target sur l'adiabate s (scan+interp)."""
    grid = []
    for lr0 in np.linspace(lr_lo, lr_hi, n):
        out = M_of(s, lr0)
        if out:
            grid.append((lr0, out[0], out[1]))
    if len(grid) < 2:
        return None
    ms = np.array([g[1] for g in grid])
    if not (ms.min() <= M_target <= ms.max()):
        return None
    # interpolation sur la branche croissante englobante
    for k in range(len(grid) - 1):
        m1, m2 = grid[k][1], grid[k + 1][1]
        if (m1 - M_target) * (m2 - M_target) <= 0:
            lo, hi = grid[k][0], grid[k + 1][0]
            for _ in range(4):          # 4 bissections (±4 %)
                mid = 0.5 * (lo + hi)
                out = M_of(s, mid)
                if out is None:
                    break
                if (out[0] - M_target) * (m1 - M_target) > 0:
                    lo = mid
                else:
                    hi = mid
            out = M_of(s, 0.5 * (lo + hi))
            if out and abs(out[0] - M_target) / M_target < 0.08:
                return 0.5 * (lo + hi), out[0], out[1]
    return None

# entropies candidates : gamme élargie vers le froid (petits objets)
_S_ALL = sorted({round(eos(10**lr, 10**lt)[1], -2)
                 for lr in np.linspace(4.0, 6.8, 7)
                 for lt in np.linspace(5.0, 6.8, 8)})
S_SET = [_S_ALL[int(q * (len(_S_ALL) - 1))] for q in np.linspace(0, 1, 20)]
S_SET = sorted(set(S_SET))

_MEMO = {}
def _memo_structure(rho0, T0, pearl):
    key = (round(math.log10(rho0), 3), round(math.log10(T0), 3), pearl)
    if key not in _MEMO:
        _MEMO[key] = structure(rho0, T0, pearl)
    return _MEMO[key]

# calibration différentielle identique au modèle complet :
# L_crit(0,075) = L_nuc,max du contrôle = 1,94e-5 Lsun (mesuré, d0cbc88)
A_CAL = 1.94e-5 * LSUN
def L_crit(M):
    return A_CAL * (M / (0.075 * MSUN)) ** 2.2

print(__doc__)
print(f"entropies balayées : {len(S_SET)} valeurs ; "
      f"L_crit(0,075) = {A_CAL/LSUN:.2e} Lsun (calibration du modèle complet)")

# ================================================== A. borne basse
# nuage de points dense (s × densité de bord) puis fenêtres de masse ±10 % —
# l'appariement de masse par bissection (v1) ratait des barreaux couverts par
# le modèle complet (lacunes → faux « éteint ») : supplanté par le nuage.
print("\n===== A. CARTOGRAPHIE DE LA BORNE BASSE (noyau de perle) =====")
CLOUD = []
for s in S_SET:
    for lr0 in np.linspace(5.0, 8.6, 13):
        out = M_of(s, lr0)
        if out and 0.003 * MSUN < out[0] < 0.12 * MSUN:
            CLOUD.append((s, lr0, out[0], out[1]))
print(f"  nuage : {len(CLOUD)} structures dans [0,003, 0,12] Msun")
print("  M[Msun]   points   L_max/L_crit   verdict")
LADDER = [0.005, 0.008, 0.012, 0.018, 0.025, 0.035, 0.050, 0.065]
results = {}
for Mt in LADDER:
    sel = [c for c in CLOUD if abs(c[2] - Mt * MSUN) / (Mt * MSUN) < 0.10]
    best = max((c[3] for c in sel), default=0.0)
    ratio = best / L_crit(Mt * MSUN)
    results[Mt] = (ratio, len(sel))
    print(f"  {Mt:6.3f}   {len(sel):5d}   {ratio:12.3f}   "
          f"{'ALLUMÉ' if ratio >= 1 else ('éteint' if sel else 'NON COUVERT')}")
results = {m: r for m, (r, n) in results.items() if n > 0}

lit = [m for m, r in results.items() if r >= 1]
if lit:
    print(f"\n  BORNE BASSE (cette grille) : allumage jusqu'à M = {min(lit):.3f} Msun"
          f" ; premier barreau éteint : "
          f"{max([m for m in LADDER if m < min(lit)], default=float('nan'))}")

# ================================================== B. stabilité thermique
print("\n===== B. STABILITÉ THERMIQUE (signe de dL_nuc/ds à M fixé) =====")
def stability(Mt, pearl):
    pts = []
    for s in S_SET:
        if pearl:
            sel = [c for c in CLOUD
                   if c[0] == s and abs(c[2] - Mt * MSUN) / (Mt * MSUN) < 0.10]
            if sel:
                cbest = min(sel, key=lambda c: abs(c[2] - Mt * MSUN))
                pts.append((s, cbest[3]))
        else:
            # contrôle : à M fixé, apparier ρ_c pour chaque s via T_adiabate
            for lrho in np.linspace(4.2, 6.8, 9):
                lt = t_adiabat(lrho, s)
                if lt >= LT[-1] - 0.02:
                    continue
                M, R, L = structure(10**lrho, 10**lt, False)
                if abs(M - Mt * MSUN) / (Mt * MSUN) < 0.06:
                    pts.append((s, L))
                    break
    pts.sort()
    return pts

for label, Mt, pearl in [("contrôle 0,090 (validation : doit être STABLE)", 0.090, False),
                          ("perle 0,050", 0.050, True),
                          ("perle 0,025", 0.025, True)]:
    pts = stability(Mt, pearl)
    if len(pts) < 3:
        print(f"  {label} : couverture insuffisante ({len(pts)} points)")
        continue
    s_arr = np.array([p[0] for p in pts])
    L_arr = np.array([p[1] for p in pts])
    Lc = L_crit(Mt * MSUN)
    # croisement L_nuc = L_crit et pente locale
    verdict = "pas de croisement dans la fenêtre"
    for k in range(len(pts) - 1):
        if (L_arr[k] - Lc) * (L_arr[k + 1] - Lc) <= 0 and L_arr[k] != L_arr[k + 1]:
            slope = (L_arr[k + 1] - L_arr[k]) / (s_arr[k + 1] - s_arr[k])
            verdict = (f"équilibre à s≈{0.5*(s_arr[k]+s_arr[k+1]):.0f} ; "
                       f"dL/ds {'< 0 → THERMOSTAT STABLE' if slope < 0 else '> 0 → INSTABLE (flash)'}")
            break
    else:
        if L_arr.min() > Lc:
            slope = np.polyfit(s_arr, L_arr, 1)[0]
            verdict = (f"L_nuc > L_crit partout (×{L_arr.min()/Lc:.0f} à ×{L_arr.max()/Lc:.0f}) ; "
                       f"pente globale dL/ds {'< 0 → se régule en montant en s' if slope < 0 else '> 0 → ATTENTION'}")
    print(f"  {label} : {len(pts)} points ; {verdict}")

# ================================================== C. population
print("\n===== C. POPULATION : budgets disponibles =====")
F_MAX = 0.01
M_HALO_DM = 1.0e12 * MSUN
M_BARY_MW = 6.0e10 * MSUN          # gaz + étoiles
M_STAR_TYP = 0.05 * MSUN
N_pearls = F_MAX * M_HALO_DM / M_P
N_bary_cap = M_BARY_MW / M_STAR_TYP
print(f"  budget NON BARYONIQUE : N_perles(halo MW, plafond f=1 %) = {N_pearls:.1e}")
print(f"  budget BARYONIQUE : {M_BARY_MW/MSUN:.1e} Msun / {M_STAR_TYP/MSUN} Msun par objet"
      f" = {N_bary_cap:.1e} objets possibles")
print(f"  → facteur limitant : {'les PERLES' if N_pearls < N_bary_cap else 'les BARYONS'}"
      f" (ratio perles/baryons-cap = {N_pearls/N_bary_cap:.2f})")
eps_disk = 4e-3 / N_pearls          # 4e-3 événement/10 Gyr (perle_germe_stellaire)
print(f"  efficacité du canal DISQUE (mesurée) : ε ≤ {eps_disk:.1e} → population tardive ~0")
print(f"  population d'étoiles-perle : N = ε_primordial × {N_pearls:.1e} —"
      f" ε_primordial (jeune Univers co-mobile) N'EST PAS calculé : cible nommée ;")
print(f"  borne de cohérence baryonique : ε_primordial ≤ {N_bary_cap/N_pearls:.2f}"
      f" (au-delà, plus de baryons que la Galaxie n'en contient)")
print("  contre-borne observationnelle (à poser) : les étoiles-perle brûlent >t_H"
      " (réservoir 0,04 Msun à ~1e-5 Lsun ~ 1e14 ans) → toutes encore allumées ;"
      " un recensement des naines « trop chaudes pour leur masse » (Gaia+WISE)"
      " bornerait ε_primordial directement.")
