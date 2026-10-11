"""Le passage de Voyager dans le plasma VLISM — la température autour de la sonde.

Simulation au niveau CINÉTIQUE (le bon régime) : libre parcours moyen
coulombien ≫ taille de la sonde → pas de choc fluide, pas de couche de
stagnation ; chaque espèce garde SA température — « la température autour
de Voyager » est un bouquet, pas un nombre. Étapes :
  1. régime d'écoulement (Mach sonique/Alfvén/magnétosonique) ;
  2. gaine électrostatique (Debye, potentiel flottant) ;
  3. bombardement des faces (bélier vs sillage) — température de surface
     équivalente vue par la face avant ;
  4. la bulle infrarouge des RTG : le SEUL chauffage réel du voisinage —
     profil T_rad(r) jusqu'au raccord CMB ;
  5. les composantes qui ne se couplent pas : nœuds sombres (température
     cinétique équivalente ~5e6 K, échange nul), trame (Unruh ~1e-27 K).
Entrées : inventaire VLISM (plasma_vlism_tqnt.py), Voyager 1 en 2026
(v=17 km/s, vent interstellaire 26 km/s, RTG ~4,9 kW thermiques restants).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / 'output'
k, mp, me, e = 1.381e-23, 1.673e-27, 9.109e-31, 1.602e-19
sigma_SB = 5.67e-8

# --- milieu (inventaire VLISM) et cinématique
n_e, n_H = 0.09e6, 0.15e6            # /m³
T_pl, T_H = 3.5e4, 7.5e3             # K (plasma ; H neutre)
B = 0.5e-9                            # T
v_sc, v_wind = 17e3, 26e3            # m/s (sonde ; vent interstellaire)
ang = np.radians(35)                  # V1 ~35° du nez
v_rel = np.sqrt(v_sc**2 + v_wind**2 + 2*v_sc*v_wind*np.cos(ang))
print(f'vitesse relative sonde-plasma : {v_rel/1e3:.0f} km/s')

# 1. régime d'écoulement
c_s = np.sqrt(5/3 * k*(2*T_pl) / mp)             # e- + p
rho_i = n_e * mp
v_A = B / np.sqrt(4e-7*np.pi * rho_i)
v_ms = np.hypot(c_s, v_A)
print(f'c_son = {c_s/1e3:.0f} km/s ; v_Alfvén = {v_A/1e3:.0f} km/s ; magnétosonique = {v_ms/1e3:.0f} km/s')
print(f'Mach magnétosonique = {v_rel/v_ms:.2f} → SUB-magnétosonique : PAS de choc d étrave')
# libre parcours moyen coulombien (protons, ~35 000 K)
lnL = 25
lam_mfp = 3.4e11 * (T_pl/1e4)**2 / (n_e/1e6) / lnL   # m (formule NRL approx)
print(f'libre parcours coulombien ~ {lam_mfp:.1e} m ({lam_mfp/1.5e11:.1f} UA) ≫ sonde (4 m) → SANS COLLISION')

# 2. gaine
lam_D = np.sqrt(8.854e-12 * k*T_pl / (n_e * e**2))
V_f = -k*T_pl/e * 0.5*np.log(mp/(2*np.pi*me))
print(f'longueur de Debye = {lam_D:.0f} m (≫ sonde) ; potentiel flottant ≈ {V_f:.1f} V')

# 3. bombardement des faces
E_ram = 0.5*mp*v_rel**2 / e                       # eV par proton
T_ram_eq = (2/3) * 0.5*mp*v_rel**2 / k
flux_ram = n_e * v_rel
print(f'face bélier : {E_ram:.1f} eV/proton (T équivalente {T_ram_eq:.0f} K) ; flux {flux_ram:.1e} /m²/s')
print(f'sillage : raréfaction ~exp(-(v_rel/v_th)²) ; v_th(p) = {np.sqrt(2*k*T_pl/mp)/1e3:.0f} km/s → cavité partielle')
# puissance déposée par le plasma sur ~10 m² : négligeable devant les RTG
P_plasma = flux_ram * E_ram*e * 10
print(f'puissance plasma interceptée (~10 m²) : {P_plasma:.1e} W (RTG : ~4 900 W)')

# 4. la bulle infrarouge des RTG
P_rtg = 7200 * 0.5**(49/87.7)                    # W thermiques restants (3× MHW-RTG)
r = np.logspace(0.3, 4.2, 300)                   # 2 m → 16 km
T_rad = (P_rtg/(16*np.pi*sigma_SB*r**2) + 2.725**4)**0.25
r_cmb = np.sqrt(P_rtg/(16*np.pi*sigma_SB*2.725**4))
print(f'RTG 2026 : {P_rtg:.0f} W thermiques → bulle IR : T(5 m)={T_rad[np.argmin(abs(r-5))]:.0f} K, '
      f'T(100 m)={T_rad[np.argmin(abs(r-100))]:.0f} K ; raccord CMB à r = {r_cmb/1e3:.1f} km')

# 5. les spectateurs
E_dark = 0.5*1.9e9*(250e3/3e8)**2                # eV
print(f'nœuds sombres : T cinétique équivalente {2/3*E_dark*e/k:.1e} K — échange NUL (σ_eq)')
a_grav = 6.674e-11*1.989e30/(165*1.496e11)**2
print(f'trame (Unruh, a_grav={a_grav:.1e} m/s²) : T = {1.055e-34*a_grav/(2*np.pi*3e8*k):.1e} K')

# figure
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.loglog(r, T_rad, '-', color='#b91c1c', lw=1.8, label='bulle IR des RTG (photons)')
for T, lab, c in [(T_pl, 'plasma e⁻/p ambiant (~35 000 K, non perturbé : sans collision)', '#1f4e79'),
                  (T_ram_eq, f'bombardement face bélier ({E_ram:.1f} eV/p équiv.)', '#7c3aed'),
                  (T_H, 'hydrogène neutre (~7 500 K)', '#059669'),
                  (2.725, 'CMB (2,7 K)', '#f59e0b')]:
    ax.axhline(T, ls='--', lw=1, color=c, label=lab)
ax.axvline(lam_D, color='#666', ls=':', lw=1, label=f'gaine de Debye ({lam_D:.0f} m)')
ax.set_xlabel('distance à la sonde (m)'); ax.set_ylabel('température (K)')
ax.set_title(f'Autour de Voyager 1 (2026) : un bouquet de températures, pas un nombre\n'
             f'(sub-magnétosonique M={v_rel/v_ms:.2f}, sans collision — chaque espèce garde la sienne)')
ax.legend(fontsize=7.5, loc='lower left'); ax.set_xlim(2, 2e4); ax.set_ylim(1, 3e5)
fig.tight_layout(); fig.savefig(OUT/'voyager_temperature_autour.png', dpi=140)
print('figure :', OUT/'voyager_temperature_autour.png')
