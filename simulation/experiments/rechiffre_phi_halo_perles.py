"""Re-chiffrage de φ_pred par composante — halo galactique vs cosmologique.

La machinerie du 15/07 (f3_lines_acpr_no_smear) ne calculait que la MOYENNE
DE HUBBLE : φ_iso = 2N_univ/(4π d_H² t_H) — un flux isotrope. Deux audits :

(1) MORPHOLOGIE : les perles de Classe A suivent le halo (n_perle ∝ ρ_NFW,
    signal de type DÉSINTÉGRATION : taux interne par perle, ∝ ρ, pas ρ²).
    La composante halo dans un ROI se calcule par le facteur D :
       φ_halo(ROI) = (L_perle f / 4π M_perle) · D_ROI,
       D_ROI = ∫_ROI dΩ ∫ ρ_NFW ds.
(2) SPECTRE : la composante cosmologique est décalée vers le rouge — les
    raies émises à z s'empilent en CONTINUUM sous E₀ ; seule la composante
    LOCALE (halo) reste une RAIE. La line search ne teste donc QUE le halo :
    φ_halo est LA prédiction de raie, φ_iso n'en a jamais été une.

Ingrédients repris à l'identique de la chaîne du 15/07 (Γ par perle,
fractions de Classe A et par état, M_perle) ; halo NFW r_s=20 kpc normalisé
à ρ_local = 0,4 GeV/cm³ en R0 = 8,2 kpc.
"""
import math
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'V5_pearl_internal'))
sys.path.insert(0, str(ROOT / 'scripts' / 'V5_Fermi_LAT'))

from theory.dm_pearl_internal import (M_PEARL_TYPICAL_MSUN, M_SUN_KG,
                                      R_HUBBLE_M, T_HUBBLE_S,
                                      entanglement_rate,
                                      n_events_over_universe_age)
import class_E_entanglement_test as CE
from class_E_six_lines_predictions import OMEGA_PER_AMPHI

DC = 22
GEV_PER_KG = 5.60959e26
CM_PER_KPC = 3.0857e21

# --- taux par perle et fractions, EXACTEMENT comme f3 (brique m/n, état 6_3+8_3)
m_canon, n_briques = 1.221, 2          # la raie 1,22 GeV (état 6_3+8_3)
m_brique = m_canon / n_briques
rate = entanglement_rate(M_PEARL_TYPICAL_MSUN, m_brique, DC)
L_perle = 2.0 * rate['Gamma_total_pearl_Hz']          # photons/s/perle
f_state = OMEGA_PER_AMPHI / CE.OMEGA_A_FRACTION       # part de l'état dans la Classe A
f_A = 0.05 * 0.265 / 0.265                            # Classe A = 5 % de la DM
M_perle_GeV = M_PEARL_TYPICAL_MSUN * M_SUN_KG * GEV_PER_KG

# --- contrôle : reproduire φ_iso du 15/07
nev = n_events_over_universe_age(M_PEARL_TYPICAL_MSUN, m_brique, DC)
N_univ = nev['N_events_total_universe'] * f_state
d_H_cm = R_HUBBLE_M * 100.0
phi_iso = N_univ * 2.0 / (4.0 * math.pi * d_H_cm**2 * T_HUBBLE_S)
print(f'contrôle φ_iso (formule 15/07)      : {phi_iso:.2e} ph/cm²/s (attendu 2,2e-10)')

# --- facteur D du halo (NFW, ρ_local = 0,4 GeV/cm³ à R0)
rs_kpc, R0_kpc, rho_loc = 20.0, 8.2, 0.4
def rho_nfw(r_kpc):
    x = r_kpc / rs_kpc
    x0 = R0_kpc / rs_kpc
    norm = rho_loc * x0 * (1 + x0)**2
    return norm / (x * (1 + x)**2)
def D_factor(psi_max_deg, n_psi=400, n_s=2000):
    psis = np.linspace(1e-3, np.radians(psi_max_deg), n_psi)
    s = np.linspace(1e-3, 100.0, n_s)                 # kpc
    D = 0.0
    for i, psi in enumerate(psis):
        r = np.sqrt(R0_kpc**2 + s**2 - 2*R0_kpc*s*np.cos(psi))
        col = np.trapz(rho_nfw(np.maximum(r, 0.05)), s) * CM_PER_KPC   # GeV/cm²
        dOmega = 2*np.pi*np.sin(psi)*(psis[1]-psis[0])
        D += col * dOmega
    return D                                          # GeV/cm² · sr

D16 = D_factor(16.0)
Omega16 = 2*np.pi*(1-np.cos(np.radians(16)))
print(f'facteur D (ROI 16°)                 : {D16:.2e} GeV/cm²·sr (Ω={Omega16:.3f} sr)')

# --- composante halo dans le ROI
phi_halo = (L_perle * f_state * f_A / (4*np.pi * M_perle_GeV)) * D16
print(f'φ_halo (raie VRAIE, ROI 16°)        : {phi_halo:.2e} ph/cm²/s')

# --- composante cosmologique dans le ROI (pour mémoire : PAS une raie — redshift)
phi_iso_roi = phi_iso * Omega16 / (4*np.pi)
print(f'φ_iso dans le ROI (continuum décalé): {phi_iso_roi:.2e} ph/cm²/s')
print(f'rapport halo/iso dans le ROI        : ×{phi_halo/max(phi_iso_roi,1e-99):.1f}')

# --- confrontation
print()
print('=== Confrontation (raie 1,22 GeV) ===')
print(f'  prédiction de RAIE (halo, ρ)      : {phi_halo:.2e} ph/cm²/s')
print(f'  UL95 NFW² (ρ², non pertinente)    : 1,6e-10')
print(f'  UL95 pow1 (ρ, la pertinente)      : voir scan_pow1 en cours')
print()
print('AUDIT SPECTRAL : la composante cosmologique (φ_iso) est décalée vers')
print('le rouge — empilée en continuum SOUS E₀, invisible à une line search.')
print('La prédiction de raie opposable est φ_halo ; le 15/07 comparait un')
print('flux isotrope moyenné (pas une raie) aux limites de raies NFW —')
print('doublement conservateur pour la levée de F3, mais à re-étiqueter.')
