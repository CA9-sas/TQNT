"""Le plasma interstellaire local (VLISM) reconstruit en TQNT — l'inventaire.

Le milieu que traversent les Voyager, relu par le document principal : chaque espèce
est un objet topologique du champ χ ; les densités observées (Voyager
PWS/PLS + littérature LIC) sont converties en nombre de NŒUDS et en
masse-énergie. Trois populations ontologiques :
  (a) ONDES JAMAIS ENTRÉES (𝒮=0) : photons (CMB + champ magnétique
      drapé = limite de jauge collective des ondes χ) — la lumière ;
  (b) NŒUDS/ENTRELACS (ondes entrées et nouées) : la matière du plasma —
      électron = trèfle 3₁ (mode Z₃), proton = entrelacs 3₁+7₂+7̄₂
      (Q = Σχ+Lk = −2+3 = +1), neutron = idem à Lk=+2 (Q=0),
      H = p+e (4 nœuds), He-4 = 2p+2n+2e (14 nœuds) ;
  (c) NŒUDS SOMBRES : réservoir c≥23 (1,9 GeV typ.), spectateur absolu
      (σ_eq ~ 1e-48) mais présent au même titre.
Masses : canoniques V7 (m_p = 938,2 base octet √σ_χ𝒮 ; m_e = 0,511
chaîne Koide/Green ; Δm(n−p) = 1,293 dérivé).
"""
import numpy as np

# densités observées (LIC/VLISM, Voyager + littérature ; /cm³)
n_e_free = 0.09      # électrons libres (V1 PWS : 0,055→0,13)
n_p_free = 0.075     # protons libres (quasi-neutralité, part He+)
n_HI     = 0.15      # hydrogène NEUTRE (ionisation 20-40 %)
n_He     = 0.015     # hélium (He/H ~ 10 %, majoritairement neutre)
n_dark   = 0.4/1.9   # nœuds sombres (0,4 GeV/cm³ / 1,9 GeV)
n_cmb    = 411.0     # photons CMB
B_nT     = 0.5

# masses canoniques (MeV)
m_e, m_p, m_H, m_He = 0.511, 938.2, 938.8, 3728.4

especes = [
    # (nom, topologie, nœuds/objet, n (/cm³), masse MeV, Q)
    ('électron libre',  'trèfle 3₁ (mode Z₃)',            1, n_e_free, m_e,  -1),
    ('proton libre',    'entrelacs 3₁+7₂+7̄₂ (Lk=+3)',     3, n_p_free, m_p,  +1),
    ('H neutre',        'entrelacs p ⊕ trèfle e',          4, n_HI,     m_H,   0),
    ('hélium-4',        '2p+2n (Y-jonctions) + 2e',       14, n_He,     m_He,  0),
    ('nœud sombre',     'nœud premier isolé c≥23',         1, n_dark,   1900,  0),
]
print(f'{"espèce":16s} {"topologie":30s} {"n (/cm³)":>9s} {"nœuds/cm³":>10s} {"ρ (MeV/cm³)":>12s}')
tot_kn_vis, tot_rho_vis = 0, 0
for nom, topo, k, n, m, Q in especes:
    print(f'{nom:16s} {topo:30s} {n:9.3f} {k*n:10.3f} {m*n:12.1f}')
    if nom != 'nœud sombre':
        tot_kn_vis += k*n; tot_rho_vis += m*n
rho_dark = 1900*n_dark
print('-'*80)
print(f'{"TOTAL visible":47s} {tot_kn_vis:10.2f} {tot_rho_vis:12.1f}')
print(f'{"TOTAL sombre":47s} {n_dark:10.2f} {rho_dark:12.1f}')

# les ondes jamais entrées (𝒮=0)
u_B = (B_nT*1e-5)**2/(8*np.pi) / 1.602e-6            # MeV/cm³
u_cmb = 411*6.34e-4*1e-6/1.602e-19/1e6 * 0 + 0.26e-6*1e6*1e-6  # 0,26 eV/cm³
u_cmb = 0.26e-6                                       # MeV/cm³
u_gcr = 1.0e-6*1e6*1e-6*0 + 1.0e-6                    # ~1 eV/cm³ → MeV/cm³
print(f'\nondes jamais entrées (𝒮=0, la lumière au sens large) :')
print(f'  photons CMB : {n_cmb:.0f}/cm³ ({u_cmb*1e6:.2f} eV/cm³) — ×{n_cmb/(tot_kn_vis+n_dark):.0f} tous les nœuds réunis')
print(f'  champ B drapé (0,5 nT) : {u_B*1e6:.0f} eV/cm³ (limite de jauge des ondes χ)')
print(f'  rayons cosmiques (nœuds rapides) : ~1 eV/cm³ en énergie')
print(f'\nBILAN TQNT du VLISM :')
print(f'  nœuds visibles {tot_kn_vis:.2f}/cm³ vs sombres {n_dark:.2f}/cm³ (rapport {tot_kn_vis/n_dark:.1f}) ;')
print(f'  masse : visible {tot_rho_vis:.0f} vs sombre {rho_dark:.0f} MeV/cm³ — le sombre GAGNE ×{rho_dark/tot_rho_vis:.1f} ;')
print(f'  et la population dominante en NOMBRE reste la lumière (CMB), jamais entrée.')
