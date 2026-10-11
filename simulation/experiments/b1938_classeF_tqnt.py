"""B1938+666-V comme objet compact topologique primordial (Classe F TQNT).

L'OBSERVATION (Powell+ 2025 ; analyse raffinée Vegetti+ 2026, modèle
favorisé « Pseudo-Jaffe + Point Mass ») : un perturbateur de ~1e6 Msun
dans la lentille B1938+666 (z=0,881) avec (i) une composante PONCTUELLE
non résolue < 10 pc de quelques 1e5 Msun (>10 % de la masse dans 10 pc :
20σ hors-CDM) ; (ii) une enveloppe étendue ; (iii) le point PUR exclu.

LES CONCURRENTS ET LEURS COÛTS (2606.12909, leurs propres aveux) :
- SIDM : σ/m ~ 100 cm²/g + instantané t=0,24 Gyr d'un emballement où la
  densité croît ×300 en 0,07 Gyr (réglage temporel) ; paramètres de halo
  « significativement contraints » pour tout reproduire ;
- CDM+BH : trou noir 1,9e5 dans le reste d'un progéniteur de 1e11
  effeuillé de 5 ORDRES (« difficile à réaliser »).

L'INTERPRÉTATION TQNT (Classe F, V5.19 — catalogue ANTÉRIEUR à
l'observation) : défaut topologique primordial compact —
m_apparent = 1e5 Msun, r_eff = 0,01 pc (canonique) — SANS progéniteur,
SANS effeuillage extrême, SANS réglage de collapse :
(1) le point = le défaut lui-même (1e5 canonique vs « quelques 1e5 »
    inféré) ;
(2) l'enveloppe = accrétion secondaire ΛCDM autour d'une masse
    ponctuelle (générique : M_halo ≈ M_seed × (1+z_eq)/(1+z)) puis
    troncature tidale ORDINAIRE dans la galaxie-lentille ;
(3) l'abondance : Ω_F ≤ 1e-4 Ω_DM (borne du catalogue) → jusqu'à ~12
    objets par zone sensible : la borne n'interdit pas la détection ; une
    détection unique MESURE Ω_F ~ 1e-5, un ordre sous la borne.
CHRONOLOGIE DÉCLARÉE : le catalogue V5.19 (juin 2026) fixe la Classe F
(1e5 Msun apparent, 0,01 pc) sur des considérations IMBH/microlentilles,
SANS entrée B1938 — postérieur à Powell+ 2025 (détection ~1e6 concentré)
mais indépendant, et la masse du POINT « quelques 1e5 » (Vegetti+ 2026,
Nat. Astron. 10.1038/s41550-025-02746-w) n'existait pas encore :
dérivation aveugle en pratique, PAS une pré-registration (motif Ω_cc).
DISCRIMINANTS vs SIDM (falsifiables) :
- sous 10 pc : TQNT = strictement ponctuel (0,01 pc) ; SIDM = cœur
  secondaire ÉTENDU (leur profil cylindrique DÉCROÎT vers le centre —
  leur propre figure) → le VLBI futur résolvant <10 pc tranche ;
- population : ~0,1-1 objet point+enveloppe par lentille, TOUS avec la
  même masse de point ~1e5 (spectre étroit du défaut) vs SIDM où la
  masse du cœur varie avec l'histoire de collapse ;
- canal fusion : bande LISA (7-22 mHz), 0,05-12 événements/mission au f_F
  pincé — LVK structurellement aveugle (correction de bande 30/07, cf.
  gwtc4_classF_confrontation.py).
"""
import numpy as np

Msun = 1.0
# (1) le point
M_pt_obs = (2e5, 5e5)          # « quelques 1e5 » (Vegetti+ 2026)
M_pt_F = 1e5                   # canonique Classe F (m_apparent)
print(f'(1) point : Classe F canonique {M_pt_F:.0e} vs inféré {M_pt_obs[0]:.0e}-{M_pt_obs[1]:.0e} Msun ✓ (facteur 2-5)')

# (2) l'enveloppe par accrétion secondaire autour d'une masse ponctuelle
z_lens, z_eq = 0.881, 3400
for M_seed in (1e5, 3e5):
    M_halo = M_seed * (1+z_eq)/(1+z_lens)
    print(f'(2) graine {M_seed:.0e} → halo accrété non tronqué {M_halo:.1e} Msun', end='')
    print(f' → troncature tidale dans la lentille → ~1e6 observé : GÉNÉRIQUE (il faut ×{M_halo/1e6:.0f} de troncature, banal en sous-halo)')

# (3) abondance
Omega_F = 1e-4                 # borne catalogue (spéculative, déclarée)
# masse de DM projetée dans l'anneau de sensibilité (~anneau d'Einstein ±1 kpc)
Sigma_cr = 3e9                 # Msun/kpc² (ordre, z~0,88 lentille)
kappa_dm = 0.3
R_E, dR = 6.4, 1.0             # kpc
M_annulus = kappa_dm*Sigma_cr * 2*np.pi*R_E*dR
N_max = Omega_F * M_annulus / 3e5          # au plafond de la borne
Omega_mes = 1.0 * 3e5 / M_annulus          # 1 objet observé → Ω_F mesuré
print(f'(3) DM projetée dans l anneau sensible : {M_annulus:.1e} Msun')
print(f'    au plafond Ω_F=1e-4 : N ≤ {N_max:.0f} par lentille — la borne N INTERDIT PAS la détection')
print(f'    1 objet observé → Ω_F(mesuré) ~ {Omega_mes:.0e} × Ω_DM : UN ORDRE SOUS la borne ✓')
print('    (grossier : 1 lentille, zone sensible ~anneau ±1 kpc ; c est un ordre de grandeur)')
print()
print('BILAN DES COÛTS :')
print('  SIDM      : σ/m~100 + instantané ±0,07 Gyr d un emballement ×300 + paramètres pincés')
print('  CDM+BH    : effeuillage de 5 ordres d un progéniteur 1e11 (« difficile »)')
print('  TQNT (F)  : un défaut primordial du catalogue (masse canonique indépendante, chronologie déclarée),')
print('              une accrétion générique, une troncature banale — ZÉRO réglage.')
print('  Réserve déclarée : Ω_F est une BORNE spéculative du catalogue, pas une prédiction fine.')
