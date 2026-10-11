"""GWTC-4 vs Classe F — le canal fusion de la population B1938/GD-1, corrigé.

CORRECTION DE BANDE (2026-07-30, prime sur le test ae3cb04 de juin) : le
forecast V5.19 « LIGO O5 : N ~ 1e-2-1 » comparait le taux de fusion Classe F
(M_typ = 1,6e4 Msun, formule binaire PBH) aux candidats IMBH LVK (~150 Msun).
Or f_ISCO ≈ 4400 Hz/(M_tot/Msun) : une fusion 2×1,6e4 sonne à ~0,14 Hz et
une fusion de graines mesurées B1938/GD-1 (2×[1-3]e5) à 7-22 mHz — TOUTES
sous le mur ~10 Hz de LIGO. LVK ne peut PAS voir une fusion Classe F,
quelle que soit l'abondance. Le forecast LIGO était incohérent en bande ;
il est REQUALIFIÉ ici :
  (1) bande LVK : prédiction d'ABSENCE structurelle (une de plus au
      registre) — GW231123 et les lourds de GWTC-4 ne sont PAS des
      Classe F (composantes ~100-140 Msun, ×1000 trop légères ; lecture
      publiée : hiérarchie stellaire, spins élevés — cohérente TQNT) ;
  (2) le canal fusion VIT DANS LA BANDE LISA (mHz) : forecast chiffré
      ci-dessous au f_F pincé par B1938+GD-1 (1e-5..1e-4).
Contrôles de cohérence : chauffage des naines ultra-faibles (Brandt),
Poisson/Lyα ; réserve CMB-accrétion déclarée.
"""
import numpy as np

MSUN = 1.0

# ---------------- 1. la bande : qui peut voir quoi
print('=== 1. Correction de bande ===')
for label, M_tot in [('GW231123 (137+101)', 238.0), ('paire M_typ V5.19 (2×1,6e4)', 3.2e4),
                     ('paire graine B1938 (2×1e5)', 2e5), ('paire graine haute (2×3e5)', 6e5)]:
    f_isco = 4400.0 / M_tot           # Hz
    f_rd = 17000.0 / M_tot            # ringdown (220, a~0,7)
    band = 'LVK' if f_rd > 10 else ('LISA' if 1e-4 < f_isco < 1 else 'deci-Hz')
    print(f'  {label:32s} f_ISCO = {f_isco:8.4f} Hz ; f_ringdown = {f_rd:8.4f} Hz → bande {band}')
print('  → mur LVK ~10 Hz : M_tot ≲ 1000 Msun. AUCUNE fusion Classe F visible LVK.')
print('  → le forecast « LIGO O5 1e-2-1 » (V5.19/ae3cb04) était incohérent en bande : REQUALIFIÉ.')

# ---------------- 2. GWTC-4 : lecture TQNT
print('\n=== 2. GWTC-4 (218 événements, O4a) — lecture ===')
print('  GW231123 : 137+101 Msun, z~0,4, taux 0,08 [+0,19 −0,07] Gpc⁻³an⁻¹, spins élevés.')
print('  Population : pas de population IMBH d effondrement (2605.05563) ; sous-population')
print('  hiérarchique identifiée (2601.07908) ; GW231123 = ~10 fusions successives (ae4102).')
print('  TQNT : cohérent — les lourds LVK sont astrophysiques (hiérarchie) ; la Classe F')
print('  contribue EXACTEMENT ZÉRO à la bande LVK (absence structurelle, pas un réglage).')
print('  Falsifiable : une composante LVK à ~1e5 Msun est impossible (bande) — trivialement')
print('  protégé ; le vrai test est ailleurs (LISA).')

# ---------------- 3. le forecast LISA (formule binaire PBH, Ali-Haïmoud+ 2017)
print('\n=== 3. Le canal fusion réel : LISA (mHz) ===')
def rate_binary(f, M):
    """R [Gpc⁻³ an⁻¹] = 3,7e6 f^(53/37) M^(-32/37) — binaires primordiales découplées."""
    return 3.7e6 * f ** (53.0 / 37.0) * M ** (-32.0 / 37.0)

V_z10 = 3700.0    # Gpc³ comobiles (z<10) — LISA voit 1e5-1e6 Msun partout
print(f'  {"f_F":>6s} {"M_F":>6s} {"R (Gpc⁻³an⁻¹)":>14s} {"N (4 ans)":>10s} {"N (10 ans)":>10s}')
for f in (1e-5, 1e-4):
    for M in (1e5, 3e5):
        R = rate_binary(f, M)
        print(f'  {f:6.0e} {M:6.0e} {R:14.2e} {R*V_z10*4:10.2f} {R*V_z10*10:10.2f}')
print("""  → N(LISA, 4-10 ans) = 0,05-12 selon (f_F, M_F) dans la gamme pincée :
    la fenêtre est OUVERTE et discriminante. Signature Classe F :
    - masse totale QUANTIFIÉE 2-6e5 Msun (spectre de graine étroit) vs
      graines MBH astrophysiques (spectre large, corrélées aux galaxies) ;
    - AUCUNE contrepartie EM (défaut nu, pas d environnement de noyau) ;
    - distribution en z étendue (population primordiale, pas de pic d
      assemblage galactique).
    Réserve : formule binaire découplée (Ali-Haïmoud+ 2017) transposée d un
    scénario PBH — perturbations de l habit r^(-9/4) non incluses (ordre).""")

# ---------------- 4. contrôles d'abondance à 1e5 Msun
print('=== 4. Contrôles d abondance (f_F = 1e-5..1e-4 à ~1e5 Msun) ===')
print('  chauffage naines ultra-faibles (Brandt 2016)   : f ≲ 3e-3   → marge ×30-300 ✓')
print('  Poisson/Lyα (isocourbure)                      : f ≲ qq e-2 → marge ×100+  ✓')
print('  CMB accrétion (Serpico+ 2020, si accrétion BH) : f ~ 1e-5 (disque, ±ordres)')
print('  → RÉSERVE DÉCLARÉE : la borne CMB mord SI le défaut accrète et rayonne comme')
print('    un trou noir ; l efficacité radiative d un défaut à surface dure (pas d')
print('    horizon) est un calcul TQNT ouvert — cible nommée. À f=1e-5 le point est')
print('    à la frontière même en lecture BH ; à 1e-4 il faut la suppression défaut.')

print('\nVERDICT : GWTC-4 ne teste PAS la Classe F (bande) — il la protège trivialement ;')
print('le canal fusion réel est LISA (0,05-12 événements mission, spectre de masse')
print('quantifié, sans contrepartie EM). Le discriminant n°3 de la lettre est CORRIGÉ :')
print('« LVK O5 » → « absence LVK + fenêtre LISA chiffrée ».')
