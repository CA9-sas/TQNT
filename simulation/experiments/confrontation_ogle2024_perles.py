"""Perles de Classe A face à OGLE-IV 20 ans (Mróz et al. 2024) — le verdict.

SOURCE (abstract exact, arXiv:2403.02386, Nature 632, 749) : 20 ans OGLE
sur 78,7 M d'étoiles du LMC, aucun événement > 1 an, les courts expliqués
par les populations stellaires → « compact objects in the mass range from
1.8e-4 to 6.3 Msun cannot compose more than 1% of dark matter » (95 %) ;
[1,3e-5, 860] Msun : < 10 %.

CLASSE A (référence V5.19) : perles à M_typ = 2,03e-3 Msun (log-normale
σ = 0,3 dex), f = 5 % de la DM. 100 % de la fonction de masse tombe dans
la fenêtre f < 1 % → EXCLUSION ×5,0 (95 %), robuste à la largeur.

REQUALIFICATIONS :
- « Class A SURVIT Subaru HSC ×5 » (V5.19, référence) : CADUC — le test de
  2026-06 utilisait une compilation 2019-2020 sans sensibilité à 2e-3 Msun
  (limite employée : 0,35, extrapolation Smyth/EROS) ; la fenêtre de masse
  des perles est celle d'OGLE, pas de Subaru HSC (1e-11..1e-5 Msun).
- Cascade Classe E : le flux des 6 raies ∝ abondance des perles → OGLE
  impose s < 0,20 INDÉPENDAMMENT de notre line search (s < 0,076) : la
  normalisation nominale de la machinerie perles est désormais exclue par
  DEUX canaux indépendants (microlentille ×5 ; γ ×13).
- Le 12e registre proposé s'élargit : « machinerie perles Classe A/E —
  fraction 5 % exclue ×5 (OGLE-IV 2024) et flux de raies nominal exclu ×13
  (Fermi 18 ans, notre analyse) ».
Ce qui survit : des perles à f < 1 % (ou hors fenêtre de masse [restent
1e-5..2e-4 à <10 % et sous 1,3e-5 non contraint]) ; le réservoir diffus
(nœuds isolés, 95 % de la DM TQNT) n'est PAS touché — seule la fraction
agrégée en perles l'est.
"""
import numpy as np
from math import erf, sqrt

M0, SIG, F_A = 2.03e-3, 0.3, 0.05
LIM = [(1.8e-4, 6.3, 0.01), (1.3e-5, 860.0, 0.10)]   # (Mmin, Mmax, f_95)

def cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

print(__doc__)
for Mmin, Mmax, f95 in LIM:
    frac = cdf((np.log10(Mmax) - np.log10(M0)) / SIG) - \
           cdf((np.log10(Mmin) - np.log10(M0)) / SIG)
    print(f'fenêtre [{Mmin:.1e}, {Mmax:.0f}] Msun (f<{f95:.0%}) : '
          f'{100*frac:.1f} % de la fonction de masse → '
          f'f_eff = {F_A*frac:.3f} → exclusion ×{F_A*frac/f95:.1f}')
print(f'cascade raies Classe E : s_perles < {LIM[0][2]/F_A:.2f} (OGLE, indépendant du γ)')
