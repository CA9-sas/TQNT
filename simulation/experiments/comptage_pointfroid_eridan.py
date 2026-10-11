"""Comptage local de chiralité — région point froid / Eridan.

Prédiction multi-jets (brouillon sablier, §8) : un jet d'éjection vers
(l,b) = (209°, -57°) marquerait le ciel en température (le point froid),
en densité (le supervide d'Eridan) ET en chiralité — un excès local de
sens de rotation des galaxies dans la même direction.
Prédiction canonique (isotropie structurelle) : aucun excès local
au-delà du hasard.

Données : le jeu « propre » de 72 888 galaxies annotées cw/ccw
(Pan-STARRS DR1, Shamir — le même « clean dataset » que ses papiers ;
récupéré via la Wayback Machine :
https://web.archive.org/web/20220601033654/https://people.cs.ksu.edu/
~lshamir/data/assym_72k/all_galaxies.xlsx).
Couverture : 0 <= RA < 360, -24,5 <= dec <= 85 — la région du point
froid (dec ~ -19,6°) est au bord sud de l'empreinte : déclaré.

Méthode :
 1. disques de rayon R = 10/15/20/30 deg autour du point froid ;
 2. excès cw mesuré CONTRE la fraction globale du catalogue (pas contre
    0,5 : les biais d'annotation globaux sont ainsi soustraits) ;
 3. p-value binomiale bilatérale + p-value EMPIRIQUE : le même excès
    est calculé sur des disques de contrôle tirés au hasard dans
    l'empreinte (mêmes coupures), et on compte la fraction de disques
    de contrôle au moins aussi asymétriques (à N comparable).

Réserves déclarées : l'annotation est celle de Shamir (dont le dipôle
GLOBAL est contesté — mais un artefact d'annotation est global, pas
localisé) ; le bord d'empreinte peut créer des systématiques locales
(les contrôles sont tirés dans la même bande de déclinaison pour les
absorber) ; test à une région, choisie a posteriori (déclaré).
"""
import os
import numpy as np

rng = np.random.default_rng(209)

# ---------------------------------------------------------------- données
SCRATCH = os.environ.get(
    'SABLIER_DATA',
    '/tmp/claude-1000/-home-euk9-projets-physique-nouvelle/'
    '4281778e-628f-46db-80e6-4e8c179f6e45/scratchpad/spin72k.npy')
a = np.load(SCRATCH)          # colonnes : RA, Dec, cw/ccw (+1/-1)
ra, dec, spin = a[:, 0], a[:, 1], a[:, 2].astype(int)
N_tot = len(a)
f_cw_glob = np.mean(spin == 1)

# ------------------------------------------------- centre : point froid
# (l, b) = (209°, -57°) -> équatorial J2000 (matrice galactique inverse)
_R = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [+0.4941094279, -0.4448296300, +0.7469822445],
    [-0.8676661490, -0.1980763734, +0.4559837762],
])
l0, b0 = np.radians(209.0), np.radians(-57.0)
v_gal = np.array([np.cos(b0) * np.cos(l0), np.cos(b0) * np.sin(l0), np.sin(b0)])
v_eq = _R.T @ v_gal
dec0 = np.degrees(np.arcsin(v_eq[2]))
ra0 = np.degrees(np.arctan2(v_eq[1], v_eq[0])) % 360.0

def angdist(ra1, dec1, ra2, dec2):
    r1, d1 = np.radians(ra1), np.radians(dec1)
    r2, d2 = np.radians(ra2), np.radians(dec2)
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

from math import lgamma, exp, log

def binom_two_sided(n, k, p):
    """P bilatérale exacte (somme des issues de proba <= P(k))."""
    def logpmf(i):
        return (lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
                + i * log(p) + (n - i) * log(1 - p))
    lp_k = logpmf(k)
    tot = 0.0
    for i in range(n + 1):
        if logpmf(i) <= lp_k + 1e-9:
            tot += exp(logpmf(i))
    return min(1.0, tot)

print("=" * 74)
print("COMPTAGE LOCAL DE CHIRALITÉ — POINT FROID / ERIDAN (tâche croisée §8)")
print("=" * 74)
print(f"\nCatalogue : {N_tot} galaxies (Pan-STARRS DR1, annotation Shamir)")
print(f"Fraction cw globale : {f_cw_glob:.4f} "
      f"(excès global {2*f_cw_glob-1:+.4f} — biais/dipôle global soustrait)")
print(f"Centre : point froid (l,b) = (209, -57) -> "
      f"(RA, dec) = ({ra0:.1f}, {dec0:.1f})")

d_cs = angdist(ra, dec, ra0, dec0)

print(f"\n{'R (deg)':>8s} {'N':>6s} {'n_cw':>6s} {'f_cw':>7s} {'excès':>7s} "
      f"{'p binom.':>9s} {'p empirique':>12s}")

for R in (10, 15, 20, 30):
    sel = d_cs < R
    n = int(sel.sum())
    if n < 20:
        print(f"{R:8d} {n:6d}   -- trop peu de galaxies --")
        continue
    k = int((spin[sel] == 1).sum())
    f = k / n
    p_bin = binom_two_sided(n, k, f_cw_glob)

    # ---- contrôles : disques aléatoires, même bande de déclinaison ----
    # centres uniformes en RA ; dec du centre = dec0 (même géométrie de
    # bord d'empreinte), en excluant tout chevauchement avec le disque test
    n_ctrl_target = 2000
    excess_obs = abs(f - f_cw_glob)
    worse = 0
    used = 0
    tries = 0
    while used < n_ctrl_target and tries < 20 * n_ctrl_target:
        tries += 1
        ra_c = rng.uniform(0, 360)
        if angdist(ra_c, dec0, ra0, dec0) < 2 * R:
            continue                      # pas de chevauchement
        d = angdist(ra, dec, ra_c, dec0)
        selc = d < R
        nc = int(selc.sum())
        if nc < max(20, n // 3):
            continue                      # empreinte trop vide (bord)
        kc = int((spin[selc] == 1).sum())
        fc = kc / nc
        # z comparable malgré N différents : normaliser par l'erreur binomiale
        z_c = abs(fc - f_cw_glob) / np.sqrt(f_cw_glob * (1 - f_cw_glob) / nc)
        z_o = excess_obs / np.sqrt(f_cw_glob * (1 - f_cw_glob) / n)
        if z_c >= z_o:
            worse += 1
        used += 1
    p_emp = (worse + 1) / (used + 1) if used else float('nan')
    print(f"{R:8d} {n:6d} {k:6d} {f:7.3f} {f-f_cw_glob:+7.3f} "
          f"{p_bin:9.3f} {p_emp:12.3f}  (contrôles : {used})")

# ------------------------------------------------- axe local (bonus)
# le multi-jets prédit un DIPÔLE local : ajuster f(theta) = a + b cos(theta)
# où theta = distance angulaire au point froid, sur 0-40 deg
sel40 = d_cs < 40
n40 = int(sel40.sum())
if n40 > 200:
    x = np.cos(np.radians(d_cs[sel40]))
    y = (spin[sel40] == 1).astype(float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    # bootstrap
    bs = []
    for _ in range(2000):
        idx = rng.integers(0, n40, n40)
        c, *_ = np.linalg.lstsq(A[idx], y[idx], rcond=None)
        bs.append(c[1])
    lo, hi = np.percentile(bs, [16, 84])
    print(f"\nGradient radial de chiralité (0-40 deg, {n40} gal.) : "
          f"pente b = {coef[1]:+.3f} (68 % : [{lo:+.3f}, {hi:+.3f}])")
    print("  (b > 0 = excès cw croissant VERS le point froid ; "
          "0 dans l'intervalle = pas de gradient)")

print("""
Interprétation :
 - p empirique ~ uniforme => la région du point froid n'est PAS spéciale
   en chiralité dans ce catalogue (point pour l'isotropie canonique) ;
 - p empirique petit (<~0,01) => excès local : premier fil réel du
   multi-jets — à confirmer sur une annotation indépendante avant tout.
Réserves : bord sud de l'empreinte à dec ~ -24,5 (contrôles appariés en
déclinaison), annotation unique (Shamir), région choisie a posteriori.""")
