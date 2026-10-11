"""Non-réplication des anomalies CMB dans le lentillage — test de la prédiction V7.

LA PRÉDICTION (V7, trio cosmo B.6, isotropie structurelle) : les anomalies
grande échelle de la température (cold spot, asymétrie hémisphérique, axe
quad/oct) sont du BRUIT de réalisation, pas des structures de la trame →
elles ne doivent PAS se répliquer dans les sondes indépendantes du même
ciel (modes E, lentillage, 21 cm). Toute réplication croisée réfuterait.

LE TEST (données locales, Planck PR3) : la carte de convergence κ du
lentillage (MV, champ moyen soustrait, ℓ=8-400) est une sonde de matière
QUASI indépendante de la température primaire (corrélation ISW-lentillage
résiduelle ~qq %, déclarée). On teste :
  (1) COLD SPOT (l,b)=(209°,−57°) : κ lissé à 5° à cette position, z-score
      contre 2000 positions aléatoires du même ciel masqué ;
  (2) ASYMÉTRIE HÉMISPHÉRIQUE : dipôle de variance locale (disques de 8°,
      bande ℓ=8-64) — angle entre l'axe κ et l'axe T (~(221°,−20°)),
      null P(<θ)=1−cosθ ; et le même calcul mené sur T (ℓ=2-64) comme
      contrôle positif (l'anomalie doit être là en T) ;
  (3) AXE QUAD/OCT : hors de portée du lentillage (reconstruction ℓ≥8) —
      déclaré ; l'arène de ce volet = modes E LiteBIRD.
Caveats déclarés : bruit de reconstruction κ inhomogène (le null par
positions aléatoires du même ciel l'absorbe au 1er ordre) ; inpainting
diffusif du T basse résolution (axes à ~10° près).
"""
import numpy as np
import healpy as hp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data' / 'planck_pr3'
rng = np.random.default_rng(7)

CS_LB = (209.0, -57.0)        # cold spot
TMOD_LB = (221.0, -20.0)      # axe de modulation dipolaire T (Planck, ℓ<64)
NSIDE = 256

def lb_to_vec(l, b):
    return hp.ang2vec(np.radians(90 - b), np.radians(l))

# ---------------------------------------------------------------- κ (MV)
print('=== chargement κ (MV, champ moyen soustrait) ===')
klm = hp.read_alm(D / 'COM_Lensing_4096_R3.00' / 'MV' / 'dat_klm.fits')
mflm = hp.read_alm(D / 'COM_Lensing_4096_R3.00' / 'MV' / 'mf_klm.fits')
klm = klm - mflm
lmax = hp.Alm.getlmax(len(klm))
ell = hp.Alm.getlm(lmax)[0]
klm_b = np.where((ell >= 8) & (ell <= 400), klm, 0)          # bande officielle
kappa = hp.alm2map(klm_b, NSIDE)
mask_k = hp.ud_grade(hp.read_map(D / 'COM_Lensing_4096_R3.00' / 'mask.fits'), NSIDE) > 0.5
print(f'  fsky lentillage = {mask_k.mean():.3f}')

# ---------------------------------------------------------------- T (SMICA)
print('=== chargement T (SMICA) + masque commun ===')
T = hp.ud_grade(hp.read_map(D / 'COM_CMB_IQU-smica_2048_R3.00_full.fits', field=0), NSIDE)
mask_t = hp.ud_grade(hp.read_map(D / 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits'), NSIDE) > 0.5
print(f'  fsky T = {mask_t.mean():.3f}')

def smooth_masked(m, mask, fwhm_deg):
    w = mask.astype(float)
    ms = hp.smoothing(m * w, fwhm=np.radians(fwhm_deg))
    ws = hp.smoothing(w, fwhm=np.radians(fwhm_deg))
    out = np.where(ws > 0.3, ms / np.maximum(ws, 1e-9), hp.UNSEEN)
    return out, ws > 0.3

# ---------------------------------------------------------------- (1) cold spot
print('\n=== (1) Cold spot : réplication dans κ ? ===')
def spot_z(m, mask, lb, fwhm):
    sm, ok = smooth_masked(m, mask, fwhm)
    pix = hp.vec2pix(NSIDE, *lb_to_vec(*lb))
    val = sm[pix]
    good = np.where(ok & (sm != hp.UNSEEN))[0]
    draws = sm[rng.choice(good, 2000)]
    return (val - draws.mean()) / draws.std()

for fwhm in (5.0, 10.0):
    zT = spot_z(T, mask_t, CS_LB, fwhm)
    zK = spot_z(kappa, mask_k, CS_LB, fwhm)
    print(f'  lissage {fwhm:4.0f}° : z(T) = {zT:+.2f} (contrôle : le spot est là) ; z(κ) = {zK:+.2f}')

# ---------------------------------------------------------------- (2) asymétrie
print('\n=== (2) Asymétrie hémisphérique : dipôle de variance locale ===')
def variance_dipole(m, mask, lmin, lmax_b):
    w = mask.astype(float)
    alm = hp.map2alm(m * w, lmax=512)
    l2 = hp.Alm.getlm(512)[0]
    mb = hp.alm2map(np.where((l2 >= lmin) & (l2 <= lmax_b), alm, 0), NSIDE)
    # variance dans des disques de 8° centrés sur les pixels nside 16
    ns_c = 16
    centers = np.arange(hp.nside2npix(ns_c))
    vecs = np.array(hp.pix2vec(ns_c, centers)).T
    var = np.full(len(centers), hp.UNSEEN)
    for i, v in enumerate(vecs):
        d = hp.query_disc(NSIDE, v, np.radians(8.0))
        d = d[mask[d]]
        if len(d) > 200:
            var[i] = np.var(mb[d])
    ok = var != hp.UNSEEN
    var_n = var[ok] / var[ok].mean() - 1.0
    # ajustement dipôle pondéré
    A = np.vstack([np.ones(ok.sum()), vecs[ok].T]).T
    coef = np.linalg.lstsq(A, var_n, rcond=None)[0]
    dip = coef[1:]
    amp = np.linalg.norm(dip)
    return dip / amp, amp

axT, ampT = variance_dipole(T, mask_t, 2, 64)
axK, ampK = variance_dipole(kappa, mask_k, 8, 64)
vT_lit = lb_to_vec(*TMOD_LB)
def angle(u, v):
    c = abs(float(np.dot(u, v)))          # axe (sans signe)
    return np.degrees(np.arccos(min(c, 1.0)))
lT = hp.vec2ang(axT, lonlat=True)
lK = hp.vec2ang(axK, lonlat=True)
print(f'  T (ℓ=2-64)  : axe (l,b) = ({float(lT[0]):.0f}°, {float(lT[1]):.0f}°), amplitude {ampT:.3f} '
      f'; angle à l axe littérature (221,−20) : {angle(axT, vT_lit):.0f}°')
print(f'  κ (ℓ=8-64)  : axe (l,b) = ({float(lK[0]):.0f}°, {float(lK[1]):.0f}°), amplitude {ampK:.3f}')
thKT = angle(axK, axT)
p_align = 1 - np.cos(np.radians(thKT))    # P(angle < θ) sous null isotrope
print(f'  angle κ-T = {thKT:.0f}° ; sous le null isotrope, P(alignement ≤ {thKT:.0f}°) = {p_align:.2f}')

# ---------------------------------------------------------------- (3) axe quad/oct
print('\n=== (3) Axe quad/oct : hors de portée du lentillage (ℓ≥8) — déclaré ===')
print('  l arène de ce volet = modes E (LiteBIRD) ; aucune conclusion ici.')

print('\nVERDICT provisoire (voir sortie ci-dessus) : la prédiction V7 exige')
print('|z(κ, cold spot)| ≲ 2 ET aucun alignement significatif du dipôle de')
print('variance κ sur l axe T. Une réplication (z(κ) < −3 au même endroit, ou')
print('alignement à P < 0,01 avec amplitude comparable) serait une TENSION.')
