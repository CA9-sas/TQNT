"""Premier regard Fermi-LAT 18 ans — bande des 6 raies TQNT (1,07-1,64 GeV).

Données : LAT Data Server (FSSC, publiques) — centre galactique r<16°,
700-2400 MeV, 2008-08-05 → 2026-07-28 (docs/references/fermi_lat_gc/wide/,
gitignoré). Coupures appliquées ICI (le serveur ne les fait pas) :
zénith < 90° (albédo terrestre — ~47 % des photons bruts !), r < 16°.

Méthode (espace de comptages, SANS exposition — premier regard, PAS une line
search publiable) : histogramme en log E ; par énergie testée, fond lisse =
log-parabole ajustée sur les ailes ±30 % (cœur ±2σ_E exclu, ailes complètes
grâce à la bande élargie) ; raie = gaussienne σ_E/E = 7 % (dispersion LAT
~1 GeV) ; amplitude en fraction du continuum local sous le cœur.
Le BALAYAGE NUL (mêmes fits à toutes les énergies) mesure le plancher
SYSTÉMATIQUE de la méthode (courbure du diffus + aire efficace non corrigée) :
c'est LUI la sensibilité réelle, pas l'erreur statistique.
Les 6 raies canoniques (ACPR 22/07) : 1,07/1,10/1,22/1,35/1,47/1,64 GeV ;
à σ_E = 7 %, 1,07 et 1,10 sont confondues (une seule structure attendue).
"""
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'fermi_lat_gc' / 'wide'
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

CACHE = DATA / 'photons_cut.npz'
if CACHE.exists():
    z = np.load(CACHE)
    e = z['E']
else:
    from astropy.io import fits
    Es, Ls, Bs, Zs = [], [], [], []
    for f in sorted(glob.glob(str(DATA / '*_PH*.fits'))):
        with fits.open(f) as h:
            d = h[1].data
            Es.append(d['ENERGY']); Ls.append(d['L']); Bs.append(d['B'])
            Zs.append(d['ZENITH_ANGLE'])
    E = np.concatenate(Es) / 1000.0
    L = np.concatenate(Ls); B = np.concatenate(Bs); Z = np.concatenate(Zs)
    L = np.where(L > 180, L - 360, L)
    keep = (Z < 90) & (np.hypot(L, B) < 16)
    e = E[keep]
    np.savez_compressed(CACHE, E=e)
    print(f'photons bruts {len(E):,} → après coupures (zénith<90°, r<16°) : {len(e):,}')

LINES = [1.07, 1.10, 1.22, 1.35, 1.47, 1.64]   # GeV (ACPR 2026-07-22)
SIG_E = 0.07

lo, hi = np.log10(0.70), np.log10(2.40)
edges = np.logspace(lo, hi, int((hi - lo) * 100) + 1)
cnt, _ = np.histogram(e, bins=edges)
ctr = np.sqrt(edges[:-1] * edges[1:])

def line_fit(E0):
    x = np.log(ctr / E0)
    w = np.abs(x) < 0.30
    core = np.abs(x) < 2 * SIG_E
    side = w & ~core
    nL, nR = np.sum(side & (x < 0)), np.sum(side & (x > 0))
    A = np.vstack([np.ones_like(x), x, x * x]).T
    p = np.linalg.lstsq(A[side], np.log(cnt[side]), rcond=None)[0]
    bkg = np.exp(A @ p)
    prof = np.exp(-0.5 * (x / SIG_E) ** 2)
    prof /= prof[w].sum()
    num = ((cnt - bkg) * prof / np.maximum(bkg, 1))[w].sum()
    den = (prof * prof / np.maximum(bkg, 1))[w].sum()
    amp, err = num / den, 1 / np.sqrt(den)
    cc = bkg[core].sum()
    return amp / cc, err / cc, (nL, nR)

print(f'{len(e):,} photons dans la bande ; bins de 1 % en log E')
print('--- les 6 raies (ailes complètes des deux côtés) :')
for E0 in LINES:
    f, df, wings = line_fit(E0)
    print(f'  {E0:.2f} GeV : ({100*f:+.3f} ± {100*df:.3f}) % du continuum '
          f'[ailes {wings[0]}/{wings[1]} bins]')

# balayage nul — le plancher systématique de la méthode
scanE = ctr[(ctr > 0.95) & (ctr < 1.85)]
vals = np.array([line_fit(E0)[:2] for E0 in scanE])
fr, dfr = vals[:, 0], vals[:, 1]
floor = np.std(fr)
print(f'--- balayage nul ({len(scanE)} points 0,95-1,85 GeV) : '
      f'plancher systématique RMS = {100*floor:.2f} % du continuum '
      f'(statistique : {100*np.median(dfr):.3f} %)')
print(f'    verdict : aucune raie ne dépasse ~{100*2*floor:.1f} % du continuum '
      f'(2×RMS) — les raies TQNT prédites (~×8-13 sous les limites 2015) sont '
      f'SOUS ce plancher : test réel = analyse Fermitools (exposition + IRF).')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})
ax1.step(ctr, cnt, where='mid', lw=0.8, color='#1f4e79')
ax1.set_yscale('log')
ax1.set_ylabel('coups / bin (1 % log E)')
ax1.set_title(f'Fermi-LAT 2008-2026, centre galactique r<16°, zénith<90° — '
              f'{len(e):,} photons')
ax2.plot(scanE, 100 * fr, '-', color='#1f4e79', lw=1.1,
         label='amplitude de raie ajustée (balayage)')
ax2.fill_between(scanE, -100 * dfr, 100 * dfr, color='#999', alpha=0.4,
                 label='±1σ statistique')
ax2.axhspan(-100 * floor, 100 * floor, color='#f59e0b', alpha=0.15,
            label=f'plancher systématique ±{100*floor:.2f} %')
ax2.axhline(0, color='k', lw=0.5)
ax2.set_xlabel('E (GeV)')
ax2.set_ylabel('raie (% du continuum)')
ax2.legend(fontsize=8, loc='upper right')
for ax in (ax1, ax2):
    for E0 in LINES:
        ax.axvline(E0, color='#b91c1c', lw=0.8, ls='--', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlim(0.8, 2.1)
    ax.set_xticks([0.8, 0.9, 1.0, 1.07, 1.22, 1.35, 1.47, 1.64, 1.8, 2.0])
    ax.set_xticklabels(['0,8', '0,9', '1,0', '1,07', '1,22', '1,35', '1,47',
                        '1,64', '1,8', '2,0'], fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'fermi_quicklook_raies.png', dpi=140)
print('figure :', OUT / 'fermi_quicklook_raies.png')
