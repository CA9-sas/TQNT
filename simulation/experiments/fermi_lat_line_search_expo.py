"""Line search Fermi-LAT 18 ans, exposition corrigée — les 6 raies TQNT.

Entrées (docs/references/fermi_lat_gc/, gitignorées) : produits du pipeline
Fermitools run_pipeline.sh — ft1_gti.fits (photons SOURCE, zénith<90°, r<16°,
GTI DATA_QUAL>0 && LAT_CONFIG==1, 4,94 M photons 0,7-2,4 GeV, 2008-2026),
expcube.fits (exposition binnée P8R3_SOURCE_V3, evtype=3).

Méthode : spectre en flux au pas de 1 % en log E (comptages / exposition ROI
interpolée / ΔE) ; par énergie testée, fond = log-parabole sur les ailes ±30 %
(cœur ±2σ_E exclu), raie gaussienne σ_E/E = 7 % ; balayage nul = plancher
systématique de la méthode. Fenêtres de recherche « trame » : voir
fermi_raies_decalages_trame.py (glissement d'ancre ±7 % cohérent, motif à
rapports fixes).

VERDICT (2026-07-29, consigné) :
- l'exposition ne réduit guère le plancher (0,38 → 0,32 % du continuum) :
  le systématique dominant est la COURBURE DU DIFFUS (bump π⁰ vers 1 GeV),
  pas l'aire efficace (22 % de variation lisse, absorbée) ;
- la seule structure au-dessus du plancher (1,08-1,17 GeV, +0,6-0,9 %,
  ~1,3e-8 ph/cm²/s) est RÉFUTÉE comme raie par trois tests :
  (1) LARGEUR : plateau ~16 % ≫ σ_E = 7 % ;
  (2) PROFIL SPATIAL : fraction PLATE à travers les anneaux r<4/4-8/8-12/
      12-16° (elle CROÎT même vers l'extérieur : +0,83/+0,54/+0,75/+1,02 %)
      et identique plan/hors-plan — c'est le profil du diffus, pas d'une
      source à facteur J concentrée au centre ;
  (3) COHÉRENCE EXTERNE : une raie réelle à 1,3e-8 violerait les limites
      Ackermann 2015 (analyse à diffus modélisé, 5,8 ans) ;
- aux 6 énergies canoniques : AUCUNE raie ; UL95 statistiques 2-15e-9
  ph/cm²/s, mais la sensibilité réelle est bornée par le plancher
  systématique ~3-7e-9 — comparable aux limites 2015, PAS meilleure ;
- les raies prédites (×8-13 sous les limites 2015, ~1e-9) restent SOUS la
  portée de cette méthode : CONFORME à la prédiction, mais pas un test.
  Le test réel = vraisemblance binnée avec gabarit de diffus (gll_iem,
  fermipy) — cible nommée, les produits du pipeline sont prêts.
"""
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data' / 'fermi_lat_gc'
OUT = ROOT / 'output'
LINES = [1.07, 1.10, 1.22, 1.35, 1.47, 1.64]
SIG_E = 0.07

with fits.open(D / 'ft1_gti.fits') as h:
    d = h[1].data
    E = d['ENERGY'] / 1000.0
    L = np.asarray(d['L']); B = np.asarray(d['B'])
L = np.where(L > 180, L - 360, L)
r = np.hypot(L, B)

with fits.open(D / 'expcube.fits') as h:
    exp = h[0].data; hdr = h[0].header
    E_exp = np.array(h['ENERGIES'].data.field(0), dtype=float) / 1000.0
ny, nx = exp.shape[1], exp.shape[2]
lon = (np.arange(nx) + 1 - hdr['CRPIX1']) * hdr['CDELT1'] + hdr['CRVAL1']
lat = (np.arange(ny) + 1 - hdr['CRPIX2']) * hdr['CDELT2'] + hdr['CRVAL2']
LON, LAT = np.meshgrid(np.where(lon > 180, lon - 360, lon), lat)
roi = np.hypot(LON, LAT) < 16
exp_roi = np.array([exp[i][roi].mean() for i in range(exp.shape[0])])

lo, hi = np.log10(0.70), np.log10(2.40)
edges = np.logspace(lo, hi, int((hi - lo) * 100) + 1)
ctr = np.sqrt(edges[:-1] * edges[1:]); dE = np.diff(edges)
cnt, _ = np.histogram(E, bins=edges)
expi = np.exp(np.interp(np.log(ctr), np.log(E_exp), np.log(exp_roi)))
flux = cnt / (expi * dE)
ferr = np.sqrt(np.maximum(cnt, 1)) / (expi * dE)

def line_fit(y, yerr, E0, halfwin=0.30, deg=2):
    x = np.log(ctr / E0)
    w = np.abs(x) < halfwin
    core = np.abs(x) < 2 * SIG_E
    side = w & ~core
    A = np.vstack([x ** k for k in range(deg + 1)]).T
    p = np.linalg.lstsq(A[side], np.log(np.maximum(y[side], 1e-300)), rcond=None)[0]
    bkg = np.exp(A @ p)
    prof = np.exp(-0.5 * (x / SIG_E) ** 2)
    prof /= prof[w].sum()
    wgt = 1.0 / np.maximum(yerr, 1e-300) ** 2
    num = ((y - bkg) * prof * wgt)[w].sum()
    den = (prof * prof * wgt)[w].sum()
    amp, err = num / den, 1 / np.sqrt(den)
    Fl = amp * (prof[w] * dE[w]).sum()
    Fle = err * (prof[w] * dE[w]).sum()
    Fc = (bkg * dE)[core].sum()
    return Fl, Fle, Fc

print(f'{len(E):,} photons GTI ; exposition ROI {exp_roi.min():.2e}-{exp_roi.max():.2e} cm² s')
print('--- les 6 raies (flux, exposition corrigée) :')
for E0 in LINES:
    Fl, Fle, Fc = line_fit(flux, ferr, E0)
    ul = max(Fl, 0) + 1.645 * Fle
    print(f'  {E0:.2f} GeV : F = ({Fl:+.2e} ± {Fle:.2e}) ph/cm²/s ; UL95 = {ul:.2e} ; '
          f'{100*Fl/Fc:+.3f} % du continuum')

scanE = ctr[(ctr > 0.95) & (ctr < 1.85)]
fr = np.array([line_fit(flux, ferr, E0) for E0 in scanE])
frac = fr[:, 0] / fr[:, 2]
print(f'--- balayage nul : plancher systématique RMS = {100*np.std(frac):.2f} % du continuum')

# test par anneaux à 1,10 GeV (structure du diffus, pas d'une source centrale)
print('--- test par anneaux (fraction à 1,10 GeV) :')
for name, m in [('r<4', r < 4), ('4-8', (r >= 4) & (r < 8)),
                ('8-12', (r >= 8) & (r < 12)), ('12-16', (r >= 12) & (r < 16))]:
    c, _ = np.histogram(E[m], bins=edges)
    Fl, Fle, Fc = line_fit(c.astype(float), np.sqrt(np.maximum(c, 1)), 1.10)
    print(f'  {name:6s}: {100*Fl/Fc:+.3f} ± {100*Fle/Fc:.3f} %')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})
ax1.errorbar(ctr, ctr**2 * flux, yerr=ctr**2 * ferr, fmt='.', ms=2, lw=0.6, color='#1f4e79')
ax1.set_yscale('log'); ax1.set_ylabel('E² dF/dE (GeV ph cm⁻² s⁻¹)')
ax1.set_title('Fermi-LAT 2008-2026, CG r<16° — spectre corrigé d\'exposition')
ax2.plot(scanE, 100 * frac, '-', color='#1f4e79', lw=1.1)
ax2.axhspan(-100 * np.std(frac), 100 * np.std(frac), color='#f59e0b', alpha=0.18,
            label=f'plancher systématique ±{100*np.std(frac):.2f} % (diffus π⁰)')
ax2.axhline(0, color='k', lw=0.5)
ax2.set_xlabel('E (GeV)'); ax2.set_ylabel('raie ajustée (% du continuum)')
ax2.legend(fontsize=8)
for ax in (ax1, ax2):
    for E0 in LINES:
        ax.axvline(E0, color='#b91c1c', lw=0.8, ls='--', alpha=0.7)
    ax.set_xscale('log'); ax.set_xlim(0.8, 2.1)
    ax.set_xticks([0.8, 0.9, 1.0, 1.07, 1.22, 1.35, 1.47, 1.64, 1.8, 2.0])
    ax.set_xticklabels(['0,8', '0,9', '1,0', '1,07', '1,22', '1,35', '1,47',
                        '1,64', '1,8', '2,0'], fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'fermi_line_search_expo.png', dpi=140)
print('figure :', OUT / 'fermi_line_search_expo.png')
