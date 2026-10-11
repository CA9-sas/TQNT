"""Line search haute latitude (|b|>30°, moitié du ciel) — les 6 raies TQNT.

La marche suivante du verdict GC : pour un signal en ρ (perles du halo), la
haute latitude offre un facteur D total ×~6 le ROI GC de 16° pour un fond
÷10 et spectralement plus lisse (isotrope + diffus faible — le bump π⁰ du
plan est absent). Méthode éprouvée du GC : spectre en flux corrigé
d'exposition au pas de 1 % en log E, fond log-parabole par fenêtre
glissante, raie gaussienne σ_E/E = 7 %, BALAYAGE NUL pour le plancher
systématique. Conversion en normalisation commune s (s = 1 ⇔ machinerie
perles Classe E nominale) via le facteur D de la région.

Entrées : data/fermi_lat_hl/{npole,spole}/*.fits (cônes 60° sur les pôles
galactiques = |b|>30°), expcube_allsky.fits (depuis le ltcube 18 ans).
"""
import glob
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data' / 'fermi_lat_hl'
OUT = ROOT / 'output'
LINES = [1.07, 1.10, 1.22, 1.35, 1.47, 1.64]
SIG_E = 0.07
PHI_GC16 = 6.25e-11        # flux halo nominal dans le ROI GC 16° (s = 1)
D_GC16 = 2.08e22           # GeV/cm²·sr (rechiffre_phi_halo_perles)
BMIN = 30.0

# ---------------------------------------------------------------- photons
CACHE = D / 'photons_hl.npz'
if CACHE.exists():
    z = np.load(CACHE)
    E, B = z['E'], z['B']
else:
    Es, Bs = [], []
    for f in sorted(glob.glob(str(D / '*pole' / '*_PH*.fits'))):
        with fits.open(f) as h:
            d0 = h[1].data
            keep = (d0['ZENITH_ANGLE'] < 90) & (np.abs(d0['B']) > BMIN)
            Es.append(d0['ENERGY'][keep]); Bs.append(d0['B'][keep])
    E = np.concatenate(Es) / 1000.0
    B = np.concatenate(Bs)
    np.savez_compressed(CACHE, E=E, B=B)
print(f'photons |b|>{BMIN:.0f}°, zénith<90° : {len(E):,}')

# ---------------------------------------------------------------- exposition
with fits.open(D / 'expcube_allsky.fits') as h:
    exp = h[0].data; hdr = h[0].header
    E_exp = np.array(h['ENERGIES'].data.field(0), dtype=float) / 1000.0
ny, nx = exp.shape[1], exp.shape[2]
lat = (np.arange(ny) + 1 - hdr['CRPIX2']) * hdr['CDELT2'] + hdr['CRVAL2']
LATg = np.broadcast_to(lat[:, None], (ny, nx))
sel = np.abs(LATg) > BMIN
w = np.cos(np.radians(LATg))          # poids de surface en CAR
exp_hl = np.array([(exp[i] * w)[sel].sum() / w[sel].sum() for i in range(exp.shape[0])])
Omega_hl = 4 * np.pi * (1 - np.sin(np.radians(BMIN)))
print(f'exposition moyenne région : {exp_hl.min():.2e}-{exp_hl.max():.2e} cm² s ; Ω = {Omega_hl:.2f} sr')

# ---------------------------------------------------------------- spectre en flux
lo, hi = np.log10(0.70), np.log10(2.40)
edges = np.logspace(lo, hi, int((hi - lo) * 100) + 1)
ctr = np.sqrt(edges[:-1] * edges[1:]); dE = np.diff(edges)
cnt, _ = np.histogram(E, bins=edges)
expi = np.exp(np.interp(np.log(ctr), np.log(E_exp), np.log(exp_hl)))
flux = cnt / (expi * dE)
ferr = np.sqrt(np.maximum(cnt, 1)) / (expi * dE)

def line_fit(E0):
    x = np.log(ctr / E0)
    wdw = np.abs(x) < 0.30
    core = np.abs(x) < 2 * SIG_E
    side = wdw & ~core
    A = np.vstack([np.ones_like(x), x, x * x]).T
    p = np.linalg.lstsq(A[side], np.log(np.maximum(flux[side], 1e-300)), rcond=None)[0]
    bkg = np.exp(A @ p)
    prof = np.exp(-0.5 * (x / SIG_E) ** 2)
    prof /= prof[wdw].sum()
    wgt = 1.0 / np.maximum(ferr, 1e-300) ** 2
    num = ((flux - bkg) * prof * wgt)[wdw].sum()
    den = (prof * prof * wgt)[wdw].sum()
    amp, err = num / den, 1 / np.sqrt(den)
    Fl = amp * (prof[wdw] * dE[wdw]).sum()
    Fle = err * (prof[wdw] * dE[wdw]).sum()
    Fc = (bkg * dE)[core].sum()
    return Fl, Fle, Fc

# ---------------------------------------------------------------- facteur D de la région
rs, R0, rho_loc = 20.0, 8.2, 0.4
CM_KPC = 3.0857e21
def rho_nfw(r):
    x, x0 = r / rs, R0 / rs
    return rho_loc * x0 * (1 + x0) ** 2 / (x * (1 + x) ** 2)
def col(psi):
    s = np.linspace(1e-3, 100, 2000)
    r = np.sqrt(R0**2 + s**2 - 2 * R0 * s * np.cos(psi))
    return np.trapezoid(rho_nfw(np.maximum(r, 0.05)), s) * CM_KPC
# moyenne de la colonne sur |b|>30 : angle ψ au CG selon (l, b) — échantillonnage
rng = np.random.default_rng(1)
ls = rng.uniform(-np.pi, np.pi, 4000)
sb = rng.uniform(np.sin(np.radians(BMIN)), 1, 4000)
bs = np.arcsin(sb) * rng.choice([-1, 1], 4000)
psi = np.arccos(np.cos(bs) * np.cos(ls))
D_hl = np.mean([col(p) for p in psi]) * Omega_hl
phi_pred_hl = PHI_GC16 * D_hl / D_GC16
print(f'facteur D (|b|>{BMIN:.0f}°) = {D_hl:.2e} GeV/cm²·sr → φ_halo nominal région = {phi_pred_hl:.2e} ph/cm²/s')
print(f'  soit {phi_pred_hl * np.median(expi):.0f} photons de raie attendus au nominal (s = 1)')

# ---------------------------------------------------------------- raies + balayage nul
print('--- les 6 raies (flux, |b|>30°) :')
rows = []
for E0 in LINES:
    Fl, Fle, Fc = line_fit(E0)
    ul = max(Fl, 0) + 1.645 * Fle
    s_ul = ul / phi_pred_hl
    rows.append((E0, Fl, Fle, ul, s_ul))
    print(f'  {E0:.2f} GeV : F = ({Fl:+.2e} ± {Fle:.2e}) ; UL95 = {ul:.2e} '
          f'({ul*np.median(expi):.0f} ph) ; s < {s_ul:.3f}')
scanE = ctr[(ctr > 0.95) & (ctr < 1.85)]
fr = []
for E0 in scanE:
    Fl, Fle, Fc = line_fit(E0)
    fr.append(Fl / phi_pred_hl)
fr = np.array(fr)
print(f'--- balayage nul : plancher systématique RMS = {np.std(fr):.3f} en unités de s '
      f'(flux RMS = {np.std(fr)*phi_pred_hl:.2e})')
# verdict joint (normalisation commune, profils ~linéaires)
s_joint = 1.645 * np.sqrt(1.0 / np.sum([(1.645/r[4])**2 for r in rows]))
print(f'--- indication jointe (stat. seule) : s < {s_joint:.3f} (95 %)')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})
ax1.errorbar(ctr, ctr**2 * flux, yerr=ctr**2 * ferr, fmt='.', ms=2, color='#1f4e79')
ax1.set_yscale('log'); ax1.set_ylabel('E² dF/dE (GeV ph cm⁻² s⁻¹)')
ax1.set_title(f'Fermi-LAT 2008-2026, |b|>{BMIN:.0f}° — {len(E):,} photons')
ax2.plot(scanE, fr, '-', color='#1f4e79', lw=1.1)
ax2.axhspan(-np.std(fr), np.std(fr), color='#f59e0b', alpha=0.18,
            label=f'plancher systématique ±{np.std(fr):.3f} (en s)')
ax2.axhline(0, color='k', lw=0.5)
ax2.axhline(0.076, color='#059669', lw=1.2, ls=':', label='plafond joint GC : s < 0,076')
ax2.set_xlabel('E (GeV)'); ax2.set_ylabel('amplitude (unités de s)')
ax2.legend(fontsize=8)
for ax in (ax1, ax2):
    for E0 in LINES:
        ax.axvline(E0, color='#b91c1c', lw=0.8, ls='--', alpha=0.7)
    ax.set_xscale('log'); ax.set_xlim(0.8, 2.1)
fig.tight_layout()
fig.savefig(OUT / 'fermi_hl_line_search.png', dpi=140)
print('figure :', OUT / 'fermi_hl_line_search.png')
