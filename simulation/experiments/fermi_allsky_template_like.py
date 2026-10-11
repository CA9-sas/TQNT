"""Vraisemblance de Poisson à gabarits, tout-ciel haute latitude (|b|>30°).

L'outil que le ROI fermipy ne couvre pas : le test propre de la composante
halo-ρ des 6 raies sur la moitié du ciel. Modèle par (pixel healpix, bin E) :

  μ(p,e) = X(p,e)·[ A_g·(E/E₀)^δg·T_gal(p,e) + A_i·(E/E₀)^δi·T_iso(e)
                    + s·φ_halo·H(p)·L(e;E₀,σ_E) ]

- X(p,e) : exposition (expcube_allsky, ltcube 18 ans) × ΔE × Ω_pix ;
- T_gal : le cube gll_iem_v07 lui-même, interpolé (pas de factorisation) ;
- T_iso : spectre isotrope tabulé iso_P8R3_SOURCE_V3 ;
- H(p)  : colonne NFW-ρ normalisée (Σ H = 1 sur la région) ;
- L(e)  : gaussienne σ_E/E = 7 %, flux total unité → s = normalisation
          en unités de la machinerie perles NOMINALE (s = 1) ;
- A_g, δg, A_i, δi libres (normes + inclinaisons spectrales — absorbent
  les résidus lisses du modèle de diffus) ; s ≥ 0 profilé → UL95
  (ΔNLL = 1,353) ; TS = 2ΔNLL(0 vs ŝ).
- Masque : |b|>30° + pixels des sources 4FGL brillantes (Flux1000 > 5e-10).

Couverture : pôles téléchargés présents (nord seul tant que le sud n'est
pas arrivé — le φ_halo de région est calculé sur la couverture RÉELLE).
Usage : micromamba run -n fermi python -m simulation.experiments.fermi_allsky_template_like
"""
import glob
import numpy as np
import healpy as hp
from astropy.io import fits
from pathlib import Path
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data' / 'fermi_lat_hl'
DIF = ROOT / 'data' / 'fermi_lat_gc' / 'diffuse'
OUT = ROOT / 'output'
NSIDE = 64
BMIN = 30.0
SIG_E = 0.07
LINES = [1.07, 1.10, 1.22, 1.35, 1.47, 1.64]
PHI_GC16, D_GC16 = 6.25e-11, 2.08e22          # ancre de normalisation (s = 1)

# ---------------------------------------------------------------- photons → (pix, e)
lo, hi = np.log10(0.70), np.log10(2.40)
NEB = int((hi - lo) * 50)                      # bins de 2 % en log E
edges = np.logspace(lo, hi, NEB + 1)
ctr = np.sqrt(edges[:-1] * edges[1:]); dE = np.diff(edges)

CACHE = D / f'counts_nside{NSIDE}.npz'
poles = [p for p in ['npole', 'spole'] if list((D / p).glob('*_PH*.fits'))]
if CACHE.exists() and np.load(CACHE)['poles'].tolist() == poles:
    z = np.load(CACHE)
    counts = z['counts']
else:
    npix = hp.nside2npix(NSIDE)
    counts = np.zeros((npix, NEB), dtype=np.int32)
    for pole in poles:
        for f in sorted((D / pole).glob('*_PH*.fits')):
            with fits.open(f) as h:
                d0 = h[1].data
                keep = (d0['ZENITH_ANGLE'] < 90) & (np.abs(d0['B']) > BMIN)
                E = d0['ENERGY'][keep] / 1000.0
                gl, gb = d0['L'][keep], d0['B'][keep]
            pix = hp.ang2pix(NSIDE, gl, gb, lonlat=True)
            eb = np.clip(np.searchsorted(edges, E) - 1, 0, NEB - 1)
            np.add.at(counts, (pix, eb), 1)
    np.savez_compressed(CACHE, counts=counts, poles=np.array(poles))
print(f'pôles : {poles} ; photons binés : {counts.sum():,}')

# ---------------------------------------------------------------- masque
npix = hp.nside2npix(NSIDE)
gl_pix, gb_pix = hp.pix2ang(NSIDE, np.arange(npix), lonlat=True)
mask = np.abs(gb_pix) > BMIN
if 'spole' not in poles:
    mask &= gb_pix > 0
# sources 4FGL brillantes
with fits.open(DIF / 'gll_psc_v35.fit') as h:
    cat = h[1].data
    bright = cat['Flux1000'] > 5e-10
    src_pix = hp.ang2pix(NSIDE, cat['GLON'][bright], cat['GLAT'][bright], lonlat=True)
mask[np.unique(src_pix)] = False
# aussi les 8 voisins des sources
for p in np.unique(src_pix):
    mask[hp.get_all_neighbours(NSIDE, p)] = False
print(f'pixels utiles : {mask.sum():,}/{npix:,} ({mask.sum()/npix:.1%} du ciel) ; '
      f'{bright.sum()} sources masquées')

# ---------------------------------------------------------------- exposition X(p,e)
with fits.open(D / 'expcube_allsky.fits') as h:
    exp = h[0].data; hdr = h[0].header
    E_exp = np.array(h['ENERGIES'].data.field(0), dtype=float) / 1000.0
ny, nx = exp.shape[1], exp.shape[2]
lon_g = (np.arange(nx) + 1 - hdr['CRPIX1']) * hdr['CDELT1'] + hdr['CRVAL1']
lat_g = (np.arange(ny) + 1 - hdr['CRPIX2']) * hdr['CDELT2'] + hdr['CRVAL2']
gl360 = np.where(gl_pix < 0, gl_pix + 360, gl_pix)
ix = np.clip(np.searchsorted(np.sort(lon_g), gl360) - 1, 0, nx - 1)
# lon_g décroît en CAR standard : retrouver l'index par valeur
order = np.argsort(lon_g)
ix = order[np.clip(np.searchsorted(lon_g[order], gl360) - 1, 0, nx - 1)]
iy = np.clip(np.searchsorted(lat_g, gb_pix) - 1, 0, ny - 1)
Xe = exp[:, iy, ix]                                  # (41, npix)
lnE = np.log(ctr); lnEx = np.log(E_exp)
X = np.empty((npix, NEB))
for e in range(NEB):
    j = np.clip(np.searchsorted(lnEx, lnE[e]) - 1, 0, len(lnEx) - 2)
    t = (lnE[e] - lnEx[j]) / (lnEx[j+1] - lnEx[j])
    X[:, e] = np.exp((1-t) * np.log(Xe[j]) + t * np.log(Xe[j+1]))
OMEGA_PIX = hp.nside2pixarea(NSIDE)
XdEo = X * dE[None, :] * OMEGA_PIX                   # cm² s GeV sr

# ---------------------------------------------------------------- gabarit diffus
with fits.open(DIF / 'gll_iem_v07.fits') as h:
    gcube = h[0].data                                # (28, 1441, 2880), ph/cm²/s/MeV/sr
    gh = h[0].header
    Eg = np.array(h['ENERGIES'].data.field(0), dtype=float) / 1000.0
gny, gnx = gcube.shape[1], gcube.shape[2]
glon_t = (np.arange(gnx) + 1 - gh['CRPIX1']) * gh['CDELT1'] + gh['CRVAL1']
glat_t = (np.arange(gny) + 1 - gh['CRPIX2']) * gh['CDELT2'] + gh['CRVAL2']
og = np.argsort(glon_t)
gx = og[np.clip(np.searchsorted(glon_t[og], gl360) - 1, 0, gnx - 1)]
gy = np.clip(np.searchsorted(glat_t, gb_pix) - 1, 0, gny - 1)
Gc = gcube[:, gy, gx] * 1000.0                       # → ph/cm²/s/GeV/sr
lnEg = np.log(Eg)
T_gal = np.empty((npix, NEB))
for e in range(NEB):
    j = np.clip(np.searchsorted(lnEg, lnE[e]) - 1, 0, len(lnEg) - 2)
    t = (lnE[e] - lnEg[j]) / (lnEg[j+1] - lnEg[j])
    T_gal[:, e] = np.exp((1-t) * np.log(np.maximum(Gc[j], 1e-30)) +
                         t * np.log(np.maximum(Gc[j+1], 1e-30)))

# ---------------------------------------------------------------- isotrope
iso = np.loadtxt(DIF / 'iso_P8R3_SOURCE_V3_v1.txt')   # E(MeV), dN/dE (MeV⁻¹cm⁻²s⁻¹sr⁻¹)
T_iso = np.exp(np.interp(lnE, np.log(iso[:, 0] / 1000.0), np.log(iso[:, 1] * 1000.0)))

# ---------------------------------------------------------------- gabarit halo H(p)
rs, R0, rho_loc = 20.0, 8.2, 0.4
CM_KPC = 3.0857e21
def rho_nfw(r):
    x, x0 = r / rs, R0 / rs
    return rho_loc * x0 * (1 + x0) ** 2 / (x * (1 + x) ** 2)
psi = np.radians(hp.rotator.angdist(np.array([gl_pix, gb_pix]),
                                    np.array([0.0, 0.0]), lonlat=True)) * 0
# angle au CG : cos ψ = cos b cos l
psi = np.arccos(np.clip(np.cos(np.radians(gb_pix)) * np.cos(np.radians(gl_pix)), -1, 1))
s_los = np.linspace(1e-3, 100, 800)
r_g = np.sqrt(R0**2 + s_los[None, :]**2 - 2*R0*s_los[None, :]*np.cos(psi[:, None]))
Hcol = np.trapz(rho_nfw(np.maximum(r_g, 0.05)), s_los, axis=1) * CM_KPC
D_region = (Hcol[mask] * OMEGA_PIX).sum()
phi_region = PHI_GC16 * D_region / D_GC16
H = Hcol / (Hcol[mask] * OMEGA_PIX).sum() / 1.0      # ∫ H dΩ = 1 sur la région (en sr⁻¹)
print(f'D(région) = {D_region:.2e} GeV/cm²·sr → φ_halo nominal (s=1) = {phi_region:.2e} ph/cm²/s')

# ---------------------------------------------------------------- vraisemblance
# Ajustement PAR BIN D'ÉNERGIE (3 amplitudes linéaires : diffus, iso, halo) —
# le fond est totalement libre spectralement ; une raie n'est détectable que
# par sa corrélation SPATIALE avec H au bin E0, et la fuite lisse (bulles de
# Fermi, Loop I) part dans la ligne de base du spectre halo a_h(e).
C = counts[mask].astype(float)
Xm = XdEo[mask]
Tg = T_gal[mask]
Hm = H[mask]

# ORTHOGONALISATION : H_perp = H - proj(H | gal, iso) pixel à pixel (poids
# exposition) — tue la dégénérescence spatiale diffus/halo ; l'efficacité
# eff = <H_perp·H>/<H·H> corrige la normalisation (le signal perdu dans la
# projection est réabsorbé par les gabarits de fond, pas compté).
gref = Tg[:, NEB//2] / Tg[:, NEB//2].mean()
ones = np.ones(mask.sum())
Bmat = np.vstack([gref, ones]).T
coef, *_ = np.linalg.lstsq(Bmat, Hm, rcond=None)
Hperp = Hm - Bmat @ coef
eff = float((Hperp * Hm).sum() / (Hm * Hm).sum())
print(f'orthogonalisation : efficacité = {eff:.3f} (part de H inimitable par le fond)')

def fit_bin(e):
    c = C[:, e]; x = Xm[:, e]
    t1, t2, t3 = Tg[:, e], np.full(mask.sum(), T_iso[e]), Hperp
    def nl(lp):
        a = np.concatenate([np.exp(lp[:2]), lp[2:]])
        mu = np.maximum(x * (a[0]*t1 + a[1]*t2 + a[2]*t3), 1e-300)
        return float((mu - c * np.log(mu)).sum())
    from scipy.optimize import minimize as mz
    r = mz(nl, np.array([0.0, 0.0, 0.0]), method='Nelder-Mead',
           options={'xatol': 1e-5, 'fatol': 1e-4, 'maxiter': 3000})
    ah_v = r.x[2] / eff                     # amplitude halo (peut être <0), corrigée d'efficacité
    med = np.median(np.abs(r.x[2])) if r.x[2] != 0 else 1e-9
    h = max(abs(r.x[2]) * 0.2, 1e-9)
    f0 = nl(r.x)
    fp = nl(r.x + np.array([0, 0, h])); fm = nl(r.x - np.array([0, 0, h]))
    curv = max((fp + fm - 2*f0) / h**2, 1e-30)
    return ah_v, (1.0 / np.sqrt(curv)) / eff, f0

print('--- ajustement 3-gabarits par bin (86 bins)…')
ah = np.zeros(NEB); ah_err = np.zeros(NEB)
for e in range(NEB):
    ah[e], ah_err[e], _ = fit_bin(e)
print('    spectre halo extrait ; médiane a_h =', f'{np.median(ah):.2e}',
      'ph/cm²/s/GeV (fuite lisse attendue : bulles/Loop I)')

# recherche de raie dans a_h(e) : ligne de base lisse (log-parabole) + gaussienne
def line_scan(E0):
    x = np.log(ctr / E0)
    wd = np.abs(x) < 0.30
    core = np.abs(x) < 2 * SIG_E
    side = wd & ~core
    A = np.vstack([np.ones_like(x), x, x * x]).T
    p, *_ = np.linalg.lstsq(A[side], np.log(np.maximum(ah[side], 1e-30)), rcond=None)
    base = np.exp(A @ p)
    prof = np.exp(-0.5 * (x / SIG_E) ** 2)
    prof /= prof[wd].sum()
    wgt = 1.0 / np.maximum(ah_err, 1e-30) ** 2
    den = (prof * prof * wgt)[wd].sum()
    amp = ((ah - base) * prof * wgt)[wd].sum() / den
    err = 1.0 / np.sqrt(den)
    Fl = amp * (prof[wd] * dE[wd]).sum()
    Fle = err * (prof[wd] * dE[wd]).sum()
    return Fl, Fle

print('--- raies dans la composante halo (en unités de s, nominal = 1) :')
svals = []
for E0 in LINES:
    Fl, Fle = line_scan(E0)
    ul = (max(Fl, 0) + 1.645 * Fle) / phi_region
    svals.append((E0, Fl / phi_region, Fle / phi_region, ul))
    print(f'  {E0:.2f} GeV : s_fit = {Fl/phi_region:+.3f} ± {Fle/phi_region:.3f} ; s_UL95 = {ul:.3f}')
scanE = [float(c) for c in ctr if 0.95 < c < 1.85]
fr = np.array([line_scan(E0)[0] / phi_region for E0 in scanE])
print(f'--- balayage nul : plancher systématique RMS = {np.std(fr):.3f} en s')
wsum = sum(1.0 / (r[2] ** 2 + np.std(fr) ** 2) for r in svals)
s_joint = 1.645 / np.sqrt(wsum)
print(f'--- limite jointe (stat ⊕ plancher, 6 raies, normalisation commune) : '
      f's < {s_joint:.3f} (95 %)')
print(f'    (verdict GC : s < 0,076)')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
ax1.errorbar(ctr, ctr**2 * ah, yerr=ctr**2 * ah_err, fmt='o', ms=3, color='#1f4e79')
ax1.set_yscale('log'); ax1.set_ylabel('E² F_halo (GeV ph cm⁻² s⁻¹)')
ax1.set_title(f'Composante spatiale halo-ρ, |b|>30° ({("N+S" if "spole" in poles else "Nord seul")}) — spectre extrait')
ax2.plot(scanE, fr, '-', color='#1f4e79')
ax2.axhspan(-np.std(fr), np.std(fr), color='#f59e0b', alpha=0.2,
            label=f'plancher ±{np.std(fr):.3f} en s')
ax2.axhline(0, color='k', lw=0.5)
ax2.set_xlabel('E (GeV)'); ax2.set_ylabel('raie (unités de s)')
ax2.legend(fontsize=8)
for ax in (ax1, ax2):
    for E0 in LINES:
        ax.axvline(E0, color='#b91c1c', ls='--', lw=0.8, alpha=0.7)
    ax.set_xscale('log'); ax.set_xlim(0.8, 2.1)
fig.tight_layout()
fig.savefig(OUT / 'fermi_allsky_halo_component.png', dpi=140)
print('figure :', OUT / 'fermi_allsky_halo_component.png')
