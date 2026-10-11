"""Line search à vraisemblance binnée (fermipy) — les 6 raies TQNT, 18 ans.

LE test réel de la prédiction (niveau Ackermann 2015, avec 18 ans au lieu de
5,8) : vraisemblance binnée 0,7-2,4 GeV, centre galactique, avec
- diffus galactique gll_iem_v07 (supprime le plancher de courbure π⁰ qui
  bornait le premier regard à ~0,3 % du continuum),
- isotrope iso_P8R3_SOURCE_V3, sources 4FGL-DR4 du champ,
- composante raie = gabarit spatial NFW (deux hypothèses : ρ² « entrelacement
  de paires », défaut, façon R16 de 2015 ; ρ « désexcitation » en contrôle)
  × spectre gaussien étroit (la dispersion d'énergie est activée).

Sorties : TS, flux ajusté et UL95 par raie ; scan fin 0,95-1,85 GeV ;
data/fermi_lat_gc/like/resultats_line_scan.json + figure output/.
Données et produits : data/fermi_lat_gc/ (montage 4 To — JAMAIS dans git).

Usage : micromamba run -n fermi python -m simulation.experiments.fermipy_line_scan
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data' / 'fermi_lat_gc'
WORK = D / 'like'
WORK.mkdir(exist_ok=True)
OUT = ROOT / 'output'

LINES = [1.07, 1.10, 1.22, 1.35, 1.47, 1.64]   # GeV, ACPR 2026-07-22
SIG_REL = 0.02          # largeur intrinsèque (fine ; l'edisp instrumental élargit)

# ---------------------------------------------------------------- gabarit NFW
def make_nfw_template(power=2, npix=257, width=34.0):
    """Carte du facteur J (power=2, annihilation/paires) ou D (power=1,
    désexcitation) en projection CAR autour du CG. NFW γ=1, r_s=20 kpc,
    R0=8,2 kpc ; normalisation arbitraire (la norme est ajustée)."""
    from astropy.io import fits as pf
    from astropy import wcs
    rs, R0 = 20.0, 8.2
    pix = width / npix
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [(npix + 1) / 2, (npix + 1) / 2]
    w.wcs.cdelt = [-pix, pix]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']
    yy, xx = np.mgrid[0:npix, 0:npix]
    lon, lat = w.wcs_pix2world(xx, yy, 0)
    lon = np.where(lon > 180, lon - 360, lon)
    psi = np.radians(np.hypot(lon, lat))
    s = np.linspace(0, 60.0, 1200)          # kpc, ligne de visée
    ds = s[1] - s[0]
    J = np.zeros_like(psi)
    cospsi = np.cos(psi)
    for i in range(psi.shape[0]):
        r = np.sqrt(R0**2 + s[None, :]**2 - 2*R0*s[None, :]*cospsi[i][:, None])
        r = np.maximum(r, 1e-3)
        rho = 1.0 / ((r/rs) * (1 + r/rs)**2)
        J[i] = (rho**power).sum(axis=1) * ds
    J /= J.mean()                     # normalisation : moyenne 1 sur la carte
    hdu = pf.PrimaryHDU(J.astype('float32'), header=w.to_header())
    path = WORK / f'nfw_pow{power}.fits'
    hdu.writeto(path, overwrite=True)
    return str(path)

# ---------------------------------------------------------------- config
def write_config():
    from astropy.io import fits as _pf
    with _pf.open(D / 'ft1_gti.fits') as _h:
        tmin = int(_h[1].header['TSTART']); tmax = int(_h[1].header['TSTOP'])
    cfg = f"""
data:
  evfile: {D/'ft1_gti.fits'}
  scfile: {D/'lat_spacecraft_merged.fits'}
  ltcube: {D/'ltcube.fits'}
binning:
  coordsys: GAL
  roiwidth: 32.0
  binsz: 0.25
  binsperdec: 55
selection:
  emin: 700
  emax: 2400
  zmax: 90
  tmin: {tmin}
  tmax: {tmax}
  evclass: 128
  evtype: 3
  glon: 0.0
  glat: 0.0
gtlike:
  edisp: true
  edisp_disable: ['isodiff']
  irfs: P8R3_SOURCE_V3
model:
  src_roiwidth: 35.0
  galdiff: {D/'diffuse'/'gll_iem_v07.fits'}
  isodiff: {D/'diffuse'/'iso_P8R3_SOURCE_V3_v1.txt'}
  catalogs: ['{D/'diffuse'/'gll_psc_v35.fit'}']
  extdir: {D/'diffuse'/'Extended_14years'}
fileio:
  outdir: {WORK}
"""
    (WORK / 'config.yaml').write_text(cfg)
    return str(WORK / 'config.yaml')

# ---------------------------------------------------------------- scan
def _patch_fermipy_catalog():
    """Correctifs locaux pour fermipy 1.4.2 : (1) alias NumPy retirés en 2.0
    (np.string_...) ; (2) astropy masque les chaînes vides des FITS et le
    chargeur 4FGL joint sur la colonne masquée Extended_Source_Name — on
    remplit la clé avant la jointure."""
    for _al, _t in [('string_', np.bytes_), ('unicode_', np.str_),
                    ('float_', np.float64), ('complex_', np.complex128)]:
        if not hasattr(np, _al):
            setattr(np, _al, _t)
    import fermipy.catalog as fcat
    from astropy.table import Table, Column

    def _load_fixed(fitsfile, key_left='Extended_Source_Name'):
        table = Table.read(fitsfile, hdu=1)
        table_extsrc = Table.read(fitsfile, 'ExtendedSources')
        table_extsrc.meta.clear()
        fcat.strip_columns(table)
        fcat.strip_columns(table_extsrc)
        for t in (table, table_extsrc):
            for c in t.colnames:
                if t[c].dtype.kind in 'SU' and hasattr(t[c], 'filled'):
                    t[c] = t[c].filled('')
        if 'Spatial_Function' not in table_extsrc.colnames:
            table_extsrc.add_column(Column(name='Spatial_Function', dtype='U20',
                                           length=len(table_extsrc)))
            table_extsrc['Spatial_Function'] = 'SpatialMap'
        table = fcat.join_tables(table, table_extsrc, key_left, 'Source_Name',
                                 fcat._EXTSRC_JOIN_COLS)
        table.sort('Source_Name')
        return table

    fcat._load_extended_tables = _load_fixed


def main():
    _patch_fermipy_catalog()
    from fermipy.gtanalysis import GTAnalysis
    tmpl2 = make_nfw_template(power=2)
    tmpl1 = make_nfw_template(power=1)
    gta = GTAnalysis(write_config(), logging={'verbosity': 2})
    gta.setup()
    # libérer le fond : diffus (norme + index), isotrope, sources brillantes <5°
    gta.free_source('galdiff')
    gta.free_source('isodiff')
    gta.free_sources(distance=5.0, pars='norm', minmax_ts=[25, None])
    fit0 = gta.fit()
    print('fit du fond : loglike =', fit0['loglike'])
    gta.write_roi('fond', make_plots=False)

    results = {'lines': {}, 'scan': [], 'profile_check': {}}
    scan_grid = list(np.round(np.arange(0.95, 1.86, 0.03), 3))
    import os
    ONLY = os.environ.get('SCAN_ONLY', '')
    plans = [('pow2', tmpl2, sorted(set(LINES + scan_grid))), ('pow1', tmpl1, LINES)]
    if ONLY:
        plans = [p for p in plans if p[0] == ONLY]
    for tag, tmpl, grid in plans:
        for E0 in grid:
            name = f'line_{tag}_{int(E0*1000)}'
            gta.add_source(name, {
                'SpatialModel': 'SpatialMap', 'Spatial_Filename': tmpl,
                'SpectrumType': 'Gaussian',
                'Mean': {'value': E0*1000, 'scale': 1.0, 'free': False},
                'Sigma': {'value': SIG_REL*E0*1000, 'scale': 1.0, 'free': False},
                'Prefactor': {'value': 1.0, 'scale': 1e-11, 'free': True,
                              'min': 0, 'max': 1e6},
            }, free=True)
            gta.free_source(name, pars='norm')
            gta.fit(update=True)
            src = gta.roi[name]
            F, dF = float(src['flux']), float(src['flux_err'])
            ts = float(src['ts'])
            if not np.isfinite(ts):
                try:
                    ts = float(gta.like.Ts(name, reoptimize=False))
                except Exception:
                    ts = float('nan')
            rec = {'E0_GeV': E0, 'ts': ts,
                   'flux': F, 'flux_err': dF,
                   'flux_ul95': max(F, 0.0) + 1.645 * dF}   # UL gaussien
            print(f'[{tag}] {E0:.2f} GeV : TS={rec["ts"]:.1f}  '
                  f'F={rec["flux"]:.2e}±{rec["flux_err"]:.2e}  UL95={rec["flux_ul95"]:.2e}')
            if tag == 'pow2' and E0 in LINES:
                results['lines'][E0] = rec
            elif tag == 'pow2':
                results['scan'].append(rec)
            else:
                results['profile_check'][E0] = rec
            gta.delete_source(name)
    (WORK / ('resultats_line_scan%s.json' % ('_' + ONLY if ONLY else ''))).write_text(json.dumps(results, indent=2))
    print('résultats :', WORK / 'resultats_line_scan.json')

if __name__ == '__main__':
    main()
