"""H₀ et S₈ depuis les cartes Planck PR3, sous les théorèmes de la trame.

Ce que la trame de l'espace-temps impose (V6, consolidé) — et qui transforme
l'ajustement en test :
  [théorème]  Ω_tot = 1 (poussée = attraction = le même champ de déplétion) ;
  [théorème]  w = −1 constant (le résidu du vide ne se dilue ni ne se retire) ;
  [théorème]  croissance freinée = ΛCDM (Birkhoff) → le σ₈ des données EST S₈.
Corollaire falsifiable : les tensions H₀ (SH0ES) et S₈ (lentillage faible)
ne peuvent pas être de la physique de trame — hérités (systématique ou vide
local en RG).

Volet A — H₀ : spectre TT pseudo-Cℓ de la carte SMICA sous le masque commun
(anafast/f_sky, faisceau 5', fenêtre de pixel) → positions des pics
acoustiques → ℓ_A → θ* ; puis fond cosmologique exact (r_s, D_M — intégrales,
pas de Boltzmann) sous PLATITUDE-THÉORÈME : θ* fixe H₀ sans liberté
géométrique. Entrées déclarées : ω_b, ω_m (paramètres de forme, hauteurs de
pics Planck). Démonstration jointe : SANS le théorème (Ω_k libre), le même θ*
accepte H₀ = 73 — c'est la trame qui fait de la valeur locale une TENSION.

Volet B — S₈ : paquet lensing PR3 (MV dat_klm − mf_klm, bruit nlkk.dat) →
C_L^κκ mesuré → amplitude  = mesuré/fiduciel sur L∈[8,400] →
S₈ = S₈^fid × √A sous la croissance-théorème. Comparaison au lentillage
faible (KiDS/DES ~0,76-0,79) : la tension héritée.

Données : data/planck_pr3/ (montage 4 To, jamais dans git).
Usage : micromamba run -n fermi python -m simulation.experiments.planck_h0_s8_trame
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data' / 'planck_pr3'
OUT = ROOT / 'output'

C_KMS = 299792.458
OMEGA_B, OMEGA_M = 0.02237, 0.1430    # paramètres de forme (hauteurs de pics Planck 2018)
OMEGA_G = 2.47e-5                     # photons
OMEGA_R = 4.15e-5                     # + neutrinos sans masse (N_eff = 3,046)
ZSTAR = 1089.92

# ---------------------------------------------------------------- volet A
def measure_theta_star():
    import healpy as hp
    print('— lecture SMICA + masque…')
    m = hp.read_map(str(D / 'COM_CMB_IQU-smica_2048_R3.00_full.fits'), field=0)
    mask = hp.read_map(str(D / 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits'), field=0)
    fsky = mask.mean()
    print(f'  f_sky = {fsky:.3f}')
    lmax = 2000
    print('— anafast (pseudo-Cℓ)…')
    cl = hp.anafast(m * mask, lmax=lmax, use_pixel_weights=True) / fsky
    ll = np.arange(lmax + 1)
    bl = hp.gauss_beam(np.radians(5.0 / 60.0), lmax=lmax)      # SMICA : 5' FWHM
    pw = hp.pixwin(2048)[:lmax + 1]
    cl_corr = cl / (bl ** 2 * pw ** 2)
    dl = ll * (ll + 1) * cl_corr / (2 * np.pi) * 1e12           # µK²
    np.save(D / 'dl_tt.npy', dl)

    # ÉTIREMENT DE GABARIT (robuste) : le spectre théorique du meilleur
    # ajustement PR3 est étiré en ℓ → αℓ ; α minimise le χ² sur ℓ∈[300,1800]
    # (au-delà du 1er pic, où le couplage de masque pèse peu). θ* = θ*_fid / α.
    th = np.loadtxt(D / 'theory_bestfit_R3.txt')
    lt, dt = th[:, 0], th[:, 1]
    from scipy.interpolate import interp1d
    from scipy.optimize import minimize_scalar
    fth = interp1d(lt, dt, bounds_error=False, fill_value=0.0)
    lfit = np.arange(300, 1600)
    dmeas = dl[lfit]
    # gabarit de bruit blanc débeamé : D_ℓ^bruit ∝ ℓ(ℓ+1)/(b_ℓ² p_ℓ²)
    nshape = lfit * (lfit + 1.0) / (bl[lfit] ** 2 * pw[lfit] ** 2)
    nshape /= nshape.mean()
    def chi2(alpha):
        m1 = fth(lfit / alpha)
        A_ = np.vstack([m1, nshape]).T
        coef, *_ = np.linalg.lstsq(A_, dmeas, rcond=None)
        return np.sum((dmeas - A_ @ coef) ** 2)
    res = minimize_scalar(chi2, bounds=(0.97, 1.03), method='bounded')
    alpha = res.x
    THETA_FID = 0.0104109
    theta_star = THETA_FID / alpha
    print(f'  étirement α = {alpha:.5f}  →  100 θ* = {100*theta_star:.4f} '
          f'(Planck 2018 : 1,0411)')
    main = []
    return theta_star, main, dl

def background_theta(h, omega_k=0.0):
    """θ* prédit pour h donné (ω_b, ω_m fixés ; platitude si omega_k=0)."""
    H0 = 100 * h
    Ok = omega_k
    Ol = 1 - (OMEGA_M + OMEGA_R) / h**2 - Ok
    def E(z):
        return np.sqrt((OMEGA_M / h**2) * (1+z)**3 + (OMEGA_R / h**2) * (1+z)**4
                       + Ok * (1+z)**2 + Ol)
    z1 = np.linspace(0, ZSTAR, 40000)
    Dc = np.trapz(1.0 / E(z1), z1) * C_KMS / H0
    if abs(Ok) < 1e-8:
        Dm = Dc
    else:
        k = np.sqrt(abs(Ok)) * H0 / C_KMS
        Dm = (np.sinh(k * Dc) / k) if Ok > 0 else (np.sin(k * Dc) / k)
    z2 = np.logspace(np.log10(ZSTAR), 8, 40000)
    R = 3 * OMEGA_B / (4 * OMEGA_G) / (1 + z2)
    cs = C_KMS / np.sqrt(3 * (1 + R))
    rs = np.trapz(cs / (H0 * E(z2)), z2)   # r_s = ∫ c_s dz / H(z)
    return rs / Dm

def solve_h0(theta_meas, omega_k=0.0):
    from scipy.optimize import brentq
    return brentq(lambda h: background_theta(h, omega_k) - theta_meas, 0.4, 1.0, xtol=1e-5)

# ---------------------------------------------------------------- volet B
def lensing_s8():
    """Amplitude de lentillage depuis les BANDPOWERS OFFICIELS PR3
    (consext8, L=8-400, dé-biaisés RDN0/N1/PS/MC par l equipe Planck) —
    la voie klm brute est documentée comme non dé-biaisée (±20-50 %)."""
    bp = np.loadtxt(D / 'planck_lensing_2018' /
                    'smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_bandpowers.dat')
    Lav, PPhat, err, Ahat = bp[:, 3], bp[:, 4], bp[:, 5], bp[:, 6]
    dA_bin = Ahat * err / PPhat
    w = 1.0 / dA_bin ** 2
    A = np.sum(w * Ahat) / np.sum(w)
    dA = 1.0 / np.sqrt(np.sum(w))
    return A, dA, (Lav, Ahat)

def main():
    print('=== Volet A — H0 sous le théorème de platitude ===')
    theta, peaks, dl = measure_theta_star()
    h_flat = solve_h0(theta)
    print(f'  θ* mesuré + platitude-THÉORÈME (Ω_k = 0) → H0 = {100*h_flat:.2f} km/s/Mpc')
    print(f'    (registre de la trame : 67,4 ; Planck 2018 : 67,36 ± 0,54)')
    # sans le théorème : quelle courbure pour H0 = 73 ?
    from scipy.optimize import brentq
    ok73 = brentq(lambda ok: background_theta(0.732, ok) - theta, -0.15, 0.15, xtol=1e-6)
    print(f'  SANS le théorème : le même θ* accepte H0 = 73,2 avec Ω_k = {ok73:+.4f}')
    print(f'    → la trame INTERDIT cette issue (Ω_tot = 1 en théorème) :')
    print(f'      c est elle qui fait du H0 local une tension, pas une solution.')

    print('=== Volet B — S8 sous la croissance-théorème ===')
    A, dA, (Ls, ratio) = lensing_s8()
    S8_fid = 0.832
    S8 = S8_fid * np.sqrt(max(A, 0))
    dS8 = S8_fid * 0.5 * dA / np.sqrt(max(A, 1e-3))
    print(f'  amplitude de lentillage A = {A:.3f} ± {dA:.3f} (relative au fiduciel PR3)')
    print(f'  → S8 = {S8:.3f} ± {dS8:.3f} (Planck 2018 : 0,832 ± 0,013 ; '
          f'lentillage faible : ~0,76-0,79)')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ll = np.arange(len(dl))
    ax1.plot(ll[2:], dl[2:], lw=0.6, color='#1f4e79')
    for p in peaks:
        ax1.axvline(p, color='#b91c1c', ls='--', lw=0.8, alpha=0.6)
    ax1.set_xlim(2, 2000); ax1.set_ylim(0, 6500)
    ax1.set_xlabel(r'$\ell$'); ax1.set_ylabel(r'$D_\ell^{TT}$ ($\mu K^2$)')
    ax1.set_title(f'SMICA TT — pics → 100θ* = {100*theta:.4f} → H0 = {100*h_flat:.2f} (Ω_tot=1, théorème)')
    ax2.errorbar(Ls, ratio, fmt='o', ms=4, color='#1f4e79')
    ax2.axhline(1, color='#666', lw=0.8)
    ax2.axhline(A, color='#b91c1c', lw=1.2, label=f'A = {A:.3f} → S8 = {S8:.3f}')
    ax2.set_xlabel('L'); ax2.set_ylabel('A par bin (officiel)')
    ax2.set_ylim(0.6, 1.4); ax2.legend()
    ax2.set_title('Lentillage PR3 (MV) — amplitude sous croissance-théorème')
    fig.tight_layout()
    fig.savefig(OUT / 'planck_h0_s8_trame.png', dpi=140)
    print('figure :', OUT / 'planck_h0_s8_trame.png')

if __name__ == '__main__':
    main()
