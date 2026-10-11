"""Borner Ω_k avec les données publiques — la certification de platitude.

Le théorème de la trame dit Ω_tot = 1 EXACTEMENT. On ne peut pas le
« prouver » — on le borne, et le théorème élève la borne en certification
falsifiable (toute courbure mesurée ≠ 0 réfute). La dégénérescence
géométrique du CMB seul (θ* accepte H0=73 avec Ω_k=+0,011) est brisée par
la MÊME règle standard (horizon sonore) vue à deux époques :
  - θ* mesuré sur la carte SMICA (z ≈ 1090) — notre chaîne indépendante,
    100θ* = 1,0417 ± 0,0010 (planck_h0_s8_trame.py) ;
  - distances BAO publiques D_M/r_d, D_H/r_d, D_V/r_d (DESI DR1 2024,
    arXiv:2404.03002, tableau 1 — dix nombres, z = 0,30-2,33).
Cohérence interne : le fond utilise w = −1 (théorème de la trame) — pas
d'échappatoire par l'énergie noire dynamique.

Méthode : fond exact (intégrales), r_s(z*) pour θ*, r_d = r_s(z_drag) pour
les BAO ; χ² sur grille (Ω_k, h), ω_m profilé sous prior de forme Planck
(0,1430 ± 0,0011), ω_b = 0,02237 (BBN). Sortie : Ω_k ± σ et la borne.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / 'output'
C_KMS = 299792.458
OMEGA_B = 0.02237
OMEGA_G = 2.47e-5
OMEGA_R = 4.15e-5
ZSTAR, ZDRAG = 1089.92, 1059.94

THETA_MEAS, THETA_SIG = 1.0417e-2, 0.0010e-2   # notre mesure SMICA (étirement)
OMEGA_M0, OMEGA_M_SIG = 0.1430, 0.0011          # prior de forme (hauteurs de pics)

# BAO DESI DR1 (arXiv:2404.03002, tab. 1) : (z, type, valeur, sigma)
BAO = [
    (0.295, 'DV', 7.93, 0.15),
    (0.510, 'DM', 13.62, 0.25), (0.510, 'DH', 20.98, 0.61),
    (0.706, 'DM', 16.85, 0.32), (0.706, 'DH', 20.08, 0.60),
    (0.930, 'DM', 21.71, 0.28), (0.930, 'DH', 17.88, 0.35),
    (1.317, 'DM', 27.79, 0.69), (1.317, 'DH', 13.82, 0.42),
    (1.491, 'DV', 26.07, 0.67),
    (2.330, 'DM', 39.71, 0.94), (2.330, 'DH', 8.52, 0.17),
]

def model(h, ok, om):
    H0 = 100 * h
    Ol = 1 - om / h**2 - OMEGA_R / h**2 - ok
    def E(z):
        return np.sqrt((om / h**2) * (1+z)**3 + (OMEGA_R / h**2) * (1+z)**4
                       + ok * (1+z)**2 + Ol)
    def Dc(z):
        zz = np.linspace(0, z, 4000)
        return np.trapz(1.0 / E(zz), zz) * C_KMS / H0
    def Dm(z):
        d = Dc(z)
        if abs(ok) < 1e-9: return d
        k = np.sqrt(abs(ok)) * H0 / C_KMS
        return np.sinh(k * d) / k if ok > 0 else np.sin(k * d) / k
    def rs(zend):
        z2 = np.logspace(np.log10(zend), 8, 6000)
        R = 3 * OMEGA_B / (4 * OMEGA_G) / (1 + z2)
        cs = C_KMS / np.sqrt(3 * (1 + R))
        return np.trapz(cs / (H0 * E(z2)), z2)
    return E, Dm, rs

def chi2(h, ok, om):
    E, Dm, rs = model(h, ok, om)
    c2 = ((rs(ZSTAR) / Dm(ZSTAR) - THETA_MEAS) / THETA_SIG) ** 2
    rd = rs(ZDRAG)
    H0 = 100 * h
    for z, kind, val, sig in BAO:
        dm = Dm(z)
        dh = C_KMS / (H0 * E(np.array([z]))[0])
        if kind == 'DM': pred = dm / rd
        elif kind == 'DH': pred = dh / rd
        else: pred = (z * dm**2 * dh) ** (1/3) / rd
        c2 += ((pred - val) / sig) ** 2
    c2 += ((om - OMEGA_M0) / OMEGA_M_SIG) ** 2
    return c2

# grille Ω_k, profil (h, ω_m)
oks = np.linspace(-0.02, 0.02, 41)
prof = []
for ok in oks:
    best = np.inf
    for om in np.linspace(OMEGA_M0 - 3*OMEGA_M_SIG, OMEGA_M0 + 3*OMEGA_M_SIG, 7):
        hs = np.linspace(0.60, 0.76, 33)
        c = [chi2(h, ok, om) for h in hs]
        best = min(best, min(c))
    prof.append(best)
prof = np.array(prof)
prof -= prof.min()
best_ok = oks[np.argmin(prof)]
# σ par interpolation Δχ²=1
from numpy.polynomial import polynomial as P
sel = prof < 9
coef = np.polyfit(oks[sel], prof[sel], 2)
ok_hat = -coef[1] / (2 * coef[0])
sig = 1.0 / np.sqrt(coef[0])
print(f'Ω_k = {ok_hat:+.4f} ± {sig:.4f}   (χ² parabolique sur la grille)')
print(f'→ |Ω_k| < {abs(ok_hat) + 2*sig:.4f} à 95 %')
print(f'théorème de la trame : Ω_k = 0 exactement — compatible à {abs(ok_hat)/sig:.1f} σ ;')
print('toute mesure future hors de zéro réfuterait (falsifiable, pas prouvable).')

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(oks, prof, '-', color='#1f4e79')
ax.axhline(1, color='#999', lw=0.8); ax.axhline(4, color='#999', lw=0.8)
ax.axvline(0, color='#b91c1c', lw=1.2, label='théorème de la trame : Ω_k = 0')
ax.set_xlabel(r'$\Omega_k$'); ax.set_ylabel(r'$\Delta\chi^2$ (profil)')
ax.set_ylim(0, 12)
ax.set_title(f'θ* (SMICA, notre mesure) + BAO DESI DR1 : Ω_k = {ok_hat:+.4f} ± {sig:.4f}')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'planck_bao_courbure.png', dpi=140)
print('figure :', OUT / 'planck_bao_courbure.png')
