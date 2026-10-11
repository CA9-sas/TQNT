"""Test T(z) = T0(1+z) — la prédiction de protection de la trame.

La trame prédit la FORME, pas la valeur : E = hf identitaire + zéro
dispersion → le spectre de Planck est un invariant adiabatique transporté
sans déformation → T(z) = T0 (1+z) EXACTEMENT (β = 0 dans la
paramétrisation standard T(z) = T0 (1+z)^{1-β}) ; distorsions μ/y
primordiales nulles. Toute déviation confirmée (photons qui « fatiguent »,
couplage caché, dérive de constantes) réfuterait la graduation en action.

Données : mesures publiées de T_CMB(z) — absorption moléculaire/atomique
(le thermomètre : niveaux rotationnels de CO et structure fine de C I
pompés par le fond) et effet SZ (rapports de raies) ; compilation
littérature ci-dessous (valeurs et incertitudes telles que publiées).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / 'output'
T0_FIRAS = 2.7255           # ± 0.0006 (Fixsen 2009)

# (z, T, sigma, ref) — compilation littérature (absorption + SZ)
DATA = [
    (0.89,   5.081, 0.10, 'Muller+2013 (PKS1830-211, molécules)'),
    (1.7293, 7.5,   1.0,  'Noterdaeme+2011 (CO)'),
    (1.7738, 7.8,   0.65, 'Noterdaeme+2011 (CO)'),
    (2.0377, 8.6,   1.05, 'Noterdaeme+2011 (CO)'),
    (2.4184, 9.15,  0.7,  'Noterdaeme+2011 (CO)'),
    (2.6896, 10.5,  0.7,  'Noterdaeme+2011 (CO)'),
    (1.973,  7.9,   1.0,  'Ge+1997 (C I)'),
    (2.3371, 10.0,  1.5,  'Srianand+2000 (C I, borne médiane)'),
]

z = np.array([d[0] for d in DATA])
T = np.array([d[1] for d in DATA])
s = np.array([d[2] for d in DATA])

# ajustement β : T(z) = T0_FIRAS (1+z)^{1-β}
from scipy.optimize import curve_fit
def law(z, beta):
    return T0_FIRAS * (1 + z) ** (1 - beta)
popt, pcov = curve_fit(law, z, T, p0=[0.0], sigma=s, absolute_sigma=True)
beta, dbeta = popt[0], np.sqrt(pcov[0, 0])
chi2 = np.sum(((T - law(z, beta)) / s) ** 2)
chi2_trame = np.sum(((T - law(z, 0.0)) / s) ** 2)
print(f'ajustement : β = {beta:+.4f} ± {dbeta:.4f}   (χ²/dof = {chi2:.1f}/{len(z)-1})')
print(f'théorème de la trame : β = 0 exactement — χ²/dof = {chi2_trame:.1f}/{len(z)} ;')
print(f'compatible à {abs(beta)/dbeta:.1f} σ.')
print('contraintes publiées (contrôle) : Hurier+2014 (SZ Planck) β = 0,009 ± 0,017 ;')
print('Noterdaeme+2011 β = −0,007 ± 0,027 — même verdict.')
print('réfutations offertes : β ≠ 0 confirmé ; distorsion μ/y primordiale (PIXIE).')

zz = np.linspace(0, 3, 100)
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.errorbar(z, T, yerr=s, fmt='o', ms=5, color='#1f4e79', label='mesures publiées (CO, C I, molécules)')
ax.plot(zz, law(zz, 0.0), '-', color='#b91c1c', lw=1.5,
        label='trame : T = T0(1+z) exactement [théorème]')
ax.plot(zz, law(zz, beta), '--', color='#666', lw=1,
        label=f'ajustement libre : β = {beta:+.3f} ± {dbeta:.3f}')
ax.errorbar([0], [T0_FIRAS], yerr=[0.0006], fmt='s', color='#059669', ms=6, label='FIRAS (z=0)')
ax.set_xlabel('z'); ax.set_ylabel('T_CMB (K)')
ax.set_title('La température du fond suit-elle T0(1+z) ? — test de la graduation en action')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'test_tz_cmb_trame.png', dpi=140)
print('figure :', OUT / 'test_tz_cmb_trame.png')
