"""B1938+666-V : le perturbateur sombre anormalement dense face à la TQNT.

Données : profil non-paramétrique ρ(r) du sous-halo (Lei+ 2025,
arXiv:2509.07808, table tab:data — Keck/NIRC2 + LensCharm ; 58 anneaux,
0,079-4,6 kpc ; erreurs stat+sys). Le VLBI (Powell/McKean) voit par
ailleurs un objet COMPACT ~1e6 Msun aux échelles mas.

Concurrents publiés et leurs coûts : SIDM (σ/m~300 cm²/g — amas : <1) ;
ψDM (m~1,3e-22 eV — Lyα : ≳1e-20) ; CDM+IMBH (effeuillage de 5 ordres).

Modèle TQNT (Classe F, V5.19) : objet compact PRIMORDIAL (défaut,
r_eff~0,01 pc → ponctuel ; gamme 1e5-1e7 Msun) + enveloppe NFW de nœuds
du réservoir (CDM standard, σ/m dérivé 2e-4). Pas d'effeuillage requis
(le compact n'est pas un reste de halo), pas de microphysique exotique.
L'objet ponctuel apparaît dans la reconstruction lissé par la résolution
effective (noyau du champ de perturbation) → gaussienne 3D de largeur
σ_sm (paramètre de nuisance, borné par le pas des anneaux).
Ajustements χ² (mêmes données/erreurs que Lei+) : NFW / cœur-SIDM /
TQNT (point lissé + NFW) — comparaison AIC/BIC.
"""
import numpy as np
from scipy.optimize import minimize

# table tab:data (r kpc, rho 1e6 Msun/kpc3, sig 1e6, M 1e8, sigM 1e8)
D = """0.079 25.285 5.123 0.001 0.0001
0.158 23.843 4.831 0.004 0.001
0.238 23.098 6.244 0.013 0.004
0.317 22.061 6.187 0.031 0.009
0.396 20.776 5.392 0.058 0.015
0.475 19.257 4.064 0.096 0.02
0.555 17.595 2.607 0.144 0.021
0.634 15.963 1.407 0.203 0.018
0.713 14.422 1.605 0.271 0.03
0.792 12.984 2.539 0.349 0.068
0.872 11.684 3.572 0.433 0.133
0.951 10.485 4.488 0.525 0.225
1.03 9.284 5.045 0.621 0.338
1.109 8.272 5.615 0.721 0.489
1.189 7.15 5.559 0.822 0.639
1.268 6.314 5.707 0.923 0.834
1.347 5.53 5.513 1.023 1.02
1.426 4.945 5.497 1.123 1.249
1.506 4.685 5.658 1.226 1.481
1.585 3.748 4.782 1.326 1.691
1.664 3.259 4.451 1.417 1.936
1.743 3.21 4.602 1.511 2.166
1.823 2.896 4.319 1.607 2.397
1.902 2.611 4.03 1.702 2.628
1.981 2.131 3.368 1.791 2.83
2.06 1.733 2.808 1.869 3.028
2.139 1.677 2.774 1.944 3.216
2.219 1.602 2.687 2.021 3.389
2.298 1.314 2.213 2.095 3.529
2.377 1.056 1.793 2.159 3.664
2.456 1.008 1.723 2.219 3.794
2.536 0.97 1.659 2.28 3.902
2.615 0.875 1.495 2.341 3.999
2.694 0.758 1.285 2.398 4.066
2.773 0.653 1.077 2.451 4.042
2.853 0.605 0.988 2.5 4.084
2.932 0.556 0.891 2.548 4.082
3.011 0.503 0.798 2.595 4.121
3.09 0.456 0.711 2.639 4.109
3.17 0.411 0.626 2.682 4.08
3.249 0.378 0.571 2.722 4.115
3.328 0.346 0.513 2.761 4.086
3.407 0.317 0.463 2.798 4.085
3.487 0.291 0.416 2.834 4.047
3.566 0.27 0.378 2.869 4.013
3.645 0.25 0.345 2.903 4.012
3.724 0.229 0.31 2.935 3.965
3.804 0.212 0.284 2.966 3.962
3.883 0.197 0.257 2.996 3.908
3.962 0.184 0.235 3.025 3.862
4.041 0.172 0.216 3.054 3.831
4.12 0.161 0.198 3.082 3.792
4.2 0.15 0.183 3.108 3.774
4.279 0.141 0.169 3.134 3.749
4.358 0.131 0.154 3.16 3.703
4.437 0.124 0.143 3.184 3.674
4.517 0.118 0.134 3.208 3.638
4.596 0.109 0.122 3.232 3.599"""
arr = np.array([[float(x) for x in l.split()] for l in D.strip().split('\n')])
r, rho, sig = arr[:, 0], arr[:, 1], arr[:, 2]        # unités 1e6 Msun/kpc3

def nfw(r, rs, rhos):
    x = r / rs
    return rhos / (x * (1 + x) ** 2)

def cored(r, rc, rhoc, xs):                          # cœur SIDM-like, r_s = xs·r_c
    rs = xs * rc
    return rhoc / ((1 + (r / rc) ** 2) * (1 + r / rs) ** 2)

def point_smear(r, Mpt, ssm):                        # point lissé (gaussienne 3D)
    return Mpt / ((2 * np.pi) ** 1.5 * ssm ** 3) * np.exp(-r ** 2 / (2 * ssm ** 2))

def chi2(model, p):
    return float(np.sum(((rho - model(r, *p)) / sig) ** 2))

def fit(model, p0, bounds):
    best = None
    for scale in (1.0, 0.5, 2.0):
        res = minimize(lambda p: chi2(model, p), np.array(p0) * scale,
                       method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 20000})
        if best is None or res.fun < best.fun:
            best = res
    return best

print('=== Ajustements sur le profil Lei+ 2025 (58 anneaux) ===')
print('IDENTITÉ D OBJET (clarifiée 30/07) : ce profil est le GROS sous-halo')
print('(M(4,6 kpc) = 3,2e8 Msun, le perturbateur historique type Vegetti 2012')
print('revu par Keck) — PAS l objet V du million de soleils (VLBI Powell+ 2025,')
print('traité dans b1938_classeF_tqnt.py). Deux objets, deux dossiers.')
out = {}
res = fit(nfw, [2.0, 5.0], None)
out['NFW (2p)'] = (res.fun, 2, res.x)
res = fit(cored, [0.5, 25.0, 3.0], None)
out['cœur SIDM-like (3p)'] = (res.fun, 3, res.x)
tqnt = lambda r, Mpt, ssm, rs, rhos: point_smear(r, Mpt, ssm) + nfw(r, rs, rhos)
# fit LIBRE (diagnostic) : σ_sm s envole → le « point » avale l enveloppe
res = fit(tqnt, [15.0, 0.4, 2.0, 1.0], None)
out['point+NFW, σ libre (diagnostic)'] = (res.fun, 4, res.x)
# fit CONTRAINT : σ_sm ≤ 0,3 kpc (résolution de la reconstruction : pas des
# anneaux 0,079 kpc, mocks de Lei+ distinguant cœur/NFW à ~0,3 kpc) ;
# M_pt ≤ 10e6 (gamme Classe F/VLBI)
from scipy.optimize import minimize as _min
best = None
for p0 in ([1.0, 0.15, 2.0, 5.0], [5.0, 0.25, 1.5, 8.0], [0.3, 0.1, 3.0, 3.0]):
    r0 = _min(lambda q: chi2(tqnt, q), p0, method='Nelder-Mead',
              bounds=[(0.05, 10.0), (0.05, 0.30), (0.1, 50.0), (1e-3, 500.0)],
              options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 40000})
    if best is None or r0.fun < best.fun:
        best = r0
out['point (Classe F) + NFW, CONTRAINT'] = (best.fun, 4, best.x)

n = len(r)
for nom, (c2, k, pp) in out.items():
    aic = c2 + 2 * k
    print(f'  {nom:38s} χ² = {c2:7.2f}  (χ²/dof {c2/(n-k):.2f})  AIC = {aic:7.2f}')
    print(f'    params : {np.round(pp, 3)}')

print()
print('VERDICT (honnête) :')
print(' - le fit σ-libre était PATHOLOGIQUE : σ = 0,89 kpc = le « point » devenu')
print('   une boule gaussienne qui absorbe tout le halo (NFW résiduel dégénéré).')
print(' - contraint à la résolution physique, le point n aide plus : ce profil')
print('   est un VRAI cœur étendu (~0,5 kpc) — AUCUN objet Classe F dedans, et')
print('   la TQNT n en réclame pas (son dossier Classe F = l objet V, séparé).')
print(' - RÉSERVE DÉCLARÉE : un sous-halo de 3e8 Msun à cœur plat serait une')
print('   tension pour tout réservoir CDM-like (TQNT incluse) SI la')
print('   reconstruction non paramétrique tient ; systématiques débattues')
print('   (SHARP IX conclut « atypique mais possible en ΛCDM ») — à suivre.')
