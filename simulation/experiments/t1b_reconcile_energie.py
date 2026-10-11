"""T1b — réconciliation des conventions d'énergie (brouillon tension Maxwell).

DIAGNOSTIC À LA LECTURE DU CANONIQUE (derive_profil_tube_k33.py) :
  E_complet = 0,5 · [alpha hbar c/(2L)] · K33h,   K33h = noyau de Neumann
  des filaments hélicoïdaux PONDÉRÉ PAR cos(3(sigma_i - sigma_j)) —
  c'est l'énergie PROJETÉE SUR LE MODE n=3 (le K_33 du programme m_e),
  PAS l'énergie magnétostatique totale.
  L'énergie totale dans le MÊME schéma est E_tot = pref · Nh
  (noyau SANS poids ; Neumann : E = (mu0 I^2/8pi) somme dl·dl'/d).

  => la « relation 16/9 » du T1 comparait DEUX OBSERVABLES :
     E_helicoidal(nous, champ total) vs E_mode(canonique).
     Si pref·Nh ≈ notre 4,05 MeV, tout se réconcilie ET notre calcul de
     champ est CROSS-VALIDÉ par un noyau indépendant.
     Le rapport 2Nh/K33h ≈ 16/9 se réduit alors à K33 ≈ 9/4 (coïncidence
     DÉJÀ consignée au canonique, règle du null) x Nh ≈ 2.

Ici : reconstruction EXACTE du noyau canonique (m = 12 filaments de
framing, fermeture par branche d'holonomie, double somme /m^2) sur la
courbe SONO en cache, poids cos3 ET sans poids ; comparaison au MC de T1.
"""
import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from simulation.ideal_embedding import resample, linking_number

ALPHA = 1 / 137.035999
HBARC = 197.32698
SQRT_SIGMA = 418.47
SIGMA_FM = SQRT_SIGMA**2 / HBARC
M_L_NU = 0.75 * SQRT_SIGMA
L_FM = M_L_NU / SIGMA_FM

CACHE = ('/tmp/claude-1000/-home-euk9-projets-physique-nouvelle/'
         '4281778e-628f-46db-80e6-4e8c179f6e45/scratchpad/trefle_ideal_sono.npz')
z = np.load(CACHE)
P_id, ell = z['P'].copy(), float(z['ell'])
P_id[:, 2] *= -1.0                      # chiralité canonique (Wr > 0)

print("=" * 74)
print("T1b — RÉCONCILIATION DES CONVENTIONS D'ÉNERGIE   (2026-08-28)")
print("=" * 74)

def frames(P):
    dP = np.roll(P, -1, 0) - P
    ds = np.linalg.norm(dP, axis=1)
    t = dP / ds[:, None]
    u = np.zeros_like(P)
    v0 = np.array([0., 0., 1.]); v0 -= (v0 @ t[0]) * t[0]
    u[0] = v0 / np.linalg.norm(v0)
    for i in range(1, len(P)):
        w = u[i-1] - (u[i-1] @ t[i]) * t[i]
        u[i] = w / np.linalg.norm(w)
    v = np.cross(t, u)
    return t, u, v, ds

def kernels(P, a, m=12):
    """Reproduction exacte du noyau canonique (K33h, Nh)."""
    t, u, v, ds = frames(P)
    n = len(P)
    s_arc = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
    L = ds.sum()
    sig = 2 * np.pi * s_arc / L
    # construction T1 (White explicite) : holonomie de Bishop absorbée,
    # rotation uniforme fermant les lignes au plus près de Tw = Lk - Wr
    w_tr = u[-1] - (u[-1] @ t[0]) * t[0]
    w_tr /= np.linalg.norm(w_tr)
    delta_seam = np.arctan2(w_tr @ v[0], w_tr @ u[0])
    # writhe de Gauss
    mid_ax = P + 0.5 * (np.roll(P, -1, 0) - P)
    dP = np.roll(P, -1, 0) - P
    Wr = 0.0
    for i in range(n):
        r = mid_ax[i] - mid_ax
        dd = np.linalg.norm(r, axis=1); dd[i] = np.inf
        Wr += np.sum(np.einsum('j,ij->i', dP[i], np.cross(dP, r, axis=1)) / dd**3)
    Wr /= 4 * np.pi
    Tw_target = 1.0 - Wr
    k_int = round(Tw_target + delta_seam / (2*np.pi))
    rot = 2 * np.pi * k_int - delta_seam
    tw_eff = rot / (2 * np.pi)
    ph = (2 * np.pi * np.arange(m)[None, :] / m + rot * (s_arc / L)[:, None])
    X = (P[:, None, :] + a * np.cos(ph)[:, :, None] * u[:, None, :]
         + a * np.sin(ph)[:, :, None] * v[:, None, :])
    lk = linking_number(X[:, 0, :], P)
    br = k_int
    dl = np.roll(X, -1, axis=0) - X
    Mid = X + 0.5 * dl
    K33h = Nh = 0.0
    for i in range(n):
        diff = Mid[i][:, None, None, :] - Mid[None, :, :, :]
        d = np.linalg.norm(diff, axis=3)
        d[d < 1e-12] = np.inf
        dot = np.einsum('kc,jlc->kjl', dl[i], dl)
        g = (dot / d).sum(axis=(0, 2)) / m**2
        K33h += (np.cos(3 * (sig[i] - sig)) * g).sum()
        Nh += g.sum()
    return K33h / L, Nh / L, tw_eff, lk, L

P = resample(P_id, 240)
dl_ax = np.roll(P, -1, 0) - P
L_geo = np.linalg.norm(dl_ax, axis=1).sum()
a_geo = L_geo / ell
pref = ALPHA * HBARC / (2 * L_FM)

for m in (12, 24):
    K33h, Nh, tw_eff, lk, _ = kernels(P, a_geo, m)
    E_mode = 0.5 * pref * K33h
    E_tot = pref * Nh
    print(f"""
m = {m:2d} filaments (rotation {tw_eff:+.3f} tours ; Lk(fonction) = {lk} — convention d'orientation, |Lk| = 1 attendu) :
   K33/L (pondéré cos3)  = {K33h:+.4f}   -> E_mode = 0,5·pref·K33 = {E_mode:.3f} MeV
      (canonique E_complet = 2,283 ; alpha·m_l = {ALPHA*M_L_NU:.3f})
   N/L   (sans poids)    = {Nh:+.4f}   -> E_tot  = pref·N       = {E_tot:.3f} MeV
      (notre MC T1 extrapolé : 4,046 MeV -> écart {100*(4.046/E_tot-1):+.1f} %)
   rapport 2N/K33 = {2*Nh/K33h:.4f}  vs 16/9 = {16/9:.4f} ({100*(2*Nh/K33h/(16/9)-1):+.2f} %)
   décomposition du rapport : K33 = {K33h:.4f} (9/4 : {100*(K33h/2.25-1):+.2f} %) ;
   N = {Nh:.4f} (2 : {100*(Nh/2-1):+.2f} %)""")

print(f"""
VERDICT T1b :
 - E_complet canonique = énergie PROJETÉE SUR LE MODE n=3 (poids cos3) ;
   l'énergie magnétostatique TOTALE du même modèle est pref·N.
 - Si pref·N ≈ notre 4,05 MeV (MC indépendant, box+masque ±10 %),
   les deux calculs se CROSS-VALIDENT — il n'y a PAS deux conventions
   incompatibles, il y avait DEUX OBSERVABLES.
 - Le « E_hel = (16/9)·E_complet » se dissout en 2N/K33 ≈ 16/9, porté
   par K33 ≈ 9/4 (coïncidence DÉJÀ consignée au canonique, jamais
   retenue) et N ≈ 2 (nouvelle coïncidence du même rang, non retenue).
   AUCUNE identité nouvelle : null exécuté.""")
