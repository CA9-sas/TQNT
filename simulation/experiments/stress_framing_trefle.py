"""T1 — stress de Maxwell complet du courant de framing sur le trèfle idéal.

Cible du brouillon tension_maxwell_trame : fermer le 16/9 -> O(1) réel et
mesurer la TENSION nette du tube (T_tt intégré) contre alpha·sigma_chi,
avec sa structure de signe (T2).

MODÈLE (le canonique du programme m_e, ordre suivant inclus) :
 - trèfle IDÉAL (SONO, 20 000 itérations), ré-échantillonné N=360 ;
 - échelle physique : L_phys = m_l(nu)/sigma_chi = 0,3537 fm ;
   rayon de tube a = L_geo/ropelength (convention L/r), contrôle D vs
   0,0214 fm canonique ;
 - courant de SURFACE porté par N_f lignes de framing (White) : repère
   de transport parallèle + taux de twist UNIFORME ajusté pour fermer
   les lignes avec Lk = +1 exactement (Tw = Lk - Wr, holonomie de
   Fuller absorbée) ; I_0 = e c / L_phys (un tour de charge par
   période), chaque ligne porte I_0/N_f ;
 - champ : Biot-Savart des 8640 segments ; intégrales volumiques par
   Monte-Carlo uniforme sur la boîte englobante (graine fixe), masque
   d'exclusion près des filaments (artefact de discrétisation de la
   nappe, stabilité contrôlée en N_f et delta).

OBSERVABLES (unités : mu_0 I_0^2/4pi = alpha hbar c/L^2 ;
 E = [alpha hbar c/(8 pi L^2)] int b^2 dV, b = somme dl x r/r^3) :
 - E_total : VALIDATION contre le canonique E = alpha m_l = 2,290 MeV
   (E_complet 2,283, -0,3 %) ;
 - tension moyenne <T> = [alpha hbar c/(8 pi L^3)] int (b_t^2-b_perp^2) dV
   (b_t = composante sur la tangente du point d'axe le plus proche),
   comparée à alpha·sigma_chi = 6,476 MeV/fm ;
 - décomposition intérieur/extérieur du tube + parts b_t/b_perp (T2).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from simulation.ideal_embedding import sono, resample
from theory.omega_ts import _knot_curve

ALPHA = 1 / 137.035999
HBARC = 197.32698          # MeV.fm
SQRT_SIGMA = 418.47        # MeV
SIGMA_FM = SQRT_SIGMA**2 / HBARC          # MeV/fm
M_L_NU = 0.75 * SQRT_SIGMA                # 313.85 MeV
L_PHYS = M_L_NU / SIGMA_FM                # fm (0.3537)

rng = np.random.default_rng(31)

print("=" * 74)
print("T1 — STRESS DE MAXWELL DU FRAMING SUR LE TRÈFLE IDÉAL (2026-08-28)")
print("=" * 74)

# ---------------------------------------------------- courbe idéale (SONO)
import os
CACHE = os.environ.get('SCRATCH_SONO',
    '/tmp/claude-1000/-home-euk9-projets-physique-nouvelle/'
    '4281778e-628f-46db-80e6-4e8c179f6e45/scratchpad/trefle_ideal_sono.npz')
if os.path.exists(CACHE):
    z = np.load(CACHE)
    P_id, ell = z['P'], float(z['ell'])
    print("courbe SONO chargée du cache")
else:
    P0 = np.asarray(_knot_curve("3_1", 200), float)
    P_id, info = sono(P0, iters=20000)
    ell = info["ropelength"]              # convention L/r
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, P=P_id, ell=ell)
P_id = P_id.copy()
P_id[:, 2] *= -1.0        # MIROIR : chiralité canonique (Wr > 0, Tw = 1-Wr < 0)
N = 360
P = resample(P_id, N)
dl_ax = np.roll(P, -1, 0) - P
L_geo = np.linalg.norm(dl_ax, axis=1).sum()
a_geo = L_geo / ell                       # rayon de tube (géométrique)
scale = L_PHYS / L_geo
P = P * scale
a = a_geo * scale
print(f"\nSONO : ropelength (L/r) = {ell:.3f} -> L/D = {ell/2:.3f} "
      f"(ACPR 16,3724 : {100*(ell/2/16.3724-1):+.2f} %)")
print(f"échelle : L = {L_PHYS:.4f} fm ; a = {a:.5f} fm ; D = {2*a:.5f} fm "
      f"(canonique 0,0214)")

# ---------------------------------------------------- repère de Bishop
dP = np.roll(P, -1, 0) - P
t_ax = dP / np.linalg.norm(dP, axis=1)[:, None]
e1 = np.zeros_like(P)
v = np.array([0.0, 0.0, 1.0])
v = v - (v @ t_ax[0]) * t_ax[0]
e1[0] = v / np.linalg.norm(v)
for i in range(1, N):
    v = e1[i-1] - (e1[i-1] @ t_ax[i]) * t_ax[i]
    e1[i] = v / np.linalg.norm(v)
# holonomie de fermeture
v_end = e1[-1] - (e1[-1] @ t_ax[0]) * t_ax[0]
v_end /= np.linalg.norm(v_end)
e2_0 = np.cross(t_ax[0], e1[0])
theta_H = np.arctan2(v_end @ e2_0, v_end @ e1[0])
e2 = np.cross(t_ax, e1)

# ---------------------------------------------------- writhe (Gauss) + Tw
mid = 0.5 * (P + np.roll(P, -1, 0))
Wr = 0.0
for i in range(N):
    r = mid[i] - mid
    d = np.linalg.norm(r, axis=1); d[i] = np.inf
    tri = np.einsum('j,ij->i', dP[i], np.cross(dP, r, axis=1))
    Wr += np.sum(tri / d**3)
Wr /= 4 * np.pi
Lk = 1.0
Tw_target = Lk - Wr
print(f"Wr(idéal) = {Wr:.4f} -> Tw = Lk - Wr = {Tw_target:+.4f} "
      f"(canonique -2,42) ; holonomie Bishop = {theta_H:+.4f} rad")

# taux de rotation uniforme FERMANT les lignes : total = 2 pi Tw_eff avec
# 2 pi Tw_eff + theta_H = 2 pi k (k entier le plus proche de Tw_target)
k_int = round(Tw_target + theta_H / (2*np.pi))
tw_total = 2 * np.pi * k_int - theta_H
print(f"rotation appliquée = {tw_total/(2*np.pi):+.4f} tours "
      f"(cible {Tw_target:+.4f} ; fermeture exacte des lignes)")

# ---------------------------------------------------- lignes de framing
s_cum = np.concatenate([[0], np.cumsum(np.linalg.norm(dP, axis=1))])[:-1]
phase = tw_total * s_cum / s_cum[-1] if False else tw_total * (s_cum / (L_PHYS))
def filaments(N_f):
    segs_mid, segs_dl = [], []
    for j in range(N_f):
        ang = 2*np.pi*j/N_f + phase
        F = P + a * (np.cos(ang)[:, None]*e1 + np.sin(ang)[:, None]*e2)
        dF = np.roll(F, -1, 0) - F
        segs_mid.append(0.5*(F + np.roll(F, -1, 0)))
        segs_dl.append(dF)
    return np.vstack(segs_mid), np.vstack(segs_dl)

# ---------------------------------------------------- Biot-Savart (chunks)
def field_and_dist(pts, segs_mid, segs_dl, chunk=400):
    sm = segs_mid.astype(np.float32); sd = segs_dl.astype(np.float32)
    b = np.zeros_like(pts, dtype=np.float64)
    dmin = np.full(len(pts), np.inf)
    for i0 in range(0, len(pts), chunk):
        p = pts[i0:i0+chunk].astype(np.float32)
        r = p[:, None, :] - sm[None, :, :]
        d = np.linalg.norm(r, axis=2)
        dmin[i0:i0+chunk] = d.min(axis=1)
        cr = np.cross(np.broadcast_to(sd[None, :, :], r.shape), r, axis=2)
        b[i0:i0+chunk] = (cr / (d**3)[:, :, None]).sum(axis=1)
        del r, d, cr
    return b, dmin

# ---------------------------------------------------- Monte-Carlo volume
lo = P.min(axis=0) - 3*a; hi = P.max(axis=0) + 3*a
V_box = np.prod(hi - lo)
M = 300_000
pts = rng.uniform(lo, hi, size=(M, 3))
# distance à l'axe + tangente locale (par blocs)
i_near = np.zeros(M, dtype=int); r_ax = np.zeros(M)
for i0 in range(0, M, 20000):
    d_ax = np.linalg.norm(pts[i0:i0+20000, None, :] - P[None, :, :], axis=2)
    i_near[i0:i0+20000] = d_ax.argmin(axis=1)
    r_ax[i0:i0+20000] = d_ax.min(axis=1)
t_loc = t_ax[i_near]

_cache = {}
def run(N_f, delta_frac):
    if N_f not in _cache:
        segs_mid, segs_dl = filaments(N_f)
        _cache[N_f] = field_and_dist(pts, segs_mid, segs_dl)
    b, dmin = _cache[N_f]
    b = b / N_f                       # chaque ligne porte I0/N_f
    delta = delta_frac * (2*np.pi*a/N_f)
    keep = dmin > delta
    b2 = (b**2).sum(axis=1)
    bt = (b * t_loc).sum(axis=1)
    bt2 = bt**2; bp2 = b2 - bt2
    w = V_box / M
    I_E   = (b2[keep]).sum() * w
    I_T   = ((bt2 - bp2)[keep]).sum() * w
    inside = keep & (r_ax < a)
    I_T_in  = ((bt2 - bp2)[inside]).sum() * w
    I_T_out = I_T - I_T_in
    pref_E = ALPHA * HBARC / (8*np.pi*L_PHYS**2)          # MeV / fm
    E_tot = pref_E * I_E                                   # MeV
    T_avg = pref_E * I_T / L_PHYS                          # MeV/fm
    return E_tot, T_avg, pref_E*I_T_in/L_PHYS, pref_E*I_T_out/L_PHYS, keep.sum()

print(f"\nMonte-Carlo : {M} points, boîte {V_box:.2e} fm^3")
print(f"{'N_f':>4s} {'delta':>6s} {'E (MeV)':>9s} {'vs a·m_l':>9s} "
      f"{'<T> MeV/fm':>11s} {'T/as':>7s} {'T_in':>8s} {'T_out':>8s}")
res = {}
for N_f, df in ((16, 0.5), (24, 0.5), (48, 0.5), (24, 0.25)):
    E, T, Tin, Tout, nk = run(N_f, df)
    res[(N_f, df)] = (E, T, Tin, Tout)
    print(f"{N_f:4d} {df:6.2f} {E:9.3f} {100*(E/(ALPHA*M_L_NU)-1):+8.1f}% "
          f"{T:11.3f} {T/(ALPHA*SIGMA_FM):7.3f} {Tin:8.3f} {Tout:8.3f}")

# extrapolation nappe : X(N_f) = X_inf + c/N_f sur (16, 24, 48) à delta=0.5
import numpy as _np
ns = _np.array([16., 24., 48.])
inv = 1./ns
for lbl, idx in (('E (MeV)', 0), ('<T> (MeV/fm)', 1), ('T_in', 2), ('T_out', 3)):
    ys = _np.array([res[(int(n), 0.5)][idx] for n in ns])
    A = _np.vstack([_np.ones(3), inv]).T
    coef, *_ = _np.linalg.lstsq(A, ys, rcond=None)
    print(f"extrapolation N_f->inf : {lbl:14s} = {coef[0]:+9.3f}"
          + (f"   ({coef[0]/(ALPHA*SIGMA_FM):+.3f} alpha·sigma_chi)" if idx else
             f"   (vs alpha·m_l {ALPHA*M_L_NU:.3f} : {100*(coef[0]/(ALPHA*M_L_NU)-1):+.1f} %)"))

print(f"""
Références : alpha·m_l = {ALPHA*M_L_NU:.3f} MeV ; alpha·sigma_chi = {ALPHA*SIGMA_FM:.3f} MeV/fm ;
identité d'ouverture (16/9)·alpha·sigma_chi = {16/9*ALPHA*SIGMA_FM:.3f} MeV/fm.
Lecture : T/as = la tension nette en unités de alpha·sigma_chi ; T_in = part
intérieure (solénoïde du twist, attendue TENSILE), T_out = part extérieure
(champ azimutal de la circulation, attendue COMPRESSIVE).""")
