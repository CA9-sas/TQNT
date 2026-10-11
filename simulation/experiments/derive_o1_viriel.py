"""Dérivation du O(1) = 0,81 — le viriel de dilatation (brouillon Maxwell).

MÉCANISME (exact, magnétostatique standard) : la tension intégrée du
tenseur de Maxwell est la force mécanique transmise le long du tube,
soit, à courant constant :

      <T> = - (dE/dl)|_I,a,Tw,forme        (viriel de dilatation)

Contrôle analytique (tube droit) : E = c_long·l + c_sol/l
  -> -dE/dl = -c_long + c_sol/l^2 = (compression log) + (tension
  solénoïde) — exactement la structure de signe mesurée au T1.

CONSÉQUENCE : le O(1) se décompose en invariants géométriques du nœud
idéal. En unités de mu0 I^2/4pi (= (16/9) alpha sigma_chi) :

      G_T = 2 pi^2 (1-Wr)^2 / x^2  -  [ n + x n'(x) ]

  x = ropelength (L/r) ; Wr = writhe idéal (Tw = 1-Wr, White) ;
  n(x) = E_long/(mu0 I^2 l/4pi) = N/2 (le noyau de Neumann par longueur)
  et x n' = sa réponse à la dilatation (axe dilaté, a fixe, tours fixes).

ICI : (1) vérification NUMÉRIQUE du viriel — dE/dl par noyau de Neumann
sur axes dilatés (lambda = 0,90..1,10) vs la tension MC du T1 ;
(2) mesure de n et x n' (courant total ET quasi-longitudinal) ;
(3) assemblage en forme close structurelle + candidats de formes pour
la part mesurée, CONSIGNÉS sans être retenus (règle du null).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from simulation.ideal_embedding import resample

ALPHA = 1 / 137.035999
HBARC = 197.32698
SQRT_SIGMA = 418.47
SIGMA_FM = SQRT_SIGMA**2 / HBARC
M_L_NU = 0.75 * SQRT_SIGMA
L_FM = M_L_NU / SIGMA_FM
UNIT = ALPHA * HBARC / L_FM**2            # mu0 I0^2/4pi en MeV/fm
AS = ALPHA * SIGMA_FM                     # alpha sigma_chi en MeV/fm

CACHE = ('/tmp/claude-1000/-home-euk9-projets-physique-nouvelle/'
         '4281778e-628f-46db-80e6-4e8c179f6e45/scratchpad/trefle_ideal_sono.npz')
z = np.load(CACHE)
P_id, ell = z['P'].copy(), float(z['ell'])
P_id[:, 2] *= -1.0

print("=" * 74)
print("O(1) EN FORME CLOSE — LE VIRIEL DE DILATATION      (2026-08-28)")
print("=" * 74)

N = 240
P0 = resample(P_id, N)
L0 = np.linalg.norm(np.roll(P0, -1, 0) - P0, axis=1).sum()
a = L0 / ell                              # rayon de tube FIXE (unités courbe)

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
    return t, u, np.cross(t, u), ds

# writhe idéal
dP0 = np.roll(P0, -1, 0) - P0
mid0 = P0 + 0.5 * dP0
Wr = 0.0
for i in range(N):
    r = mid0[i] - mid0
    d = np.linalg.norm(r, axis=1); d[i] = np.inf
    Wr += np.sum(np.einsum('j,ij->i', dP0[i], np.cross(dP0, r, axis=1)) / d**3)
Wr /= 4 * np.pi
x_rope = ell                              # L/r

def neumann(P, m, k_turns_extra):
    """Énergie de Neumann N/L (sans dimension) des m lignes de framing,
    rotation totale = -holonomie + 2 pi k (k = tours entiers appliqués)."""
    t, u, v, ds = frames(P)
    n = len(P)
    s_arc = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
    L = ds.sum()
    w_tr = u[-1] - (u[-1] @ t[0]) * t[0]
    w_tr /= np.linalg.norm(w_tr)
    e2_0 = np.cross(t[0], u[0])
    th_H = np.arctan2(w_tr @ e2_0, w_tr @ u[0])
    rot = 2 * np.pi * k_turns_extra - th_H
    ph = 2 * np.pi * np.arange(m)[None, :] / m + rot * (s_arc / L)[:, None]
    X = (P[:, None, :] + a * np.cos(ph)[:, :, None] * u[:, None, :]
         + a * np.sin(ph)[:, :, None] * v[:, None, :])
    dl = np.roll(X, -1, axis=0) - X
    Mid = X + 0.5 * dl
    Nh = 0.0
    for i in range(n):
        diff = Mid[i][:, None, None, :] - Mid[None, :, :, :]
        d = np.linalg.norm(diff, axis=3)
        d[d < 1e-12] = np.inf
        dot = np.einsum('kc,jlc->kjl', dl[i], dl)
        Nh += (dot / d).sum() / m**2
    return Nh / L, rot / (2 * np.pi), L

# tours entiers : framing physique Tw = 1 - Wr -> k = round(Tw + th/2pi)
# (mesuré au T1 : rotation appliquée -2.4598 tours = k...)
m = 24
lams = np.array([0.90, 0.95, 1.00, 1.05, 1.10])
print(f"\nWr = {Wr:.4f} ; Tw = 1-Wr = {1-Wr:+.4f} ; x = ropelength = {x_rope:.3f}")
print(f"terme solénoïde clos : 2 pi^2 (1-Wr)^2/x^2 = "
      f"{2*np.pi**2*(1-Wr)**2/x_rope**2:.4f}")

# ---------------- CONTRÔLE EXACT : la boucle circulaire ----------------
R_c = L0 / (2 * np.pi)
th = np.linspace(0, 2*np.pi, N, endpoint=False)
ncirc, nslope = [], []
for lam in (0.95, 1.0, 1.05):
    Pc = np.stack([R_c*lam*np.cos(th), R_c*lam*np.sin(th),
                   np.zeros_like(th)], axis=1)
    Nh, _, _ = neumann(Pc, m, 0)
    ncirc.append((lam, Nh/2))
n_th = np.log(8*R_c/a) - 2
slope_num = (ncirc[2][1]-ncirc[0][1])/0.10
print(f"\nCONTRÔLE cercle (même L, même a) : n = {ncirc[1][1]:.4f} vs "
      f"ln(8R/a)-2 = {n_th:.4f} ({100*(ncirc[1][1]/n_th-1):+.2f} %) ; "
      f"lambda·dn/dlambda = {slope_num:.4f} vs 1 exact "
      f"({100*(slope_num-1):+.2f} %)")

print(f"\n--- balayage de dilatation (axe x lambda, a fixe, tours fixes) ---")
print(f"{'lambda':>7s} {'n=N/2 (framing)':>16s} {'n (quasi-long)':>15s}")
res_f, res_l = [], []
for lam in lams:
    Pl = P0 * lam
    # framing physique : k tel que rotation ~ Tw (indépendant de lambda)
    t_, u_, v_, ds_ = frames(Pl)
    # même construction que T1 : k = round(Tw + th_H/2pi) — th_H invariant
    # par dilatation (géométrie angulaire) ; on passe k directement.
    # essais : trouver k donnant rot ~ -2.46 et k donnant rot minimale
    got = {}
    for k in (-3, -2, -1, 0, 1):
        Nh, tours, L = neumann(Pl, m, k)
        got[k] = (Nh, tours)
    k_fr = min(got, key=lambda k: abs(got[k][1] - (1 - Wr)))
    k_lg = min(got, key=lambda k: abs(got[k][1]))
    res_f.append(got[k_fr][0] / 2)
    res_l.append(got[k_lg][0] / 2)
    print(f"{lam:7.2f} {res_f[-1]:16.4f} {res_l[-1]:15.4f}"
          f"   (rot {got[k_fr][1]:+.3f} / {got[k_lg][1]:+.3f} tours)")

res_f = np.array(res_f); res_l = np.array(res_l)
# n(lambda) ; E = UNIT * lambda*L0_phys * n(lambda) ; T = -dE/dl
# dE/dl = UNIT*[n + lambda dn/dlambda] (l = lambda L0)
for name, arr in (("framing complet", res_f), ("quasi-longitudinal", res_l)):
    c = np.polyfit(lams, arr, 2)
    n1 = np.polyval(c, 1.0)
    dn = np.polyval(np.polyder(c), 1.0)
    GT = -(n1 + dn)
    print(f"""
{name} :
   n(1) = {n1:.4f} ; lambda·dn/dlambda = {dn:+.4f}
   G_T(viriel, sans solénoïde explicite) = -(n + n') = {GT:+.4f}
   T = {GT*UNIT:+.3f} MeV/fm = {GT*UNIT/AS:+.4f} alpha·sigma_chi""")

# le solénoïde est DANS le framing complet (différence des deux)
c_f = np.polyfit(lams, res_f, 2); c_l = np.polyfit(lams, res_l, 2)
n_f, dn_f = np.polyval(c_f, 1.0), np.polyval(np.polyder(c_f), 1.0)
n_l, dn_l = np.polyval(c_l, 1.0), np.polyval(np.polyder(c_l), 1.0)
GT_f = -(n_f + dn_f)
print(f"""
=== ASSEMBLAGE ===
   T(viriel, framing complet) = {GT_f*UNIT:+.3f} MeV/fm = {GT_f*UNIT/AS:+.4f} as
   T(stress MC, T1)           = -5.27 MeV/fm  = -0.813 as  (+/- 15-20 %)
   -> VERDICT viriel vs stress : écart {100*(GT_f*UNIT/(-5.27)-1):+.1f} %
   part solénoïdale (framing - quasi-long) : n_sol = {n_f-n_l:+.4f},
   réponse {dn_f-dn_l:+.4f} (attendu {-(n_f-n_l):+.4f} si E_sol ~ 1/l)

FORME CLOSE STRUCTURELLE (unités (16/9) alpha sigma_chi) :
   G_T = 2pi^2(1-Wr)^2/x^2 - [n_long + x n_long']
       = {2*np.pi**2*(1-Wr)**2/x_rope**2:+.4f} - [{n_l:.4f} + {dn_l:+.4f}]
   O(1) final = (16/9)|G_T| = {16/9*abs(GT_f):.4f}

Candidats de forme pour la part mesurée [n+n']_long = {n_l+dn_l:.4f}
(CONSIGNÉS, AUCUN retenu — règle du null) :
   9/16 = 0.5625 ({100*((n_l+dn_l)/0.5625-1):+.1f} %) ; 4/7 = 0.5714
   ({100*((n_l+dn_l)/(4/7)-1):+.1f} %) ; ln(2)-1/8 = 0.5681
   ({100*((n_l+dn_l)/0.5681-1):+.1f} %)""")
