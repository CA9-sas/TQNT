"""Pincer le facteur d'inductance de H2 (fluxoide) depuis la theorie ?

Suite de derive_beta_otimes_test.py : E(dLk=1) = Phi_chi^2/(2 L_ind) encadrait la cible
1.293 MeV a +8.1%/-6.2% selon la convention c in [7/4, 2] de L = mu0 r (ln(8r/a) - c).

TROIS PINCEMENTS THEORIQUES TESTES :
 (A) Quel courant porte le fluxoide ? Le courant de TWIST a un courant net NUL
     (I_enc(rho) = 2 pi omega rho^2 B_z / mu0 dans le coeur, mais B_phi saute a 0 a la
     frontiere -> courant de retour de surface = -I_coeur : total 0, comme il se doit
     pour un tube isole). Le fluxoide est donc porte par la CIRCULATION DE CHARGE
     (chaine jonction mu_p-mu_n : q a v=c). Des charges mobiles sur une boucle se
     repoussent -> equilibre a la SURFACE du tube (+ image Meissner duale : l'ecrantage
     vit a la frontiere). => c = 2 PINCE (courant de surface, terme interne Y=0).
 (B) Correction d'anneau epais : a/r = 0.209/0.99 = 0.21, la formule mince est limite.
     Calcul exact du L de surface par moyenne GMD sur le tore (integrales elliptiques).
 (C) Vraie geometrie de la boucle : le circuit qui repond est le COEUR 3_1 (porteur du
     moment magnetique, V6 torsion) ou un 7_2 -- pas un cercle. Inductance de Neumann
     regularisee sur la courbe reelle, tube a = d_tube, aux echelles de contact
     (min-distance = 2a) et de ropelength (L_tot = ropelength_diam x 2a).

Validation : la routine de Neumann est validee sur le cercle contre mu0 R (ln(8R/a)-2).
"""
import math
import os
import sys

import numpy as np
from scipy.special import ellipk, ellipe

sys.path.insert(0, os.path.dirname(__file__))
from repair_curves_braid import braid_curve, alexander, BRAID_WORDS  # noqa: E402

MU0 = 4e-7 * math.pi
ALPHA = 1.0 / 137.0
H_OVER_E = 4.135667696e-15
J_PER_MEV = 1.602176634e-13
FM = 1e-15

R_LINK = 0.99          # fm
D_TUBE = 0.209         # fm (rayon du tube)
TARGET = 1.293         # MeV
ROPE_DIAM = {"3_1": 16.372, "7_2": 32.96}   # ropelength en unites de DIAMETRE

PHI = ALPHA * H_OVER_E


def energy_MeV(L_H):
    return PHI**2 / (2 * L_H) / J_PER_MEV


def L_star():
    """Inductance requise pour la cible exacte."""
    return PHI**2 / (2 * TARGET * J_PER_MEV)


# ------------------------------------------------------------------ (B) tore
def mutual_rings(r1, r2, dz):
    """Inductance mutuelle de deux anneaux coaxiaux (formule de Maxwell)."""
    k2 = 4 * r1 * r2 / ((r1 + r2) ** 2 + dz**2)
    k = math.sqrt(k2)
    return MU0 * math.sqrt(r1 * r2) * ((2 / k - k) * ellipk(k2) - (2 / k) * ellipe(k2))


def L_torus_surface(R, a, n=720):
    """L exact d'un anneau a courant de surface : moyenne GMD sur le tore."""
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    vals = [mutual_rings(R, R + a * math.cos(t), a * math.sin(t)) for t in th]
    return float(np.mean(vals))


# --------------------------------------------------------- (C) Neumann knots
def resample(P, n):
    """Reechantillonnage uniforme en abscisse curviligne (courbe fermee)."""
    P = np.asarray(P, float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    t = np.linspace(0, L, n, endpoint=False)
    Q = np.empty((n, 3))
    for c in range(3):
        Q[:, c] = np.interp(t, s, np.concatenate([P[:, c], P[:1, c]]))
    return Q


def fourier_smooth(P, nharm):
    n = len(P)
    Q = np.empty_like(P)
    for c in range(3):
        F = np.fft.rfft(P[:, c])
        F[nharm + 1:] = 0
        Q[:, c] = np.fft.irfft(F, n)
    return Q


def curve_length(P):
    return float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum())


def min_nonlocal_dist(P):
    """Distance minimale entre points d'ecart curviligne > 1/8 de la longueur locale."""
    n = len(P)
    guard = max(8, n // 12)
    dmin = np.inf
    for i in range(n):
        j = np.arange(i + guard, min(n, i + n - guard))
        if len(j) == 0:
            continue
        dmin = min(dmin, float(np.linalg.norm(P[j] - P[i], axis=1).min()))
    return dmin


def bishop_frame(P):
    """Repere de Bishop (transport parallele, pas de glitch de Frenet)."""
    n = len(P)
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.zeros_like(P)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, T[0])) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    N[0] = seed - np.dot(seed, T[0]) * T[0]
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, n):
        v = N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
        N[i] = v / np.linalg.norm(v)
    B = np.cross(T, N)
    return T, N, B


def mutual_neumann(P, Q):
    """Mutuelle de Neumann entre deux courbes fermees (integrande finie)."""
    dlP = np.roll(P, -1, 0) - P
    dlQ = np.roll(Q, -1, 0) - Q
    mP = P + 0.5 * dlP
    mQ = Q + 0.5 * dlQ
    tot = 0.0
    for i in range(len(P)):
        diff = mQ - mP[i]
        tot += float((dlQ @ dlP[i] / np.linalg.norm(diff, axis=1)).sum())
    return MU0 / (4 * math.pi) * tot


def L_gmd(P, a, ntheta=8):
    """L d'un tube (courant de surface) : moyenne GMD axe <-> filaments de surface."""
    _, N, B = bishop_frame(P)
    vals = []
    for th in np.linspace(0, 2 * math.pi, ntheta, endpoint=False):
        Q = P + a * (math.cos(th) * N + math.sin(th) * B)
        vals.append(mutual_neumann(P, Q))
    return float(np.mean(vals)), (max(vals) - min(vals)) / abs(np.mean(vals)) * 100


def trefoil(n=4096):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    P = np.stack([np.sin(t) + 2 * np.sin(2 * t),
                  np.cos(t) - 2 * np.cos(2 * t),
                  -np.sin(3 * t)], axis=1)
    return P


def report_knot(name, P, a):
    P = resample(P, 3000)
    ltot0, dmin0 = curve_length(P), min_nonlocal_dist(P)
    rows = []
    for label, scale in (("contact (min dist = 2a)", 2 * a / dmin0),
                         ("ropelength (L = rl x 2a)", ROPE_DIAM[name] * 2 * a / ltot0)):
        Q = P * scale
        lq = curve_length(Q)
        Lmean, spread = L_gmd(Q * FM, a * FM)
        E = energy_MeV(Lmean)
        rows.append((label, lq, Lmean, spread, E))
    print(f"\n--- {name} (tube a = {a} fm) ---")
    print(f"   brute : longueur {ltot0:.2f} (u.a.), min-dist non locale {dmin0:.3f} (u.a.)")
    for label, lq, Lm, sp, E in rows:
        print(f"   {label:26}: l = {lq:6.2f} fm ; L = {Lm/MU0/FM:6.3f} mu0.fm "
              f"(var-theta {sp:4.1f}%) ; E = {E:6.3f} MeV ({(E/TARGET-1)*100:+5.1f}%)")


def run():
    Lstar = L_star()
    print("=== Cible : E = 1.293 MeV  <=>  L* = %.3f mu0.fm ===" % (Lstar / MU0 / FM))

    print("\n(A) Courant du fluxoide = circulation de charge (le twist a courant net 0)")
    print("    -> charges mobiles se repoussent -> courant de SURFACE -> c = 2 PINCE.")

    print("\n(B) Anneau epais, courant de surface (moyenne GMD exacte, tore R=0.99, a=0.209) :")
    Lthin = MU0 * R_LINK * FM * (math.log(8 * R_LINK / D_TUBE) - 2)
    Lthick = L_torus_surface(R_LINK * FM, D_TUBE * FM)
    for lab, L in (("formule mince (c=2)", Lthin), ("tore exact (GMD)  ", Lthick)):
        print(f"    {lab}: L = {L/MU0/FM:.3f} mu0.fm ; E = {energy_MeV(L):.3f} MeV "
              f"({(energy_MeV(L)/TARGET-1)*100:+.1f}%)")

    # cercle a l'echelle ropelength du trefle (l = 16.372 x 2a)
    R_rope = ROPE_DIAM["3_1"] * 2 * D_TUBE / (2 * math.pi)
    Lrope = L_torus_surface(R_rope * FM, D_TUBE * FM)
    print(f"    cercle rope-3_1 (R={R_rope:.2f} fm) : L = {Lrope/MU0/FM:.3f} mu0.fm ; "
          f"E = {energy_MeV(Lrope):.3f} MeV ({(energy_MeV(Lrope)/TARGET-1)*100:+.1f}%)")

    # validation de la routine GMD-Bishop sur le cercle (vs GMD-tore exact)
    n = 2000
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    C = np.stack([R_LINK * np.cos(t), R_LINK * np.sin(t), 0 * t], 1) * FM
    Lnum, _ = L_gmd(C, D_TUBE * FM)
    print(f"\n    [validation GMD-Bishop/cercle : {Lnum/MU0/FM:.3f} vs tore exact {Lthick/MU0/FM:.3f} mu0.fm]")

    print("\n(C) Vraie geometrie de la boucle compagne (Neumann, courant de surface) :")
    report_knot("3_1", trefoil(), D_TUBE)

    word, ns, target_alex = BRAID_WORDS["7_2"]
    P72, _, _ = braid_curve(word, ns, pts_per_seg=40)
    P72 = resample(P72, 4096)
    P72s = fourier_smooth(P72, 48)
    try:
        alex = alexander(P72s)
        print(f"\n   [garde Alexander 7_2 lisse : {alex} (cible {target_alex})]")
    except Exception as e:  # pragma: no cover
        print(f"\n   [garde Alexander indisponible : {e}]")
    report_knot("7_2", P72s, D_TUBE)

    print("\n=== Lecture ===")
    print("L* = %.3f mu0.fm ; le candidat retenu est la geometrie qui atterrit dessus" % (Lstar / MU0 / FM))
    print("sans reglage (c pince par (A), echelle par contact/ropelength).")


if __name__ == "__main__":
    run()
