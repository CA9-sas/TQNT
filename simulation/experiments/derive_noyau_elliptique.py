"""Noyau elliptique : inductance EXACTE de la nappe uniforme (sections moyennees).

LE CALCUL RESTANT (V6 §voie de reduction, brouillon beta_otimes §10) : remplacer
le tube-mince par le noyau exact anneau-anneau. Objet :

  L = (mu0/4pi) oint oint ds ds' [T(s).T(s')] < w w' / |r - r'| >_{theta,theta'}

  r = P(s) + a u(theta) (section droite, repere de Bishop) ;
  w = 1 - a kappa(s).u(theta)  (element de longueur du filament de surface --
  EXACT en Bishop : les tangentes des filaments sont paralleles a l'axe, toute
  la geometrie de section entre par ce poids scalaire et la moyenne de 1/r).

La double moyenne de section = le noyau 'elliptique' (forme close K(k), E(k)
de Maxwell pour deux anneaux) -- calcule ici par quadrature directe sur
(theta, theta') avec grilles DECALEES d'un demi-pas (aucune evaluation
singuliere ; la singularite log en |s-s'| est integrable et resolue par
ds << a). Contrairement a la decomposition en filaments (log-divergente,
ECARTEE au §10), cette formulation est bien posee : c'est la double integrale
de nappe exacte, sans auto-terme ad hoc.

VALIDATION (test acide, celui qui a tue les filaments) : cercle r/a = 3 et 5
-> tore analytique (L_torus_surface) a <0.3% EN CONVERGENCE (n_s, n_q).
Puis : coeur 3_1 du nucleon (brut + regularise sigma = 0.3a).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from derive_inductance_factor_test import (  # noqa: E402
    bishop_frame, L_torus_surface, L_gmd, energy_MeV, D_TUBE, MU0, FM, TARGET,
)
from simulation.ideal_embedding import resample, curve_length  # noqa: E402


def _G_cyl(delta, a):
    """Noyau elliptique coaxial : << 1/|Delta z + a(u-u')| >> pour deux
    sections paralleles de rayon a separees de Delta.

    Forme close : (2/pi) K(m) / sqrt(Delta^2 + 4a^2), m = 4a^2/(Delta^2+4a^2).
    Limite Delta >> a : 1/Delta (tube-mince retrouve)."""
    from scipy.special import ellipk
    den = np.sqrt(delta ** 2 + 4 * a ** 2)
    m = 4 * a ** 2 / den ** 2
    return (2 / math.pi) * ellipk(m) / den


def _diag_cell(ds, a, ngauss=64):
    """Contribution exacte de la cellule diagonale (s = s', carre de cote ds) :
    integrale triangulaire du noyau coaxial, int_{-ds}^{ds} (1-|u|/ds) G(u) du.

    La singularite log du continu est integrable ; rectitude locale exacte a
    O((ds/a)^2) pres (ds << a en pratique)."""
    # Gauss-Legendre sur (0, ds), symetrie x2
    x, w = np.polynomial.legendre.leggauss(ngauss)
    u = 0.5 * ds * (x + 1)
    wu = 0.5 * ds * w
    return 2.0 * float(((1 - u / ds) * _G_cyl(u, a) * wu).sum())


def L_sheet_exact(P, a, nq=12, chunk=40):
    """Inductance exacte de la nappe uniforme (double quadrature de sections).

    P : axe ferme (n, 3) ; a : rayon du tube ; nq : points de quadrature par
    cercle (grilles theta et theta' decalees d'un demi-pas). La cellule
    DIAGONALE (log-divergente en continu, source de la derive ln(nq) de la
    v1) est exclue de la quadrature et remplacee par son integrale
    ANALYTIQUE (noyau coaxial elliptique, _diag_cell).
    """
    P = np.asarray(P, float)
    n = len(P)
    T, N, B = bishop_frame(P)
    dl = np.linalg.norm(np.roll(P, -1, 0) - np.roll(P, 1, 0), axis=1) / 2
    # courbure vectorielle kappa = dT/ds
    dT = (np.roll(T, -1, 0) - np.roll(T, 1, 0)) / (2 * dl)[:, None]
    k1 = np.einsum('ij,ij->i', dT, N)
    k2 = np.einsum('ij,ij->i', dT, B)
    th = 2 * math.pi * np.arange(nq) / nq            # grille source
    thp = th + math.pi / nq                          # grille cible (decalee)
    cs, sn = np.cos(th), np.sin(th)
    csp, snp = np.cos(thp), np.sin(thp)
    # points de surface et poids : (n, nq, 3), (n, nq)
    S1 = P[:, None, :] + a * (cs[None, :, None] * N[:, None, :] +
                              sn[None, :, None] * B[:, None, :])
    W1 = 1.0 - a * (k1[:, None] * cs[None, :] + k2[:, None] * sn[None, :])
    S2 = P[:, None, :] + a * (csp[None, :, None] * N[:, None, :] +
                              snp[None, :, None] * B[:, None, :])
    W2 = 1.0 - a * (k1[:, None] * csp[None, :] + k2[:, None] * snp[None, :])
    tot = 0.0
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        # distances (i, j, q, q')
        d = S1[i0:i1, None, :, None, :] - S2[None, :, None, :, :]
        d = np.linalg.norm(d, axis=-1)
        ww = W1[i0:i1, None, :, None] * W2[None, :, None, :]
        kern = (ww / d).mean(axis=(2, 3))            # (chunk, n)
        tt = T[i0:i1] @ T.T                          # (chunk, n)
        blk = tt * kern * dl[i0:i1, None] * dl[None, :]
        # cellule diagonale : retiree ici, remplacee analytiquement ci-dessous
        for i in range(i0, i1):
            blk[i - i0, i] = 0.0
        tot += float(blk.sum())
    # remplacement analytique de la diagonale (T.T = 1, poids ~ 1 a O((ds/a)^2))
    tot += float(sum(dl[i] * _diag_cell(dl[i], a) for i in range(n)))
    return MU0 / (4 * math.pi) * tot


def _ring_kernel_weighted(X, c, Nv, Bv, a, k1, k2):
    """< w(theta)/|x - c - a u(theta)| >_theta EXACT (potentiel d'anneau pondere).

    X : points d'evaluation (..., 3) ; anneau de centre c, plan (Nv, Bv),
    rayon a ; poids w = 1 - a(k1 cos + k2 sin). Formes closes :
      Phi0 = (2/pi) K(m)/S ;  I_c = (2/(pi B S)) [A K(m) - S^2 E(m)] ;
      resultat = Phi0 - a (k1 cos psi + k2 sin psi) I_c,
    avec A = z^2 + rho^2 + a^2, B = 2 a rho, S = sqrt(A+B), m = 2B/(A+B),
    (rho, psi, z) coordonnees cylindriques de x dans le repere de l'anneau."""
    from scipy.special import ellipk, ellipe
    d = X - c
    Tv = np.cross(Nv, Bv)
    z = d @ Tv
    xn = d @ Nv
    xb = d @ Bv
    rho = np.sqrt(xn ** 2 + xb ** 2)
    A = z ** 2 + rho ** 2 + a ** 2
    Bq = 2 * a * rho
    S = np.sqrt(A + Bq)
    m = np.clip(2 * Bq / (A + Bq), 0.0, 1.0 - 1e-13)
    K, E = ellipk(m), ellipe(m)
    phi0 = (2 / math.pi) * K / S
    with np.errstate(divide='ignore', invalid='ignore'):
        Ic = np.where(Bq > 1e-30 * A,
                      (2 / math.pi) * (A * K - (A + Bq) * E) / (Bq * S),
                      0.0)
    cpsi = np.where(rho > 0, xn / np.maximum(rho, 1e-300), 0.0)
    spsi = np.where(rho > 0, xb / np.maximum(rho, 1e-300), 0.0)
    return phi0 - a * (k1 * cpsi + k2 * spsi) * Ic


def _selftest_ring_kernel(tol=1e-9):
    """Verifie la forme close contre la quadrature brute (512 pts)."""
    rng = np.random.default_rng(3)
    a = 0.7
    c = np.zeros(3)
    Nv, Bv = np.array([1., 0, 0]), np.array([0, 1., 0])
    k1, k2 = 0.31, -0.44
    th = 2 * math.pi * np.arange(512) / 512
    ring = c + a * (np.cos(th)[:, None] * Nv + np.sin(th)[:, None] * Bv)
    w = 1 - a * (k1 * np.cos(th) + k2 * np.sin(th))
    for _ in range(6):
        x = rng.normal(size=3) * 1.5
        if np.linalg.norm(x - c) < 1.2 * a:
            x *= 2.5
        brute = float((w / np.linalg.norm(x - ring, axis=1)).mean())
        exact = float(_ring_kernel_weighted(x[None, :], c, Nv, Bv, a, k1, k2)[0])
        assert abs(exact - brute) < tol * max(1.0, abs(brute)), (exact, brute)
    return True


def L_sheet_ring(P, a, nq=48, chunk=8):
    """v3 : reduction 1D par le potentiel d'anneau (interieur EXACT).

    G(s,s') = (1/nq) sum_theta' w'(theta') * [noyau d'anneau pondere exact
    en x(theta')] : quadrature 1D d'une fonction lisse (singularite log
    integrable seulement au contact exact). Diagonale : cellule analytique
    coaxiale (_diag_cell), comme en v2."""
    P = np.asarray(P, float)
    n = len(P)
    T, N, B = bishop_frame(P)
    dl = np.linalg.norm(np.roll(P, -1, 0) - np.roll(P, 1, 0), axis=1) / 2
    dT = (np.roll(T, -1, 0) - np.roll(T, 1, 0)) / (2 * dl)[:, None]
    k1 = np.einsum('ij,ij->i', dT, N)
    k2 = np.einsum('ij,ij->i', dT, B)
    thp = 2 * math.pi * (np.arange(nq) + 0.5) / nq
    csp, snp = np.cos(thp), np.sin(thp)
    # points cibles et poids : (n, nq, 3), (n, nq)
    S2 = P[:, None, :] + a * (csp[None, :, None] * N[:, None, :] +
                              snp[None, :, None] * B[:, None, :])
    W2 = 1.0 - a * (k1[:, None] * csp[None, :] + k2[:, None] * snp[None, :])
    tot = 0.0
    for i in range(n):           # anneau source i (noyau exact), cibles j
        X = S2.reshape(-1, 3)
        ker = _ring_kernel_weighted(X, P[i], N[i], B[i], a, k1[i], k2[i])
        ker = ker.reshape(n, nq)
        Gi = (W2 * ker).mean(axis=1)          # (n,)
        Gi[i] = 0.0                            # diagonale : analytique ci-dessous
        tt = T[i] @ T.T if False else T @ T[i]
        tot += float((tt * Gi * dl).sum() * dl[i])
    tot += float(sum(dl[i] * _diag_cell(dl[i], a) for i in range(n)))
    return MU0 / (4 * math.pi) * tot


def circle(r, n):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t), 0 * t], axis=1)


def smooth_gauss(P, sigma):
    n = len(P)
    ds = curve_length(P) / n
    sig = sigma / ds
    k = np.fft.fftfreq(n) * n
    ker = np.exp(-0.5 * (2 * np.pi * k / n) ** 2 * sig ** 2)
    return np.real(np.fft.ifft(np.fft.fft(P, axis=0) * ker[:, None], axis=0))


def run():
    a = D_TUBE * FM
    print("=== Noyau elliptique : nappe exacte par sections moyennees ===\n", flush=True)

    print("[1] Test acide (cercle vs tore analytique) :", flush=True)
    for ra in (3.0, 5.0):
        R = ra * a
        L_ana = L_torus_surface(R, a)
        for ns in (400, 800, 1600):
            for nq in (8, 12, 16):
                L = L_sheet_exact(circle(R, ns), a, nq=nq)
                print("   r/a=%.0f n_s=%4d n_q=%2d : L = %.4f mu0.fm "
                      "(analytique %.4f, %+.2f%%)" % (
                          ra, ns, nq, L / MU0 / FM, L_ana / MU0 / FM,
                          (L / L_ana - 1) * 100), flush=True)

    print("\n[2] Coeur 3_1 du nucleon :", flush=True)
    root = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    C0 = np.load(os.path.join(root, "nucleon_ideal_0.npy"))
    for tag, sig in (("brut", None), ("regularise sigma=0.3a", 0.3)):
        for ns in (800, 1600):
            P = resample(C0, ns)
            if sig is not None:
                P = smooth_gauss(P, sig)
            X = P * D_TUBE * FM
            Lg, _ = L_gmd(X, a)
            L = L_sheet_exact(X, a, nq=12)
            print("   %-22s n_s=%4d : nappe exacte L = %.4f -> E = %.3f MeV "
                  "(%+.1f%%) | gmd %.4f (E %+.1f%%)" % (
                      tag, ns, L / MU0 / FM, energy_MeV(L),
                      (energy_MeV(L) / TARGET - 1) * 100,
                      Lg / MU0 / FM, (energy_MeV(Lg) / TARGET - 1) * 100), flush=True)

    print("\nLECTURE : si [1] converge a <0.3%% et [2] est stable en n_s et sous", flush=True)
    print("regularisation, c'est E_exact -- le verdict final du modele de courant.", flush=True)


if __name__ == "__main__":
    run()
