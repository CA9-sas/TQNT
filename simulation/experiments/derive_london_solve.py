"""Resolution de London : la distribution physique du fluxoide (marche restante).

ETAT (brouillon beta_otimes §12) : encadrement E in [-1.6% (nappe uniforme,
exacte v3) ; +1.4% (flux d'axe, gmd)] -- l'objet correct est la distribution
de LONDON (flux egal a travers tous les contours de surface), entre les deux.

TROIS ETAGES :
  1. TORE COAXIAL, EXACT 1D : strips poloidaux = anneaux coaxiaux
     (mutual_rings de Maxwell, auto-termes de bande GMD g = e^{-3/2} w) ;
     M I = Phi 1, somme(I) = I  =>  L_London = 1/(1^T M^{-1} 1).
     Cross-checks : L_unif(1D) = v3 (machinerie 3D independante) ;
     L_axe(1D) = L_torus_surface. Ecart uniforme->London en (a kappa)^2.
  2. VARIATIONNEL DIPOLAIRE sur le tore : base {uniforme, cos(theta)} ->
     L_var = L_unif - E01^2/E11 ; compare a l'exact 1D = qualite de l'ansatz
     (le dipole doit capturer l'essentiel du gap).
  3. NUCLEON, variationnel dipolaire aligne sur la courbure locale :
     lambda(s,theta) = (I/2pi)[1 + c * (kappa_hat(s).u(theta))] -- les
     elements de matrice utilisent les formes closes du potentiel d'anneau
     (le moment I_c EST le canal dipolaire ; les corrections w x dipole sont
     d'ordre (a kappa)^3, negligees). Minimisation en c -> L_var(nucleon),
     BORNE SUPERIEURE rigoureuse de L_London => E_London >= E_var ;
     le residu de troncature est calibre a l'etage 2.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from derive_inductance_factor_test import (  # noqa: E402
    bishop_frame, mutual_rings, L_torus_surface, energy_MeV,
    D_TUBE, MU0, FM, TARGET,
)
from derive_noyau_elliptique import (  # noqa: E402
    L_sheet_ring, _diag_cell, circle,
)
from simulation.ideal_embedding import resample  # noqa: E402

GMD_STRIP = math.exp(-1.5)


def ring_self(r, w):
    """Auto-inductance d'une bande annulaire (rayon r, largeur w) :
    anneau de rayon de fil effectif g = e^{-3/2} w (GMD de bande)."""
    return MU0 * r * (math.log(8 * r / (GMD_STRIP * w)) - 2)


def torus_london_1d(R, a, N=128):
    """Tore coaxial : L_axe, L_unif, L_London (exact 1D) + variationnel dipole."""
    th = 2 * math.pi * (np.arange(N) + 0.5) / N
    r = R + a * np.cos(th)
    z = a * np.sin(th)
    w = 2 * math.pi * a / N
    M = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i, j] = ring_self(r[i], w)
            else:
                M[i, j] = mutual_rings(r[i], r[j], z[i] - z[j])
    ones = np.ones(N)
    L_unif = float(ones @ M @ ones) / N ** 2
    sol = np.linalg.solve(M, ones)
    L_london = 1.0 / float(ones @ sol)
    # variationnel {1, cos theta} : lambda ~ 1 + c cos ; <cos> = 0
    d = np.cos(th)
    E00, E01, E11 = (ones @ M @ ones) / N**2, (ones @ M @ d) / N**2, (d @ M @ d) / N**2
    L_var = E00 - E01 ** 2 / E11
    L_axe = L_torus_surface(R, a)
    return L_axe, L_unif, L_london, L_var


def L_nucleon_var_dipole(P, a, nq=48):
    """Nucleon : variationnel dipolaire aligne sur kappa_hat(s).

    E(c) = E00 + 2c E01 + c^2 E11 avec la base {1, kappa_hat.u} ;
    interieur exact (Phi0, I_c), exterieur quadrature 1D (v3).
    Retourne (L_unif=E00_complet_avec_poids, L_var)."""
    from derive_noyau_elliptique import _ring_kernel_weighted
    from scipy.special import ellipk, ellipe
    P = np.asarray(P, float)
    n = len(P)
    T, N, B = bishop_frame(P)
    dl = np.linalg.norm(np.roll(P, -1, 0) - np.roll(P, 1, 0), axis=1) / 2
    dT = (np.roll(T, -1, 0) - np.roll(T, 1, 0)) / (2 * dl)[:, None]
    k1 = np.einsum('ij,ij->i', dT, N)
    k2 = np.einsum('ij,ij->i', dT, B)
    kmag = np.sqrt(k1 ** 2 + k2 ** 2)
    kh1 = np.where(kmag > 1e-12, k1 / np.maximum(kmag, 1e-300), 0.0)
    kh2 = np.where(kmag > 1e-12, k2 / np.maximum(kmag, 1e-300), 0.0)
    thp = 2 * math.pi * (np.arange(nq) + 0.5) / nq
    csp, snp = np.cos(thp), np.sin(thp)
    S2 = P[:, None, :] + a * (csp[None, :, None] * N[:, None, :] +
                              snp[None, :, None] * B[:, None, :])
    W2 = 1.0 - a * (k1[:, None] * csp[None, :] + k2[:, None] * snp[None, :])
    D2 = kh1[:, None] * csp[None, :] + kh2[:, None] * snp[None, :]   # dipole cible

    def ring_kernels(i, X):
        """(uniforme pondere, dipole) de l'anneau i evalue en X."""
        d = X - P[i]
        Tv = T[i]
        z = d @ Tv
        xn = d @ N[i]
        xb = d @ B[i]
        rho = np.sqrt(xn ** 2 + xb ** 2)
        A = z ** 2 + rho ** 2 + a ** 2
        Bq = 2 * a * rho
        S = np.sqrt(A + Bq)
        m = np.clip(2 * Bq / (A + Bq), 0.0, 1.0 - 1e-13)
        K, E = ellipk(m), ellipe(m)
        phi0 = (2 / math.pi) * K / S
        with np.errstate(divide='ignore', invalid='ignore'):
            Ic = np.where(Bq > 1e-30 * A,
                          (2 / math.pi) * (A * K - (A + Bq) * E) / (Bq * S), 0.0)
        cpsi = np.where(rho > 0, xn / np.maximum(rho, 1e-300), 0.0)
        spsi = np.where(rho > 0, xb / np.maximum(rho, 1e-300), 0.0)
        unif = phi0 - a * (k1[i] * cpsi + k2[i] * spsi) * Ic
        dip = (kh1[i] * cpsi + kh2[i] * spsi) * Ic          # <kappa_hat.u / r>
        return unif, dip

    E00 = E01 = E11 = 0.0
    for i in range(n):
        X = S2.reshape(-1, 3)
        unif, dip = ring_kernels(i, X)
        unif = unif.reshape(n, nq)
        dip = dip.reshape(n, nq)
        G00 = (W2 * unif).mean(axis=1)
        G01 = (D2 * unif).mean(axis=1)          # dipole cote cible
        G10 = (W2 * dip).mean(axis=1)           # dipole cote source
        G11 = (D2 * dip).mean(axis=1)
        for G in (G00, G01, G10, G11):
            G[i] = 0.0
        tt = T @ T[i]
        E00 += float((tt * G00 * dl).sum() * dl[i])
        E01 += 0.5 * float((tt * (G01 + G10) * dl).sum() * dl[i])
        E11 += float((tt * G11 * dl).sum() * dl[i])
    # diagonale analytique : canal uniforme seulement (le dipole s'y moyenne a 0
    # au premier ordre ; residu (a kappa)^3)
    diag = float(sum(dl[i] * _diag_cell(dl[i], a) for i in range(n)))
    E00 += diag
    E11 += diag                                  # meme noyau local pour cos^2 ~ 1/2 x2
    pref = MU0 / (4 * math.pi)
    E00, E01, E11 = pref * E00, pref * E01, pref * E11
    L_var = E00 - E01 ** 2 / E11 if E11 > 0 else E00
    return E00, L_var


def run():
    a = D_TUBE * FM
    print("=== Resolution de London ===\n", flush=True)

    print("[1] Tore coaxial, exact 1D (N=128) + variationnel dipole :", flush=True)
    for ra in (2.0, 3.0, 5.0):
        R = ra * a
        L_axe, L_unif, L_london, L_var = torus_london_1d(R, a, N=128)
        v3 = L_sheet_ring(circle(R, 600), a, nq=48)
        print("  r/a=%.0f : axe %.4f | unif-1D %.4f (v3 3D : %.4f, ecart %+.2f%%) |"
              " LONDON %.4f | var-dipole %.4f" % (
                  ra, L_axe/MU0/FM, L_unif/MU0/FM, v3/MU0/FM,
                  (L_unif/v3-1)*100, L_london/MU0/FM, L_var/MU0/FM), flush=True)
        gap = (L_unif - L_london) / L_unif * 100
        cap = (L_unif - L_var) / max(L_unif - L_london, 1e-30) * 100
        print("       gap uniforme->London = %.2f%% de L ; le dipole en capture %.0f%% ;"
              " London vs axe : %+.2f%%" % (
                  gap, cap, (L_london/L_axe-1)*100), flush=True)

    print("\n[2] Convergence N (tore r/a=3) :", flush=True)
    for N in (64, 128, 256):
        _, Lu, Ll, _ = torus_london_1d(3.0 * a, a, N=N)
        print("  N=%3d : unif %.4f ; London %.4f" % (Lu/MU0/FM, Ll/MU0/FM) if False
              else "  N=%3d : unif %.4f ; London %.4f" % (N, Lu/MU0/FM, Ll/MU0/FM),
              flush=True)

    print("\n[3] Nucleon : variationnel dipolaire (kappa-aligne) :", flush=True)
    C0 = np.load(os.path.join(os.path.dirname(__file__), "..", "..",
                              "output", "nucleon_ideal_0.npy"))
    for ns in (800, 1600):
        X = resample(C0, ns) * D_TUBE * FM
        L_unif, L_var = L_nucleon_var_dipole(X, a, nq=48)
        Eu, Ev = energy_MeV(L_unif), energy_MeV(L_var)
        print("  n_s=%4d : L_unif = %.4f (E %+.2f%%) -> L_var = %.4f "
              "(E_var = %.3f MeV, %+.2f%%)" % (
                  ns, L_unif/MU0/FM, (Eu/TARGET-1)*100,
                  L_var/MU0/FM, Ev, (Ev/TARGET-1)*100), flush=True)
    print("\nLECTURE : E_London >= E_var (borne variationnelle) ; le residu de", flush=True)
    print("troncature dipolaire est calibre en [1] (part du gap non capturee).", flush=True)


if __name__ == "__main__":
    run()
