"""Saturation nucleaire : le VRAI volume exclu de l'entrelacs (19/07).

La resbox << Saturation nucleaire emergente >> (V6, sec:F3-BW) utilise une
sphere dure v_c = (4/3) pi r_excl^3 = 2,57 fm^3 et ANNONCE que le calcul
correct doit partir de la geometrie reelle de l'entrelacs 3_1+7_2+7b_2.
Ce script FAIT ce calcul, par mesure Monte-Carlo directe sur deux copies
RIGIDES de l'entrelacs nucleon relaxe (SONO, output/nucleon_ideal_*.npy,
unites = diametres de tube, D = 0,2103 fm = lambdabar_C(p)).

Le volume exclu d'une paire d'entrelacs a DEUX composantes :
  (a) STERIQUE : configurations ou les tubes se recouvrent
      (distance < D entre axes) ;
  (b) TOPOLOGIQUE (P16) : configurations SANS contact mais ENFILEES
      (enlacement mutuel Lk != 0 entre composantes des deux nucleons) --
      inaccessibles sans reconnexion : exclues de l'espace des phases.
v_c(vrai) = V_echantillon x [P(sterique) + P(enfile sans contact)],
moyenne sur les orientations relatives (rotations aleatoires uniformes).

Le v_c mesure est ensuite reinjecte dans le MEME modele E/A que la V6
(bethe_weizsacker_F3 : T_F(rho_eff) + rho J_att/2) -> nouveaux
rho_0, r_0, k_F, a_V, compares a la sphere et a l'empirique.
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from simulation.experiments.bethe_weizsacker_F3 import (
    attraction_integral, energy_per_A, R_EXCL)

D_TUBE = 0.2103          # fm (lambdabar_C(p))
R_MAX = 3.2              # fm : rayon de la region d'echantillonnage
N_ORIENT = 40
N_DISP = 4000
SEED = 2026
EMP = {"rho0": 0.16, "r0": 1.20, "kF": 1.33, "aV": 15.8}


def load_entrelacs():
    """Les 3 composantes, en fm, re-echantillonnees au pas D/2, centrees."""
    comps = []
    for i in range(3):
        C = np.load("output/nucleon_ideal_%d.npy" % i) * D_TUBE
        # re-echantillonnage a pas ~D/2
        P = np.vstack([C, C[:1]])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(24, int(s[-1] / (D_TUBE / 2)))
        u = np.linspace(0, s[-1], n, endpoint=False)
        idx = np.searchsorted(s, u, side="right") - 1
        f = ((u - s[idx]) / seg[idx])[:, None]
        comps.append(P[idx] * (1 - f) + P[idx + 1] * f)
    allp = np.vstack(comps)
    c0 = allp.mean(0)
    return [c - c0 for c in comps]


def rand_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def decimate(C, n=60):
    idx = np.linspace(0, len(C), n, endpoint=False).astype(int)
    return C[idx]


def gauss_linking(A, B):
    """Nombre d'enlacement (Gauss discretise) de deux courbes fermees."""
    dA = np.roll(A, -1, 0) - A
    dB = np.roll(B, -1, 0) - B
    mA = A + dA / 2
    mB = B + dB / 2
    r = mA[:, None, :] - mB[None, :, :]
    n3 = np.linalg.norm(r, axis=2) ** 3 + 1e-12
    cr = np.cross(dA[:, None, :], dB[None, :, :])
    return float(np.sum(np.einsum("ijk,ijk->ij", r, cr) / n3)) / (4 * np.pi)


def run():
    print("=== Volume exclu de l'entrelacs nucleon : mesure directe ===\n")
    comps = load_entrelacs()
    pts_A = np.vstack(comps)
    print("entrelacs relaxe : %d points (pas D/2), etendue %.2f fm" % (
        len(pts_A), np.ptp(pts_A, axis=0).max()))
    print("tube D = %.4f fm ; sphere V6 : v_c = %.2f fm^3 (r_excl = %.2f)\n" % (
        D_TUBE, 4 * np.pi * R_EXCL ** 3 / 3, R_EXCL))

    tree_A = cKDTree(pts_A)
    dec_A = [decimate(c) for c in comps]
    rng = np.random.default_rng(SEED)
    V_samp = 4 * np.pi * R_MAX ** 3 / 3

    n_st = n_topo = n_tot = 0
    n_topo_checked = 0
    for _ in range(N_ORIENT):
        R = rand_rotation(rng)
        pts_B0 = pts_A @ R.T
        dec_B0 = [c @ R.T for c in dec_A]
        # deplacements uniformes dans la boule R_MAX
        d = rng.normal(size=(N_DISP, 3))
        d *= (rng.random(N_DISP) ** (1 / 3) * R_MAX / np.linalg.norm(d, axis=1))[:, None]
        for k in range(N_DISP):
            n_tot += 1
            dmin, _ = tree_A.query(pts_B0 + d[k], k=1)
            if dmin.min() < D_TUBE:
                n_st += 1
                continue
            if np.linalg.norm(d[k]) > 2.6:
                continue
            n_topo_checked += 1
            linked = False
            for ca in dec_A:
                for cb in dec_B0:
                    if abs(gauss_linking(ca, cb + d[k])) > 0.5:
                        linked = True
                        break
                if linked:
                    break
            if linked:
                n_topo += 1

    p_st = n_st / n_tot
    p_topo = n_topo / n_tot
    v_st = V_samp * p_st
    v_topo = V_samp * p_topo
    v_true = v_st + v_topo
    err = V_samp * np.sqrt(p_st * (1 - p_st) / n_tot)
    print("configurations : %d (topologie testee sur %d sans contact)" % (
        n_tot, n_topo_checked))
    print("  v_sterique     = %.2f fm^3  (P = %.3f)" % (v_st, p_st))
    print("  v_topologique  = %.2f fm^3  (enfile sans contact, P = %.4f)" % (
        v_topo, p_topo))
    print("  v_c(entrelacs) = %.2f +- %.2f fm^3   [sphere V6 : 2,57]\n" % (
        v_true, err))

    # reinjection dans le modele E/A de la V6 (memes conventions)
    J_att, lam = attraction_integral(R_EXCL)
    print("modele E/A (conventions bethe_weizsacker_F3, J_att = %.0f MeV.fm^3) :" % J_att)
    print("%-24s %8s %8s %8s %8s" % ("volume exclu", "rho_0", "r_0", "k_F", "a_V"))
    print("%-24s %8s %8s %8s %8s" % ("", "fm^-3", "fm", "fm^-1", "MeV"))
    rows = [("sphere (V6, 2,57 fm^3)", 4 * np.pi * R_EXCL ** 3 / 3),
            ("entrelacs (mesure)", v_true)]
    for lab, vc in rows:
        rho = np.linspace(0.005, 0.98 / vc, 6000)
        EA = energy_per_A(rho, J_att, vc)
        i0 = np.argmin(EA)
        rho0 = rho[i0]
        r0 = (3 / (4 * np.pi * rho0)) ** (1 / 3)
        kF = (1.5 * np.pi ** 2 * rho0) ** (1 / 3)
        print("%-24s %8.3f %8.2f %8.2f %8.1f" % (lab, rho0, r0, kF, -EA[i0]))
    print("%-24s %8.3f %8.2f %8.2f %8.1f   (empirique)" % (
        "", EMP["rho0"], EMP["r0"], EMP["kF"], EMP["aV"]))

    # phase 3 : inversion (v_c requis pour rho_0 empirique) + volume de matiere
    from scipy.optimize import brentq

    def rho0_of(vc):
        r = np.linspace(0.005, 0.98 / vc, 8000)
        return r[np.argmin(energy_per_A(r, J_att, vc))]

    v_eff = brentq(lambda v: rho0_of(v) - EMP["rho0"], 2.6, 6.0, xtol=1e-3)
    r = np.linspace(0.005, 0.98 / v_eff, 8000)
    EA2 = energy_per_A(r, J_att, v_eff)
    i0 = np.argmin(EA2)
    L_tot = len(pts_A) * D_TUBE / 2
    V_tube = L_tot * np.pi * (D_TUBE / 2) ** 2
    print("\nINVERSION : v_c requis pour rho_0 = %.2f -> %.2f fm^3" % (
        EMP["rho0"], v_eff))
    print("  -> a_V = %.1f MeV (empirique %.1f : la sur-liaison x1,5 se" % (
        -EA2[i0], EMP["aV"]))
    print("     dissout), k_F exact par construction.")
    print("  volume de MATIERE du tube (L = %.1f fm, D = %.4f) : %.2f fm^3" % (
        L_tot, D_TUBE, V_tube))
    print("  v_eff = %.1f x V_tube ; interpenetration de la silhouette" % (
        v_eff / V_tube))
    print("  rigide : %.0f %%. VERDICT : l'entrelacs RIGIDE est refute" % (
        100 * (1 - v_eff / v_true)))
    print("  (rho_0 s'effondrerait x30) ; l'echelle correcte du volume exclu")
    print("  est le volume de MATIERE du tube (x~2, facteur d'objets")
    print("  flexibles interpenetres -- O(1) restant a deriver).")


if __name__ == "__main__":
    run()
