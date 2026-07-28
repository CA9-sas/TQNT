"""Le menu des modes de surface contraints — spectre E(m) du canal
n=3 (27/07/2026, suite du juge azimutal).

RAFFINEMENT DE L'INCOMPRESSIBILITÉ : la contrainte n'interdit pas
toute modulation longitudinale — elle interdit le BOSSELAGE de
charge. Un courant de surface SANS DIVERGENCE peut moduler
longitudinalement si l'azimut compense : fonction de courant
    psi(s, phi) = cos(3 sigma + m phi),
    K_s = (1/a) d psi/d phi = -(m/a) sin(3 sigma + m phi),
    K_phi = -d psi/ds     = (6 pi/L) sin(3 sigma + m phi),
divergence de surface NULLE pour tout m. Le canal du mode de
génération est donc un MENU discret :
    m = 0   : anneaux purs (le canal azimutal jugé — x0,36) ;
    |m| >= 1: modes mixtes (longitudinal licite, azimut compensant) ;
    m -> inf: longitudinal quasi pur (la référence interdite x4,5
              n'est atteinte qu'asymptotiquement).
Le signe de m est physique (le tube est CHIRAL) : +m et -m sont
calculés séparément.

CALCUL : énergie magnétique exacte de chaque mode sur le tube idéal
(coquille, grille n x m_phi, double somme de Green complète, vecteurs
de courant réels) — deux normalisations DÉCLARÉES :
    (N1) norme L² égale à celle du courant de base (contenu égal) ;
    (N2) amplitude de crête égale au courant de base (la convention
         eta=1 des calculs précédents).
CONTRÔLES : (a) le mode m=0 doit retrouver le juge azimutal ;
(b) la référence longitudinale interdite (pattern cos(3sigma) t)
recalculée dans la même machinerie doit redonner E_complet.

CONFRONTATION : jeu fermé des amplitudes canoniques {1 ; 1/2 (le
demi-tour S4'', LE quantum canonique du cadre)} x menu E(m) vs
m_e = 0,511 MeV. Règle du test nul : tout accord est étiqueté
candidat conditionnel (mécanisme de sélection de m requis).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from simulation.ideal_embedding import sono, resample
from theory.omega_ts import _knot_curve
from simulation.experiments.derive_profil_tube_k33 import frames

M_E = 0.51099895
ALPHA = 1 / 137.035999
HBARC = 197.32698
MU = 17.715562
M_L = MU ** 2
SQRT_SIGMA = 418.47


def mode_energy(P, a, mmode, n_phi=12, longitudinal_ref=False):
    """énergie de Green du mode (réel, stationnaire) ; renvoie
    (E_raw, norme L² brute) — E_raw pour courant unité de base."""
    P = np.asarray(P, float)
    P = P - P.mean(0)
    t, u, v, ds = frames(P)
    n = len(P)
    s = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
    L = ds.sum()
    sig = 2 * np.pi * s / L
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    X = (P[:, None, :] + a * np.cos(phi)[None, :, None] * u[:, None, :]
         + a * np.sin(phi)[None, :, None] * v[:, None, :])
    e_phi = (-np.sin(phi)[None, :, None] * u[:, None, :]
             + np.cos(phi)[None, :, None] * v[:, None, :])
    # champ de courant du mode (unités : K_base = 1/(2 pi a))
    ph = 3 * sig[:, None] + mmode * phi[None, :]
    if longitudinal_ref:
        Ks = np.cos(ph if mmode == 0 else 3 * sig[:, None] + 0 * phi)
        Kf = np.zeros_like(Ks)
    else:
        Ks = -(mmode / a) * np.sin(ph)
        Kf = (6 * np.pi / L) * np.sin(ph)
    J = Ks[:, :, None] * t[:, None, :] + Kf[:, :, None] * e_phi   # (n,nphi,3)
    dA = (ds[:, None] * (2 * np.pi * a / n_phi)
          * np.ones((1, n_phi)))                                  # aires
    JdA = J * dA[:, :, None]
    norm2 = (np.sum(J * J, axis=2) * dA).sum()
    E = 0.0
    for i in range(n):
        diff = X[i][:, None, None, :] - X[None, :, :, :]
        d = np.linalg.norm(diff, axis=3)
        d[d < 1e-12] = np.inf
        dots = np.einsum('kc,jlc->kjl', JdA[i], JdA)
        # exclusion du même anneau (i=j) pour l'auto-terme des nappes
        contrib = (dots / d)
        contrib[:, i, :] = 0.0
        E += contrib.sum()
    return 0.5 * E, norm2, L


def main():
    print(__doc__)
    P0 = np.asarray(_knot_curve("3_1", 200), float)
    P_id, info = sono(P0, iters=20000)
    ell = info["ropelength"]
    P = resample(P_id, 240)
    dl = np.roll(P, -1, 0) - P
    a = np.linalg.norm(dl, axis=1).sum() / ell
    L_fm = M_L / SQRT_SIGMA ** 2 * HBARC
    # calibration physique : la référence longitudinale interdite doit
    # redonner E_complet = 2,283 MeV (convention crête, temps 1/2)
    E_ref, n2_ref, L = mode_energy(P, a, 0, longitudinal_ref=True)
    CAL = 2.283 / (0.5 * E_ref)          # MeV par unité de E_raw (avec 1/2 tps)
    print("=" * 78)
    print("CALIBRATION : référence longitudinale interdite -> E_complet")
    print("  E_raw(ref) = %.4f ; norme² = %.4f ; calibration fixée" % (
        E_ref, n2_ref))

    print("=" * 78)
    print("LE MENU E(m) — modes sans divergence, n=3")
    print("=" * 78)
    print("  %-6s %12s %12s %14s %14s" % (
        "m", "E(N2 crête)", "E(N1 norme)", "N2 x 1/2 ampl.", "N1 x 1/2 ampl."))
    res = {}
    for mmode in [0, 1, -1, 2, -2, 3, -3]:
        E_raw, n2, _ = mode_energy(P, a, mmode)
        # N2 crête : amplitude du champ telle que max|K| ~ K_base ->
        #   pour le mode, |K| = sqrt(Ks²+Kf²) ; crête = sqrt((m/a)²+(6pi/L)²)
        peak = np.sqrt((mmode / a) ** 2 + (6 * np.pi / L) ** 2)
        E_N2 = 0.5 * E_raw * CAL / peak ** 2
        # N1 : norme L² égale à celle de la réf. longitudinale
        E_N1 = 0.5 * E_raw * CAL * (n2_ref / n2)
        res[mmode] = (E_N2, E_N1)
        print("  %-6d %9.3f MeV %9.3f MeV %11.3f MeV %11.3f MeV" % (
            mmode, E_N2, E_N1, E_N2 / 4, E_N1 / 4))
    print("  (m_e = %.3f MeV ; réf. interdite : 2,283 crête / %.3f norme)"
          % (M_E, 2.283 * 2))

    print("=" * 78)
    print("CONFRONTATION (jeu fermé : amplitudes {1, 1/2}, m = 0..±3)")
    print("=" * 78)
    for mmode, (E_N2, E_N1) in res.items():
        for lab, val in [("crête x1", E_N2), ("crête x1/2 ampl", E_N2 / 4),
                         ("norme x1", E_N1), ("norme x1/2 ampl", E_N1 / 4)]:
            dev = 100 * (val / M_E - 1)
            if abs(dev) < 25:
                print("  m=%+d, %-16s : %.3f MeV  (%+.1f %%)  <— dans ±25 %%"
                      % (mmode, lab, val, dev))
    print("""
  LECTURE : les entrées ci-dessus (s'il y en a) sont des CANDIDATS
  CONDITIONNELS — il manque le mécanisme de sélection de m et la
  justification de l'amplitude ; règle du test nul. Les chiffres du
  menu complet bornent dans tous les cas la réponse.""")


if __name__ == "__main__":
    main()
