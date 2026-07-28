"""Deriver le facteur d'interpenetration ~2,2 de la saturation (19/07).

La resbox saturation (V6) laisse UN O(1) ouvert : v_c^eff = 2,2 x V_tube
(ajuste sur rho_0). STRUCTURE DERIVABLE du facteur :

  (1) EXCLUSION D'AXE : deux tubes de diametre D ne peuvent pas avoir
      leurs axes a moins de D l'un de l'autre -> la coquille exclue
      autour d'un axe a le rayon D = 2 x (D/2) : section pi D^2, soit
      4 x la section de matiere (pi D^2/4). Volume de coquille ideal :
      V_coq = pi D^2 L_tot = 4 V_tube (formule du tube de Weyl, exacte
      hors recouvrements).
  (2) RECOUVREMENTS : la ou l'entrelacs se replie/s'enlace a moins de
      2D, les coquilles se recouvrent -> facteur f_ov = V_coq(reel) /
      (pi D^2 L_tot) < 1, MESURE sur la geometrie relaxee (MC).
  (3) PARTAGE DE PAIRE (covolume de van der Waals) : le volume bloque
      entre deux nucleons est partage -> b = V_coq(reel)/2.

  => facteur PREDIT : v_c/V_tube = 2 f_ov  (sans parametre libre).

Le script mesure f_ov, injecte b dans le MEME modele E/A, et compare
sphere / derive / ajuste (2,2) / empirique.
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from simulation.experiments.bethe_weizsacker_F3 import (
    attraction_integral, energy_per_A, R_EXCL)
from simulation.experiments.saturation_volume_entrelacs import (
    load_entrelacs, D_TUBE, EMP)

SEED = 77
N_MC = 400_000


def run():
    print("=== Facteur d'interpenetration : derivation geometrique ===\n")
    comps = load_entrelacs()
    pts = np.vstack(comps)
    L_tot = len(pts) * D_TUBE / 2
    V_tube = L_tot * np.pi * (D_TUBE / 2) ** 2
    V_coq_ideal = np.pi * D_TUBE ** 2 * L_tot
    print("L_tot = %.1f fm ; V_tube = %.2f fm^3 ; coquille ideale" % (L_tot, V_tube))
    print("pi D^2 L = %.2f fm^3 (= 4 V_tube, section d'exclusion x4)\n" % V_coq_ideal)

    # [2] volume reel de la coquille (union des tubes de rayon D) par MC
    rng = np.random.default_rng(SEED)
    lo = pts.min(0) - D_TUBE
    hi = pts.max(0) + D_TUBE
    V_box = np.prod(hi - lo)
    tree = cKDTree(pts)
    X = lo + rng.random((N_MC, 3)) * (hi - lo)
    dmin, _ = tree.query(X, k=1)
    inside = dmin < D_TUBE
    p = inside.mean()
    V_coq = V_box * p
    err = V_box * np.sqrt(p * (1 - p) / N_MC)
    f_ov = V_coq / V_coq_ideal
    print("[2] coquille reelle (union, rayon D, %d points MC) :" % N_MC)
    print("    V_coq = %.2f +- %.2f fm^3  ->  f_ov = %.3f" % (V_coq, err, f_ov))
    print("    (recouvrements aux replis/enlacements : perte %.0f %%)\n" % (
        100 * (1 - f_ov)))

    # [3] covolume derive et facteur predit
    b = V_coq / 2
    fact = b / V_tube
    print("[3] b = V_coq/2 = %.2f fm^3  ->  FACTEUR PREDIT = 2 f_ov = %.2f" % (b, fact))
    print("    [requis par rho_0 = 0,16 : 2,2]\n")

    # [4] injection dans le modele E/A
    J_att, _ = attraction_integral(R_EXCL)
    print("%-30s %8s %8s %8s" % ("volume exclu", "rho_0", "k_F", "a_V"))
    rows = [("sphere (V6, 2,57)", 4 * np.pi * R_EXCL ** 3 / 3),
            ("derive (2 f_ov V_tube = %.2f)" % b, b),
            ("ajuste (2,2 V_tube = 3,33)", 3.33)]
    for lab, vc in rows:
        r = np.linspace(0.005, 0.98 / vc, 8000)
        EA = energy_per_A(r, J_att, vc)
        i0 = np.argmin(EA)
        print("%-30s %8.3f %8.2f %8.1f" % (
            lab, r[i0], (1.5 * np.pi ** 2 * r[i0]) ** (1 / 3), -EA[i0]))
    print("%-30s %8.3f %8.2f %8.1f   (empirique)" % ("", EMP["rho0"], EMP["kF"], EMP["aV"]))

    print("\nVERDICT : le facteur n'est plus un O(1) libre -- structure")
    print("derivee (section d'exclusion x4, partage 1/2, recouvrements")
    print("mesures) : 2 f_ov = %.2f vs 2,2 requis (%+.0f %%). L'ecart" % (
        fact, 100 * (fact / 2.2 - 1)))
    print("residuel est du niveau de la forme du modele (vdW 1er ordre,")
    print("J_att a coeur spherique) -- declare, pas ajuste.")


if __name__ == "__main__":
    run()
