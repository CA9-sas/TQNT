"""Mesurer le circuit d'enlacement relaxe (outillage entrelacs ideal) — beta_otimes.

QUESTION (suite pin_r_link_test) : le modele « cercle a r_link » du fluxoide porte
+6.6% ; le circuit PHYSIQUE est la boucle compagne telle que deformee par l'entrelacs.
On le MESURE au lieu de le modeliser :

  A. 3_1 (x) anneau libre, Lk=1 : l'anneau libre s'ETREINT sur le brin (rayon -> 2a).
     C'est la BORNE INFERIEURE du circuit (E maximal), pas le circuit physique.
  B. 3_1 (x) 7_2, Lk=1 : le compagnon reel a une ropelength propre — sa fenetre ne
     peut pas se refermer sous sa geometrie serree. Candidat physique.

Pour chaque configuration relaxee (sono_link, contraintes inter-composantes,
validation Hopf 8*pi a +0.07%) : L_gmd de chaque composante a l'echelle physique
(tube a = d_tube = 0.209 fm), E = Phi_chi^2/(2L), compare a la cible 1.293 MeV et
au modele cercle-r_link (1.379, +6.6%).

Gardes : Lk preserve (Gauss), Alexander des composantes.

RESULTATS (2026-07-13, gardes toutes OK) :
  A. anneau libre : s'etreint exactement a r_eff = 2a (solution locale type Hopf,
     circonference 4*pi*R) -> E = 6.16 MeV (+376%) = borne d'etreinte, comme prevu.
  B. 3_1 (x) 7_2 (Lk=-1, alexander (1,-1,1)/(3,-5,3) preserves) :
     coeur 3_1 lie : E = 1.509 MeV (+16.7%, var-theta 30%) ;
     compagnon 7_2 : E = 0.614 MeV (-52.5%) -> rejete (comme le 7_2 isole).
  C. NUCLEON 3_1+7_2+7_2 (run_nucleon ; motif neutron |Lk|=(1,1,0), construction C3
     -- le trefle torique est C3-invariant donc Lk(K,S2)=Lk(K,S1) automatiquement) :
     gardes toutes vertes (Lk -1/-1/0, Alexander (1,-1,1)/(3,-5,3)/(3,-5,3)) ;
     coeur 3_1 : L = 1.738 mu0.fm (L* requis 1.750) -> E = 1.302 MeV = +0.7% !
     (+0.1% a a=lambdabar_C(p) exact 0.2103 ; sensibilite E ~ 1/a).
  SERIE COMPLETE (coeur 3_1) : isole +33% -> (x)anneau +22.8% -> (x)7_2 +16.7%
  -> NUCLEON +0.7% : chaque compagnon qui traverse le coeur l'OUVRE vers la cible ;
  la sur-determination de r_link (pin_r_link_test) se dissout (le 'cercle a 1.04 fm'
  est la fonctionnelle d'inductance de la geometrie mesuree). Reserves : var-theta
  46% (tube-mince aux contacts), serrage 7_2 +11%, motif proton (1,1,1) a construire,
  une graine. beta_otimes implique : E/6.15 = 0.212 (vs 0.210) -- RESTE COMPTE CALIBRE
  en attendant la robustesse. Consigne : brouillon beta_otimes §7 + V6 §Comptage.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ideal_embedding import (  # noqa: E402
    sono, sono_link, torus_knot, twist_knot_seed, resample,
    linking_number, curve_length,
)
from derive_inductance_factor_test import (  # noqa: E402
    L_gmd, energy_MeV, L_star, TARGET, D_TUBE, FM, MU0,
)
from repair_curves_braid import alexander  # noqa: E402


def circle3(r, center, normal, n=140):
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1, 0])
    u = np.cross(normal, a)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.asarray(center, float) + r * np.outer(np.cos(t), u) + r * np.outer(np.sin(t), v)


def report_component(tag, C):
    X = C * D_TUBE * FM
    L, var = L_gmd(X, D_TUBE * FM)
    E = energy_MeV(L)
    r_eff = curve_length(C) * D_TUBE / (2 * np.pi)
    print(f"   {tag:24}: l = {curve_length(C)*D_TUBE:6.2f} fm (r_eff = {r_eff:.3f}) ; "
          f"L = {L/MU0/FM:7.3f} mu0.fm (var {var:4.0f}%) ; E = {E:6.3f} MeV "
          f"({(E/TARGET-1)*100:+6.1f}%)", flush=True)
    return E


def run():
    print(f"=== Mesure du circuit d'enlacement (cible E = {TARGET} MeV ; "
          f"L* = {L_star()/MU0/FM:.3f} mu0.fm ; modele cercle-r_link : +6.6%) ===", flush=True)

    # --- A. 3_1 (x) anneau libre (borne d'etreinte) -------------------------
    K = torus_knot(2, 3, n=240)
    ring = circle3(1.3, (0, 2.6, 0), (0, 1, 0))
    print(f"\nA. 3_1 (x) anneau libre : Lk initial = {linking_number(K, ring)}", flush=True)
    comps, info = sono_link([K, ring], iters=15000)
    print(f"   relaxe : rho_tot = {info['ropelength']:.2f}, gap = {info['min_gap']:.3f}, "
          f"Lk = {linking_number(*comps)}", flush=True)
    report_component("coeur 3_1 (dans le lien)", comps[0])
    report_component("anneau etreint (borne)", comps[1])

    # --- B. 3_1 (x) 7_2 (candidat physique) ---------------------------------
    # Placement DIRIGE (le tirage aleatoire echoue : fenetres du 7_2 etroites) :
    # viser la grande fenetre de l'agrafe (boucle connecteur+bande, normale ~y),
    # y faire passer le brin du trefle dont la tangente est alignee, balayer
    # les petits offsets jusqu'a Lk=+-1 avec clairance.
    print("\nB. 3_1 (x) 7_2 : placement dirige (fenetre de l'agrafe)...", flush=True)
    K = torus_knot(2, 3, n=240)
    s7 = 2.5
    S = resample(twist_knot_seed(9, hand=+1, thread=+1, x_eye=0.10), 300)
    S = (S - S.mean(0)) * s7
    W0 = np.array([-1.1, 0.0, 0.0]) * s7          # centre de la fenetre d'agrafe
    T = np.roll(K, -1, 0) - np.roll(K, 1, 0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    i0 = int(np.argmax(np.abs(T[:, 1])))          # brin a tangente ~y
    placed, best_d = None, 0.0
    for dx in np.linspace(-1.2, 1.2, 7):
        for dz in np.linspace(-1.2, 1.2, 7):
            for dy in np.linspace(-1.5, 1.5, 5):
                Kp = K - K[i0] + W0 + np.array([dx, dy, dz])
                d = np.linalg.norm(Kp[:, None, :] - S[None, :, :], axis=2).min()
                if d < 0.35 or d <= best_d:
                    continue
                if abs(linking_number(Kp, S)) == 1:
                    placed, best_d = Kp, d
    if placed is None:
        print("   ECHEC de placement Lk=1 — a reprendre", flush=True)
        return
    print(f"   placement dirige : clairance {best_d:.2f}", flush=True)
    K, placed = placed, S     # K = trefle deplace ; placed = compagnon 7_2
    comps, info = sono_link([K, placed], iters=25000)
    al = tuple(int(x) for x in alexander(comps[1])[0])
    print(f"   relaxe : rho_tot = {info['ropelength']:.2f}, gap = {info['min_gap']:.3f}, "
          f"Lk = {linking_number(*comps)}, alexander(7_2) = {al}", flush=True)
    report_component("coeur 3_1 (dans le lien)", comps[0])
    report_component("compagnon 7_2 relaxe", comps[1])
    np.save("/tmp/link_31_72.npy", np.vstack([np.hstack([comps[0], np.zeros((len(comps[0]), 1))]),
                                              np.hstack([comps[1], np.ones((len(comps[1]), 1))])]))

    print("\n=== Lecture : le circuit physique est-il entre l'etreinte (A) et le "
          "cercle-r_link (+6.6%) ? ===", flush=True)


def run_nucleon():
    """C. Nucleon complet 3_1+7_2+7_2 (motif neutron, construction C3)."""
    K = torus_knot(2, 3, n=240)
    s7 = 2.5
    S = resample(twist_knot_seed(9, hand=+1, thread=+1, x_eye=0.10), 300)
    S = (S - S.mean(0)) * s7
    W0 = np.array([-1.1, 0.0, 0.0]) * s7
    T = np.roll(K, -1, 0) - np.roll(K, 1, 0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    i0 = int(np.argmax(np.abs(T[:, 1])))
    off = np.array([0.40, -0.75, 1.20])          # placement valide (clairance 0.78)
    S1 = S + (K[i0] - W0 - off)
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    S2 = S1 @ Rz.T                                # C3 : Lk(K,S2) = Lk(K,S1) = -1
    print(f"Lk initiaux : {linking_number(K, S1)}, {linking_number(K, S2)}, "
          f"{linking_number(S1, S2)}", flush=True)
    comps, info = sono_link([K, S1, S2], iters=25000)
    print(f"relaxe : rho = {info['ropelength']:.2f} gap = {info['min_gap']:.3f} ; "
          f"Lk {linking_number(comps[0], comps[1])}/{linking_number(comps[0], comps[2])}/"
          f"{linking_number(comps[1], comps[2])} ; alexander "
          f"{[tuple(int(x) for x in alexander(C)[0]) for C in comps]}", flush=True)
    for tag, C in (("coeur 3_1 (nucleon)", comps[0]), ("7_2 a", comps[1]), ("7_2 b", comps[2])):
        report_component(tag, C)
    out = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    os.makedirs(out, exist_ok=True)
    for i, C in enumerate(comps):
        np.save(os.path.join(out, f"nucleon_ideal_{i}.npy"), C)


if __name__ == "__main__":
    run()
    run_nucleon()
