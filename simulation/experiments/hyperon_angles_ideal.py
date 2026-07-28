"""Angles des boucles bloquees (theta_Sigma, theta_Xi) sur entrelacs IDEAUX — reprise.

CONTEXTE : la premiere tentative (blocked_loop_angles_attempt.py) avait echoue sur
3 requis manquants : (i) relaxation ideale convergee ; (ii) geometrie du systeme LIE ;
(iii) axe de spin du compose. Les outils (i)+(ii) existent desormais (sono/sono_link).

PIPELINE (reproduit ici ; resultats du 2026-07-14) :
  1. Sigma = 3_1+6_1+7_2 et Xi = 3_1+6_2+7_2 : construction par placements diriges
     (fenetre d'agrafe pour les graines twist ; scan rotations pour le 6_2 tresse),
     motif |Lk| = (1,1,0) ; relaxation sono_link 25k ; gardes TOUTES VERTES :
       Sigma : rho = 204.22 ; Xi : rho = 257.28 ; Lk (-1,-1,0) ; Alexander
       (1,-1,1)/(2,-5,2)/(3,-5,3) et (1,-1,1)/(1,-3,3,-3,1)/(3,-5,3).
  2. Angles : normale d'aire vectorielle du coeur 3_1 vs axes candidats, avec le
     NUCLEON relaxe comme CONTROLE (attendu theta ~ 0).

RESULTAT — LE CONTROLE ECHOUE POUR TOUTES LES DEFINITIONS STATIQUES :
  (a) axes globaux (aire totale ; 3 axes d'inertie) : nucleon 44-133 deg, jamais ~0 ;
  (b) normales des boucles compagnes : nucleon 85/95 deg — quasi PERPENDICULAIRES
      (structurel : le plan d'une boucle enfilante CONTIENT le brin enfile ->
      cos(theta)~0 statiquement, incompatible avec cos(theta)~1 du nucleon).
VERDICT : avec (i)+(ii) satisfaits, l'echec isole le requis (iii) — l'axe de spin est
DYNAMIQUE (circulation chi), pas une caracteristique du plongement serre.
theta_Sigma ~ 67 deg et theta_Xi ~ 97 deg RESTENT DES LECTURES.
Consigne : brouillon_angles_boucles_bloquees.tex §Reprise.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ideal_embedding import (  # noqa: E402
    sono_link, torus_knot, twist_knot_seed, resample, linking_number,
)
from repair_curves_braid import braid_curve, alexander, BRAID_WORDS  # noqa: E402


def vec_area(P):
    dl = np.roll(P, -1, 0) - P
    return 0.5 * np.cross(P, dl).sum(0)


def angle_deg(u, v):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def analyse(name, comps):
    core_n = vec_area(comps[0])
    a_tot = sum(vec_area(C) for C in comps)
    X = np.vstack(comps)
    X = X - X.mean(0)
    _, V = np.linalg.eigh(X.T @ X)
    print(f"--- {name} ---")
    for lab, ax in (("aire totale", a_tot), ("inertie min", V[:, 0]),
                    ("inertie med", V[:, 1]), ("inertie max", V[:, 2])):
        a = angle_deg(core_n, ax)
        print(f"  normale(coeur) vs {lab:12}: {a:6.1f} deg (ou {180-a:6.1f})")
    for i in (1, 2):
        a = angle_deg(core_n, vec_area(comps[i]))
        print(f"  normale(coeur) vs compagnon {i}: {a:6.1f} deg")


def run():
    # les etats relaxes du 2026-07-14 sont dans /tmp (reproduire : voir docstring)
    for name, pref in (("NUCLEON (controle)", "nucleon"), ("SIGMA", "sigma"), ("XI", "xi")):
        try:
            comps = [np.load(f"/tmp/{pref}_relaxed_{i}.npy") for i in range(3)]
        except FileNotFoundError:
            print(f"--- {name} : etats relaxes absents (relancer la construction) ---")
            continue
        analyse(name, comps)


if __name__ == "__main__":
    run()
