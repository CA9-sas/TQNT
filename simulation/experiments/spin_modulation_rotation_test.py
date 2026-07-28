"""Test : la modulation « creux/bosses » vue sous rotation est-elle l'origine du spin ?

Idee testee (brouillon brouillon_spin_modulation_rotation.tex) : un noeud/entrelacs qui
tourne presente a l'observateur une variation transverse (creux aux croisements, bosses aux
lobes). Deux questions :
  (Q1) cette variation perturbe-t-elle le moment angulaire ? -> OUI (tenseur d'inertie
       anisotrope ; c'est la correction de jonction Y deja derivee, E=S(S+1) Lambda/8).
  (Q2) cette variation est-elle l'ORIGINE du spin ? -> NON (ce script).

Mesure : modulation transverse m=2 = ||Q||/(|m| R_g) (quadrupole/dipole du coeur, invariant
d'echelle). 0 exact pour les toriques C_{q>=3} (protege par symetrie), O(1) pour les C_1.

RESULTAT : la modulation NE trace PAS le spin.
  - electron 3_1 (spin-1/2, l'archetype) : modulation 0.00  -> spin-1/2 SANS aucune variation.
  - nucleon-composant 7_2 (spin-1/2 du noeud... porte le nucleon spin-1/2) : 0.93.
  - deux objets spin-1/2 a 0.00 et 0.93 -> la variation est necessaire NI suffisante au spin.
  - elle trace la symetrie de rotation C_q (type : lepton torique vs twist), pas le spin.
Conclusion : la variation N'EST PAS l'origine du spin ; elle est (a) un classifieur de TYPE
(C_q) et (b) la source de la correction tensorielle de bas-J (Q1). Le spin vient de l'indice
de tresse / double revetement, independamment de la modulation.
"""
import numpy as np
from theory.knot_catalog import knot_curve
from theory.jones import topological_spin, braid_index
from simulation.experiments.repair_curves_braid import braid_curve, multipoles, BRAID_WORDS

TORUS = ['3_1', '5_1', '7_1', '9_1']       # courbes valides (analytiques)
REPAIRED = ['5_2', '7_2', '8_3']           # courbes reparees (fermeture de tresse, Alexander OK)


def modulation(K):
    """Modulation transverse m=2 = ||Q||/(|m| R_g) sur une courbe VALIDE."""
    if K in BRAID_WORDS:
        w, ns, _ = BRAID_WORDS[K]
        P, _, _ = braid_curve(w, ns, pts_per_seg=60)
    else:
        P = np.asarray(knot_curve(K, 2000), float)
    _, _, _, ratio = multipoles(P)
    return ratio


def run():
    rows = TORUS + ['4_1'] + REPAIRED
    print(f"{'K':5}{'b':>3}{'s_min':>6}{'modul m=2':>11}  lecture")
    data = []
    for K in rows:
        mod = modulation(K)
        b = braid_index(K)
        s = topological_spin(K)
        lect = 'torique C_{q>=3} -> 0' if K in TORUS else ('C2' if K == '4_1' else 'C1 (twist)')
        data.append((K, b, s, mod))
        print(f"{K:5}{b:>3}{s:>6}{mod:>11.2f}  {lect}")
    # le point decisif
    m31 = dict((d[0], d[3]) for d in data)
    print(f"\nElectron 3_1 (spin-1/2) modulation = {m31['3_1']:.2f}")
    print(f"Nucleon-porteur 7_2       modulation = {m31['7_2']:.2f}")
    print("=> spin-1/2 existe A 0.00 (electron) : la variation n'est PAS necessaire au spin.")
    print("=> la modulation trace la symetrie C_q (type), pas le spin.")
    print("Q2 : la variation N'EST PAS l'origine du spin. (Q1 : elle perturbe l'energie de")
    print("     rotation = correction tenseur jonction Y, E=S(S+1) Lambda_chi/8, deja derivee.)")


if __name__ == "__main__":
    run()
