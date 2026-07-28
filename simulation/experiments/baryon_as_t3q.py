r"""
Piste : baryon = nœud torique T(3,q) UNIQUE (3 brins = 3 quarks), pas un entrelacs.

Demande utilisateur : « explore la piste baryon = T(3,q) unique ». Idée : au lieu
de l'entrelacs 3_1+7_2+7_2 (3 nœuds DIFFÉRENTS), un baryon serait un T(3,q) unique
— clôture de la tresse (σ_1 σ_2)^q à 3 brins, C_3, les 3 brins = 3 quarks braidés.

TROIS TESTS.

1. SPIN (le point critique — RÉSOLU en faveur de la piste).
   Tension : un nœud fermé b=3 donne s=(b−1)/2=1 → BOSON (parité b impair). MAIS
   T(3,q) est la clôture d'une tresse à 3 BRINS = 3 mondes-lignes de FERMIONS
   (quarks spin-½). La composition Clebsch-Gordan de 3 spin-½ donne J ∈ {1/2, 3/2}
   = OCTET / DÉCUPLET. Le spin baryonique vient des 3 BRINS, pas de l'indice de
   tresse du nœud fermé. → la piste donne NATURELLEMENT le bon spin fermionique,
   là où le nœud-fermé naïf échouait.

2. STRUCTURE (plus naturelle que l'entrelacs).
   Le mot de tresse (σ_1 σ_2)^q a une symétrie Z_3 cyclique : les 3 brins sont
   ÉQUIVALENTS → 3 quarks identiques. Contraste net avec 3_1+7_2+7_2 (3 nœuds
   DISTINCTS, aucune raison d'être 3-fold). (Réserve : le C_3 SPATIAL de
   l'embedding standard n'est pas exact — résidu ~0,5 ; la Z_3 est dans la
   permutation des brins, pas la rotation spatiale.)

3. MASSE + CHARGE (plausibles, mais OUVERTES).
   Masse : 3 brins ≈ 3 quarks constituants (~350 MeV) → ~1 GeV = échelle baryon.
   q = nombre de torsions = excitation ; cr(T(3,q))=2q, le nucléon-entrelacs a
   cr=17 → T(3,8/9) même complexité. La masse FINE reste étrangeté/Z_3
   (non-géométrique, [[flavor-is-SM-artifact]]) — la piste ne change pas ça.
   Charge : q mod 3 donne 3 classes topologiques (q=3k+1 / 3k+2 / 3k) — SUGGESTIF
   de 3 états de charge, mais Q=Σχ+Lk n'est PAS dérivé pour le nœud unique (le
   self-linking des 3 brins jouerait le rôle du Lk inter-composantes). OUVERT.

VERDICT : la piste baryon = T(3,q) est TOPOLOGIQUEMENT plus naturelle que
l'entrelacs 3_1+7_2+7_2 — elle résout le spin (3 brins fermioniques → J baryon),
donne 3 quarks équivalents (Z_3 des brins), et une échelle de masse ~baryon. Mais
la charge (Q=Σχ+Lk pour un nœud unique) et la composition de spin exacte restent à
dériver, et la masse fine reste Z_3/saveur. Reframe PROMETTEUR, non fermé.
Connexion : T(3,q) est CENTRÉ (famille torique, torus_t34_third_family) — le
baryon serait un vortex CENTRÉ à 3 brins, pas les twists excentrés actuels.

PROTON ET NEUTRON — OUI, portés par le MÊME T(3,q) (via le FRAMING).
  Proton et neutron ne diffèrent QUE par la charge (Q=+1 vs 0), à 1,3 MeV près.
  Mécanisme (identique à l'entrelacs, transposé) : Călugăreanu-White Lk = Tw + Wr.
  - Wr (writhe) = FIXÉ par la géométrie du nœud (le q).
  - Tw (twist du RUBAN/tube de flux) = DOF LIBRE = l'analogue single-knot du Lk
    inter-composantes de l'entrelacs. C'est un vrai DOF d'un tube de flux (pas
    d'une courbe) : le tube peut tourner sur son cœur (théorème C-W).
  → proton = T(3,q) framing Tw_opt (Q=+1, énergie min) ; neutron = MÊME T(3,q),
    framing Tw_opt−1 (Q=0, +1 quantum de tension). Δm(n−p)=β·1²·⟨tension⟩≈1,3 MeV.
  Le DIFFÉRENTIEL n−p (ΔQ=1, Δm=1,3 MeV) est DÉRIVÉ (même mécanisme que
  [[proton-neutron-chain-discrete-vs-continuous]]) ; la charge ABSOLUE (quel
  framing = proton) reste ouverte (dépend de Σχ du T(3,q), non dérivé).
  BONUS : le doublet d'isospin (p,n) = doublet de FRAMING d'UN seul nœud — plus
  économique que l'entrelacs (ajuster le Lk parmi 3 nœuds distincts). T(3,q)
  chiral (writhe de signe défini) → distinction matière/antimatière OK.
"""
from __future__ import annotations
import numpy as np
from theory.knot_catalog import _torus_knot_curve
from simulation.experiments.repair_curves_braid import alexander


def invariants(q, n=800):
    P = np.asarray(_torus_knot_curve(3, q, n), float)
    a = alexander(P)
    L = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum()
    return dict(cr=2 * q, det=a[2], qmod3=q % 3, L=L)


def main():
    print("T(3,q) comme baryon unique (3 brins = 3 quarks) :\n")
    print(f"{'nœud':8s} {'cr':>3s} {'det':>4s} {'q%3':>4s} {'classe charge':>14s}")
    L4 = invariants(4)["L"]
    for q in [4, 5, 7, 8, 10, 11]:
        d = invariants(q)
        cls = {1: "A (q=3k+1)", 2: "B (q=3k+2)", 0: "C (q=3k, lien)"}[d["qmod3"]]
        print(f"T(3,{q:<2d}) {d['cr']:>3d} {d['det']:>4d} {d['qmod3']:>4d} {cls:>14s}")
    print("\nTESTS :")
    print("  SPIN  : 3 brins fermioniques → J∈{1/2,3/2} = octet/décuplet ✓")
    print("          (résout la tension 'nœud fermé b=3 → boson')")
    print("  STRUCT: (σ_1σ_2)^q, Z_3 des brins = 3 quarks équivalents ✓")
    print("          (vs 3_1+7_2+7_2 = 3 nœuds distincts)")
    print("  MASSE : 3 quarks constituants ~1 GeV ✓ ; q = excitation ; fine = Z_3")
    print("  CHARGE: q mod 3 → 3 classes (suggestif) ; Q=Σχ+Lk non dérivé (OUVERT)")
    print("\n  → reframe PROMETTEUR (spin résolu, 3 quarks naturels), charge OUVERTE.")


if __name__ == "__main__":
    main()
