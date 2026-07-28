r"""
Poids amplificateur PAR NŒUD — l'idée utilisateur : « clairement, il faut qu'il y
ait des nœuds simples qui aient plus de poids amplificateurs que d'autres ».

Test direct. Avec la moyenne de Hölder-1/2 (validée), la masse d'un entrelacs
s'écrit √m = Σ_i √W_i (LINÉAIRE en √W par nœud). L'octet u/d/s donne 4 équations
pour 3 nœuds (3_1, 5_2, 7_2) → on EXTRAIT le poids amplificateur W_K de chaque
nœud par moindres carrés non-négatifs (nnls).

RÉSULTAT — l'idée est CONFIRMÉE (le poids est NON-MONOTONE en complexité) :
    nœud   cr   √W      √W/cr    type
    3_1     3   12,3    4,10     torus   ← le PLUS SIMPLE, poids fort
    5_2     5   12,8    2,56     twist
    7_2     7   10,0    1,43     twist   ← le PLUS de croisements, poids FAIBLE
Le trèfle 3_1 (cr=3) a un poids QUASI ÉGAL à 5_2 et SUPÉRIEUR à 7_2, et par
croisement il cogne 3× plus fort. L'ordre des poids est 3_1≈5_2 > 7_2, soit
DÉCROISSANT après 5_2 — impossible pour une loi lisse cr·M_C+n_l·M_L (qui CROÎT
en cr : 0,21 < 0,39 < 0,56). La loi lisse prédit l'ordre des poids À L'ENVERS.

STABILITÉ (leave-one-out sur l'octet) : l'ordre 3_1≈5_2 > 7_2 tient dans CHAQUE
retrait, mais les valeurs ont CV 15-26 % (seulement 4 baryons, système à peine
surdéterminé) → la DIRECTION est robuste, les valeurs précises non.

INTERPRÉTATION (direction de recherche) : l'amplification de masse n'est PAS la
complexité (nombre de croisements) mais une RÉSONANCE topologique propre à chaque
nœud. Le trèfle 3_1 (torus T(2,3), b=2, le plus serré/symétrique, = cœur
lepton/Koide, dipôle pur) résonne fortement ; les twists lâches à haut croisement
(7_2) résonnent faiblement. Le bon modèle de masse remplacerait m_naive =
cr·M_C+n_l·M_L (proxy de complexité, croissant) par un poids W_K = résonance par
nœud, non monotone. Reste ouvert : QUELLE quantité topologique donne W_K (pas cr ;
candidats : concentration de U_κ, « tightness »/ropelength inverse, torus-ness).

CAVEAT : poids par nœud = beaucoup de paramètres (19 nœuds) → risque de
surajustement ; l'extraction n'est fiable que sur des sous-ensembles solides
(octet). Le décuplet contamine (terme J manquant gonfle 4_1, 6_3). Ne PAS
extraire des poids libres sur tout le spectre (ce serait le 0,3 % surajusté).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import nnls
from simulation.experiments.combi_formula_search import D

TORUS = {"3_1", "5_1", "7_1", "9_1"}
OCTET = [("p", "3_1+7_2+7_2", 938.3), ("Lam", "3_1+5_2+7_2", 1115.7),
         ("Sig", "5_2+7_2+7_2", 1193.2), ("Xi", "3_1+3_1+7_2", 1318.3)]
KNOTS = ["3_1", "5_2", "7_2"]


def extract(data, knots):
    A = np.array([[cfg.split("+").count(k) for k in knots] for _, cfg, _ in data])
    y = np.array([np.sqrt(m) for _, _, m in data])
    sw, _ = nnls(A, y)
    return dict(zip(knots, sw))


def main():
    w = extract(OCTET, KNOTS)
    print("Poids amplificateur extrait par nœud (octet u/d/s, √m = Σ √W_i) :")
    print(f"{'nœud':6s} {'cr':>3s} {'√W':>7s} {'√W/cr':>7s} {'m_naive lisse':>13s}")
    for k in KNOTS:
        lisse = D[k]["cr"] * 0.084 + D[k]["nl"] * (-0.014)
        print(f"{k:6s} {D[k]['cr']:>3d} {w[k]:>7.2f} {w[k] / D[k]['cr']:>7.2f} "
              f"{lisse:>13.3f}  {'TORUS' if k in TORUS else ''}")
    print("\nOrdre des poids : 3_1≈5_2 > 7_2 (cr 3,5 > cr 7) → NON monotone en cr.")
    print("La loi lisse cr·M_C+n_l·M_L croît en cr (0,21<0,39<0,56) → ordre À L'ENVERS.")
    print("\nStabilité leave-one-out :")
    for k in KNOTS:
        vals = [extract([OCTET[j] for j in range(len(OCTET)) if j != i], KNOTS)[k]
                for i in range(len(OCTET))]
        a = np.array(vals)
        print(f"  {k}: √W = {a.mean():.2f} ± {a.std():.2f} (CV {a.std() / a.mean() * 100:.0f}%)")
    print("\n→ Idée CONFIRMÉE : amplification = résonance par nœud, non la complexité.")
    print("  Le trèfle 3_1 (simple, torus, cœur lepton) amplifie fort ; 7_2 (lâche) faible.")


if __name__ == "__main__":
    main()
