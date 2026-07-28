"""
Mécanisme unknot -> 3_1 (A3) : comment le « mou » devient un électron.

OBJECTION. Le scénario de boucle piégée veut que le « mou » (brin en excès,
topologiquement un nœud TRIVIAL) se noue en électron 3_1 lors de la β. Or on
ne passe PAS du nœud trivial au trèfle par déformation continue (isotopie) :
ce sont deux types topologiques distincts.

RÉSOLUTION. La désintégration β n'est pas une isotopie : c'est une
RECONNEXION (passage de brin / changement de croisement), la seule opération
qui change le type topologique. Deux faits la rendent cohérente :

  1. NOMBRE DE DÉNOUEMENT. u(3_1) = 1 : UNE seule reconnexion sépare le nœud
     trivial du trèfle. Plus généralement, pour le nœud torique T(2,n)
     (= la famille des fermions, 3_1,5_1,7_1) :  u = (n-1)/2.
     -> 3_1 : u=1,  5_1 : u=2,  7_1 : u=3.

  2. COMPTAGE. La β fait ΔLk = ±1 = EXACTEMENT une reconnexion. Elle peut
     donc nouer/dénouer précisément les nœuds de u=1, i.e. le 3_1 = électron
     (génération 1). Les nœuds u>1 (gén. 2,3) exigent plusieurs reconnexions
     -> barrière ∝ exp(-2 u k π) (limbox), fortement supprimés en un événement.

  3. RELATION DE CALUGAREANU  Lk = Tw + Wr. Le neutron sous-enlacé (Lk=+2 <
     Lk_opt=+3) stocke son déficit en TWIST + WRITHE : le « mou » est un brin
     COILÉ (writhe non nul), pas un cercle plat. C'est ce writhe pré-chargé
     qui, libéré par la reconnexion unique, se cristallise en les croisements
     du trèfle. On vérifie ci-dessous que le 3_1 porte un writhe d'ordre 3.

On calcule le writhe (intégrale de Gauss) des nœuds toriques pour ancrer le
point 3, et on relie u(K) à la barrière.

Usage :  python3 -m simulation.experiments.unknot_to_trefoil_A3
"""
from __future__ import annotations

import numpy as np

from theory.knot_catalog import knot_curve
from theory.postulates import K_CS


def writhe(curve):
    """Writhe par l'intégrale de Gauss discrète (somme sur paires de segments).

    Wr = (1/4π) Σ_{i≠j} (Δr_i × Δr_j)·(r_i - r_j)/|r_i - r_j|^3 .
    """
    r = np.asarray(curve, float)
    n = len(r)
    mids = 0.5 * (r + np.roll(r, -1, axis=0))     # milieux de segments
    segs = np.roll(r, -1, axis=0) - r              # vecteurs segments
    total = 0.0
    for i in range(n):
        d = mids[i] - mids                         # (n,3)
        cross = np.cross(np.broadcast_to(segs[i], segs.shape), segs)
        num = np.einsum("kj,kj->k", cross, d)
        dist = np.linalg.norm(d, axis=1)
        mask = dist > 1e-9
        total += np.sum(num[mask] / dist[mask] ** 3)
    return total / (4 * np.pi)


def main():
    print("=" * 70)
    print("  MÉCANISME unknot -> 3_1 (A3) : le « mou » devient un électron")
    print("=" * 70)

    print("\n  La β n'est pas une isotopie mais une RECONNEXION (passage de")
    print("  brin) : la seule opération qui change le type topologique.\n")

    # --- (1)(2) nombre de dénouement de la famille torique ---
    print("  (1-2) Nombre de dénouement u(K) = nb de reconnexions :")
    print(f"    {'nœud':>6} {'T(2,n)':>8} {'croisements':>12} {'u(K)':>5} "
          f"{'génération':>11}")
    fam = [("3_1", 3, 1), ("5_1", 5, 2), ("7_1", 7, 3)]
    for name, n, gen in fam:
        u = (n - 1) // 2
        print(f"    {name:>6} {'T(2,'+str(n)+')':>8} {n:>12} {u:>5} {gen:>11}")
    print("    -> u(3_1)=1 = LE seul nœud atteignable en UNE reconnexion (ΔLk=±1)")
    print("       = l'électron (gén. 1). Les gén. 2,3 (u=2,3) exigent 2,3")
    print("       reconnexions : fortement supprimées en un événement faible.")

    # --- (3) writhe pré-chargé dans le mou ---
    print("\n  (3) Writhe porté par chaque nœud (intégrale de Gauss) :")
    print("      (le « mou » doit coiler à ce writhe pour se nouer ainsi)")
    print(f"    {'nœud':>6} {'writhe Wr':>11} {'|Wr|/croisements':>17}")
    for name, n, gen in fam:
        c = knot_curve(name, 600)
        wr = writhe(c)
        print(f"    {name:>6} {wr:>11.2f} {abs(wr)/n:>17.2f}")
    print("    -> le 3_1 porte |Wr| ~ 3 (ordre du nombre de croisements) : le")
    print("       mou coilé (Călugăreanu Lk=Tw+Wr, neutron sous-enlacé) stocke")
    print("       ce writhe ; la reconnexion unique le fige en trèfle.")

    # --- barrière vs u(K) ---
    print("\n  Lien avec la barrière (limbox) : barrière ∝ exp(-2 u k π).")
    print(f"    Avec k = k_CS = {K_CS} (P20) :")
    print(f"    {'nœud':>6} {'u':>3} {'suppression relative exp(-2ukπ)':>34}")
    for name, n, gen in fam:
        u = (n - 1) // 2
        # rapport à u=1 (en log10, sinon underflow) : exp(-2(u-1)kπ)
        log10rel = -2 * (u - 1) * K_CS * np.pi / np.log(10)
        print(f"    {name:>6} {u:>3}   ~10^{log10rel:>8.0f} (relatif au 3_1)")
    print("    -> écart ASTRONOMIQUE : la β ne produit QUE le 3_1 (u=1).")

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(11, 4.3))
        # (a) le trèfle 3D
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        c = knot_curve("3_1", 400)
        ax1.plot(c[:, 0], c[:, 1], c[:, 2], color="red", lw=2)
        ax1.set_title("électron $= 3_1$ : $u=1$, $|Wr|\\approx 3$")
        ax1.set_axis_off()
        # (b) u(K) et accessibilité faible
        ax2 = fig.add_subplot(1, 2, 2)
        ns = [3, 5, 7]
        us = [(n - 1) // 2 for n in ns]
        labels = ["$3_1$\n(e, gén.1)", "$5_1$\n(gén.2)", "$7_1$\n(gén.3)"]
        colors = ["red", "orange", "gray"]
        ax2.bar(range(3), us, color=colors, alpha=0.8)
        ax2.axhline(1, color="green", ls="--", lw=1.2,
                    label="1 reconnexion = $\\Delta Lk=\\pm1$ (β)")
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.set_ylabel("nombre de dénouement $u(K)$")
        ax2.set_title("Seul $u=1$ ($3_1$) est atteignable en une β")
        ax2.legend(fontsize=9)
        for i, u in enumerate(us):
            ax2.text(i, u + 0.05, f"$u={u}$", ha="center", fontsize=9)
        fig.tight_layout()
        out = "output/unknot_to_trefoil_A3.png"
        fig.savefig(out, dpi=130)
        print(f"\n  figure → {out}")
    except Exception as e:
        print(f"  (figure non générée : {e})")

    print("\n  CONCLUSION :")
    print("  • « unknot -> 3_1 » n'est PAS une déformation interdite : c'est une")
    print("    RECONNEXION (passage de brin), l'opération qui change la topologie.")
    print("  • u(3_1)=1 : une seule reconnexion sépare nœud trivial et trèfle —")
    print("    exactement le ΔLk=±1 d'UN événement faible. D'où l'électron 3_1.")
    print("  • Le neutron sous-enlacé stocke (Călugăreanu Lk=Tw+Wr) le writhe ~3")
    print("    dans le mou coilé ; la reconnexion le fige en trèfle.")
    print("  • Les nœuds u>1 (gén. 2,3) exigent plusieurs reconnexions -> barrière")
    print("    exp(-2ukπ) écrasante : la β ne crée QUE le 3_1.")
    print("=" * 70)


if __name__ == "__main__":
    main()
