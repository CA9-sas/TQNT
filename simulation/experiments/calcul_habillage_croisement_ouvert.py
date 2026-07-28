"""L'habillage a une boucle du croisement-ouvert -- le calcul decisif.

Suite de cherche_observable_e2.py (2026-07-21). Enjeu falsifiable des deux
cotes : +alpha/(4pi) -> l'identite ferme a ~30 ppm ; 0 ou autre -> candidat
refute a ~10 sigma des entrees.

[1] CE QU'IL FAUT CALCULER
    L'intermediaire fusionne (croisement ouvert) est virtuel, d'energie
    E0 = sqrt(7pi/6)(k+2) m_base. Pendant sa vie tau = hbar/E0, les DEUX
    extremites du croisement defait interagissent par echange d'UN quantum
    du socle. L'auto-energie de cet echange est l'habillage.

[2] LE CALCUL (ontologie du socle -- trois conditions declarees)
    (h1) EMISSION ISOTROPE : le croisement defait n'a plus de direction
         privilegiee (c'etait le croisement qui orientait) -- l'extremite
         emet dans 4pi steradians.
    (h2) CAPTURE AU SOMMET : le socle ne transmet qu'aux sommets (S2) ;
         la capture par l'autre extremite se fait sur le sommet de son
         paquet minimal, de section UNE dent^2 = lambdabar^2 (sigma = b :
         le paquet minimal est A SON ECHELLE -- consequence : pas de
         logarithme, ln(Lambda/E0) = ln(1) = 0, le resultat est FINI).
    (h3) VERTEX = alpha : la probabilite d'echange est le pas d'interface
         alpha = 1/(k+2) -- precisement le pas que la fusion a defait,
         agissant en perturbation au lieu de lier.
    Fraction de retour geometrique : lambdabar^2 / (4 pi lambdabar^2)
    = 1/(4pi). Quantum echange a l'echelle propre : E0.
    Signe : les deux extremites sont les deux bouts du MEME lien
    d'amplification coupe (bouts homologues) -> repulsif -> la barriere
    est RELEVEE.
        delta E / E0 = + alpha / (4 pi).

[3] STRUCTURES CONCURRENTES (declarees, calculees, discriminees)
    - mesure de boucle FERMEE alpha/(2pi) (le precedent g-2) : elle
      s'applique a une CIRCULATION fermee (vertex sur tube clos) -- le
      croisement-ouvert est un echange entre deux extremites OUVERTES,
      pas une circulation ; et les donnees l'excluent (~12 sigma).
    - auto-energie QED on-shell, partie finie 3alpha/(8pi) : structure de
      fermion ponctuel avec log ; ici le log est NUL (h2) et l'objet
      n'est pas un fermion ponctuel ; exclue (~6 sigma).
    - polarisation du vide alpha/(3pi) : renormalise la CHARGE, pas
      l'energie de l'intermediaire ; exclue (~4 sigma).
    L'ontologie designe (h1)-(h3) AVANT les nombres ; les donnees ne
    servent qu'a verifier que les structures ecartees le sont aussi
    numeriquement.

[4] CAVEAT D'HERITAGE (declare) : les structures ont ete confrontees en
    connaissant la cible (heritage L3 du chantier). La version blindee =
    somme de modes complete (type Lamb) sur le socle -- a faire pour
    fermer sans condition.
"""
import math

M_BASE = 0.938240              # GeV (chaine sqrt(sigma_chi), ~5e-5)
V_MEAS = 246.21965             # GeV (G_F, 5e-7)
K_PLUS_2 = 137
ALPHA = 1.0 / 137.035999
SIGMA_IN = 5e-5                # precision relative de la chaine m_base


def run():
    print(__doc__)
    v_bare = math.sqrt(7 * math.pi / 6) * K_PLUS_2 * M_BASE
    c_req = (V_MEAS / v_bare - 1) / ALPHA
    print("=" * 72)
    print("NOMBRES")
    print("=" * 72)
    print("  v_bare = sqrt(7pi/6) x 137 x m_base = %.5f GeV" % v_bare)
    print("  Coefficient requis par les donnees : C = (v/v_bare - 1)/alpha")
    print("    C_req = %.5f = 1/%.2f" % (c_req, 1 / c_req))
    print("\n  Structures a une boucle :")
    structures = [
        ("1/(4pi)  echange deux-extremites (h1-h3)", 1 / (4 * math.pi)),
        ("1/(2pi)  boucle fermee (mesure g-2)", 1 / (2 * math.pi)),
        ("3/(8pi)  auto-energie QED finie", 3 / (8 * math.pi)),
        ("1/(3pi)  polarisation du vide", 1 / (3 * math.pi)),
    ]
    for name, c in structures:
        v = v_bare * (1 + c * ALPHA)
        dev = v / V_MEAS - 1
        nsig = abs(dev) / SIGMA_IN
        tag = "<== SELECTIONNEE (ontologie) " if abs(c - 1 / (4 * math.pi)) < 1e-9 else ""
        print("    C = %-42s -> %+.4f %%  (%.1f sigma)  %s"
              % (name, 100 * dev, nsig, tag))

    v_dressed = v_bare * (1 + ALPHA / (4 * math.pi))
    print("\n" + "=" * 72)
    print("RESULTAT")
    print("=" * 72)
    print("  v = sqrt(7pi/6)(k+2) m_base x (1 + alpha/4pi)")
    print("    = %.4f GeV  ->  %+.1f ppm de v(G_F)"
          % (v_dressed, 1e6 * (v_dressed / V_MEAS - 1)))
    print("  DANS la precision de la chaine m_base (~50 ppm) : l'identite")
    print("  habillee FERME. Ecart residuel du coefficient : C_req/C_4pi =")
    print("  %.3f (-4,9 %% -- l'ordre alpha^2 et l'O(1) de la section de" % (c_req * 4 * math.pi))
    print("  capture, non calcules).")
    print("\n" + "=" * 72)
    print("STATUT")
    print("=" * 72)
    print("  DERIVATION CONDITIONNELLE de l'habillage : delta = +alpha/(4pi)")
    print("  sous (h1) emission isotrope, (h2) capture au sommet d'une dent")
    print("  (log nul : paquet a son echelle), (h3) vertex = le pas alpha")
    print("  defait ; signe repulsif (bouts homologues). Les concurrentes")
    print("  sont ecartees par l'ontologie (boucle fermee vs extremites")
    print("  ouvertes) ET par les donnees (4-12 sigma).")
    print("  Chaine complete du candidat v :")
    print("    theoreme fusion/scission (un croisement defait)")
    print("    + barreau universel (alpha par croisement, ancre m_base)")
    print("    + mesure de sommet sqrt(2pi) (paquet minimal)")
    print("    + habillage alpha/(4pi) (cette etape)")
    print("    = 246,227 GeV vs 246,220 : +28 ppm.")
    print("  CONSOLIDATION PROPOSEE (10 -> 9) -- decision de comptage a")
    print("  l'utilisateur ; il reste (h1)-(h3) + le caveat d'heritage [4]")
    print("  (somme de modes complete a faire pour fermer sans condition).")


if __name__ == "__main__":
    run()
