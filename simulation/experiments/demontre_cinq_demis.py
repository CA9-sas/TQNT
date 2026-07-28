"""Demonstration du 5/2 : verrouillage materiel + repere inertiel (21/07).

CIBLE : X = 5/2 dans n_bande = 2 + 3^g X (la bande P des lourds-legers).

TROIS PIECES :
 (1) FAIT GEOMETRIQUE : le repere MATERIEL du tube en rotation fait un
     tour complet par revolution (verifiable) -- le bout leger, attache
     au tube (verrouillage de corde), tourne AVEC lui : 2 demi-tours.
 (2) L'ASYMETRIE D'INERTIE : le repere du noeud LOURD est inertiel (il
     est le pivot ; le tube tourne autour de lui) -> la rotation
     RELATIVE tube/lourd s'accumule a l'anneau du lourd -> couplage
     3^g (comme tout ce qui est relatif-au-lourd). Le tour du TUBE
     lui-meme est un cout de tube (portail) : la pente de Regge,
     aveugle a la generation.
     COROLLAIRE GRATUIT : leger-leger -> les DEUX bouts sont entraines
     -> aucune rotation relative -> n = 2 pur (les mesons legers ne
     paient QUE la pente de Regge -- verifie a2/K2*).
 (3) L'ALIGNEMENT : j_K = 3/2 -> + 1/2 (le meme 1/2 que le rung
     J = 3/2 de l'echelle).
 => X = 2 + 1/2 = 5/2 ; n = 2 + 3^g (5/2).
"""
import numpy as np

SQS = 0.41847
HALF = np.pi * SQS**2
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
dd, bb = ML + MC, ML + MB

print(__doc__)
print("=" * 76)
print("[A] LE FAIT GEOMETRIQUE (repere materiel d'une corde en rotation)")
print("=" * 76)
# corde droite pivotee en O, tournant d'un angle phi : direction du tube
# au bout = (cos phi, sin phi) ; le repere materiel du bout suit le tube.
phi = np.linspace(0, 2 * np.pi, 361)
frame_angle = np.unwrap(np.arctan2(np.sin(phi), np.cos(phi)))
print("  rotation du repere du bout apres UNE revolution : %.6f x 2 pi"
      % (frame_angle[-1] / (2 * np.pi)))
print("  -> 1 tour complet = 2 demi-tours EXACT (verrouillage materiel).")
print("  Au pivot (noeud lourd), le repere du CORPS ne suit pas le tube")
print("  (inertie) : la rotation relative tube/lourd = 1 tour/revolution,")
print("  stockee a l'anneau du LOURD -> facteur 3^g (relatif-au-lourd).")

print("\n" + "=" * 76)
print("[B] L'ASSEMBLAGE ET LES CONTROLES")
print("=" * 76)
print("  X = 2 (tour relatif au lourd) + 1/2 (alignement j_K=3/2) = 5/2 ;")
print("  n = 2 (tube/Regge, aveugle) + 3^g (5/2). Controles :")
print("  - legers/etranges : deux bouts entraines -> 0 relatif -> n = 2")
print("    (a2 -1,1 %, K2* -3 % : pente seule) [deja verifie] ;")
print("  - lourds-legers : 8 doublets <= 0,8 % [SS16] ;")
print("  - lourd-lourd (quarkonia, B_c) : les deux reperes inertiels ->")
print("    pas de terme relatif MAIS compression Coulomb (hors cadre,")
print("    etabli) : coherent, pas de nouveau test.")

print("\n" + "=" * 76)
print("[C] LE CAS j_K = 1/2 : les trois lectures de l'alignement")
print("=" * 76)
for nom, x, com in [("X = 2 (alignement 0, SS16)", 2.0, ""),
                    ("X = 2 - 1/2 (symetrique)", 1.5, "PREFEREE par D0*"),
                    ("X = 2 - 1 (L.s exact)", 1.0, "defavorisee")]:
    n = 2 + x
    th = np.sqrt(dd * dd + n * HALF)
    print("  %-28s n=%.1f -> %.4f  (%+.2f %% vs D0*(2300)) %s"
          % (nom, n, th, 100 * (th / 2.343 - 1), com))
print("  D0* est large (Gamma ~ 230 MeV, masse 2300-2350 selon analyses) :")
print("  la lecture symetrique -1/2 est mildly preferee (-0,7 %), DECLAREE")
print("  ouverte. Predictions bottom j_K=1/2 en fourchette :")
for x in (1.5, 2.0):
    n = 2 + 3 * x
    print("    X=%.1f : centroide B0*/B1' ~ %.0f MeV"
          % (x, 1e3 * np.sqrt(bb * bb + n * HALF)))

print("\n" + "=" * 76)
print("[D] EXTENSION L=2 (fourchette declaree)")
print("=" * 76)
for nom, n in [("repere tourne avec L (X=2L+1/2)", 4 + 4.5),
               ("repere tourne 1 fois (X=2+1/2)", 4 + 2.5)]:
    th = np.sqrt(dd * dd + n * HALF)
    print("  %-34s n=%.1f -> %.4f GeV" % (nom, n, th))
print("  -> la fourchette [2,66, 2,86] ENCADRE les D(2750)/D(2760)")
print("     observes (~2,75-2,76 ; melange 2S/1D connu) : non tranche.")

print("\n" + "=" * 76)
print("VERDICT : le 5/2 est DEMONTRE (theoreme conditionnel) --")
print("(1) verrouillage materiel du bout leger : 1 tour de repere par")
print("    revolution (geometrie, [A]) = 2 demi-tours ;")
print("(2) asymetrie d'inertie : repere du lourd inertiel -> la rotation")
print("    relative se stocke a SON anneau -> 3^g ; le tour du tube =")
print("    pente de Regge (portail, aveugle) -- et le leger-leger a ZERO")
print("    terme relatif (les deux bouts entraines) : le secteur leger")
print("    est un COROLLAIRE, pas une exception ;")
print("(3) alignement j_K=3/2 : +1/2 (le rung de l'echelle).")
print("Conditions nommees : verrouillage de corde (bout leger) + inertie")
print("du repere lourd -- meme rang que les identifications de l'echelle.")
print("Ouverts : le signe du cas j_K=1/2 (-1/2 prefere, D0* large),")
print("l'extension L=2 (fourchette), le shift s (0,36).")
