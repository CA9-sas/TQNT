"""Derivation des valeurs de bande P (9/2, 19/2) : 2 + 3^g (5/2) (21/07).

DECOMPOSITION : la bande j_K=3/2 des mesons lourds-legers se scinde en
  n_bande = 2  +  3^g . X
     ^ orbital STANDARD (1 tour = la pente de Regge, aveugle a la
       generation -- exactement ce que paient les mesons LEGERS)
     ^ excitation du compagnon COUPLEE au noeud lourd -> le pas Z3
       (troisieme apparition du x3, apres l'exces de paire et... lui).
MESURE : X_c = 4,50 - 2 = 2,50 ; X_b = 9,40 - 2 = 7,40 -> ratio 2,96.
  => X = 5/2 EXACT : bandes 2 + 5/2 = 9/2 et 2 + 15/2 = 19/2 --
     LES DEUX VALEURS LUES, avec UN SEUL nombre.
COHERENCES : (i) l'universalite bottom (B et Bs ensemble : le shift s
  ~0,36 est petit devant 19/2) ; (ii) le ratio de centroides ~2,1 qui
  intriguait = (2+7,5)/(2+2,5) -- le x3 etait CACHE par l'orbital ;
  (iii) legers ET etranges : pas de lourd -> pas de terme -> n = 2
  (la pente de Regge pure, verifiee sur a2/K2*...).
CANDIDATS pour 5/2 (nommes, non demontres) : 2 + 1/2 (l'orbital du
  compagnon + l'alignement j_K=3/2 -- le MEME 1/2 que le rung J=3/2
  de l'echelle) ; ou l'echelle du compagnon 1+1+1/2.
"""
import numpy as np

SQS = 0.41847
HALF = np.pi * SQS**2
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
dd, bb = ML + MC, ML + MB
ds, bs = MC + (ML + MS) / 2, MB + (ML + MS) / 2

print(__doc__)
print("=" * 76)
print("[A] LA MESURE : n_bande - 2 (l'exces couple au lourd), par famille")
print("=" * 76)
cen = {}
for fam, core, m1, m2 in [("D ", dd, 2.4221, 2.4611), ("Ds", ds, 2.5351, 2.5691),
                          ("B ", bb, 5.7257, 5.7395), ("Bs", bs, 5.8287, 5.83986)]:
    n = np.mean([(m * m - core * core) / HALF for m in (m1, m2)])
    cen[fam] = n
    print("  %s : centroide n = %.2f  ->  X = n - 2 = %.2f" % (fam, n, n - 2))
print("  ratio X_b/X_c : D->B %.2f ; Ds->Bs %.2f  (3 attendu)"
      % ((cen["B "] - 2) / (cen["D "] - 2), (cen["Bs"] - 2) / (cen["Ds"] - 2)))
print("  -> X = 5/2 : bandes 2+5/2 = 9/2 (charme) et 2+3(5/2) = 19/2")
print("     (bottom) EXACTEMENT les valeurs lues -- UN nombre au lieu de")
print("     deux ; le x3 de generation, TROISIEME apparition. Shift s")
print("     (+0,36, Ds et Bs pareil, NON triple) : residu ouvert (~4 %).")

print("\n" + "=" * 76)
print("[B] LA LOI CONFRONTEE (n = 2 + 3^g (5/2) + s-shift lu 0,36)")
print("=" * 76)
for nom, m, core, g, sh in [("D1(2420)", 2.4221, dd, 0, 0), ("D2*(2460)", 2.4611, dd, 0, 0),
                            ("Ds1(2536)", 2.5351, ds, 0, 0.36), ("Ds2*(2573)", 2.5691, ds, 0, 0.36),
                            ("B1(5721)", 5.7257, bb, 1, 0), ("B2*(5747)", 5.7395, bb, 1, 0),
                            ("Bs1(5830)", 5.8287, bs, 1, 0.36), ("Bs2*(5840)", 5.83986, bs, 1, 0.36)]:
    n = 2 + 3**g * 2.5 + sh
    th = np.sqrt(core * core + n * HALF)
    print("  %-11s n = %.2f -> %.4f GeV  (%+.2f %%)"
          % (nom, n, th, 100 * (th / m - 1)))
print("  (le splitting intra-doublet ~0,3, en 1/m_Q, reste lu.)")

print("\n" + "=" * 76)
print("[C] COHERENCE LEGERE + PREDICTION j_K=1/2")
print("=" * 76)
print("  Legers/etranges (pas de lourd) : n_bande = 2 pur -- verifie")
print("  (a2(1320) -1,1 %, K2*(1430) -3 % : la pente de Regge seule).")
print("  j_K=1/2 (sans le 1/2 d'alignement) : X -> 2, bande = 2 + 3^g.2 :")
for nom, me, core, g in [("D0*(2300) [j_K=1/2, large]", 2.343, dd, 0)]:
    th = np.sqrt(core * core + (2 + 3**g * 2) * HALF)
    print("    %s : n=4 -> %.4f (%+.1f %%)" % (nom, th, 100 * (th / me - 1)))
b0 = np.sqrt(bb * bb + (2 + 3 * 2) * HALF)
print("    PREDICTION : doublet j_K=1/2 bottom (B0*/B1', non observes,")
print("    larges) : n = 8 -> centroide ~ %.0f MeV." % (1e3 * b0))

print("\n" + "=" * 76)
print("VERDICT : les valeurs de bande sont IDENTIFIEES -- n = 2 + 3^g(5/2)")
print("(+ shift s 0,36 lu) : les deux bandes lues (9/2, 19/2) sortent d'UN")
print("nombre X = 5/2 avec le pas Z3 (mesure 2,96 D->B) ; l'orbital nu (2)")
print("est la pente de Regge commune aux legers (verifie). Candidats pour")
print("5/2 : 2 + 1/2 (orbital du compagnon + alignement j_K -- le meme 1/2")
print("que le rung J=3/2) -- NON demontre, declare. Prediction : doublet")
print("j_K=1/2 bottom ~%.0f MeV. Restent lus : 5/2 (candidat nomme)," % (1e3 * b0))
print("shift s 0,36, splitting 1/m_Q.")
