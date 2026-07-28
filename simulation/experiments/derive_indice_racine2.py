"""Derivation de l'indice rac2-1 : la dimension quantique de la paire (21/07).

CIBLE : le dernier element non derive de l'echelle de recouplage --
l'exces d'echange des paires de noeuds IDENTIQUES, mesure ~0,42
demi-tour, forme close candidate rac2-1 (rang d'indice).

MECANISME CANDIDAT (identification conditionnelle) :
  (1) Une paire de noeuds identiques a DEUX canaux de fusion --
      symetrique et antisymetrique (pour ss/cc l'antisym fondamental
      est verrouille par le Pauli tresse, mais le CANAL existe).
  (2) Theoreme des dimensions quantiques (consistance de fusion
      s x s = 1 + psi, l'unique vecteur propre positif) :
      d = sqrt(2 canaux) = rac2 -- FORCE, pas choisi. C'est d_1/2(k=2),
      LE MEME nombre que l'amplitude de Koide A = rac2 : les deux
      indices du document principal sont UNE SEULE structure (le doublement de
      canal des objets topologiques identiques).
  (3) Maillon conditionnel (nomme, non demontre) : l'anneau
      peripherique partage de la paire ne se referme qu'en
      transportant le ruban dans l'espace des canaux double ->
      twist residuel = (d - 1) demi-tour.
  => exces = rac2 - 1 = 0,4142 ; deux criteres SEPARES :
     exces <- meme TYPE de noeud (geometrie du ruban ; ll : u/d
     distingues par Lk ont l'exces SANS le verrou) ;
     verrou <- meme etat complet (type+Lk : ss, cc).
"""
import numpy as np

SQS = 0.41847
HALF = np.pi * SQS**2
R2 = np.sqrt(2)
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
LC, XC, OC = MC + 2 * ML, MC + ML + MS, MC + 2 * MS
OB = MB + 2 * MS

print(__doc__)
print("=" * 76)
print("[A] LA MESURE : trois determinations independantes de l'exces")
print("=" * 76)
flip_ls = (2.5784**2 - 2.46790**2) / HALF          # Xi_c' - Xi_c (nue)
e_sig = (2.4535**2 - 2.28646**2) / HALF - flip_ls  # Sigma_c : bascule ll - ls
e_omc = (2.6952**2 - OC * OC) / HALF - 1           # Omega_c : offset - base
e_omb = ((6.0452**2 - OB * OB) / HALF - 1) / 3     # Omega_b : (offset-base)/3
vals = np.array([e_sig, e_omc, e_omb])
print("  Sigma_c (bascule ll - bascule ls)     : %.4f" % e_sig)
print("  Omega_c (offset - base 1)             : %.4f" % e_omc)
print("  Omega_b ((offset - base)/3, pas Z3)   : %.4f" % e_omb)
print("  => exces = %.4f +- %.4f (stat) +- ~0,01 (coeurs)"
      % (vals.mean(), vals.std()))
print("  Trois familles, deux generations, une valeur : l'exces est une")
print("  CONSTANTE STRUCTURELLE (aveugle a la saveur, x3 par generation).")

print("\n" + "=" * 76)
print("[B] CONFRONTATION DES FORMES CLOSES (null declare)")
print("=" * 76)
m = vals.mean()
for nom, v, com in [
        ("rac2 - 1 (d-1, MECANISME)", R2 - 1, "SELECTIONNE (canal double)"),
        ("5/12 (arithmetique)", 5 / 12, "plus proche mais SANS mecanisme"),
        ("ln(3/2)", np.log(1.5), "defavorise"),
        ("1/2 (poids Ising h_psi)", 0.5, "EXCLU (~16 %)"),
        ("2 - phi", 2 - (1 + np.sqrt(5)) / 2, "EXCLU (~10 %)"),
        ("e/(2 pi)", np.e / (2 * np.pi), "sans mecanisme")]:
    print("  %-28s %.4f  (%+5.1f %%)  %s" % (nom, v, 100 * (m / v - 1), com))
print("  Regle du null : l'accord numerique seul ne selectionne pas (5/12")
print("  bat rac2-1 de peu) ; ce qui selectionne = le MECANISME (d force")
print("  par la fusion) + l'unification avec Koide (un indice, pas deux).")

print("\n" + "=" * 76)
print("[C] MECANISMES CONCURRENTS ELIMINES")
print("=" * 76)
for ang in [120, 90, 0]:
    print("  co-rotation L2 (axes a %3d deg) : cout = %.3f"
          % (ang, np.sqrt(2 + 2 * np.cos(np.radians(ang)))))
print("  -> la co-rotation exigerait des axes ORTHOGONAUX (90 deg) pour")
print("     donner rac2 ; la jonction Y les met a 120 deg (cout 1, pas")
print("     rac2) : ECARTEE par la geometrie. Le poids d'Ising h_psi = 1/2")
print("     est EXCLU par la mesure (~4 sigma). Survit : d = rac2 (fusion).")

print("\n" + "=" * 76)
print("[D] CONSEQUENCE : le verrou cc et l'attachement du x3")
print("=" * 76)
xcc_core = 2 * MC + ML
n_xcc = (3.6216**2 - xcc_core**2) / HALF
for nom, n in [("verrou seul  1+(rac2-1)", R2),
               ("verrou x3    1+3(rac2-1)", 1 + 3 * (R2 - 1)),
               ("bascule      1+rac2 (contredit le verrou)", 1 + R2)]:
    mm = np.sqrt(xcc_core**2 + n * HALF)
    print("  Xi_cc %s : n=%.3f -> %.4f GeV (%+.2f %%)"
          % (nom, n, mm, 100 * (mm / 3.6216 - 1)))
print("  n mesure %.2f ; sensibilite 0,43/%% de coeur : les trois lectures" % n_xcc)
print("  tiennent dans +-0,7 %% -- l'attachement du 3^g n'est PINNE que")
print("  pour paire legere + spectateur lourd ; Xi_cc reste AMBIGU")
print("  (declare) ; les fourchettes Omega_cc/Omega_ccc sont inchangees.")

print("\n" + "=" * 76)
print("VERDICT : l'indice rac2-1 devient une IDENTIFICATION -- exces =")
print("d - 1 avec d = rac2 la dimension quantique FORCEE de la paire de")
print("noeuds identiques (2 canaux de fusion) ; le MEME rac2 que Koide")
print("(A = d_1/2(k=2)) : les deux indices du document principal = UNE structure, le")
print("doublement de canal. Mesure 0,420+-0,005 vs 0,414 (+1,4 %%, dans")
print("les coeurs). Maillon restant nomme : twist residuel = (d-1)")
print("demi-tour (non demontre) -- l'echelle n'a plus d'INCONNU, elle a")
print("UN maillon.")
