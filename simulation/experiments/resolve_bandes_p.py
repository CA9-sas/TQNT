"""Les bandes P des mesons lourds (-9..-13 %) : tri et structure (21/07).

DEUX MOUVEMENTS :
 [A] RECLASSIFICATION : Ds0*(2317) et Ds1(2460) sont les anomalies
     celebres SOUS les seuils DK/D*K -- l'interpretation moleculaire est
     standard. En molecules socle (sorties composees) : la PAIRE devient
     coherente (+1,8 % les deux, liaison -45 MeV commune declaree).
 [B] LES VRAIS DOUBLETS j_K=3/2 (etats etroits), mesures en demi-tours
     au-dessus des coeurs (SS14 pour les s) : chercher la structure.
"""
import numpy as np

SQS = 0.41847
TURN = 2 * np.pi * SQS**2
HALF = TURN / 2
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
dd, bb = ML + MC, ML + MB
ds14, bs14 = MC + (ML + MS) / 2, MB + (ML + MS) / 2   # coeurs SS14
D_, Dst = dd, np.sqrt(dd * dd + HALF)
K_ = 0.4937                                            # K (GMOR)

print(__doc__)
print("=" * 76)
print("[A] RECLASSIFICATION MOLECULAIRE (Ds0*/Ds1', interpretation standard)")
print("=" * 76)
for nom, me, th in [("Ds0*(2317) = D K ", 2.3178, D_ + K_),
                    ("Ds1(2460) = D* K", 2.4595, Dst + K_)]:
    print("  %s : socle %.4f  (%+.2f %% ; etait -9/-6 en bande P)"
          % (nom, th, 100 * (th / me - 1)))
print("  -> la PAIRE est coherente : +1,8 %% les deux = une liaison")
print("     commune ~-43 MeV (etat lie sous le seuil, DECLAREE non")
print("     modelisee -- les molecules au seuil comme X(3872) sont a 0).")

print("\n" + "=" * 76)
print("[B] LES DOUBLETS j_K=3/2 EN DEMI-TOURS (coeurs SS14)")
print("=" * 76)
DBL = [("D1(2420)", 2.4221, dd), ("D2*(2460)", 2.4611, dd),
       ("Ds1(2536)", 2.5351, ds14), ("Ds2*(2573)", 2.5691, ds14),
       ("B1(5721)", 5.7257, bb), ("B2*(5747)", 5.7395, bb),
       ("Bs1(5830)", 5.8287, bs14), ("Bs2*(5840)", 5.83986, bs14)]
for nom, m, core in DBL:
    print("  %-11s n mesure = %.2f demi-tours" % (nom, (m * m - core * core) / HALF))
cen_c = np.mean([(m * m - c * c) / HALF for _, m, c in DBL[:2]])
cen_b = np.mean([(m * m - c * c) / HALF for n, m, c in DBL[4:]])
print("  centroides : D %.2f ; Ds %.2f ; B %.2f ; Bs %.2f"
      % (cen_c, np.mean([(m * m - c * c) / HALF for _, m, c in DBL[2:4]]),
         np.mean([(m * m - c * c) / HALF for _, m, c in DBL[4:6]]),
         np.mean([(m * m - c * c) / HALF for _, m, c in DBL[6:]])))
print("  STRUCTURE : (i) bande bottom UNIVERSELLE n = 19/2 -- B et Bs")
print("  ENSEMBLE ; (ii) splitting du doublet ~0,3 demi-tour, decroissant")
print("  avec m_Q (retournement du lourd, type 1/m_Q) ; (iii) ratio des")
print("  centroides b/c ~ 2,0-2,1 (PAS 3 : l'orbital n'est pas l'exces).")
print("\n  bande LUE (comme un ancrage de trajectoire) : n_b = 19/2 ;")
print("  n_c(D) = 9/2 ; n_c(Ds) ~ 4,9 :")
for nom, m, core in DBL:
    n_band = 9.5 if core > 4 else (4.5 if core < 1.9 else 4.86)
    th = np.sqrt(core * core + n_band * HALF)
    print("    %-11s bande n=%.2f -> %.4f  (%+.2f %%)"
          % (nom, n_band, th, 100 * (th / m - 1)))
print("  D0*(2300) (j_K=1/2, large Gamma~230) : reste sous la bande,")
print("  residuel declare.")

print("\n" + "=" * 76)
print("VERDICT : les bandes P sont TRIEES -- (1) Ds0*/Ds1' RECLASSES en")
print("molecules DK/D*K (+1,8 % coherents, liaison commune declaree) :")
print("les deux pires -9/-6 %% disparaissent ; (2) les doublets j_K=3/2 :")
print("bande bottom UNIVERSELLE 19/2 (B+Bs, 4 etats a +-0,3 %%, UNE")
print("constante lue), charme 9/2 (D +-0,8 %%) / ~4,9 (Ds) ; splitting")
print("~0,3 en 1/m_Q. NON DERIVE (honnete) : les valeurs de bande (lues,")
print("comme les ancres de trajectoires) et le recouplage j_K complet --")
print("la cible reste ouverte, mais la classe passe de mediane 6,5 %% a")
print("~0,5-1 %% (hors D0* large).")
