"""Torsion des sommets peripheriques du socle : le facteur qui categorise (21/07).

IDEE (utilisateur) : les masses les plus en ecart alterent les sommets
peripheriques du socle autour des noeuds -- une TORSION des courbes de
compensation qui altere les masses ; il doit exister un facteur lie au
socle qui CATEGORISE les masses.

QUANTUM CANDIDAT (zero parametre nouveau) : la torsion d'un anneau
peripherique est quantifiee par la pente derivee --
    1 tour complet  -> Delta M^2 = 2 pi sigma_chi = 1,1003 GeV^2 (= la
                       pente de Regge : l'orbital L = L tours) ;
    1 DEMI-tour     -> Delta M^2 = pi sigma_chi = 0,5501 GeV^2
                       (= le retournement du RUBAN : le meme fait 4pi
                       que le spin -- retourner le spin d'un compagnon
                       = un demi-tour du repere de ruban peripherique).
FACTEUR DE CATEGORIE : tau = Delta M^2 / (2 pi sigma_chi), en TOURS.
"""
import numpy as np

SQS = 0.41847
TURN = 2 * np.pi * SQS**2          # 1 tour  = 1,1003 GeV^2 (pente derivee)
HALF = TURN / 2                    # 1/2 tour = 0,5501 GeV^2

print(__doc__)
print("=" * 74)
print("[A] LE TEST-CLE : retournement de spin vecteur-pseudoscalaire")
print("    (retourner un compagnon = 1/2 tour du ruban peripherique)")
print("=" * 74)
VP = [("pi ->rho ", 0.13957, 0.77526), ("K  ->K*  ", 0.49368, 0.89555),
      ("D  ->D*  ", 1.86966, 2.01026), ("Ds ->Ds* ", 1.96835, 2.11220),
      ("B  ->B*  ", 5.27934, 5.32471), ("Bs ->Bs* ", 5.36692, 5.41540)]
taus = []
for nom, mp, mv in VP:
    dm2 = mv * mv - mp * mp
    taus.append(dm2 / TURN)
    mv_pred = np.sqrt(mp * mp + HALF)
    print("  %s dM2 = %.4f GeV^2 = %.3f tour  | M_V predit %.4f (%+.2f %%)"
          % (nom, dm2, dm2 / TURN, mv_pred, 100 * (mv_pred / mv - 1)))
print("  moyenne : %.4f GeV^2 = %.3f +- %.3f tour ;  1/2 tour = %.4f GeV^2"
      % (np.mean([mv * mv - mp * mp for _, mp, mv in VP]),
         np.mean(taus), np.std(taus), HALF))
print("  -> UNIVERSEL du pion au B_s (6 secteurs) : %.3f vs 0,500," % np.mean(taus))
print("     ecart %.1f %% -- le celebre 'dM2(V-P) ~ constant' de la" % (100 * abs(np.mean(taus) / 0.5 - 1)))
print("     phenomenologie EST le demi-tour du socle, VALEUR pi sigma_chi")
print("     DERIVEE (antecedents modeles de cordes ~1/(2 alpha') : declare ;")
print("     ici la valeur sort de sigma_chi recalibree, sans ajustement).")

print("\n" + "=" * 74)
print("[B] retournements de spin des baryons (1/2 -> 3/2, meme contenu)")
print("=" * 74)
BF = [("N   ->Delta   ", 0.93827, 1.23200), ("Sg  ->Sg(1385)", 1.18937, 1.38370),
      ("Xi  ->Xi(1530)", 1.31486, 1.53180), ("Sc  ->Sc*     ", 2.45290, 2.51750),
      ("Xic'->Xic*    ", 2.57840, 2.64590), ("Oc  ->Oc*     ", 2.69520, 2.76590),
      ("Sb  ->Sb*     ", 5.81060, 5.83210)]
for nom, m1, m3 in BF:
    dm2 = m3 * m3 - m1 * m1
    print("  %s dM2 = %.4f = %.2f tour" % (nom, dm2, dm2 / TURN))
print("  -> legers : 0,45-0,58 tour (~1/2 a +-15 %, la jonction Y du document principal) ;")
print("     lourds : derive 0,20-0,32 (l'inertie du coeur lourd amortit la")
print("     torsion -- categorie distincte, coherent avec tau_perp ~ 1/m).")

print("\n" + "=" * 74)
print("[C] exotiques LHC : la torsion en TOURS ENTIERS (molecules)")
print("=" * 74)
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
dd, ds = ML + MC, MS + MC
EX = [("X(3872)  ", 3.87165, 2 * dd), ("Tcc(3875)", 3.87480, 2 * dd),
      ("Zc(3900) ", 3.88750, 2 * dd), ("Zc(4020) ", 4.02430, 2 * dd),
      ("Zcs(3985)", 3.98270, dd + ds), ("Pc(4312) ", 4.31190, 2 * MC + 3 * ML),
      ("Pc(4440) ", 4.44030, 2 * MC + 3 * ML), ("Pc(4457) ", 4.45730, 2 * MC + 3 * ML),
      ("Zb(10610)", 10.6072, 2 * (MB + ML)), ("Zb(10650)", 10.6522, 2 * (MB + ML)),
      ("X(6600)  ", 6.63800, 4 * MC), ("X(6900)  ", 6.89900, 4 * MC)]
dist = []
for nom, m, core in EX:
    tau = (m * m - core * core) / TURN
    d = abs(tau - np.round(tau))
    dist.append(d)
    print("  %s coeur %.3f  tau = %5.2f tour  (au plus proche entier : %.2f)"
          % (nom, core, tau, d))
print("  distance moyenne a l'entier : %.3f (uniforme attendrait 0,25) --"
      % np.mean(dist))
print("  SUGGESTIF a basse masse (X(3872)/Tcc/Zc ~ 1 ; Zc(4020)/Pc(4312) ~ 2 ;")
print("  Pc(4440) ~ 3 ; X(6900) = 8,00) mais DECLARE : a haute masse le pas")
print("  1,1 GeV^2 devient dense (test faible) + selection des coeurs.")

print("\n" + "=" * 74)
print("[D] la categorisation complete (le facteur socle tau, en tours)")
print("=" * 74)
print("""  tau = 0        fondamentaux (coeurs de secteur, catalogue)
  tau = 1/2      retournement de spin MESON (V-P) : UNIVERSEL pi->Bs,
                 valeur pi sigma_chi a -1 %% -- le ruban 4pi mecanise
  tau ~ 1/2->0   retournement BARYON : 1/2 amorti par la jonction Y
                 (legers ~0,5, lourds ->0,2 : categorie a part)
  tau = L        bandes orbitales (la pente de Regge = 1 tour/unite L)
  tau = 1,38 n   radiales (ratio calibre du document principal ; ~ 1 tour + flexion)
  tau = entiers  molecules exotiques LHC (tours complets, suggestif)
  tau < 0        quarkonia : COMPRESSION (Coulomb-domine, secteur
                 Cornell calibre) -- la torsion ne s'y applique pas ;
                 le signe de tau categorise torsion vs compression.""")
print("  -> OUI : il existe un facteur automatique lie au socle -- le nombre")
print("     de (demi-)tours de torsion des anneaux peripheriques, quantifie")
print("     par LA pente derivee 2 pi sigma_chi. Il categorise les masses")
print("     et REPARE les pires ecarts de la table 224 (cf. mise a jour).")
