"""Reparation du coeur Sigma_c par la torsion : l'echelle des demi-tours (21/07).

PROBLEME : les coeurs constituants nus des baryons lourds (m_c+2m_l etc.)
manquent -4 a -11 % -- le pire residu de la table 225 (les pentaquarks
P_c en heritent leurs -6 %).

HYPOTHESE (prolonge le Verdict 7) : le coeur nu ignore la torsion de
RECOUPLAGE DE SPIN des rubans peripheriques. Chaque demi-tour coute
pi sigma_chi = 0,5501 GeV^2 dans M^2 (le quantum verifie V-P a 0,7 %).
REGLE candidate, fixee par les nombres quantiques (PAS par etat) :
    n = S_paire + J + 1/2
(paire legere antisymetrique S=0, J=1/2 -> n=1 ; symetrique S=1,
 J=1/2 -> n=2 ; S=1, J=3/2 -> n=3 -- une unite de spin = un demi-tour,
 exactement comme le retournement V-P des mesons.)
"""
import numpy as np

SQS = 0.41847
TURN = 2 * np.pi * SQS**2
HALF = TURN / 2
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
LC, XC, OC = MC + 2 * ML, MC + ML + MS, MC + 2 * MS
LB, XB, OB = MB + 2 * ML, MB + ML + MS, MB + 2 * MS
XCC = 2 * MC + ML

print(__doc__)
print("=" * 76)
print("[A] L'ECHELLE MESUREE puis LA REGLE (16 baryons lourds)")
print("=" * 76)
# (nom, m_exp, coeur, S_paire, J)
B = [("Lambda_c ", 2.28646, LC, 0, 0.5), ("Sigma_c  ", 2.45290, LC, 1, 0.5),
     ("Sigma_c* ", 2.51750, LC, 1, 1.5), ("Xi_c     ", 2.46790, XC, 0, 0.5),
     ("Xi_c'    ", 2.57840, XC, 1, 0.5), ("Xi_c*    ", 2.64590, XC, 1, 1.5),
     ("Omega_c  ", 2.69520, OC, 1, 0.5), ("Omega_c* ", 2.76590, OC, 1, 1.5),
     ("Xi_cc    ", 3.62160, XCC, 1, 0.5),
     ("Lambda_b ", 5.61960, LB, 0, 0.5), ("Sigma_b  ", 5.81060, LB, 1, 0.5),
     ("Sigma_b* ", 5.83210, LB, 1, 1.5), ("Xi_b     ", 5.79190, XB, 0, 0.5),
     ("Xi_b'    ", 5.93502, XB, 1, 0.5), ("Xi_b*    ", 5.95533, XB, 1, 1.5),
     ("Omega_b  ", 6.04520, OB, 1, 0.5)]
errs, errs_anti = [], []
for nom, m, core, sp, j in B:
    n_mes = (m * m - core * core) / HALF
    n_rule = int(sp + j + 0.5)
    m_pred = np.sqrt(core * core + n_rule * HALF)
    e = 100 * (m_pred / m - 1)
    errs.append(abs(e))
    if sp == 0:
        errs_anti.append(abs(e))
    print("  %s coeur %.3f  n mesure %5.2f | regle n=%d -> %.4f  (%+.2f %%)"
          % (nom, core, n_mes, n_rule, m_pred, e))
print("\n  mediane |err| avec la REGLE : %.2f %% (coeurs nus : 4-11 %%)"
      % np.median(errs))
print("  PAIRES ANTISYMETRIQUES (n=1, 4 etats Lc/Xic/Lb/Xib) : |err|")
print("  moyenne %.2f %% -- Lambda_b a +0,03 %%. Symetriques charme +" % np.mean(errs_anti))
print("  Omega_b : n=2/3 a 0,2-3 %%. ANOMALIE DECLAREE : Sigma_b/Xi_b'")
print("  mesurent n~4-5 (la regle n=2 laisse -1,4 a -2,4 %%) -- ouverte.")
print("  NULL (honnete) : barreau dM = %0.0f MeV au charme (~1,1 %% max)," % (1e3 * HALF / (2 * 2.5)))
print("  %.0f MeV au bottom (~0,4 %%) -- la regle est fixee par (S,J) a" % (1e3 * HALF / (2 * 5.9)))
print("  travers les 16 etats, PAS ajustee par etat (sinon vide au bottom).")

print("\n" + "=" * 76)
print("[B] LA CASCADE : pentaquarks P_c/P_cs avec le coeur Sigma_c repare")
print("=" * 76)
sc1 = np.sqrt(LC * LC + 2 * HALF)      # Sigma_c socle (n=2) = 2,4166
sc3 = np.sqrt(LC * LC + 3 * HALF)      # Sigma_c* socle (n=3)
xc1 = np.sqrt(XC * XC + 1 * HALF)      # Xi_c socle (n=1)
dd = ML + MC
dst = np.sqrt(dd * dd + HALF)          # D* socle
kst = np.sqrt(0.4976**2 + HALF)        # K* socle (demi-tour sur K)
PC = [("Pc(4312) = Sc D    ", 4.31190, sc1 + dd),
      ("Pc(4380) = Sc* D   ", 4.38000, sc3 + dd),
      ("Pc(4440) = Sc D*   ", 4.44030, sc1 + dst),
      ("Pc(4457) = Sc D*   ", 4.45730, sc1 + dst),
      ("Pcs(4338) = Xic D  ", 4.33820, xc1 + dd),
      ("Pcs(4459) = Xic D* ", 4.45880, xc1 + dst),
      ("X0(2900) = D* K*   ", 2.86600, dst + kst)]
for nom, m, pred in PC:
    print("  %s socle %.4f  (%+.2f %%)" % (nom, pred, 100 * (pred / m - 1)))
print("  -> LA FAMILLE PENTAQUARK FERME A <1 %% (etait -5 a -8 %%) : les")
print("     -6 %% des P_c etaient ENTIEREMENT le coeur Sigma_c nu ;")
print("     contenus moleculaires PDG standard, quantum unique, zero")
print("     parametre. X0(2900) = D*K* a +1,3 %% (interpretation standard).")

print("\n" + "=" * 76)
print("VERDICT : coeur Sigma_c REPARE (n=2 : 2,417, -1,5 %) ; regle")
print("n = S_paire + J + 1/2 (une unite de spin = un demi-tour, comme le")
print("V-P meson) ; mediane 16 baryons lourds ~1 %% ; cascade P_c <1 %%.")
print("Ouverts : Sigma_b/Xi_b' (n mesure 4-5) ; Omega_c legerement +2-3 %%.")
print("Propose : Lambda_c catalogue (-4,8 %%) -> constituant+torsion n=1")
print("(+0,6 %%) -- a valider avant toute retouche du catalogue canonique.")
