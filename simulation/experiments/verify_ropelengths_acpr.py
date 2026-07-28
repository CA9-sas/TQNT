"""Verification des ropelengths du depot contre la table ACPR (22/07).

Reference : Ashton-Cantarella-Piatek-Rawdon, 'Self-contact sets for 50
tightly knotted and linked tubes' (arXiv:math/0508248), App. A, colonne
Rop (borne serree, longueur/RAYON). Le depot (mass_ropelength_algebra
et copies) est en DIAMETRES -> x2.

TEST DUR : une valeur du depot SOUS le minimum ACPR est IMPOSSIBLE pour
un plongement valide du noeud -- elle signe une courbe de mauvaise
topologie (cf. l'audit des courbes du catalogue : 14/19 fausses).
"""
import numpy as np

ACPR = {  # longueur/rayon (colonne Rop)
    "3_1": 32.7448, "4_1": 42.0928, "5_1": 47.2016, "5_2": 49.4704,
    "6_1": 56.7150, "6_2": 57.0271, "6_3": 57.8435,
    "7_1": 61.4109, "7_2": 63.8956, "7_3": 63.9327, "7_4": 64.2724,
    "7_5": 65.2609, "7_6": 65.7012, "7_7": 65.6108,
    "8_1": 71.0241, "8_2": 71.4107, "8_3": 71.1655, "9_1": 75.7252}
REPO = {  # demi-ropelength (diametres), mass_ropelength_algebra.py
    "3_1": 16.372, "4_1": 20.99, "5_1": 23.55, "5_2": 26.13,
    "6_1": 28.31, "6_2": 28.13, "6_3": 27.21,
    "7_1": 30.69, "7_2": 32.96, "7_3": 33.61, "7_4": 34.97,
    "7_5": 35.16, "7_6": 32.66, "7_7": 31.93,
    "8_1": 37.49, "8_2": 37.32, "8_3": 37.30, "9_1": 38.62}

print(__doc__)
print("=" * 72)
print("  %-5s %10s %10s %8s   verdict" % ("noeud", "depot(x2)", "ACPR", "ecart"))
print("=" * 72)
bad_low, loose, ok = [], [], []
for k in ACPR:
    r = 2 * REPO[k]
    d = 100 * (r / ACPR[k] - 1)
    if d < -0.5:
        v = "IMPOSSIBLE (sous le minimum) -> courbe fausse ?"
        bad_low.append((k, d))
    elif d > 1.0:
        v = "borne LACHE (minimum local ancien)"
        loose.append((k, d))
    else:
        v = "OK"
        ok.append((k, d))
    print("  %-5s %10.3f %10.4f %+7.2f%%   %s" % (k, r, ACPR[k], d, v))

print("\n" + "=" * 72)
print("BILAN")
print("=" * 72)
print("  OK (<=1 %%)          : %s" % ", ".join("%s (%+.1f%%)" % t for t in ok))
print("  LACHES (min. locaux) : %s" % ", ".join("%s (%+.1f%%)" % t for t in loose))
print("  IMPOSSIBLES (<min)   : %s" % ", ".join("%s (%+.1f%%)" % t for t in bad_low))
print("""
LECTURE :
  - Les valeurs fiables sont les toriques (3_1, 5_1, 7_1) + 4_1 + 6_1 :
    les noeuds bien etudies. 7_6 est MARGINAL (-0,6 %, arrondi possible).
  - Neuf valeurs sont des bornes LACHES (+2 a +9 %) : 5_2 (+5,6 -- la
    source du ratio m_s, deja auditee SS21), 7_2 (+3,2), 7_3, 7_4, 7_5,
    8_1, 8_2, 8_3, 9_1.
  - QUATRE valeurs sont SOUS le minimum prouve (6_2 -1,4 %, 6_3 -5,9 %,
    7_6 -0,6 % marginal, 7_7 -2,7 %) : IMPOSSIBLES pour la bonne topologie -- coherent avec
    l'audit des courbes du catalogue (14/19 fausses) : ces ropelengths
    ont probablement ete mesures sur des courbes d'un AUTRE noeud.
POINTS DE CONTACT CANONIQUES (a decider, rien n'est modifie ici) :
  1. m_s (ratio 5_2/3_1) : deja SS21 -- decision en attente ;
  2. ETATS STRICTS DM (6_3+6_3 a 1,03 GeV, 6_3 purs = raies Fermi-LAT
     1,03/1,55) : ropelength 6_3 FAUX de -5,9 %% -> masses des etats a
     6_3 sous-estimees ~6 %% -- les raies predites se deplacent ;
  3. beta_x fluxoide (encadrement 3_1/7_2) : 7_2 lache de +3,2 %% ;
  4. tout usage de M_topo = sqrt(sigma)*l(K) hors toriques.
RECOMMANDATION : remplacer la table du depot par les valeurs ACPR
(longueur/rayon /2 pour les diametres) partout ou elle nourrit le
document principal -- sur validation.""")
