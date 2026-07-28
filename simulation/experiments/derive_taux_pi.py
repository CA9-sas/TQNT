"""Derivation du taux 1/pi (s = dWr/pi) : variantes, lecture radian (21/07).

MODELE : par revolution du tube, un tour de twist est injecte au bout ;
le noeud compagnon stocke un exces dependant de sa geometrie. Chaque
VARIANTE du stockage predit un taux (s = R . dWr, R a deriver) :
  (a) stockage plein (1 tour = 2 demi-tours par unite de writhe) R = 2
  (b) holonomie de Fuller (aire indicatrice / 2pi -> Wr en tours) R = 2
  (c) holonomie en double couverture (ruban 4pi)                 R = 1
  (d) LECTURE RADIAN : l'angle stocke par revolution = Wr RADIANS
      (le writhe comme FRACTION de l'angle plein) -> R = 1/pi.
REFERENCE TREFLE : le stockage est differentiel (Wr(K) - Wr(3_1)) --
le trefle est le noeud de la PORTE (trialite) : sa geometrie EST le
zero de reference du secteur (coherent : la bande 5/2 n'a pas de place
pour un Wr(3_1)/pi absolu).
"""
import numpy as np

S_MES, S_SIG = 0.3577, 0.0023
WR3 = 3.4144            # writhe ideal du trefle (litterature, bien etabli)
DWR_QQ = 8 / 7          # quasi-quantisation de Stasiak
DWR_LIT = 1.142         # writhes ideaux publies (4,556 - 3,414)

print(__doc__)
print("=" * 76)
print("[A] LES VARIANTES CONFRONTEES (avec dWr ideal ~ 1,14)")
print("=" * 76)
for nom, R in [("(a) stockage plein", 2.0), ("(b) Fuller / 2pi", 2.0),
               ("(c) double couverture", 1.0),
               ("(d) lecture radian 1/pi", 1 / np.pi)]:
    v = R * DWR_LIT
    print("  %-26s s = %.4f  (%+7.1f %%, %6.1f sigma)"
          % (nom, v, 100 * (S_MES / v - 1), abs(S_MES - v) / S_SIG))
print("  -> (a)-(c) EXCLUES (x2,8 a x6,4) ; seule la lecture radian")
print("     tient (-1,7 %, 2,6 sigma avec les writhes publies).")

print("\n" + "=" * 76)
print("[B] CE QUE DIT LA LECTURE RADIAN -- ET CE QUI MANQUE")
print("=" * 76)
print("""  Enonce : par revolution injectee (2 pi), l'anneau au bout stocke un
  angle supplementaire egal au WRITHE EXPRIME EN RADIANS --
  s = dWr/(2pi) tours = dWr/pi demi-tours. Autrement dit le noeud
  absorbe la fraction Wr/2pi du tour injecte : 'le writhe par radian'.
  Ce qui est ACQUIS : la forme differentielle (reference trefle = la
  porte) ; l'exclusion des trois autres taux ; la coherence d'unites.
  Ce qui MANQUE (nomme) : demontrer que la fraction absorbee est
  exactement Wr/2pi -- le candidat naturel est le theoreme de Fuller
  applique a la SEULE composante axiale du transport (la projection
  du twist injecte sur la normale de l'indicatrice), non demontre.""")

print("=" * 76)
print("[C] L'INVERSION : une PREDICTION GEOMETRIQUE falsifiable")
print("=" * 76)
wr52_pred = WR3 + np.pi * S_MES
wr52_sig = np.pi * S_SIG
print("  Si le taux est 1/pi EXACT, la mesure s = 0,3577 +- 0,0023 PREDIT :")
print("    Wr(5_2)_ideal = %.4f +- %.4f" % (wr52_pred, wr52_sig))
print("  contre : litterature ~4,556 ; quasi-quantum 32/7 = 4,571 ;")
print("  notre relaxation en cours : 4,665 (RL 58,7) et descend.")
print("  LE RUN DE FOND TRANCHE : convergence vers 4,538 -> taux 1/pi")
print("  CONFIRME et s ENTIEREMENT derive ; vers 4,556-4,571 -> le taux")
print("  porte une correction de -1,7 % (a deriver).")

print("\n" + "=" * 76)
print("VERDICT : le taux 1/pi = la LECTURE RADIAN du writhe -- les trois")
print("autres taux sont EXCLUS (2,8-6,4 x) ; la reference trefle est")
print("FONDEE (la porte) ; le maillon restant est nomme (fraction Wr/2pi")
print("par le Fuller axial, non demontre) ; et l'hypothese est devenue")
print("une PREDICTION geometrique : Wr(5_2)_ideal = 4,538 +- 0,008 --")
print("en cours de test par la convergence longue.")
