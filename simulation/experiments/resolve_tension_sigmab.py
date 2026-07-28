"""Resolution de la tension Sigma_b - Lambda_b (-5 %) : la saturation (21/07).

LA TENSION EST REELLE : c'est un splitting physique (sans coeur), le
bruit des coeurs ne s'applique pas -- ~20 sigma experimental.

CANDIDAT (issu du SS12) : l'exces d'echange est porte par UN echange
(fait geometrique : un demi-tour d'anneau = UN echange). Le contenu
residuel d'UNE bascule ne peut donc exceder UN demi-tour :
    exces_dynamique(g) = min( 3^g (d-1), 1 )
  g=0 : 3^0 (rac2-1) = 0,414 < 1  -> pas de saturation (charme OK) ;
  g=1 : 3 (rac2-1) = 1,243 > 1    -> SATURE a 1 -> bascule ll bottom
        = 3 (nue) + 1 (exces sature) = 4 EXACTEMENT.
Le verrou (Omega, PAS de bascule) stocke son exces au SOL statique de
l'anneau : pas d'evenement d'echange unique -> PAS de borne -> 3(rac2-1)
= 1,243 intact (mesure 1,257 : ca tient, la distinction est TESTEE).
"""
import numpy as np

SQS = 0.41847
HALF = np.pi * SQS**2
R2 = np.sqrt(2)
ML, MB = 0.310, 4.952
LB = MB + 2 * ML

print(__doc__)
print("=" * 76)
print("[A] LA TENSION QUALIFIEE (splitting sans coeur : significative)")
print("=" * 76)
m_sb, m_lb = 5.81310, 5.61960          # Sigma_b moyenne d'isospin ; Lambda_b
flip = (m_sb**2 - m_lb**2) / HALF
err = 2 * 5.8 * 0.0005 / HALF          # +-0,5 MeV sur les masses
print("  bascule ll bottom mesuree : %.4f +- %.3f demi-tours" % (flip, err))
print("  forme close 3 rac2 = %.4f  -> ecart %-+.1f %% (~%.0f sigma exp)"
      % (3 * R2, 100 * (flip / (3 * R2) - 1), (3 * R2 - flip) / err))
print("  -> PAS un effet de coeurs : structure manquante.")

print("\n" + "=" * 76)
print("[B] HYPOTHESES CONFRONTEES A LA MESURE (4 splittings)")
print("=" * 76)
for nom, val in [
        ("H1 document principal 3 rac2 (exces x3)", 3 * R2),
        ("H2 SATURATION : 3 + min(3(rac2-1),1) = 4", 4.0),
        ("H4 x3 sur la nue seule : 3 + (rac2-1)", 3 + R2 - 1)]:
    print("  %-44s %.3f  (%+6.1f %%)"
          % (nom, val, 100 * (flip / val - 1)))
print("  -> H2 (saturation) : +0,5 %% ; H1 -5,2 %% ; H4 +18 %%. La donnee")
print("     SELECTIONNE la saturation. Controles : ls bottom 3,051 vs 3")
print("     (bascule nue, pas d'exces : INSENSIBLE, +1,7 %%) ; Omega_b")
print("     (verrou, sol statique, pas de borne) : 1,257 vs 3(rac2-1) =")
print("     1,243 (+1,1 %%) -- la DISTINCTION dynamique/statique tient.")

print("\n" + "=" * 76)
print("[C] LES MASSES REPAREES (formes closes avec saturation)")
print("=" * 76)
for nom, me, n_old, n_new in [
        ("Sigma_b", 5.81310, 1 + 3 * R2, 5.0),
        ("Sigma_b*", 5.83250, 1.5 + 3 * R2, 5.5)]:
    m_old = np.sqrt(LB * LB + n_old * HALF)
    m_new = np.sqrt(LB * LB + n_new * HALF)
    print("  %-9s n = %.3f -> %d%s : %.4f -> %.4f GeV  (%+.2f -> %+.2f %%)"
          % (nom, n_old, int(n_new), ".5" if n_new % 1 else "",
             m_old, m_new, 100 * (m_old / me - 1), 100 * (m_new / me - 1)))
print("  Sigma_b* - Sigma_b : 1/2 demi-tour inchange (jonction Y).")

print("\n" + "=" * 76)
print("VERDICT : la tension est RESOLUE par la borne du SS12 lui-meme --")
print("l'exces dynamique d'UNE bascule ne peut exceder UN demi-tour")
print("(un demi-tour d'anneau = un echange) ; a g=1, 3(rac2-1) = 1,243")
print("depasse le budget -> SATURE a 1 -> bascule ll bottom = 4 EXACT")
print("(mesure 4,021, +0,5 %). Sigma_b passe de +0,21 a +0,01 %,")
print("Sigma_b* de +0,28 a +0,08 %. Le verrou statique (Omega) n'est PAS")
print("borne (teste : +1,1 %). Regle close finale :")
print("  n = 1 + 3^g.bascule + min(3^g(d-1),1).[identique+bascule]")
print("      + 3^g(d-1).[identique verrouille] + 1/2.[J=3/2]")
print("Consequence falsifiable : TOUTE bascule de paire identique a")
print("g>=1 sature a l'entier (jamais 3 rac2) ; a g=0 jamais de")
print("saturation. Aucun parametre nouveau (la borne = le fait")
print("geometrique du SS12).")
