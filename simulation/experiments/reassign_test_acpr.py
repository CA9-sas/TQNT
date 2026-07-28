"""Test de reattribution sous ACPR : les ecarts >2 % sont-ils corrigeables ? (22/07)

QUESTION : apres remplacement par les valeurs ACPR, changer les
attributions de noeuds/entrelacs corrigerait-il les ecarts >2 % ?
Trois cas a examiner :
 [A] m_s (l'ecart -5,4 %) : existe-t-il un noeud (ou entrelacs) dont
     l(K)/l(3_1) ~ 1,596 (cible 52,27 rayons) ?
 [B] etats stricts DM : y a-t-il une liberte d'attribution ?
 [C] les autres ecarts >2 % du spectre : dependent-ils de ropelengths ?
"""
import numpy as np

L31 = 32.7448
TARGET = L31 * (494.8 / 310.0)          # 52,27
KNOTS = {"3_1": 32.7448, "4_1": 42.0928, "5_1": 47.2016, "5_2": 49.4704,
         "6_1": 56.7150, "6_2": 57.0271, "6_3": 57.8435,
         "7_1": 61.4109, "7_2": 63.8956, "7_3": 63.9327, "7_4": 64.2724,
         "7_5": 65.2609, "7_6": 65.7012, "7_7": 65.6108,
         "8_1": 71.0241, "8_2": 71.4107, "8_3": 71.1655, "9_1": 75.7252}
LINKS = {"2^2_1": 25.1388, "4^2_1": 40.0169, "5^2_1": 49.7723,
         "6^2_1": 54.3893, "6^2_2": 56.7028, "6^2_3": 58.1044,
         "6^3_1": 57.8170, "6^3_2": 58.0145, "6^3_3": 50.5745,
         "7^2_1": 64.2353}

print(__doc__)
print("=" * 74)
print("[A] m_s : la fenetre autour de la cible %.2f rayons" % TARGET)
print("=" * 74)
for pct in (2, 5):
    lo, hi = TARGET * (1 - pct / 100), TARGET * (1 + pct / 100)
    kin = [(k, v, 100 * (v / TARGET - 1)) for k, v in KNOTS.items() if lo <= v <= hi]
    lin = [(k, v, 100 * (v / TARGET - 1)) for k, v in LINKS.items() if lo <= v <= hi]
    print("  fenetre +-%d %% [%.1f, %.1f] : NOEUDS = %s ; ENTRELACS = %s"
          % (pct, lo, hi,
             ", ".join("%s (%+.1f%%)" % (k, d) for k, v, d in kin) or "AUCUN",
             ", ".join("%s (%+.1f%%)" % (k, d) for k, v, d in lin) or "aucun"))
print("""  -> la fenetre +-2 %% est VIDE (noeuds ET entrelacs) : le spectre des
  longueurs SAUTE de 5_2 (49,47, -5,3 %) a 6_1 (56,72, +8,5 %). A +-5 %,
  seuls des ENTRELACS multi-brins entrent (5^2_1, 6^3_3, 6^2_1) -- tous
  EXCLUS par le role : le compagnon etrange est UN brin (un noeud).
  AUCUNE reattribution ne donne 1,596. COROLLAIRE (positif) : la fenetre
  vide RENFORCE l'epinglage 5_2 (le plus proche, unique a -5,3 %) --
  la geometrie dit '5_2 avec +5,6 % d'habillage', pas 'un autre noeud'.""")

print("=" * 74)
print("[B] ETATS STRICTS DM : pas de liberte d'attribution")
print("=" * 74)
print("""  Les etats stricts = combinaisons des AMPHICHIRAUX purs -- et
  l'amphichiralite EPINGLE l'ensemble {4_1, 6_3, 8_3} (les seuls a
  <=8 croisements ; suivants a 10). Il n'y a RIEN a reattribuer : les
  masses suivent les longueurs corrigees (raies 6_3 -> 1,095/1,643 GeV).
  Et ce ne sont PAS des ecarts : etats non observes (predictions).""")

print("=" * 74)
print("[C] LES AUTRES ECARTS >2 %% DU SPECTRE")
print("=" * 74)
print("""  eta +3,5 %% (GMO), rho/omega -2,6/-3,5 %% (intercept Regge a0),
  Upsilon +3,6 %% (Cornell/compression), Xi +1,4-1,9 %% (loi hyperons),
  D_s* -0,8 %% (coeur demi-subst.) : AUCUN ne depend d'une ropelength --
  les reattributions ACPR n'y changent rien. Les 'configurations' du
  tableau (3_1+7_6 pour D, etc.) portent spin et charge, PAS la masse
  (lois de secteur) : les corriger ne deplace aucune masse.""")

print("=" * 74)
print("VERDICT : NON -- la reattribution ne corrige pas les ecarts.")
print("  m_s : fenetre ACPR VIDE a +-2 %% (et les entrelacs a +-5 %% sont")
print("  exclus par le role) -- mais l'epinglage 5_2 en sort RENFORCE ;")
print("  stricts DM : epingles par l'amphichiralite (pas d'ecart, pas de")
print("  liberte) ; autres >2 %% : non-ropelength. La question m_s reste")
print("  donc : declassement honnete (9->10) ou fonctionnelle de longueur")
print("  habillee (+5,6 %% a etablir).")
