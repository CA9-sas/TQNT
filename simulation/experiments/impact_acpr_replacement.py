"""Impact du remplacement de la table de ropelengths par ACPR (22/07).

ANALYSE (rien n'est applique) : que change le passage aux valeurs ACPR
partout ou la table du depot nourrit le document principal ?
Principe : M_topo(K) = m_l * l(K)/l(3_1) -- seuls les usages de
ropelengths NON toriques bougent ; les masses de noeuds calibrees
(m_l, m_c, m_b) et m_s=494,8 (donnees) ne sont PAS des ropelengths.
"""
import numpy as np

ML = 310.0
ACPR = {"3_1": 32.7448, "4_1": 42.0928, "5_2": 49.4704, "6_3": 57.8435,
        "7_2": 63.8956, "8_3": 71.1655}
REPO = {"3_1": 32.744, "4_1": 41.98, "5_2": 52.26, "6_3": 54.42,
        "7_2": 65.92, "8_3": 74.60}


def mt(t, k):
    return ML * t[k] / t["3_1"]


print(__doc__)
print("=" * 76)
print("[1] CE QUI NE BOUGE PAS (le gros du document principal)")
print("=" * 76)
print("""  - TOUTE la spectroscopie : table 225 (1,5 %), echelle de recouplage,
    bandes, molecules, demi-substitution, score catalogue 89@0,13 % --
    les coeurs utilisent les MASSES DE NOEUDS (m_l/m_c/m_b calibrees,
    m_s=494,8 par les donnees), PAS des ropelengths ;
  - le reservoir DM (T(2,q)/T(3,q) TORIQUES -- valeurs justes, et
    valides par notre SONO sur T(2,23) a +0,4 %) : bord 1,7-2,2 GeV,
    sigma_EM, Omega : INCHANGES ;
  - nucleon, EW, leptons, G, socle, cosmologie : INCHANGES ;
  - l'indice m_l = ropelength 3_1 (411/(8 pi)) : 3_1 est EXACT.""")

print("=" * 76)
print("[2] m_s : le STATUT bouge, pas la valeur")
print("=" * 76)
r_new = ACPR["5_2"] / ACPR["3_1"]
print("  ratio vrai %.4f -> m_s^geo = %.1f vs 494,8 des donnees (-5,4 %%)."
      % (r_new, ML * r_new))
print("""  Options : (i) DECLASSEMENT m_s 'derive' -> 'calibre + indice
  geometrique (+5,6 %)' => COMPTAGE 9 -> 10 (impact textuel large :
  abstract, avant-propos, section Comptage, CLAUDE.md, ~8 endroits) ;
  (ii) fonctionnelle de longueur != ropelength nue : AUCUN candidat
  propre trouve (Wr, croisements, det : coefficients negatifs ou
  arbitraires) -- audit ouvert, pas un sauvetage ce soir.
  La VALEUR 494,8 et toute la spectroscopie qui l'utilise : INCHANGEES.""")

print("=" * 76)
print("[3] ETATS STRICTS DM : les raies bougent")
print("=" * 76)
etats = [("6_3+6_3 (RAIE fine)", ["6_3", "6_3"]),
         ("4_1+8_3", ["4_1", "8_3"]),
         ("6_3+8_3", ["6_3", "8_3"]),
         ("8_3+8_3", ["8_3", "8_3"]),
         ("4_1+4_1+8_3", ["4_1", "4_1", "8_3"]),
         ("6_3+6_3+6_3 (RAIE fine)", ["6_3", "6_3", "6_3"]),
         ("4_1+8_3+8_3 (ext. EW)", ["4_1", "8_3", "8_3"]),
         ("6_3+8_3+8_3 (ext. EW)", ["6_3", "8_3", "8_3"])]
for nom, ks in etats:
    old = sum(mt(REPO, k) for k in ks)
    new = sum(mt(ACPR, k) for k in ks)
    print("  %-26s %7.0f -> %7.0f MeV  (%+.1f %%)"
          % (nom, old, new, 100 * (new / old - 1)))
print("""  -> les DEUX RAIES FINES Fermi-LAT (6_3 purs) montent de +6,3 % :
  1,03 -> 1,095 GeV et 1,55 -> 1,64 GeV ; les etats a 8_3 descendent
  de -2 a -4,6 % (mais ils sont ETALES par le writhe : pas des raies) ;
  6_3+8_3 quasi inchange (compensation accidentelle). Les marges F3
  (x8-12 sous les limites 2015) survivent a un deplacement de 6 %
  (limites lentement variables) -- a re-chiffrer si applique.""")

print("=" * 76)
print("[4] beta_x (fluxoide) et divers")
print("=" * 76)
print("""  - encadrement 3_1 (+33 %)/7_2 (-53 %) : le 7_2 etait +3,2 % lache ;
    un 7_2 plus court RESSERRE legerement la borne basse -- l'encadrement
    qualitatif (sans parametre libre) survit largement ;
  - scripts historiques (em_energy_*, mass_pieranski, mtopo_heavy) :
    archives, gardent leurs valeurs (convention declaree du depot).""")

print("=" * 76)
print("BILAN D'IMPACT")
print("=" * 76)
print("""  FAIBLE sur les nombres du document principal : seuls les ETATS
  STRICTS DM bougent significativement (raies 6_3 +6,3 %) -- tout le
  reste est soit torique (juste), soit fonde sur les masses de noeuds
  calibrees. FORT sur un STATUT : m_s perdrait 'derive' (comptage
  9 -> 10) sauf sauvetage par une fonctionnelle de longueur a etablir.
  Le remplacement lui-meme est sur : il ne casse aucun accord existant,
  il deplace deux predictions (raies) et un statut (m_s).""")
