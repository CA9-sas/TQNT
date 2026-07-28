"""Peut-on deduire S (l'equation de masse) depuis les rails ? (23/07)

QUESTION : m = sqrt(sigma_chi) x S -- le facteur S est-il deductible de
la combinatoire des rails ?

LE PERIMETRE (net) : S a six secteurs (BASE 0 ; LEPTONS Koide ;
BARYONS Regge+Z3+tenseur ; GOLDSTONE GMOR ; LOURDS constituant ;
EW VEV). Les lois de secteur sont de la dynamique d'ETAT (modes Z3,
torsion de jonction, GMOR, VEV) : les rails sont du cote FORMATION --
ils ne peuvent pas les produire. MAIS ils touchent S en TROIS points :
[1] LA BASE : S = 0 pour les ondes libres = pas de fermeture, pas de
    verrou, pas de masse -- le zero de S est un theoreme de rails.
[2] LA BRIQUE CONSTITUANT DU PLATEAU LEGER : M_topo = sqrt(sigma) l(K)
    (ropelength). La longueur MINIMALE de rails N_min(K) (polygones
    minimaux sur reseau cubique, valeurs publiees : Diao 1993 ;
    Scharein et al. 2009) est le PRECURSEUR combinatoire de l(K) : au
    verrouillage, le tube se resserre du chemin de rails vers le
    plongement ideal -- si N_min/l est constant, LES RATIOS DE MASSE
    DU PLATEAU SONT DES RATIOS D'ENTIERS DE RAILS.
[3] L'ECHELLE m_l : les rails sont invariants d'echelle (aucune
    longueur propre) -> ils ne peuvent PAS fixer le barreau zero --
    exactement le statut etabli (m_l = constante de transmutation).
"""
import sys, os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from theory.ropelength_acpr import ROPELENGTH_ACPR  # noqa: E402

# minima publies sur reseau cubique (pas de rail)
N_MIN = {"3_1": 24, "4_1": 30, "5_1": 34, "5_2": 36,
         "6_1": 40, "6_2": 40, "6_3": 40}


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] N_min(rails) vs ropelength ACPR : la constante de resserrage")
    print("=" * 78)
    ratios = []
    for k, nm in N_MIN.items():
        l = ROPELENGTH_ACPR[k]
        ratios.append(nm / l)
        print(f"  {k} : N_min = {nm:2d} pas, l_ACPR = {l:7.3f} -> "
              f"N_min/l = {nm / l:.4f}")
    r = np.array(ratios)
    print(f"""  -> N_min/l = {r.mean():.4f} +- {r.std():.4f} ({100 * r.std() / r.mean():.1f} % de
  dispersion sur 7 noeuds) : LA CONSTANTE DE RESSERRAGE EST QUASI
  UNIVERSELLE -- le chemin de rails minimal et le tube ideal portent
  la MEME information de longueur (le verrou resserre d'un facteur
  ~{1 / r.mean():.2f} sans changer les rapports).""")

    print("=" * 78)
    print("[B] LE TEST DE MASSE : les ratios du plateau leger en ENTIERS")
    print("=" * 78)
    l31, l52 = ROPELENGTH_ACPR["3_1"], ROPELENGTH_ACPR["5_2"]
    print(f"""  m_s/m_l (donnees : 494,8/310)          = {494.8 / 310:.4f}
  ratio geometrique ACPR l(5_2)/l(3_1)   = {l52 / l31:.4f}
  RATIO DE RAILS N_min(5_2)/N_min(3_1)   = 36/24 = {36 / 24:.4f}  (ENTIERS)
  -> le ratio de rails (3/2 EXACT) est a {100 * (1.5 / (l52 / l31) - 1):+.1f} % du geometrique
  ACPR et a {100 * (1.5 / (494.8 / 310) - 1):+.1f} % des donnees : LES RAILS DONNENT LE RATIO
  GEOMETRIQUE, PAS L'HABILLE -- ils confirment le declassement de m_s
  (l'habillage +5,6 % n'est pas geometrique : il n'est pas non plus
  combinatoire). NOUVEL INDICE : le ratio geometrique pourrait etre
  EXACTEMENT 3/2 au niveau combinatoire (l'ACPR 1,5107 = 3/2 + 0,7 %,
  bornes ACPR laches sur 5_2 declarees).
  Autres ratios de rails (predictions du plateau, si secteur etendu) :
     m(4_1)/m_l = 30/24 = 5/4 -> {310 * 30 / 24:.1f} MeV (ACPR : {310 * ROPELENGTH_ACPR['4_1'] / l31:.1f})
     m(5_1)/m_l = 34/24 = 17/12 -> {310 * 34 / 24:.1f} MeV (ACPR : {310 * ROPELENGTH_ACPR['5_1'] / l31:.1f})
     m(6_x)/m_l = 40/24 = 5/3 -> {310 * 40 / 24:.1f} MeV (ACPR : {310 * ROPELENGTH_ACPR['6_1'] / l31:.1f}-{310 * ROPELENGTH_ACPR['6_3'] / l31:.1f})""")

    print("=" * 78)
    print("[C] CE QUE LES RAILS NE PEUVENT PAS DONNER (perimetre declare)")
    print("=" * 78)
    print("""  - L'ECHELLE m_l (le barreau zero) : la combinatoire est invariante
    d'echelle -- c'est une NOUVELLE preuve du statut etabli (m_l =
    constante de transmutation, hors de portee de toute couche sans
    echelle propre) ;
  - LES LOIS DE SECTEUR (Koide, Regge+Z3, GMOR, VEV, recouplage) :
    dynamique d'ETAT du systeme verrouille (modes propres, torsion,
    condensat) -- les rails s'arretent au verrou ;
  - LES NOEUDS LOURDS c/b : plafond geometrique refute (etabli) --
    la loi l(K) ne s'y applique deja plus ;
  - la BASE S = 0 est, elle, un theoreme de rails (pas de fermeture,
    pas de masse) -- trivial mais exact.""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print(f"""  S ENTIER : NON -- les lois de secteur sont de l'etat, pas de la
  formation. MAIS trois acquis :
  (1) la brique CONSTITUANT du plateau leger est combinatoire : les
      ratios de masse sont des RATIOS D'ENTIERS DE RAILS
      (constante de resserrage N_min/l = {r.mean():.3f} +- {100 * r.std() / r.mean():.0f} %%),
      et m_s/m_l = 3/2 EXACT en rails (ACPR +0,7 %%) -- l'habillage
      +5,6 %% des donnees n'est ni geometrique ni combinatoire
      (le declassement de m_s est CONFORTE) ;
  (2) S(base) = 0 est un theoreme de rails ;
  (3) l'echelle m_l est HORS DE PORTEE par invariance d'echelle --
      la transmutation recoit une preuve de plus.
  Statut : etude (brouillon) ; si le 3/2 exact se confirme (resserrer
  la borne ACPR du 5_2), le ratio m_s/m_l geometrique serait un
  RATIONNEL DE RAILS -- la brique constituant de S deviendrait
  arithmetique.""")


if __name__ == "__main__":
    main()
