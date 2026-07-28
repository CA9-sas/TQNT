"""Structure universelle du vide au barreau 8 : le croisement ouvert sur l'ancre.

Suite de derive_designation_porteur.py (2026-07-21). Contrainte heritee :
la barriere doit etre une propriete du vide chi (pas de comptage de hote),
couvrant d'un meme objet le dLk baryonique ET le mode-flip du muon.

ASSEMBLAGE -- E_relax = le cout d'UN CROISEMENT TENU OUVERT dans le vide :

  P1 (theoreme fusion/scission, 20/07) : l'intermediaire de tout dLk = +-1
      est la configuration fusionnee, avec EXACTEMENT UN croisement defait
      (une reconnexion faite, une a faire). Le nombre de pas n'est PAS un
      choix : il est fixe par le theoreme -> UN barreau.
  P2 (echelle par croisement, raccord 19/07, PREEXISTANT) : chaque
      croisement vaut le MEME facteur alpha d'amplification -- le facteur
      par croisement est UNIVERSEL (c'est la construction de l'echelle).
      D'ou l'universalite : ouvrir le croisement du muon ou celui du
      neutron coute le meme barreau, car alpha est le meme partout.
  P3 (ancre unique, raccord 19/07, PREEXISTANT) : le vide n'a qu'UNE ancre
      d'echelle -- m_base (la chaine G est ancree sur le nucleon de base
      pour TOUTE matiere ; << PAS un G par particule >>, exclu x14 ordres).
      Le prix du croisement ouvert se compte depuis l'ancre, pas depuis le
      hote -- par le MEME argument d'unicite que pour G.
  P4 (mesure de sommet, 20/07, conditionnel c1-c2) : l'intermediaire
      fusionne est UN SEUL brin -- la partition coeur/compagnons sqrt(12/7)
      de l'ancre n'a plus de sens pour lui (theoreme : une composante).
      Son amplitude est la norme du paquet minimal : sqrt(2pi).

  =>  E_relax = [sqrt(2pi) / sqrt(12/7)] x m_base / alpha
             = sqrt(7 pi / 6) x (k+2) x m_base
             = sqrt(2pi) alpha^8 M_Pl.

  Universalite : m_base (P3) et alpha (P2) sont independants du hote ;
  le nombre de pas (P1) est un theoreme. Le muon et le neutron paient la
  meme barriere : c'est le MEME croisement-ouvert du vide.

CE QUI RESTE LECTURE (declare) :
  (L1) le prix se compte depuis l'ancre (P3 est un PRECEDENT -- l'argument
       d'unicite de la chaine G -- pas une demonstration pour E_relax) ;
  (L2) les conditions c1-c2 de la mesure de sommet ;
  (L3) l'assemblage est POST-HOC (la grappe fut trouvee par scan le 19/07) :
       la promotion en << derive >> exige une prediction independante (m3).
"""
import math

M_PL_GEV = 1.220890e19
M_BASE_GEV = 0.938240
M_P_GEV = 0.93827209
V_MEAS_GEV = 246.21965
ALPHA_MEAS = 1.0 / 137.035999
ALPHA_EXACT = 1.0 / 137.0
K_PLUS_2 = 137
AMP = math.sqrt(2 * math.pi)
PART = math.sqrt(12.0 / 7.0)


def run():
    print(__doc__)
    print("=" * 72)
    print("NOMBRES")
    print("=" * 72)
    ratio = AMP / PART
    v1 = ratio * M_BASE_GEV * K_PLUS_2
    print("  E_relax = sqrt(7pi/6) x 137 x m_base")
    print("          = %.5f x 137 x %.5f GeV = %.3f GeV  (%+.3f %%)"
          % (ratio, M_BASE_GEV, v1, 100 * (v1 / V_MEAS_GEV - 1)))
    v1p = ratio * M_P_GEV * K_PLUS_2
    print("  (variante m_p : %.3f GeV, %+.3f %% -- l'ancre m_base est la"
          % (v1p, 100 * (v1p / V_MEAS_GEV - 1)))
    print("   convention canonique 19/07)")
    v2 = AMP * ALPHA_MEAS**8 * M_PL_GEV
    print("  Forme M_Pl (alpha mesure)  : %.3f GeV (%+.3f %%)"
          % (v2, 100 * (v2 / V_MEAS_GEV - 1)))
    v3 = AMP * ALPHA_EXACT**8 * M_PL_GEV
    print("  Forme M_Pl (alpha = 1/137) : %.3f GeV (%+.3f %%)"
          % (v3, 100 * (v3 / V_MEAS_GEV - 1)))
    print("  NB : les deux formes sont liees par l'identite de hierarchie")
    print("  m_base/M_Pl = sqrt(12/7) alpha^9 -- une seule relation physique.")

    print("\n  Controle des pas (P1 fixe le nombre -- verification) :")
    for steps, label in [(0, "0 pas (ancre elle-meme)"),
                         (1, "1 pas (theoreme)"),
                         (2, "2 pas")]:
        e = ratio * M_BASE_GEV * K_PLUS_2**steps
        print("    %-24s -> %10.4g GeV %s"
              % (label, e, "<== v" if steps == 1 else ""))

    print("\n  Residu -0,055 %% : au-dela de la precision de m_base (5e-5) et")
    print("  de G_F (1e-5) -- OUVERT ; taille des habillages radiatifs")
    print("  (parallele : g_topo = 0,99890, ecart 0,11 %% = corr. QED).")

    print("\n" + "=" * 72)
    print("SIGNIFICATION ET STATUT")
    print("=" * 72)
    window = 2 * 0.005
    p_one = window / math.log(137.0)
    n_amp = 2      # amplitudes encore en concurrence : sqrt(2pi) (forcee par
    #                P4 si c1-c2 tiennent) vs une alternative de paquet
    n_base = 1     # baseline : l'ancre est UNIQUE (P3)
    p = p_one * n_amp * n_base
    print("  Liberte residuelle : pas = theoreme (P1) ; baseline = ancre")
    print("  unique (P3) ; amplitude = quasi forcee (P4, conditions c1-c2).")
    print("  p(hasard) ~ %.1f %% -- MAIS (L3) : assemblage post-hoc, le" % (100 * p))
    print("  null certifie exige une prediction independante avant promotion.")
    print("\n  STATUT PROPOSE : CANDIDAT FORT non consolide (rang m_l-Koide).")
    print("  v RESTE au comptage (10+2). Si (L1)-(L3) tombent : 10 -> 9.")
    print("  Voie (m3) designee : le croisement-ouvert du vide devrait")
    print("  reapparaitre partout ou UNE reconnexion isolee est virtuelle --")
    print("  p.ex. les oscillations K0/B0 (double reconnexion V5p27-p33) :")
    print("  verifier si leurs echelles portent le meme objet.")


if __name__ == "__main__":
    run()
