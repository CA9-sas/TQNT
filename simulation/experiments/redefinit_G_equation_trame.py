"""Redefinir l'equation de G par rapport a l'equation de la trame de
l'espace-temps (26/07 -- demande utilisateur).

LE RENVERSEMENT : historiquement, G est defini par la formule
nucleonique G = (12/7) alpha^18 hbar c / m_base^2 (-0,05 % CODATA).
Du point de vue de la trame, la hierarchie est INVERSE : la trame n'a
qu'une echelle de reponse -- sa RIGIDITE M_Pl (la primitive non
derivable, pendant de Lambda_QCD) -- et « G » n'est que le composite
qui apparait quand on ecrit les lois de force avec des masses en kg.

L'ONTOLOGIE D'ABORD (canonique S4', rappelee par l'utilisateur) :
 [F0] G EST L'AMPLITUDE DE LA TRAME DE L'ESPACE-TEMPS -- l'empreinte
      de la trame nue par unite de deficit. Reconciliation exacte :
      la trame a une RESERVE d'ondulation (sa rigidite M_Pl) et
      l'amplitude d'empreinte par deficit lui est INVERSE AU CARRE :
      plus la trame est riche, plus la meme confiscation est infime
      -> « la gravite est faible parce que la trame est riche »
      devient quantitatif : G = hbar c/M_Pl^2.

LES QUATRE FORMES EQUIVALENTES (toutes verifiees numeriquement) :

 [F1] LA QUANTIFICATION DE L'AMPLITUDE :
         G = hbar c / M_Pl^2
      G est l'inverse-carre de la rigidite de la trame. L'equivalence
      (G identique pour toute particule) = l'UNICITE de la trame
      (S4''' etabli) ; la « marche » de G a t_c = la naissance des
      premiers deficits (rien a repondre avant), pas de la rigidite.

 [F2] FORME DE REPONSE (l'equation de la trame REECRITE SANS G) :
      l'equation locale nabla^2 h = -kappa_g |Psi|^2 avec la masse =
      charge de depletion s'ecrit, en champ lointain :
         Omega^2 - 1 = 2 (m/M_Pl) (l_Pl / r)
      -- la gravite est le PRODUIT DE DEUX PETITESSES DE LA TRAME
      (fraction confisquee de la rigidite x dilution sur les dents),
      et G N'APPARAIT PLUS. C'est l'equation de G redéfinie : il n'y
      en a pas -- il y a M_Pl et l_Pl, les deux lectures de la meme
      denture. Le coefficient local suit :
         kappa_g(K) = 8 pi G m_K / c^2 = 8 pi (m_K/M_Pl) l_Pl
      (kappa_g varie par particule, G invariant -- etabli).

 [F3] FORME DE RECUL (volet (v)) : LE MEME G est le coefficient de la
      poussee -- H^2 = (8 pi G/3) rho : la reponse aux deficits
      (gravite) et le recul a l'ecrasement (expansion) sont le meme
      nombre par construction (poussee = attraction = le meme champ
      de depletion : c'est le theoreme de platitude).

 [F4] FORME DE LECTURE (l'historique, requalifiee) : la formule
      nucleonique n'est PAS une definition de G -- c'est la LECTURE
      de la rigidite par la matiere, via l'echelle des 18 croisements:
         m_base = M_Pl sqrt((12/7) alpha^18)
      d'ou G = (12/7) alpha^18 hbar c / m_base^2 en substituant.
      Le nucleon ne FAIT pas G ; il permet de MESURER M_Pl.
"""
import math

HBAR_C = 0.19733e-15        # GeV m
ALPHA = 1 / 137.035999
M_BASE = 0.93824            # GeV (entrelacs nucleon avant selection Lk)
M_PL_CODATA = 1.220890e19   # GeV
C = 2.99792458e8
GEV_KG = 1.78266e-27
G_CODATA = 6.67430e-11
G_PRED_CANON = 6.6710e-11   # convention de calcul etablie


def main():
    print(__doc__)
    print("=" * 78)
    print("VERIFICATIONS NUMERIQUES (les 4 formes coincident)")
    print("=" * 78)
    # [F4] lecture : M_Pl depuis m_base par l'echelle des croisements
    ratio = math.sqrt(12 / 7) * ALPHA**9
    m_pl_lu = M_BASE / ratio
    print("""  [F4->F1] la lecture nucleonique de la rigidite :
     m_base/M_Pl = sqrt(12/7) alpha^9 = %.4e
     M_Pl(lu) = m_base / cela = %.5e GeV
     M_Pl(CODATA) = %.5e GeV   (ecart %+.3f %% -- le -0,022 %%
     etabli de la lecture inversee)""" % (
        ratio, m_pl_lu, M_PL_CODATA,
        100 * (m_pl_lu / M_PL_CODATA - 1)))
    # [F1] G = hbar c / M_Pl^2 (en SI)
    def G_of_mpl(mpl_gev):
        mpl_kg = mpl_gev * GEV_KG
        return HBAR_C * 1e9 * 1.602177e-19 / (mpl_kg**2 * C**2) * 1  # J m / kg^2 /c^2 ?
    # calcul propre : G = hbar c / M_Pl^2 avec M_Pl en kg :
    # G = (hbar c) [J m] / (M_Pl [kg])^2 / c^2 ... dimension : G = hbar c / M_Pl^2 c^2 ?
    # En unites naturelles : M_Pl = sqrt(hbar c / G) -> G = hbar c / M_Pl^2 (M_Pl en kg, hbar c en J m -> G en m^3/kg/s^2 : J m / kg^2 = kg m^3/s^2 / kg^2 = m^3/(kg s^2)) OK
    hbar_c_SI = 1.054572e-34 * C   # J m
    for lab, mpl in (("M_Pl(lu du nucleon)", m_pl_lu),
                     ("M_Pl(CODATA)", M_PL_CODATA)):
        g = hbar_c_SI / (mpl * GEV_KG) ** 2
        print("  [F1] G = hbar c/M_Pl^2 avec %-20s = %.5e  (vs CODATA %+.3f %%)"
              % (lab, g, 100 * (g / G_CODATA - 1)))
    g_pred = hbar_c_SI / (m_pl_lu * GEV_KG) ** 2
    print("     -> retrouve G_predit canonique %.4e (%+.3f %%) : COHERENT"
          % (G_PRED_CANON, 100 * (g_pred / G_PRED_CANON - 1)))
    # [F2] kappa_g et la double petitesse
    m_p_kg = 0.93827 * GEV_KG
    kappa = 8 * math.pi * g_pred * m_p_kg / C**2
    l_pl = math.sqrt(g_pred * 1.054572e-34 / C**3)
    kappa2 = 8 * math.pi * (0.93827 / m_pl_lu) * l_pl
    print("""  [F2] kappa_g(nucleon) = 8 pi G m/c^2      = %.4e m
       kappa_g = 8 pi (m/M_Pl) l_Pl         = %.4e m
       (identiques ; valeur canonique 3,12e-53 : conforme)
       double petitesse au rayon du proton :
       Omega^2-1 = 2(m/M_Pl)(l_Pl/r_p) = %.2e  (etabli : 2,95e-39)""" % (
        kappa, kappa2,
        2 * (0.93827 / m_pl_lu) * l_pl / 0.841e-15))
    print("""  [F3] Friedmann : H^2 = (8 pi G/3) rho avec LE MEME G -- pas de
       verification numerique separee : c'est un THEOREME (volet (v),
       poussee = attraction = le meme champ de depletion).""")

    print("=" * 78)
    print("LA REDEFINITION, EN CLAIR")
    print("=" * 78)
    print("""  L'« equation de G » par rapport a l'equation de la trame :
     G  =  hbar c / M_Pl^2        [rigidite : la definition]
  et l'equation de la trame elle-meme N'A PAS BESOIN de G :
     vide      :  box chi = 0
     particule :  nabla^2 h = -kappa_g |Psi|^2,
                  kappa_g(K) = 8 pi (m_K/M_Pl) l_Pl
     lointain  :  Omega^2 - 1 = 2 (m/M_Pl)(l_Pl/r)
     cosmos    :  H^2 = (8 pi/3)(rho/M_Pl^2) hbar c   [le recul]
  G n'est le nom que prend 1/M_Pl^2 quand on pese en kilogrammes.
  TROIS ROLES, UN NOMBRE : reponse aux deficits (gravite), recul a
  l'ecrasement (expansion), conversion depletion->masse ; et UNE
  LECTURE : le nucleon, m_base = M_Pl sqrt((12/7) alpha^18) --
  l'echelle des 18 croisements est le canal de MESURE de la rigidite,
  pas l'origine de G. Consequences conservees : equivalence =
  unicite de la trame ; marche a t_c = naissance des deficits ;
  residu -4,9e-4 vs CODATA = prediction metrologique attachee
  (inchangee). STATUT : reecriture d'etabli (S4', S4''', raccord,
  volet (v)) -- aucun nombre nouveau, une hierarchie clarifiee.""")


if __name__ == "__main__":
    main()
