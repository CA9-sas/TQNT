"""Trancher la famille des metastables : qui porte le canal 7Be ?
(24/07 -- suite du maillon (b)).

TROIS FAMILLES de paires enlacees au gel (cascade mesuree : 3,45 %
des noeuds finals naissent enlaces |Lk| >= 1) :
 [LL] legere-legere (trefle x trefle, dominantes a la formation --
      rails : trefle 59 %) ;
 [DD] sombre-sombre (bord c = 23-25) ;
 [MX] MIXTE legere x sombre : un trefle enlace a un noeud du bord.
POUR CHACUNE : E_gamma du pas de desexcitation, duree de vie Sargent,
sort cosmologique. LE POINT NOUVEAU : la PROTECTION PAR ENLACEMENT --
le trefle lie (electron-like) ne peut PAS s'annihiler proprement avec
les positrons du plasma : la boucle fusionnee e+e- garderait le Lk a
travers le noeud sombre (l'enlacement est entre BOUCLES ; la fusion
n'en libere pas) -> se dissoudre en ondes libres exige l'effacement
du Lk = P15 = LA barriere. L'annihilation est BOTTLENECKEE par le
meme processus que la desexcitation : survie et emission sont le MEME
evenement P15. Le membre leger survit a l'ere des positrons, et sa
liberation EST le photon du mecanisme.
"""
import math

BETA = 0.210            # MeV (quantum d'enlacement)
TAU_MU = 2.197e-6       # s
M_MU = 105.66           # MeV
CJ_L = 3.3              # trefle
CJ_D = (17.5, 26.0, 34.0)   # gamme du bord sombre (cJ ~ 1,1-1,2 c)
S_BE7, S_D = 1.587, 2.224   # MeV
T_EFF = 2.3e4           # s (survie des photons, E_c > seuil 7Be)
F_PAIR = 0.0345         # mesure (cascade enlacement)
N_D_NB = 2.5
NX_REQ = 1e-7


def tau_sargent(Q_MeV):
    return TAU_MU * (M_MU / Q_MeV) ** 5


def e_c_of_t(t_s):
    """Seuil de survie des photons E_c = m_e^2/(22 T), T(t) radiation."""
    T = (2.4 / math.sqrt(3.36) / t_s) ** 0.5   # MeV
    return 0.511**2 / (22 * T)


def main():
    print(__doc__)
    print("=" * 78)
    print("[LL] TREFLE x TREFLE : le canal est MORT (secteur visible)")
    print("=" * 78)
    print("""  Les paires legeres dominent la formation, mais un trefle est un
  objet du secteur VISIBLE (c = 3 << 23 : couple au plasma). Une
  paire LL entiere est visible : elle thermalise/s'annihile avec le
  plasma pendant l'ere chaude comme le reste du cote visible (la
  miette CP ne garde que les baryons) -- aucun vestige a t ~ 1e4 s.
  (La protection par enlacement ne s'applique pas : les DEUX boucles
  peuvent fusionner avec leurs miroirs et effacer le Lk ensemble.)""")

    print("=" * 78)
    print("[DD] SOMBRE x SOMBRE : desexcitation PRECOCE ET INOFFENSIVE")
    print("=" * 78)
    for cj in CJ_D:
        e1 = BETA * cj
        tau = tau_sargent(e1)
        ec = e_c_of_t(tau)
        print("  cJ = %.1f : pas k=1 -> 0 : E = %.2f MeV ; tau = %.0f s ;"
              " E_c(tau) = %.3f MeV" % (cj, e1, tau, ec))
    print("""  -> E ~ 3,7-7 MeV (> seuil D !) MAIS tau ~ 1-30 s : les photons
  sont emis quand E_c ~ 0,01-0,05 MeV << tous les seuils -- ils se
  degradent thermiquement AVANT de voir un noyau. Les paires DD se
  vident tot et proprement : NI solution, NI danger pour le D.""")

    print("=" * 78)
    print("[MX] MIXTE trefle x bord sombre : LE CANAL DE LA FENETRE")
    print("=" * 78)
    print("  E(k=1) = beta sqrt(cJ_l cJ_d) ; fenetre 7Be = [%.3f, %.3f] :"
          % (S_BE7, S_D))
    for cjd in CJ_D:
        e1 = BETA * math.sqrt(CJ_L * cjd)
        tau = tau_sargent(e1)
        inwin = S_BE7 <= e1 <= S_D
        print("  cJ_d = %.1f : E = %.2f MeV %s ; tau_nu = %.0f s" % (
            cjd, e1, "DANS la fenetre" if inwin else "hors fenetre", tau))
    cj_lo = S_BE7**2 / (BETA**2 * CJ_L)
    cj_hi = S_D**2 / (BETA**2 * CJ_L)
    print("""  -> la fenetre en cJ_d est [%.1f, %.1f] : elle couvre TOUT le
  bord du reservoir (c = 23-25, cJ ~ 20-30) avec marge. La paire la
  plus NATURELLE du gel (le noeud dominant de la formation, le
  trefle, enlace au noeud dominant du reservoir) émet PILE dans la
  fenetre chirurgicale. Ce n'est pas un choix : c'est la geometrie
  de sqrt(3,3 x ~26) = ~9,3.""" % (cj_lo, cj_hi))

    print("=" * 78)
    print("PROTECTION PAR ENLACEMENT (l'argument qui sauve le membre leger)")
    print("=" * 78)
    print("""  Objection : le trefle lie est electron-like ; les positrons du
  plasma (abondants jusqu'a T ~ 20 keV, t ~ 1e4 s) devraient
  l'annihiler bien avant t_eff. REPONSE TOPOLOGIQUE : l'enlacement
  est entre BOUCLES. La fusion e+ e- forme UNE boucle qui porte
  ENCORE Lk = 1 a travers le noeud sombre (le e+ n'etait pas
  enlace ; la somme l'est). Une boucle enlacee ne peut PAS se
  dissoudre en ondes libres : il faut d'abord EFFACER le Lk -- un
  changement de croisement -- c'est P15, LA barriere de Fermi. Donc :
   - l'annihilation EM rapide est BLOQUEE (elle ne menerait qu'a un
     anneau lie, pas au vide) ;
   - la liberation passe par l'evenement P15 Lk : 1 -> 0, dont le
     Q est l'energie de liaison beta sqrt(cJ cJ') ~ 1,9 MeV ;
   - SURVIE ET EMISSION SONT LE MEME EVENEMENT : le canal est
     auto-protege. (Statut : argument topologique, a demontrer
     proprement -- variante du theoreme de superselection Lk.)""")

    print("=" * 78)
    print("LA CHAINE FERMEE (famille MX seule)")
    print("=" * 78)
    e_mx = BETA * math.sqrt(CJ_L * 26.0)
    tau_nu = tau_sargent(e_mx)
    n_x0 = N_D_NB * F_PAIR
    surv_req = NX_REQ / n_x0
    tau_req = T_EFF / math.log(1 / surv_req)
    ratio = tau_req / tau_nu
    print("""  E_gamma = %.2f MeV (cJ_d = 26) ; tau_nu(Sargent, amplitude muon)
  = %.0f s. Abondance MX : par noeud SOMBRE, P(porter un trefle
  enlace) = f_mesure = %.3f (le partenaire d'un noeud du bord est
  presque surement leger) -> n_MX/n_b = 2,5 x %.4f = %.2e.
  Survie requise pour n_eff = 1e-7 : %.1e -> tau* = %.0f s.
  tau*/tau_nu = %.2f : l'element de matrice P15 d'entrelacs doit
  REDUIRE le taux d'un facteur %.2f (amplitude x%.2f) par rapport au
  muon -- ordre naturel (recouvrement d'un entrelacs lourd < point),
  valeur fine NON derivee : c'est le maillon unique restant, et il
  est FALSIFIABLE (si l'element de matrice derive differait, le
  mecanisme meurt -- rien d'autre a regler).""" % (
        e_mx, tau_nu, F_PAIR, F_PAIR, n_x0, surv_req, tau_req,
        ratio, 1 / ratio, (1 / ratio) ** 0.2))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  LA FAMILLE EST TRANCHEE : le canal 7Be = les paires MIXTES
  trefle x bord sombre --
   (1) les LL meurent avec le cote visible (declare) ;
   (2) les DD se desexcitent a t ~ 1-30 s, photons degrades sous
       tous les seuils : inoffensives (le danger « 5 MeV > seuil D »
       du brouillon est DISSOUS par la chronologie elle-meme) ;
   (3) les MX emettent E = beta sqrt(cJ_l cJ_d) ~ 1,9 MeV, PILE dans
       la fenetre pour tout le bord c = 23-25 -- la paire la plus
       naturelle du gel fait le photon le plus utile ;
   (4) le membre leger est PROTEGE par son enlacement (l'annihilation
       est bottleneckee par P15 : survie = emission) ;
   (5) abondance mesuree 8,6e-2/baryon, survie exponentielle ->
       tau* = 1 683 s vs tau_nu = 1 067 s : facteur 1,6 sur le taux
       (x1,1 en amplitude) = LE maillon restant.
  Le mecanisme tient sur UNE inconnue O(1) falsifiable. Prochaine
  etape possible : demontrer la protection par enlacement (theoreme
  de superselection Lk, outillage du chapitre intrication) et/ou
  deriver l'element de matrice P15 d'entrelacs.""")


if __name__ == "__main__":
    main()
