"""L'origine du passage d'une onde dans un rail (brouillon, 23/07).

QUESTION : qu'est-ce qui fait qu'une onde libre de la trame de
l'espace-temps ENTRE dans un rail (se confine, se magnetise) ?

THESE (a etablir) : L'ENTREE N'EST JAMAIS SPONTANEE -- c'est toujours
un FRANCHISSEMENT DE SEUIL EN ACTION, fourni par un evenement. Quatre
volets :
[A] LE THEOREME DU NON-PASSAGE SPONTANE : une onde libre seule
    n'entre jamais (spectatrice, version dynamique) -- sinon la
    lumiere s'eteindrait en vol : les photons du CMB voyagent 13,8
    milliards d'annees sans perte -> borne dure sur tout taux
    d'entree spontanee ; et par principe : l'entree spontanee d'une
    onde uniforme definirait un repere (protection de Lorentz).
[B] LE SEUIL : entrer = se fermer en boucle VERROUILLEE, ce qui exige
    (i) au moins un quantum d'action a deposer au portail (la
    graduation h), (ii) une topologie fermable NEUTRE (Q et quantite
    de mouvement) -> l'entree minimale = une PAIRE noeud + miroir :
    la porte du secteur EM est gamma gamma -> e+ e- (Breit-Wheeler,
    seuil 2 m_e = 1,022 MeV -- OBSERVE, STAR 2021) ; la version
    assistee par champ est Schwinger (exposant 2pi/xi, identite
    canonique deja derivee).
[C] LE CANAL PRIMORDIAL : au gel (T_c), les parois des domaines Z3 du
    condensat S'ANCRENT SUR LES SURFACES NODALES (cout de paroi
    minimal la ou l'ondulation est nulle -- calculable), donc leurs
    jonctions triples sur les LIGNES nodales = LES RAILS : les cordes
    de la condensation naissent DANS les rails -- l'entree primordiale
    est un ancrage, et le percage Z3 (offre 8/9) en est la
    statistique etablie.
[D] SYNTHESE : quatre canaux d'entree, tous canoniques -- gel
    (chaleur latente), champ critique (Schwinger), horizon (Hawking),
    collision (hadronisation) ; un seul mecanisme : l'evenement paie
    le seuil d'action.
"""
import numpy as np

HBAR = 1.0546e-34
C = 2.9979e8
ME = 9.109e-31
QE = 1.602e-19
ANNEE = 3.156e7


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] LE NON-PASSAGE SPONTANE : la borne du vol de la lumiere")
    print("=" * 78)
    t_cmb = 13.8e9 * ANNEE
    print(f"""  Les photons du CMB voyagent t = 13,8 Ga = {t_cmb:.1e} s sans entree
  (spectre de corps noir intact : distorsions |mu| < 9e-5) ->
     Gamma_entree(spontanee) < ~1e-4 / t = {1e-4 / t_cmb:.1e} s^-1
  soit un temps de vie du photon libre > ~1e21 s. ET PAR PRINCIPE :
  une entree spontanee en mouvement uniforme definirait un repere
  absolu (quelle cellule ? a quel instant ?) -- la protection de
  Lorentz l'interdit STRUCTURELLEMENT : le taux exact est ZERO.
  L'entree exige un EVENEMENT (un tiers : champ, paroi, collision,
  horizon). C'est la version dynamique du theoreme de la spectatrice.""")

    print("=" * 78)
    print("[B] LE SEUIL : la porte minimale est une PAIRE")
    print("=" * 78)
    seuil = 2 * ME * C**2 / QE / 1e6
    print(f"""  Entrer = se fermer en boucle verrouillee. Deux exigences :
   (i) deposer au portail au moins un quantum d'action (la
       graduation h : rien de moins ne s'inscrit dans l'engrenage) ;
   (ii) fermer une topologie NEUTRE : une onde seule ne porte ni la
       boucle (il faut revenir sur soi) ni la neutralite (Q = 0
       global) -> l'entree minimale = une PAIRE noeud + miroir
       (trefle + trefle miroir : e+ e-).
  LA PORTE DU SECTEUR EM : gamma gamma -> e+e-, seuil
  sqrt(s) >= 2 m_e = {seuil:.3f} MeV -- le processus de Breit-Wheeler,
  OBSERVE (STAR, 2021) : deux ondes libres fournissent ENSEMBLE
  l'action et la fermeture -- c'est l'entree dans les rails en
  laboratoire. Version assistee par champ : Schwinger -- l'exposant
  2pi/xi = pi m^2 c^3/(e E hbar) est l'identite canonique deja
  derivee (P22) : le champ paie le seuil par effet tunnel.
  COROLLAIRE : le spectre de l'entree est le spectre des paires --
  d'abord les trefles (la combinatoire des rails, S43), puis le
  reste : la porte et la combinatoire racontent la meme histoire.""")

    print("=" * 78)
    print("[C] LE CANAL PRIMORDIAL : l'ancrage des parois Z3 sur les rails")
    print("=" * 78)
    # cout d'une paroi plane perpendiculaire a x, posee en x0, dans le fond
    # stationnaire u = cos(pi x) cos(pi y) cos(pi z) : cout local
    # ~ sigma_paroi (1 + beta u^2) -> integre sur (y, z) :
    x0 = np.linspace(0, 1, 201)
    beta = 1.0
    cout = 1 + beta * np.cos(np.pi * x0)**2 * 0.25   # <cos^2 cos^2> = 1/4
    i_min = int(np.argmin(cout))
    print(f"""  Cout d'une paroi de domaine posee en x0 (fond stationnaire,
  beta = 1) : C(x0) = 1 + (beta/4) cos^2(pi x0)
     minimum a x0 = {x0[i_min]:.2f} (le PLAN NODAL), profondeur
     {100 * (max(cout) - min(cout)) / max(cout):.0f} % -- LA PAROI S'ANCRE SUR LA SURFACE NODALE
  (elle coute le moins la ou l'ondulation qu'elle doit traverser est
  nulle). Les jonctions TRIPLES de trois parois s'ancrent donc sur
  les intersections de surfaces nodales = LES LIGNES NODALES = LES
  RAILS : au gel, les cordes du condensat naissent DANS les rails.
  L'ENTREE PRIMORDIALE EST UN ANCRAGE, pas un accident -- et sa
  statistique est la porte etablie (percage Z3, offre 8/9 exacte,
  denombree). La chaleur latente du gel (~Lambda_chi^4 par domaine)
  paie le seuil d'action en masse.""")

    print("=" * 78)
    print("[D] SYNTHESE : quatre canaux, un mecanisme")
    print("=" * 78)
    print("""  L'ORIGINE DU PASSAGE = UN EVENEMENT QUI PAIE LE SEUIL :
   1. LE GEL (primordial) : ancrage des parois Z3 sur les rails,
      chaleur latente -> la quasi-totalite de la matiere ;
   2. LA COLLISION (aujourd'hui) : l'evenement local fournit l'action
      (hadronisation, creation de paires aux collisionneurs) ;
   3. LE CHAMP CRITIQUE : Schwinger (tunnel, exposant derive) ;
   4. L'HORIZON : Hawking (dents arretees : l'onde ne peut plus
      rester libre -- entree forcee, meme fonctionnelle P22).
  Et JAMAIS spontanement : la lumiere du fond diffus traverse
  l'univers parce qu'aucun evenement ne lui presente le seuil.
  Les trois regimes de P22 (deja une seule fonctionnelle dans le
  document principal) sont exactement les canaux 3-4 + la nucleation
  aux frontieres : P22 EST la theorie de l'entree -- la question
  « pourquoi une onde entre-t-elle ? » recoit la reponse « quand un
  evenement paie 2pi/xi », avec xi le gradient d'ecrasement local.""")

    print("=" * 78)
    print("BILAN (brouillon)")
    print("=" * 78)
    print("""  L'origine du passage d'une onde dans un rail est DETERMINEE dans
  le cadre : (a) jamais spontane (borne CMB + protection de Lorentz :
  taux exactement nul) ; (b) toujours par EVENEMENT franchissant le
  seuil d'action, avec entree minimale = une paire noeud+miroir (la
  porte EM = Breit-Wheeler 1,022 MeV, observee ; assistee par champ =
  Schwinger, identite derivee) ; (c) le canal primordial est un
  ANCRAGE : les parois Z3 se posent sur les surfaces nodales (cout
  minimal, calcule), leurs jonctions sur les rails -- les cordes
  naissent dans les rails, et le percage 8/9 en est la statistique ;
  (d) P22 unifie les canaux non primordiaux. Statut : etude --
  identifications candidates (ancrage = entree primordiale ; porte =
  paire), zero constante nouvelle.""")


if __name__ == "__main__":
    main()
