"""delta de Koide décomposé — l'électron au nœud de l'onde stationnaire
(27/07/2026, suite du magnétochiral).

L'IDÉE : dans la lecture magnétique, sqrt(m_k) = projection de l'état
sur la direction du condensat, 1 + A cos(theta_k), avec A = sqrt(2)
porté par le doublement de sens. Le ZÉRO de cette amplitude (le nœud
de l'onde stationnaire enroulée) est alors FIXÉ PAR LE MÊME sqrt(2) :
    1 + sqrt(2) cos(theta*) = 0  <=>  theta* = 3 pi / 4   (EXACT).
Or l'électron est QUASI NODAL (sqrt(m_e)/mu ~ 0,04). D'où la
décomposition candidate :
    delta = theta* - 2 pi/3 - eps = pi/12 - eps,
où pi/12 = 3pi/4 - 2pi/3 est DÉRIVÉ (A = sqrt(2) + position Z3 de
l'électron, k = 1) et eps = l'écart nodal de l'électron (petit,
lié à sqrt(m_e)/mu).

CE QUE ÇA CHANGE : le mystère « pourquoi delta = 2/9 ? » (rationnel
sans mécanisme) devient « pourquoi l'électron est-il quasi nodal ? »
— question MIEUX POSÉE : dans la lecture des rails, la génération au
nœud a un recouvrement de condensat minimal => la plus légère,
quasi protégée ; sa masse résiduelle = l'écart au nœud, UN petit
nombre (eps) au lieu d'un rationnel mystérieux.

PROGRAMME :
 [1] inversion exacte : (m_e, m_mu, m_tau) -> (mu, A, delta) — pas
     d'hypothèse ;
 [2] théorème du nœud : theta* = 3pi/4 depuis A = sqrt(2) (exact) ;
     écart nodal de l'électron mesuré ;
 [3] la décomposition delta = pi/12 - eps : identité vérifiée,
     contenu discuté honnêtement (reparamétrisation exacte — le
     GAIN est le mécanisme : pi/12 dérivé, eps petit à expliquer) ;
 [4] eps au jeu fermé des petits nombres canoniques (règle du test
     nul) ;
 [5] confrontation des deux ontologies : « delta = 2/9 exact » vs
     « électron au nœud » — discriminables ou non ?

GARDE-FOU : [3] est une identité algébrique (aucun accord numérique
nouveau n'y est revendiqué) ; seul [2] (nœud à 3pi/4 = le même
sqrt(2)) et la lecture mécanique (nœud = protection) sont du contenu.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.optimize import fsolve

M_E, M_MU, M_TAU = 0.51099895, 105.6583755, 1776.86   # MeV (PDG)
ALPHA = 1 / 137.035999


def koide_params():
    """inversion exacte (mu, A, delta) depuis les 3 masses, k=(1,2,0)
    pour (e, mu, tau) — l'assignation canonique (tau = classe 0)."""
    r = np.sqrt([M_E, M_MU, M_TAU])

    def eqs(p):
        mu, A, d = p
        return [mu * (1 + A * np.cos(2 * np.pi * 1 / 3 + d)) - r[0],
                mu * (1 + A * np.cos(2 * np.pi * 2 / 3 + d)) - r[1],
                mu * (1 + A * np.cos(0.0 + d)) - r[2]]

    mu, A, d = fsolve(eqs, [17.7, 1.41, 0.22], full_output=False)
    return mu, A, d, r


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] INVERSION EXACTE (m_e, m_mu, m_tau) -> (mu, A, delta)")
    print("=" * 78)
    mu, A, d, r = koide_params()
    Q = (r ** 2).sum() / (r.sum()) ** 2
    print("  mu = %.6f MeV^1/2 ; A = %.6f ; delta = %.6f rad" % (mu, A, d))
    print("  A vs sqrt(2)   : %+.4f %%   (l'identification établie)" % (
        100 * (A / np.sqrt(2) - 1)))
    print("  delta vs 2/9   : %+.4f %%   (l'indice historique)" % (
        100 * (d / (2.0 / 9.0) - 1)))
    print("  Q de Koide (somme m / (somme sqrt m)^2) : %.8f  (2/3 = %.8f)"
          % (Q, 2 / 3))
    print("  NB : Q = (1 + A^2/2)/3 EXACTEMENT, quel que soit delta =>")
    print("  Q = 2/3 <=> A = sqrt(2). delta ne porte QUE les rapports")
    print("  individuels — c'est lui, la hiérarchie des générations.")

    print("=" * 78)
    print("[2] LE THÉORÈME DU NŒUD : theta* = 3pi/4 depuis A = sqrt(2)")
    print("=" * 78)
    theta_star = np.arccos(-1 / np.sqrt(2))
    theta_e = 2 * np.pi / 3 + d
    eps = theta_star - theta_e
    print("  1 + sqrt(2) cos(theta*) = 0  =>  theta* = %.6f = 3pi/4 = %.6f"
          % (theta_star, 3 * np.pi / 4))
    print("  angle de l'électron theta_e = 2pi/3 + delta = %.6f" % theta_e)
    print("  écart nodal eps = theta* - theta_e = %.6f rad" % eps)
    print("  contrôle au premier ordre : sqrt(m_e)/mu = %.6f (~ eps)" % (
        r[0] / mu))
    print("""
  => le NŒUD de l'amplitude est fixé par LE MÊME sqrt(2) que le
  doublement de sens (porteur établi) — et l'électron s'en écarte
  de eps = 0,04 rad seulement : l'électron est QUASI NODAL.""")

    print("=" * 78)
    print("[3] LA DÉCOMPOSITION delta = pi/12 - eps")
    print("=" * 78)
    print("  pi/12 = 3pi/4 - 2pi/3 = %.6f rad   (DÉRIVÉ : A=sqrt(2) + Z3)"
          % (np.pi / 12))
    print("  delta reconstruit = pi/12 - eps = %.6f  vs delta exact %.6f"
          "  (identité : %.1e)" % (np.pi / 12 - eps, d,
                                   abs(np.pi / 12 - eps - d)))
    print("""
  HONNÊTETÉ : c'est une IDENTITÉ (reparamétrisation exacte, aucun
  accord nouveau). Le CONTENU est ailleurs : (i) pi/12 est DÉRIVÉ
  (l'angle du nœud vient du même sqrt(2) que les canaux) ; (ii) le
  mystère change de nature — « pourquoi le rationnel 2/9 ? » devient
  « pourquoi l'électron est-il quasi nodal ? », et LÀ il y a un
  mécanisme : la génération au nœud a un recouvrement de condensat
  MINIMAL => la plus légère (la légèreté de l'électron = protection
  nodale) ; eps != 0 = le petit brisement résiduel à expliquer.""")

    print("=" * 78)
    print("[4] eps AU JEU FERMÉ DES PETITS NOMBRES CANONIQUES")
    print("=" * 78)
    cands = [("2 pi alpha", 2 * np.pi * ALPHA),
             ("pi alpha / 2 (gonflement)", np.pi * ALPHA / 2),
             ("alpha", ALPHA),
             ("sqrt(alpha)", np.sqrt(ALPHA)),
             ("3 alpha", 3 * ALPHA),
             ("alpha * 2e (Napier)", ALPHA * 2 * np.e),
             ("1/(8 pi)", 1 / (8 * np.pi))]
    for nom, v in cands:
        print("    %-26s = %.5f   vs eps = %.5f   (%+.1f %%)" % (
            nom, v, eps, 100 * (v / eps - 1)))
    print("""
  VERDICT (règle du test nul) : le plus proche (1/(8pi), %+.1f %% ;
  2 pi alpha, %+.1f %%) est SANS mécanisme => rien n'est retenu.
  eps reste OUVERT — mais c'est désormais LE bon inconnu (l'écart
  nodal de l'électron), pas un rationnel orphelin.""" % (
        100 * (1 / (8 * np.pi) / eps - 1),
        100 * (2 * np.pi * ALPHA / eps - 1)))

    print("=" * 78)
    print("[5] LES DEUX ONTOLOGIES")
    print("=" * 78)
    d_rat = 2.0 / 9.0
    eps_rat = np.pi / 12 - d_rat
    me_rat = (mu * (1 + np.sqrt(2) * np.cos(2 * np.pi / 3 + d_rat))) ** 2
    print("  (a) « delta = 2/9 exact »  : eps induit = %.6f ; m_e prédit"
          " = %.5f MeV (%+.2f %%)" % (eps_rat, me_rat,
                                      100 * (me_rat / M_E - 1)))
    print("  (b) « électron au nœud + brisement eps » : delta composite,")
    print("      pi/12 dérivé, eps mesuré = %.6f" % eps)
    print("""
  Les deux reproduisent les masses (paramétrisations équivalentes —
  indiscernables par les masses seules). MAIS : (a) pose un rationnel
  SANS mécanisme (le test nul l'a toujours laissé indice) ; (b) DÉRIVE
  la grosse part (pi/12, du même sqrt(2) déjà porté) et isole le
  reste dans un petit nombre à mécanisme plausible (protection
  nodale, brisement résiduel). La lecture magnétique FAVORISE (b).
  Discriminant potentiel : une dérivation de eps (candidats
  physiques : couplage T-impair au sens lui-même, auto-énergie EM du
  mode nodal) qui donnerait delta = pi/12 - eps(theorie) SANS
  rationnel — cible nommée.""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""   delta N'EST PLUS UN RATIONNEL ORPHELIN : delta = pi/12 - eps,
   avec pi/12 DÉRIVÉ (l'angle du nœud, fixé par le même sqrt(2) que
   le doublement de sens) et eps = 0,0396 = l'écart nodal de
   l'électron (sa masse résiduelle). La légèreté de l'électron reçoit
   son mécanisme de rails : la génération au nœud de l'onde
   stationnaire a un recouvrement de condensat minimal. Reste UNE
   cible : eps (le brisement résiduel).""")


if __name__ == "__main__":
    main()
