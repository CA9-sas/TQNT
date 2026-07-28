"""epsilon nodal — le circuit minimal à 24 pas et le squelette discret
de Koide (27/07/2026, suite de la décomposition delta = pi/12 - eps).

OBSERVATION STRUCTURELLE : le circuit de rails minimal du trèfle
compte N_min = 24 pas (Diao, canonique §48). Sur ce circuit :
  - le pas angulaire vaut 2 pi / 24 = pi/12 — EXACTEMENT le pi/12
    de la décomposition de delta ;
  - les positions Z3 tombent aux pas {0, 8, 16} (24/3 = 8, entier) ;
  - le nœud theta* = 3 pi/4 tombe au pas 9 (24 x 3/8 = 9, ENTIER).
Toute la structure angulaire de Koide est COMMENSURABLE avec le
circuit minimal : le nœud est un SOMMET de rail, un pas exactement
après le sommet Z3 de l'électron. delta = 1 pas - eps.

DEUX CONTENUS À TESTER :
 [1] COHÉRENCE de A = sqrt(2) avec la discrétisation : un nœud
     verrouillé sur un sommet n'autorise que les amplitudes
     A = 1/|cos(2 pi n/24)| — ensemble discret ; sqrt(2) (pas 9) en
     fait partie (avec 2 au pas 8, 2/sqrt(3) au pas 10...). Le
     sqrt(2) DÉRIVÉ (doublement de sens) tombe PILE sur un sommet :
     vérification de cohérence non triviale (pas une sélection).
 [2] LE SQUELETTE AU VERROU : si le nœud se verrouille au sommet
     (delta = pi/12 exact, ordre du réseau), alors m_e = 0 et les
     masses mu/tau sont PRÉDITES sans eps. Mesurer ce que porte le
     squelette discret seul, et ce que eps habille.
 [3] eps <-> m_e : l'équivalence exacte ; jeu fermé sur m_e/m_l
     (règle du test nul).

GARDE-FOUS : l'égalité pi/12 = 2 pi/24 repose sur N_min = 24 (établi)
et sur 3/4 - 2/3 = 1/12 (arithmétique) — c'est un ÉCHO structurel ;
le lien mécanique (le nœud de l'onde DOIT se verrouiller sur un
sommet de rail) est une hypothèse nommée, pas un théorème.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

M_E, M_MU, M_TAU = 0.51099895, 105.6583755, 1776.86   # MeV (PDG)
ALPHA = 1 / 137.035999
N_MIN = 24                       # pas de rails du trèfle minimal (Diao)
MU = 17.715562                   # MeV^1/2 (inversion exacte, script delta)
M_L = 313.84                     # MeV — mu^2 = (3/4) sqrt(sigma_chi)


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] LA COMMENSURABILITÉ (le nœud est un sommet de rail)")
    print("=" * 78)
    pas = 2 * np.pi / N_MIN
    print("  pas du circuit minimal : 2pi/24 = %.6f = pi/12 = %.6f (EXACT)"
          % (pas, np.pi / 12))
    print("  positions Z3 : 2pi k/3 = pas x {0, 8, 16} (entiers exacts)")
    theta_star = 3 * np.pi / 4
    print("  nœud theta* = 3pi/4 = pas x %.6f (ENTIER : 9)" % (
        theta_star / pas))
    print("  amplitudes autorisées par un nœud-sommet (1 + A cos = 0) :")
    for n in [7, 8, 9, 10, 11]:
        c = np.cos(n * pas)
        if c < 0:
            print("    pas %2d : A = %.5f%s" % (
                n, -1 / c, "   <-- sqrt(2) (le porteur dérivé)"
                if abs(-1 / c - np.sqrt(2)) < 1e-9 else ""))
    print("""
  => le sqrt(2) DÉRIVÉ (doublement de sens) place le nœud PILE sur un
  sommet du circuit minimal (pas 9), un pas exactement après le
  sommet Z3 de l'électron (pas 8). Cohérence non triviale : parmi les
  amplitudes possibles, seule une poignée correspond à un sommet —
  la dérivée en fait partie. delta = 1 pas - eps.""")

    print("=" * 78)
    print("[2] LE SQUELETTE AU VERROU (delta = pi/12 exact, eps = 0)")
    print("=" * 78)
    d0 = np.pi / 12
    A = np.sqrt(2)
    ks = {"tau": 0, "e": 1, "mu": 2}
    mes = {"e": M_E, "mu": M_MU, "tau": M_TAU}
    print("  %-4s %14s %14s %10s" % ("", "m (verrou)", "m (PDG)", "écart"))
    for nom, k in ks.items():
        amp = 1 + A * np.cos(2 * np.pi * k / 3 + d0)
        m = (MU * amp) ** 2
        if abs(amp) < 1e-12:
            print("  %-4s %14s %14.5f %10s" % (nom, "0 (nodal)", mes[nom], "-"))
        else:
            print("  %-4s %14.3f %14.4f %+9.1f %%" % (
                nom, m, mes[nom], 100 * (m / mes[nom] - 1)))
    r_pred = ((1 + A * np.cos(4 * np.pi / 3 + d0)) /
              (1 + A * np.cos(d0))) ** 2
    print("  rapport mu/tau au verrou : %.5f vs mesuré %.5f (%+.1f %%)" % (
        r_pred, M_MU / M_TAU, 100 * (r_pred / (M_MU / M_TAU) - 1)))
    print("""
  => le squelette DISCRET seul (nœud au sommet, zéro paramètre au-delà
  de mu et A) donne : m_e = 0 (protection nodale exacte à l'ordre du
  réseau), m_tau à -1 %, m_mu à +19 % (rapport mu/tau +21 %). La
  hiérarchie leptonique est portée par le RÉSEAU à ~20 % près ; eps
  (0,0396 rad, un sixième de pas) habille les trois masses d'un coup
  — dont TOUTE la masse de l'électron.""")

    print("=" * 78)
    print("[3] eps <-> m_e : l'équivalence, et le jeu fermé sur m_e/m_l")
    print("=" * 78)
    eps = theta_star - (2 * np.pi / 3 + 2.0 / 9.0)
    ratio = M_E / M_L
    print("  eps = 0,0396 rad = %.3f pas ; 1 - cos(eps) + sin(eps) = "
          "sqrt(m_e)/mu (exact)" % (eps / pas))
    print("  dériver eps <=> dériver m_e/m_l = %.4e ; jeu fermé :" % ratio)
    for nom, v in [("3 alpha/(4 pi) (auto-énergie QED)",
                    3 * ALPHA / (4 * np.pi)),
                   ("alpha/(2 pi)", ALPHA / (2 * np.pi)),
                   ("(1/(8 pi))^2", (1 / (8 * np.pi)) ** 2),
                   ("(2 e alpha)^2", (2 * np.e * ALPHA) ** 2),
                   ("alpha^2 (k+2)/2", ALPHA ** 2 * 137 / 2)]:
        print("    %-32s = %.4e   (%+.1f %%)" % (nom, v,
                                                 100 * (v / ratio - 1)))
    print("""
  VERDICT (règle du test nul) : le plus proche à mécanisme PLAUSIBLE
  est 3 alpha/(4 pi) (le coefficient d'auto-énergie de masse QED) à
  +7 % — mais SANS le logarithme qui l'accompagne toujours : non
  retenu en l'état ; les autres sont sans mécanisme. eps reste
  OUVERT. Piste physique la plus cohérente avec la protection
  nodale : la masse résiduelle de l'électron = auto-énergie EM du
  mode nodal (le nœud tue le canal condensat, PAS le canal charge)
  — à calculer avec la vraie normalisation du circuit (cible).""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""   Le pi/12 de delta EST un pas de rail (2pi/24, circuit minimal du
   trèfle) ; toute la structure angulaire de Koide est commensurable
   avec le circuit (Z3 = pas {0,8,16}, nœud = pas 9) et le sqrt(2)
   dérivé tombe pile sur un sommet. Le squelette discret (verrou au
   sommet) porte la hiérarchie à ~20 % avec m_e = 0 exact ; eps = un
   sixième de pas habille tout, et l'équivalence eps <-> m_e/m_l en
   fait LE dernier nombre du secteur leptonique. Hypothèse nommée
   (verrouillage du nœud sur un sommet) + cible nommée (auto-énergie
   EM du mode nodal) — rien de retenu au-delà (test nul).""")


if __name__ == "__main__":
    main()
