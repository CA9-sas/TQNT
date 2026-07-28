"""L'echelle d'alignement (x1, x2) du multiplet Omega : derivation (22/07).

CIBLE : le dernier nombre mesure de la regle de multiplet -- l'echelle
x = (0, x1, x2) des niveaux j = 0, 1, 2 (mesuree au charme : 0,661 /
1,189 ; pas bottom j1->j2 : 0,470).

LE CANDIDAT A MECANISME (canonique, zero nombre nouveau) :
    x_j = j x (1/2)  --  LE COMPTAGE DES RETOURNEMENTS DE MEMBRES.
La paire ss = DEUX rubans verrouilles paralleles ; s'aligner contre L
coute LE RUNG (1/2) PAR MEMBRE -- exactement le +1/2 du S17 (l'alignement
de la bande mesonique, un seul ruban). j = nombre de membres alignes :
j=0 -> 0 ; j=1 -> 1 flip -> 1/2 ; j=2 -> 2 flips -> 1. Echelle (0, 1/2, 1).

CONCURRENT NUMERIQUE (sans mecanisme) : (2/3, 7/6) -- x1 a +0,9 %,
x2 a -1,9 % du mesure charme ; slots (2/3 = quantum Z3, 1/2 = rung)
mais AUCUNE derivation : regle du null.
"""
import numpy as np

SIGMA = 0.41847 ** 2
PS = np.pi * SIGMA
MS_, MC, MB = 0.4948, 1.557, 4.952
D1 = np.sqrt(2) - 1
OB_CORE = MB + 2 * MS_
DN_XI_B = 6.497
# mesures (S30-31)
X_C = {"x1": 0.6611, "x2": 1.1894}      # charme (assignation ordre de masse)
STEP_B = 0.4698                          # bottom j1->j2
N_J1_B = 8.5579


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] LE TEST DES DEUX ECHELLES")
    print("=" * 78)
    print("""            mesure(c)   flips (0,1/2,1)   (2/3,7/6) [null]
  x1         %.3f       0,500  (%+.0f %%)     0,667  (%+.1f %%)
  x2         %.3f       1,000  (%+.0f %%)     1,167  (%+.1f %%)
  pas j1->j2 (c) %.3f   0,500  (%+.0f %%)     0,500  (%+.0f %%)
  pas j1->j2 (b) %.3f   0,500  (%+.0f %%)     0,500  (%+.0f %%)
  -> le PAS j1->j2 encadre 1/2 aux deux generations (+6/-6 %%) : le
  deuxieme flip vaut LE RUNG, aux deux secteurs -- c'est le morceau
  robuste. La tension est CONCENTREE sur x1 charme (+32 %% du rung),
  qui repose sur UN etat (Omega_c(3000) = j0) dont l'assignation est
  le suspect declare du S30-31 (les spins du quintet scrambleraient
  l'echelle ; la parite interdit l'excuse du melange 2S : c'est bien
  l'ASSIGNATION, pas la dynamique).""" % (
        X_C["x1"], 100 * (0.5 / X_C["x1"] - 1), 100 * (2 / 3 / X_C["x1"] - 1),
        X_C["x2"], 100 * (1.0 / X_C["x2"] - 1), 100 * (7 / 6 / X_C["x2"] - 1),
        X_C["x2"] - X_C["x1"], 100 * (0.5 / (X_C["x2"] - X_C["x1"]) - 1),
        100 * (0.5 / (X_C["x2"] - X_C["x1"]) - 1),
        STEP_B, 100 * (0.5 / STEP_B - 1), 100 * (0.5 / STEP_B - 1)))

    print("=" * 78)
    print("[2] LE MECANISME DES FLIPS : POURQUOI C'EST LE RUNG, ET RIEN D'AUTRE")
    print("=" * 78)
    print("""  (i) le rung 1/2 est DEJA le quantum d'alignement du document principal : bande
      mesonique j_K=3/2 -> +1/2 (un ruban aligne, S17 demontre) ;
      J=3/2 baryonique -> +1/2 (liaison jonction) ; V-P -> +1 (le flip
      complet de la paire mesonique) ;
  (ii) la paire verrouillee = 2 rubans : l'alignement se paie PAR
      MEMBRE -> x_j = j/2, echelle (0, 1/2, 1) -- aucun nombre nouveau ;
  (iii) monotonie OK (0 au sol) ; heavy-blindness OK (le rung est un
      quantum leger) ;
  (iv) le concurrent (2/3, 7/6) : meilleur numeriquement au charme mais
      SANS mecanisme (pourquoi 2/3 au premier flip et 1/2 au second ?)
      -- regle du null : non retenu.""")

    print("=" * 78)
    print("[3] LES DISCRIMINANTS (l'observation tranchera)")
    print("=" * 78)
    n_j0_flip = N_J1_B - 0.5
    n_j0_meas = N_J1_B - X_C["x1"]
    m_flip = np.sqrt(OB_CORE**2 + PS * n_j0_flip)
    m_meas = np.sqrt(OB_CORE**2 + PS * n_j0_meas)
    print("""  (a) LE 5e Omega_b DISCRIMINE : echelle des flips -> j0 = n(j1)-1/2
      -> M = %.0f MeV ; echelle mesuree charme -> M = %.0f MeV.
      Les deux lectures different de %.0f MeV : la masse du cinquieme
      etat selectionnera l'echelle (prediction affinee : %.0f MeV si
      les flips, %.0f si l'echelle charme) ;
  (b) LES SPINS DU QUINTET Omega_c : une re-assignation (le second
      J=1/2 n'est peut-etre pas 3050) recalcule x1 -- si l'assignation
      corrigee donne x1 ~ 1/2, l'echelle des flips ferme TOUTE la
      regle de multiplet sans nombre mesure ;
  (c) les ondes P de Xi_cc/Omega_cc (paire cc) : meme echelle predite
      (0, 1/2, 1) + verrou plat (d-1).""" % (
        1e3 * m_flip, 1e3 * m_meas, 1e3 * abs(m_flip - m_meas),
        1e3 * m_flip, 1e3 * m_meas))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  L'echelle d'alignement a son MECANISME : x_j = j x (rung 1/2) -- le
  comptage des retournements de membres de la paire verrouillee, avec
  le quantum d'alignement DEJA canonique (S17). SUPPORT : le pas
  j1->j2 encadre 1/2 aux deux generations (+6/-6 %) ; TENSION : x1
  charme (+32 %), concentree sur l'assignation d'UN etat (suspect
  declare) ; le concurrent numerique (2/3, 7/6) est ecarte par la
  regle du null. STATUT : identification CONDITIONNELLE (favorisee),
  discriminee par la masse du 5e Omega_b (6304 flips vs 6297 mesuree)
  et par les spins du quintet charme. La regle de multiplet est
  potentiellement COMPLETE (zero nombre mesure) si les flips tiennent.""")


if __name__ == "__main__":
    main()
