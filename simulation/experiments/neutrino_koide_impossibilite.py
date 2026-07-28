"""L'impossibilité de Koide pour les neutrinos — une absence
structurelle prédite par la TQNT, vérifiée par les données
(28/07/2026, suite du chantier neutrino-trame).

LA PRÉDICTION STRUCTURELLE : la forme de Koide des leptons chargés,
    sqrt(m_k) = mu (1 + sqrt(2) cos(2 pi k/3 + delta)),  Q = 2/3,
est portée par un MÉCANISME précis (consolidé V6) : les générations
sont les modes Z3 du CIRCUIT trèfle, et sqrt(m) est la projection de
l'état sur le condensat, avec A = sqrt(2) = le doublement de canal
des deux sens de circulation. Le neutrino N'A PAS de nœud (onde
auto-rotative ouverte, jamais entrée dans un rail) : pas de circuit,
pas de modes Z3 de circuit, pas de projection de condensat
=> LA FORME DE KOIDE NE DOIT PAS S'APPLIQUER AUX NEUTRINOS.

LE TEST (données seules) : les écarts d'oscillation mesurés fixent
m2, m3 en fonction de la plus légère (m1 en ordre normal, m3 en
inversé). On calcule Q(m_min) = Sigma m / (Sigma sqrt(m))^2 sur tout
l'intervalle : si Q = 2/3 est INATTEIGNABLE, la prédiction d'absence
est vérifiée — et tout modèle qui étend Koide (A = sqrt(2)) aux
neutrinos est déjà exclu par les données.

DONNÉES (NuFit, 3-sigma approx) :
  Dm2_21 = 7,53e-5 eV² ; Dm2_31 = 2,53e-3 eV² (NO) ;
  DESI : Sigma m_nu < 72 meV (NO favorisé).

GARDE-FOUS : règle du test nul sur toute valeur « remarquable » de
Q_max ou du A effectif ; la prédiction TQNT est l'ABSENCE de la
forme (dichotomie nœud / onde ouverte), pas une valeur de Q.
"""
import numpy as np

DM2_21 = 7.53e-5             # eV^2
DM2_31 = 2.53e-3             # eV^2
DESI = 0.072                 # eV


def masses_NO(m1):
    return m1, np.sqrt(m1**2 + DM2_21), np.sqrt(m1**2 + DM2_31)


def masses_IO(m3):
    m2 = np.sqrt(m3**2 + DM2_31)
    m1 = np.sqrt(m3**2 + DM2_31 - DM2_21)
    return m1, m2, m3


def Q(masses):
    m = np.array(masses)
    return m.sum() / np.sqrt(m).sum() ** 2


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] Q(m_min) SUR TOUT L'INTERVALLE (les deux ordres)")
    print("=" * 78)
    for lab, f in [("ordre NORMAL (m1 libre)", masses_NO),
                   ("ordre INVERSÉ (m3 libre)", masses_IO)]:
        grid = np.concatenate([[0.0], np.logspace(-5, -0.5, 400)])
        qs = np.array([Q(f(m)) for m in grid])
        i = int(np.argmax(qs))
        print("  %-26s : Q_max = %.4f (à m_min = %.2f meV) ; "
              "Q -> 1/3 en dégénéré" % (lab, qs[i], grid[i] * 1e3))
        # Sigma m_nu à Q_max et sous DESI
        s0 = sum(f(0.0))
        print("      Sigma m_nu(m_min=0) = %.1f meV ; DESI < %.0f meV"
              % (s0 * 1e3, DESI * 1e3))
    print("""
  => Q_nu est BORNÉ : Q <= 0,585 (NO, atteint à m1 = 0) et
     Q <= 0,500 (IO). LA VALEUR DE KOIDE Q = 2/3 = 0,667 EST
     INATTEIGNABLE, quel que soit m_min et quel que soit l'ordre.
     THÉORÈME (données seules) : la forme de Koide des leptons
     chargés NE PEUT PAS tenir pour les neutrinos.""")

    print("=" * 78)
    print("[2] LE A EFFECTIF MAXIMAL (et la règle du test nul)")
    print("=" * 78)
    # Q = (1 + A^2/2)/3  =>  A = sqrt(2(3Q - 1))
    for lab, qmax in [("NO", 0.5853), ("IO", 0.5000)]:
        A = np.sqrt(2 * (3 * qmax - 1))
        print("  %-3s : A_eff <= %.4f   (vs sqrt(2) = 1.4142 des leptons"
              " chargés)" % (lab, A))
    print("""  curiosité balayée (règle du test nul) : A = sqrt(3/2) = 1,2247
  (soit Q = 7/12) est atteignable à m1 ~ 0 — SANS mécanisme (aucune
  dimension quantique canonique à cette valeur) : NON retenu.""")

    print("=" * 78)
    print("[3] LA LECTURE TQNT (l'absence est PRÉDITE)")
    print("=" * 78)
    print("""  Koide (chargés) = modes Z3 du CIRCUIT trèfle + projection de
  condensat + doublement de sens (A = sqrt(2)) — trois ingrédients
  qui exigent un NŒUD. Le neutrino est une onde OUVERTE (jamais
  entrée dans un rail) : aucun des trois ingrédients n'existe.
  => la TQNT PRÉDIT l'absence de la forme de Koide chez les
  neutrinos ; les données la VÉRIFIENT (Q <= 0,585 < 2/3, théorème
  ci-dessus). C'est une absence structurelle du même type que le
  verrou Lambda_ss : la dichotomie nœud / onde ouverte est
  DISCRIMINANTE, et elle tombe du bon côté.
  Corollaire : tout modèle qui étend Koide (A = sqrt(2)) aux
  neutrinos est DÉJÀ exclu par les écarts d'oscillation mesurés.""")

    print("=" * 78)
    print("[4] CE QUI FIXE LES TROIS MASSES (état honnête)")
    print("=" * 78)
    m1, m2, m3 = masses_NO(0.005)
    print("  lecture V6 (« boomerang », NO) : m1 ~ 5 meV -> (%.1f, %.1f,"
          " %.1f) meV, Sigma = %.0f meV" % (
              m1 * 1e3, m2 * 1e3, m3 * 1e3, (m1 + m2 + m3) * 1e3))
    print("""  Sigma m_nu ~ 66 meV vs DESI < 72 meV : la prédiction est à
  ~9 %% de la borne — TEST IMMINENT (DESI resserré : < 60 meV
  réfuterait). Les trois R d'auto-rotation restent AJUSTÉS (l'écart
  entre eux est fixé par les oscillations, le zéro par Sigma) :
  l'absence de Koide confirme qu'aucune loi de circuit ne DOIT les
  fixer — le mécanisme des masses ouvertes est une cible propre,
  distincte du secteur noué.""")


if __name__ == "__main__":
    main()
