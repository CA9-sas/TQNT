"""Prédictions des nœuds LIBRES (cœurs réalisables) : masse + durée de vie
(28/07/2026, prolongement du théorème du désert — brouillon).

DEMANDE : les cœurs réalisables (4_1, 5_2, 6_1, ... — nœuds n=1 sous le
seuil sombre, sans état libre réalisé) doivent recevoir une MASSE et une
DURÉE DE VIE prédites, pour que leur absence (ou une détection P22)
devienne OPPOSABLE.

LES TROIS PIÈCES (toutes canoniques, aucune constante nouvelle) :

  1. MASSE = loi géométrique du plateau léger (canonique pour un nœud
     statique isolé) : M_topo(K) = m_l * l(K)/l(3_1), ropelengths ACPR,
     m_l = 310 MeV (calibrée, transmutation). Validité : plateau léger
     (c <= 8 ici, loin du plafond ~1,4 GeV).

  2. CANAL = DÉNOUAGE FAIBLE P15 : un nœud n=1 n'a rien à larguer (pas
     de cascade — théorème du désert inapplicable) ; son seul canal est
     la reconnexion P15 (changement de croisement). Fait mathématique :
     u(4_1) = u(5_2) = u(6_1) = u(6_2) = u(6_3) = u(7_2) = 1 (nombre de
     dénouage) -> UNE reconnexion suffit à dénouer complètement -> toute
     la masse part en rayonnement chi (+ e± si la variante est chargée).
     Les canaux partiels (5_2 -> 3_1 + rad., Q=158 MeV) sont
     Sargent-écrasés ((Q_part/Q_tot)^5 ~ 4e-3) : négligés.

  3. TAUX = la structure de Fermi DÉRIVÉE (P15 quantitative, g_topo=1) :
         Gamma = N * Q^5 / (384 pi^3 v^4),   v = 246,22 GeV (dérivé).
     N = 1 (canal de rayonnement minimal, déclaré -> tau est une borne
     SUPÉRIEURE, tau ~ 1/N). CONTRÔLE : la même formule sur le muon
     (Q = m_mu, N=1) doit redonner tau_mu = 2,20e-6 s.

CHARGE : Q = chi + Lk (entier). Amphichiraux (4_1, 6_3, 8_3) : chi = 0
-> neutre naturel à Lk = 0. Chiraux : chi != 0, la variante NEUTRALISÉE
(Lk = -chi) existe toujours (logique des « réalisables ») ; les
variantes chargées different par Lk. La cinématique du dénouage est la
même (Q_kin ~ m, moins m_e si chargé).

PRODUCTION (pourquoi jamais vus, re-dit quantitativement) : l'entrée est
par PAIRE (théorème §54) -> au collisionneur la paire K+Kbar naît à
distance fm et se lie (construction méson : hadronisation) ; un nœud
LIBRE n'existe que si un champ critique sépare la paire (Schwinger P22 :
horizons, champs extrêmes) — flux non calculé (cible nommée).

GARDE-FOUS : m_l calibrée (±4-7 % d'ancre -> ±5 % sur m, x1,3 sur tau) ;
habillage +5,6 % du 5_2 mesuré EN COMPOSITE (le libre = géométrique nu,
bande [468, 495] déclarée) ; mesure 3-corps m^5 déclarée (structure
Fermi) ; 3_1 EXCLU de la table (son état libre EST le lepton chargé —
modes Z3 + projection condensat, PAS M_topo=310) ; toriques 5_1/7_1
= classe leptonique, loi d'état libre OUVERTE (échelle M_topo seule).
"""
import math

from theory.ropelength_acpr import ROPELENGTH_ACPR, m_topo

HBAR = 6.582119569e-25       # GeV s
V_EW = 246.22                # GeV (dérivé, croisement-ouvert)
M_E = 0.000511               # GeV
C = 2.99792458e8             # m/s
TAU_MU_PDG = 2.1969811e-6    # s
M_MU = 0.1056584             # GeV

DENOM = 384 * math.pi**3 * V_EW**4   # GeV^4

# nombre de dénouage u(K) (tables mathématiques standard)
UNKNOTTING = {"4_1": 1, "5_2": 1, "6_1": 1, "6_2": 1, "6_3": 1,
              "7_2": 1, "7_3": 2, "7_4": 2, "7_5": 2, "7_6": 1,
              "7_7": 1, "8_1": 1, "8_2": 2, "8_3": 2}
AMPHI = {"4_1", "6_3", "8_3"}
ROLE = {"4_1": "coeur meson / DM stricte (4_1+8_3)",
        "5_2": "noeud etrange (constituant, m_s habille 494,8)",
        "6_1": "orphelin (minimalite : exclu de m_s)",
        "6_2": "orphelin", "6_3": "orphelin / DM stricte (6_3+6_3)",
        "7_2": "compagnon du nucleon", "8_3": "coeur EW (quartet)"}


def tau_p15(Q_GeV, N=1):
    """durée de vie du dénouage P15 (structure de Fermi dérivée)."""
    return HBAR * DENOM / (N * Q_GeV**5)


def main():
    print(__doc__)
    print("=" * 78)
    print("[0] CONTRÔLE : le muon sous la même formule")
    print("=" * 78)
    t = tau_p15(M_MU)
    print("  tau(Q=m_mu, N=1) = %.3e s  vs PDG %.3e s  (%+.1f %%)" % (
        t, TAU_MU_PDG, 100 * (t / TAU_MU_PDG - 1)))
    print("  => la mesure de Fermi dérivée est ancrée à -0,4 %.")

    print("=" * 78)
    print("[1] LA TABLE DES NŒUDS LIBRES (prédictions opposables)")
    print("=" * 78)
    print("  %-5s %-9s %6s %4s %-7s %12s %10s %9s" % (
        "K", "role", "m", "u", "Q_top", "tau [s]", "c*tau [m]", "Gamma"))
    knots = ["4_1", "5_2", "6_1", "6_2", "6_3", "7_2", "8_3"]
    for k in knots:
        m = m_topo(k) / 1000.0                       # GeV
        u = UNKNOTTING[k]
        qtop = "0 (amph)" if k in AMPHI else "chi+Lk"
        # variante neutre : Q_kin = m ; chargée : m - m_e (identique a 0,1 %)
        tau = tau_p15(m) * u                          # u étapes en série
        ctau = C * tau
        gam = HBAR / tau
        print("  %-5s %-9s %6.1f %4d %-7s %12.2e %10.2f %9.1e" % (
            k, ROLE[k][:9], m * 1000, u, qtop, tau, ctau, gam))
    print("""
  masses en MeV (géométrique ACPR, m_l=310) ; u = nb de reconnexions
  P15 pour dénouer ; tau = u * tau_P15(Q=m), N=1 (borne sup.) ;
  Gamma en GeV (largeurs ~1e-15 : états ultra-étroits, PAS des
  résonances larges — la non-détection vient de la PRODUCTION).""")

    print("=" * 78)
    print("[2] LES TROIS PRÉDICTIONS NOMMÉES (4_1, 5_2, 6_1)")
    print("=" * 78)
    for k in ["4_1", "5_2", "6_1"]:
        m = m_topo(k)
        tau = tau_p15(m / 1000.0) * UNKNOTTING[k]
        print("  %s libre : m = %.0f MeV ; tau = %.1e s ; c*tau = %.2f m"
              % (k, m, tau, C * tau))
    print("""  5_2 : bande déclarée [468, 495] MeV (géométrique nu vs habillé
  composite +5,6 % — le libre est attendu NU : 468).""")

    print("=" * 78)
    print("[3] SIGNATURES ET FALSIFIABILITÉ")
    print("=" * 78)
    print("""  SIGNATURE (variante chargée) : trace de masse anormale
  0,40-0,67 GeV, interaction hadronique (tube chi) MAIS désintégration
  en e± + énergie manquante (rayonnement chi), vertex déplacé
  c*tau ~ 0,2-0,9 m — aucun état du MS n'a cette combinaison.
  SIGNATURE (variante neutre) : énergie manquante pure (invisible).
  PRODUCTION : par paires seulement (§54) -> collisionneurs :
  hadronisation immédiate (PRÉDICTION D'ABSENCE maintenue) ; libres
  UNIQUEMENT si un champ critique sépare la paire (P22 : horizons,
  champs Schwinger) -> chercher dans les environnements extrêmes,
  flux non calculé (cible nommée).
  RÉFUTATIONS POSSIBLES : (i) détection d'un état STABLE libre à ces
  masses (contredit le dénouage P15) ; (ii) détection au collisionneur
  d'un célibataire (contredit la porte à paires) ; (iii) état libre à
  ces masses avec tau tres différent de la table (contredit g_topo=1
  ou N=1 a un facteur pres).""")


if __name__ == "__main__":
    main()
