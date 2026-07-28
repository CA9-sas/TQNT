"""Vérification SN1987A du retard/étalement du neutrino TQNT
(27/07/2026, contrôle du brouillon neutrino-trame).

CE QU'ON VÉRIFIE : la masse prédite (m_nu ~ 50 meV, oscillations +
auto-rotation) est-elle COMPATIBLE avec SN1987A ?

POINT DE RIGUEUR : SN1987A ne mesure PAS un « retard absolu » d'un
neutrino massif vs la lumière — les neutrinos ont devancé la lumière
optique de ~3 h pour une raison SANS RAPPORT avec la masse (l'onde de
choc met ~heures à traverser l'étoile ; les neutrinos s'échappent du
coeur tout de suite). Ce qui est OBSERVABLE et contraint la masse,
c'est l'ÉTALEMENT ÉNERGIE-DÉPENDANT de la bouffée :
    Dt(E) = (L/c) * (1/2) * (m/E)^2,
un neutrino de basse E arrive plus tard qu'un de haute E. La bouffée
observée (~13 s, E = 7,5-40 MeV) borne cet étalement.

DONNÉES SN1987A (23 fév. 1987, LMC) :
  - distance L = 51,4 kpc ;
  - Kamiokande-II : 11 evts, E = 7,5-36 MeV, sur ~12,4 s ;
  - IMB : 8 evts, E = 20-40 MeV ; Baksan : 5 evts ;
  - durée totale de la bouffée ~ 13 s (dispersion INTRINSÈQUE de
    l'émission -- c'est elle qui domine, pas la masse) ;
  - borne historique sur la masse : m_nu < 5,7 eV (Loredo & Lamb
    2002, 95 %) ; ~30 eV (analyses simples 1987).

CALCULS : (1) retard absolu (échelle, non observable seul) ;
(2) étalement 7->40 MeV (observable) ; (3) masse à laquelle
l'étalement = durée de bouffée (=> retrouve la borne SN1987A) ;
(4) verdict de compatibilité.
"""
import math

HBAR = 6.582119e-22          # MeV s
C = 2.99792458e8             # m/s
PC = 3.0856776e16            # m
KPC = 1e3 * PC
L = 51.4 * KPC               # m (distance LMC)
L_over_c = L / C             # s (temps de vol lumière)

M_NU = 50.3e-9               # MeV (m_3 garanti, oscillations)
E_LO, E_HI = 7.5, 40.0       # MeV (gamme SN1987A)
T_BURST = 13.0               # s (durée de la bouffée)
M_BOUND_HIST = 5.7e-6        # MeV (borne Loredo-Lamb 2002, 5,7 eV)
KATRIN = 0.45e-6             # MeV (0,45 eV, borne labo actuelle)


def delay(m_MeV, E_MeV):
    """retard d'un neutrino massif vs sans masse (s)."""
    return L_over_c * 0.5 * (m_MeV / E_MeV) ** 2


def main():
    print(__doc__)
    print("=" * 78)
    print("[0] ÉCHELLES")
    print("=" * 78)
    print("  L = %.1f kpc = %.3e m ; L/c = %.3e s = %.0f ans (vol lumière)"
          % (51.4, L, L_over_c, L_over_c / 3.156e7))

    print("=" * 78)
    print("[1] RETARD ABSOLU (échelle ; NON observable seul)")
    print("=" * 78)
    for E in (E_LO, 15.0, E_HI):
        print("  m = %.1f meV, E = %4.1f MeV : Dt = %.3e s" % (
            M_NU * 1e9, E, delay(M_NU, E)))
    print("  => ~1e-4 s : le neutrino massif traîne de ~0,1 ms sur"
          " 168 000 ans.")

    print("=" * 78)
    print("[2] ÉTALEMENT ÉNERGIE-DÉPENDANT (OBSERVABLE, contraint la masse)")
    print("=" * 78)
    spread = delay(M_NU, E_LO) - delay(M_NU, E_HI)
    print("  Dt(%.1f MeV) - Dt(%.0f MeV) = %.3e s" % (E_LO, E_HI, spread))
    print("  vs durée de bouffée %.0f s : rapport %.1e" % (
        T_BURST, spread / T_BURST))
    print("  => l'étalement de masse est ~%.0e fois PLUS PETIT que la"
          " bouffée" % (T_BURST / spread))
    print("     => totalement invisible : la dispersion observée (~13 s)"
          " est INTRINSÈQUE.")

    print("=" * 78)
    print("[3] À QUELLE MASSE SN1987A DEVIENDRAIT SENSIBLE ?")
    print("=" * 78)
    # etalement(m) = L/c * 1/2 * m^2 * (1/E_lo^2 - 1/E_hi^2) = T_burst
    k = L_over_c * 0.5 * (1 / E_LO ** 2 - 1 / E_HI ** 2)
    m_sens = math.sqrt(T_BURST / k)          # MeV
    print("  étalement = durée de bouffée (%.0f s) pour m = %.1f eV" % (
        T_BURST, m_sens * 1e6))
    print("  => borne « au doigt » SN1987A ~ %.0f eV (analyses fines :"
          " 5,7 eV)" % (m_sens * 1e6))

    print("=" * 78)
    print("[4] VERDICT DE COMPATIBILITÉ")
    print("=" * 78)
    print("  prédiction TQNT     : m_nu = %6.3f eV" % (M_NU * 1e6))
    print("  borne SN1987A (L-L) : m_nu < %6.2f eV  (%.0fx au-dessus de la"
          " prédiction)" % (M_BOUND_HIST * 1e6, M_BOUND_HIST / M_NU))
    print("  borne KATRIN 2024   : m_nu < %6.2f eV  (%.0fx au-dessus)" % (
        KATRIN * 1e6, KATRIN / M_NU))
    print("""
  => COHÉRENT. La prédiction (0,050 eV) est ~110x sous la borne
     SN1987A et ~9x sous KATRIN : SN1987A ne pouvait RIEN voir à cette
     masse, et n'a rien vu -- l'étalement de masse (0,1 ms) est noyé
     dans la dispersion intrinsèque (13 s) d'un facteur 1e5. La
     non-observation d'un étalement est exactement ce que TQNT prédit.
  NB honnête : SN1987A CONFIRME la borne (compatibilité), il ne
     TESTE pas la valeur 50 meV (trop lâche de ~2 ordres). Le test
     fin de m_nu reste cosmologique (DESI/Planck) et labo (KATRIN,
     futur Project 8).""")


if __name__ == "__main__":
    main()
