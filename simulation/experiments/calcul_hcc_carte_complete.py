"""La carte complete du dibaryon base-2 : M, J^P, Q, I, tau, canaux (22/07).

REVISION DE PARTITION (imposee par la question du spin) : sous Steiner
(double-Y + pont), les 6 jambes se PARTITIONNENT. Deux configurations :
 - (cll)-(cll) : les deux c sur des jonctions SEPAREES -> pas de cable
   cc -> PAS de verrou ; Pauli s'accommode en spatial symetrique +
   spin cc ANTISYMETRIQUE = 0 -> J^P = 0+ ; n = 2 (bases seules) ;
 - (ccl)-(lll) : cable cc a une jonction -> verrou 3(d-1), spin cc = 1
   -> J^P = 1+ ; n = 2 + 3(d-1)  [la lecture S36].
LE FONDAMENTAL = min = (cll)-(cll) : PLUS BAS de ~75 MeV. Le S36
decrivait l'ETAT EXCITE. La coherence (somme m)^2 est LA loi du document principal
(mesons/baryons du catalogue) : appliquee au reseau connexe, elle LIE
le dibaryon lourd compact.
"""
import numpy as np

SIGMA = 0.175117
PS = np.pi * SIGMA
ML, MS_, MC, MB = 0.310, 0.4948, 1.557, 4.952
D1 = np.sqrt(2) - 1
HBARC = 0.19733
EBR = SIGMA * 0.0214 / HBARC            # pont max ~19 MeV
HBAR_EVS = 6.582e-16


def m_st(core, n, br):
    return np.sqrt((core + br)**2 + PS * n)


def card(nom, core, n_g, n_x, seuils, taus, q, iso):
    lo, hi = m_st(core, n_g, 0), m_st(core, n_g, EBR)
    ce = m_st(core, n_g, EBR / 2)
    print("  === %s ===" % nom)
    print("    FONDAMENTAL (cll)-type : J^P = 0+, Q = %+d, I = %s" % (q, iso))
    print("    M = %.0f MeV  [%.0f, %.0f]  (pont [0, 19 MeV])" % (
        1e3 * ce, 1e3 * lo, 1e3 * hi))
    for snom, sm in seuils:
        print("      vs %-24s %+.0f MeV (central)" % (
            snom + " (%.0f)" % (1e3 * sm), 1e3 * (ce - sm)))
    if n_x:
        cex = m_st(core, n_x, EBR / 2)
        print("    EXCITE (cable lourd, verrou) : J^P = 1+, M = %.0f MeV" % (
            1e3 * cex))
    if taus:
        tau, meres = taus
        print("    tau ~ %.1e s (2 desintegrations lourdes ~ tau(%s)/2)," % (
            tau, meres))
        print("    c.tau ~ %.0f um, Gamma ~ %.0e eV" % (
            1e6 * 3e8 * tau, HBAR_EVS / tau))
    print()


def main():
    print(__doc__)
    print("=" * 78)
    print("LES CARTES (partition Steiner ; pont [0, +19 MeV])")
    print("=" * 78)
    card("H (uuddss)", 2 * MS_ + 4 * ML, 2, 2 + D1,
         [("Lambda Lambda", 2.23136)], None, 0, "0")
    card("H_cc++ (ccuudd)", 2 * MC + 4 * ML, 2, 2 + 3 * D1,
         [("Xi_cc N", 4.5599), ("Lambda_c Lambda_c", 4.57292)],
         (2.02e-13 / 2, "Lambda_c"), +2, "0")
    card("H_bb0 (bbuudd)", 2 * MB + 4 * ML, 2, 2 + 9 * D1,
         [("Lambda_b Lambda_b", 11.23920), ("Xi_bb N", 11.2789)],
         (1.47e-12 / 2, "Lambda_b"), 0, "0")

    print("=" * 78)
    print("SYNTHESE ET SIGNATURES")
    print("=" * 78)
    print("""  H LEGER : fondamental 2473 [2464, 2483] -- +242 au-dessus de
  Lambda Lambda : ABSENT (robuste aux deux partitions ET au pont --
  la postdiction des 40 ans de nulls tient).

  H_cc++ : FONDAMENTAL 0+ a 4488 +- 10 MeV -- LIE PROFOND (-85 sous
  Lambda_c Lambda_c, -72 sous Xi_cc N) : desintegration FAIBLE seule
  (2 charmes) -- tau ~ 1,0e-13 s, c.tau ~ 30 um : VERTEX DEPLACE, la
  signature Xi_cc++, a Q = +2 aussi ; canaux : Lambda_c+ p K- pi+,
  Lambda_c+ Lambda pi+, Xi_c+ p Kbar0 (les cascades doublement
  charmees) ; production double-charme (la ou LHCb a vu Xi_cc et
  Omega_cc). L'EXCITE 1+ (cable cc, S36-37) a 4563 +- 10 : entre les
  seuils -- resonance etroite Xi_cc N ou cascade gamma vers le
  fondamental. DEUX etats predits, hierarchie 0+ < 1+ (~75 MeV).

  H_bb0 : fondamental 0+ a 11203 +- 10 -- LIE (-36 sous Lambda_b
  Lambda_b) : REVISION (le verrou du S36 le donnait non lie ; la
  partition Steiner le lie) -- neutre, faible seule, tau ~ 7,3e-13 s,
  c.tau ~ 220 um ; production double-beaute (rare).

  FALSIFIABLES : (i) une structure a VERTEX DEPLACE dans les spectres
  Lambda_c+ p K- pi+ / Lambda_c Lambda pi a 4488 +- 10 (LHCb Run 3,
  evenements double-charme) ; (ii) le doublet 0+/1+ separe de
  ~75 MeV ; (iii) JAMAIS de pic au-dessus de Lambda_c Lambda_c
  (4573) ; (iv) l'absence stricte du H leger (toujours).
  IDENTIFICATION DECLAREE : la coherence (somme m)^2 du reseau
  connexe (LA loi du catalogue) appliquee au double-Y -- c'est elle
  qui lie ; sa validation directe = l'observation du fondamental.""")


if __name__ == "__main__":
    main()
