"""Chiffrer la piste : E_relax (= v = 246 GeV) = energie du doublet de writhe du 8_3 ?

Le 8_3 (present dans les 4 assignations EW) porte un doublet miroir Wr = +-7.24 ;
a framing fixe, le flip coute dTw = 14.5. Trois routes testees contre 246 GeV.
"""
import math

HBARC = 0.1973269804      # GeV.fm
SQSIG = 0.41765           # GeV
SIGMA = SQSIG**2 / HBARC  # GeV/fm (0.884)
RHO83 = 74.6              # ropelength (rayons)
DTW = 2 * 7.24            # flip complet
ALPHA = 1 / 137.035999
MU0 = 4e-7 * math.pi
H_OVER_E = 4.135667696e-15
HBAR_SI = 1.054571817e-34
C = 2.99792458e8
PHI = ALPHA * H_OVER_E
V_EW = 246.22


def run():
    print("=== Piste E_relax = doublet de writhe du 8_3 : trois routes ===\n")

    print("1. Energie elastique de twist : E = 2 pi^2 sigma_chi a dTw^2 / rho")
    for name, a in (("d_tube nucleon (0.209 fm)", 0.209),
                    ("largeur de corde hbar c/sqrt(sigma) (0.472 fm)", HBARC / SQSIG),
                    ("lambdabar_C(e) (386 fm)", 386.16)):
        E = 2 * math.pi**2 * SIGMA * a * DTW**2 / RHO83
        print("   a = %-38s: E = %8.1f GeV (x%.1f de 246)" % (name, E, 246/E if E else 0))

    print("\n2. Fluxoide auto-coherent (boucle de Compton r = hbar c/E, a = r/4.7) :")
    # identite exacte : Phi_chi^2/(2 mu0 hbar c) = alpha*pi/2
    ident = PHI**2 / (2 * MU0 * HBAR_SI * C)
    print("   identite : Phi_chi^2/(2 mu0 hbar c) = %.5f ; alpha*pi/2 = %.5f (EXACTE)" % (
        ident, ALPHA * math.pi / 2))
    lnf = math.log(8 * 4.7) - 2
    dtw_needed = math.sqrt(2 * lnf / (ALPHA * math.pi))
    print("   condition E_flip = E (auto-coherence) : dTw = sqrt(2(ln-2)/(alpha pi)) = %.1f" % dtw_needed)
    print("   mesure : dTw = %.1f (ecart %.0f%%) -- MAIS la condition est INVARIANTE" % (
        DTW, (DTW/dtw_needed - 1) * 100))
    print("   d'echelle (E se simplifie) : elle ne peut PAS produire 246 GeV.")

    print("\n3. Rapports directs (discipline du null : coincidences isolees non porteuses) :")
    target = V_EW / SQSIG
    print("   cible : v/sqrt(sigma_chi) = %.1f" % target)
    for name, val in (("dTw^2", DTW**2), ("dTw^2 * e", DTW**2 * math.e),
                      ("dTw^2 * pi", DTW**2 * math.pi), ("rho * dTw / 2", RHO83 * DTW / 2),
                      ("2 pi^2 dTw^2 / rho * rho", 2 * math.pi**2 * DTW**2)):
        print("   %-16s = %8.1f (ecart %+5.1f%%)" % (name, val, (val/target - 1) * 100))

    print("\n=== VERDICT ===")
    print("NEGATIF. (1) L'elastique donne 10-23 GeV avec les rayons naturels (facteur")
    print("10-24 trop court). (2) Le fluxoide auto-coherent est degenere en echelle --")
    print("il livre en passant une identite exacte Phi_chi^2/(2 mu0 hbar c) = alpha*pi/2")
    print("et une condition dTw~11.9 vs 14.5 mesure (22%) : curiosite a noter, pas une")
    print("ancre. (3) Les rapports directs n'offrent que des coincidences isolees")
    print("(dTw^2*e a 3%) que le null ecarte. La piste ne produit pas 246 GeV.")


if __name__ == "__main__":
    run()
