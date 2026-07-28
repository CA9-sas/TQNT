"""C1 : non-contamination de la liaison nucleaire par P22 (20/07).

TEST (item C1 du chapitre interactions, **) : le gradient adimensionne
xi = |grad Omega^2| lambda_C / Omega^2 doit rester << 1 au coeur d'un
noyau -- sinon la nucleation P22 (Gamma = Gamma_P e^{-2pi/xi}) parasite
la liaison et devrait etre soustraite.

METRIQUE EMERGENTE d'un noyau (A, Z) : sphere uniforme de rayon
R = 1,2 A^{1/3} fm, masse M = A m_N ; en champ faible
  Omega^2 - 1 = 2 G M(<r)/(r c^2)   (exterieur : r >= R)
              = (G M/c^2) (3 R^2 - r^2)/R^3   (interieur)
Le gradient maximal de Omega^2 est a la surface : |grad Omega^2|_max =
2 G M/(R^2 c^2). lambda_C = Compton reduite de l'ELECTRON (canal P22
DERIVE : l'espece la plus legere). xi_max = 2 G M lambda_C/(R^2 c^2).

On calcule xi_max et le taux P22 relatif e^{-2pi/xi} pour des noyaux
representatifs, puis les cas extremes (etoile a neutrons) et le seuil
xi ~ 1 pour situer la marge. Reference etablie : le meme calcul sur la
metrique gravitationnelle avait tue l'hypothese << peau baryonique >>
(p22_dark_energy_noknob : trou de ~36 decades).
"""
import math

G = 6.6710e-11            # G_predit (convention du document)
C = 2.99792458e8
M_N = 1.674e-27           # kg (nucleon)
LAM_C_E = 3.8616e-13      # m (Compton reduite de l'electron)
FM = 1e-15

NOYAUX = [("H (p)", 1), ("He-4", 4), ("O-16", 16), ("Fe-56", 56),
          ("Pb-208", 208), ("U-238", 238)]


def run():
    print("=== C1 : xi de P22 au coeur/surface des noyaux ===\n")
    print("%-10s %8s %12s %12s %14s" % ("noyau", "R (fm)", "Omega^2-1", "xi_max",
                                        "log10 Gamma/Gamma_P"))
    xi_pire = 0.0
    for nom, A in NOYAUX:
        R = 1.2 * A ** (1 / 3) * FM
        M = A * M_N
        h = 2 * G * M / (R * C ** 2)
        xi = 2 * G * M * LAM_C_E / (R ** 2 * C ** 2)
        xi_pire = max(xi_pire, xi)
        log_rate = -2 * math.pi / xi / math.log(10)
        print("%-10s %8.2f %12.2e %12.2e %14s" % (
            nom, R / FM, h, xi, "-%.1e" % (-log_rate)))
    print("\n  -> xi_max(noyaux) = %.1e : marge de %.0f ORDRES sous xi ~ 1 ;" % (
        xi_pire, -math.log10(xi_pire)))
    print("     le taux P22 est e^{-2pi/xi} ~ 10^{-10^36} : ZERO physique.")
    print("     Aucune nucleation parasite a soustraire de la liaison.\n")

    # extremes, pour situer
    print("Extremes (perspective) :")
    for nom, M_kg, R_m in [("etoile a neutrons (surface)", 2.8e30, 1.2e4),
                           ("horizon Schwarzschild 3 M_sol", 6.0e30, 8.86e3)]:
        h = 2 * G * M_kg / (R_m * C ** 2)
        xi = 2 * G * M_kg * LAM_C_E / (R_m ** 2 * C ** 2)
        print("  %-28s Omega^2-1 = %.2f ; xi = %.1e" % (nom, h, xi))
    print("  -> meme l'etoile a neutrons reste a xi ~ 1e-17 ; xi ~ 1 n'est")
    print("     atteint qu'aux horizons/champs critiques (P22 etabli).\n")

    print("VERDICT C1 : RESOLU -- la liaison nucleaire est propre de toute")
    print("contamination P22 par ~35 ordres de grandeur ; coherent avec le")
    print("calcul sans parametre libre de l'audit energie noire (trou de 36")
    print("decades de l'hypothese << peau baryonique >>) et le profil du")
    print("nucleon (xi ~ 1e-40, omega_profile_neutron).")


if __name__ == "__main__":
    run()
