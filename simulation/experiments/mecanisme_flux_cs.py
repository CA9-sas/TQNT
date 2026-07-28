"""Le mecanisme de flux CS : compression 1/(k+2) du coeur -- test (17/07).

CIBLE : promouvoir D = 2 pi hbarc/((k+2) sqrt(sigma_chi)) de coincidence
a derivation. Test decisif : le compte de cellules transverses
    N = 2 pi lambda_sigma / D   (lambda_sigma = hbarc/sqrt(sigma_chi))
doit valoir k+2 = 137 EXACTEMENT si le mecanisme est le bon.

TROIS MECANISMES CANDIDATS :
  (A) Minimisation de tension d'un tube charge par la loi de Gauss CS
      (flux Phi lie une charge lineique ~ k Phi/2pi ; Coulomb en ln).
  (B) Cellule d'espace de phase : le quantum de phase de SU(2)_k est
      pi/(k+2) (l'argument de q) ; le tube = une cellule angulaire
      2 pi/(k+2) du cercle transverse de rayon lambda_sigma.
  (C) Largeur quantique de Luscher : w^2 = ln(L/D)/(2 pi sigma).
"""
import math

HBARC = 0.19733
SQS = 0.41765
SIGMA_PHYS = SQS ** 2 / HBARC          # GeV/fm
M_L = 0.310
L31 = 16.372
K = 135


def run():
    lam_s = HBARC / SQS
    L_phys = M_L / SIGMA_PHYS
    D = L_phys / L31
    N = 2 * math.pi * lam_s / D
    print("=== Mecanisme de flux CS : la compression 1/(k+2) au banc ===\n")
    print("D (mesure via m_l = 310) = %.5f fm ; lambda_sigma = %.5f fm" % (D, lam_s))
    print("N = 2 pi lambda_sigma / D = %.2f   (mecanisme B : k+2 = %d)\n" % (N, K + 2))

    print("[A] Tube charge par la loi de Gauss CS (minimisation de tension) :")
    print("    sigma(R) = Phi^2/(2 pi R^2 g^2) + eps pi R^2 + C k^2 alpha ln(Lam/R)")
    print("    -> equilibre condensation/Coulomb : R* ~ k sqrt(alpha) ~ sqrt(k) x")
    print("    le tube GROSSIT avec k : MAUVAIS SENS -- REFUTE comme mecanisme")
    print("    de compression.\n")

    print("[B] Cellule d'espace de phase (quantum de phase pi/(k+2)) :")
    for n in (K, K + 1, K + 2, K + 3, K + 4):
        ml = 2 * math.pi * L31 * SQS * 1000 / n
        print("    N = %3d : m_l = %.1f MeV (%+.2f %% vs 310)" % (
            n, ml, 100 * (ml / 310 - 1)))
    print("    N mesure = %.2f : l'entier k+2 = 137 est a -1.2 %% ; 138/139" % N)
    print("    collent aussi bien ou mieux. Resolution des donnees : un pas")
    print("    d'entier = 0.72 %% sur m_l -- l'incertitude de calibration de")
    print("    m_l (~0.5-1 %%, non quantifiee) NE PINCE PAS l'entier.\n")

    w2 = math.log(L_phys / D) / (2 * math.pi * SIGMA_PHYS / HBARC)
    print("[C] Largeur quantique de Luscher : w = %.3f fm (log-dependante)" % math.sqrt(w2))
    print("    -> 15x plus grande que D et pas une ECHELLE fixe : REFUTEE")
    print("    comme identification directe du coeur.\n")

    print("VERDICT :")
    print(" - Le mecanisme de compression 1/(k+2) N'EST PAS ETABLI :")
    print("   (A) energetique CS : mauvais sens (R ~ sqrt(k)) -- refute ;")
    print("   (C) Luscher : structure log, pas une echelle -- refutee ;")
    print("   (B) cellule de phase : viable en FORME mais les donnees mesurent")
    print("   N = %.1f sans resolution entiere (137 a -1.2 %%) : l'identification" % N)
    print("   reste au NIVEAU COINCIDENCE, legerement defavorisee.")
    print(" - Pour trancher : (i) pincer m_l a < 0.3 %% (fit du plateau leger")
    print("   avec barres d'erreur -- un pas d'entier = 0.72 %%) ; (ii) deriver")
    print("   le RAYON du cercle transverse (lambda_sigma est pose, pas derive ;")
    print("   Luscher montre que la largeur naturelle est une autre structure).")
    print(" - m_l reste la constante de transmutation (statut inchange).")


if __name__ == "__main__":
    run()
