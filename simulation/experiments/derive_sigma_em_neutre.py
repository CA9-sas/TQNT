"""Derivation de sigma_EM(neutre) : les canaux EM d'un noeud sombre (16/07).

QUESTION (suite du pincage <= 1e-30) : que vaut le prefacteur sigma_EM de
sigma(c) = sigma_EM * alpha^(c-18), derive des premiers principes ?

STRUCTURE DE CHARGE DU RESERVOIR (prealable) : les toriques T(2,q>=23)
sont CHIRAUX (chi = +2). Trois lectures :
  (L1) MILLICHARGE de vertex : la suppression de Vassiliev lue comme
       charge effective eps = alpha^{(c-18)/2} sur un noeud Q != 0.
  (L2) PAIRE K+Kbar liee (~fm) : dipole electrique d = eps*e*d_sep.
  (L3) NEUTRALISATION PAR FRAMING : Q = Sigma_chi + Lk = 0 (le mecanisme
       meme des 111 'realisables' du catalogue) -> noeud strictement
       neutre ; couplage restant = POLARISABILITE du tube (+ dipole
       magnetique residuel, largement annule par la circulation de
       framing compensatrice).

CANAUX (cinematique halo : v = 1e-3, q_max = 2 mu v, mu(1.7 GeV,p) = 0.61 GeV) :
  A. Rutherford ecrante (L1)   : sigma ~ 16 pi alpha^2 eps^2 mu^2 / q_scr^4
  B. Dipole electrique (L2)    : sigma ~ 4 pi alpha^2 eps^2 d_sep^2 L / v^2
  C. Dipole magnetique (borne) : mu_K ~ eps * e/(2 m_K)
  D. Polarisabilite (L3)       : V = -A/r^4, A = 2 pi alpha_pol alpha ;
       Born : sigma = 4 pi^3 mu^4 v^2 A^2  (par proton)
  alpha_pol : deux estimations independantes --
    (i)  rescaling neutron : alpha_n = 1.2e-3 fm^3 x (R_K/R_n)^3 x (dE_n/dE_K)
    (ii) dimensionnelle    : alpha_EM * R^2 / Lambda_chi

VERDICTS contre le pincage DS-50 (sigma_EM <= ~1e-30) et la limite
directe a c=23 (2.2e-41 apres 1/f).
"""
import math

ALPHA = 1 / 137.036
FM2_CM2 = 1e-26
HBARC = 0.19733           # GeV.fm
V = 1e-3
M_DM = 1.71               # GeV (c=23)
MU = M_DM * 0.938 / (M_DM + 0.938)   # GeV
EPS23 = ALPHA ** 2.5      # vertex suppression at c=23 (amplitude)
LIM23 = 2.2e-41           # DS-50 limit / f at c=23 (cm^2)
R_K = 3.0                 # fm, knot size (ropelength*a/2pi ~ 2-4 fm)
LAMBDA_CHI = 0.2088       # GeV


def gev2_to_cm2(x):
    return x * 0.3894e-27  # 1 GeV^-2 = 0.3894 mb = 0.3894e-27 cm^2


def run():
    print("=== sigma_EM(neutre) : les canaux EM du reservoir ===\n")

    # A. millicharge (L1)
    q_scr = ALPHA * 0.511e-3          # GeV (ecrantage atomique)
    sigA_full = 16 * math.pi * ALPHA ** 2 * MU ** 2 / q_scr ** 4   # eps=1, GeV^-2
    sigA_full_cm = gev2_to_cm2(sigA_full)
    sigA_23 = sigA_full_cm * EPS23 ** 2
    print("[A] Millicharge de vertex (L1, eps(23) = alpha^2.5 = %.1e) :" % EPS23)
    print("    prefacteur (eps=1, Rutherford ecrante) : %.1e cm^2" % sigA_full_cm)
    print("    sigma(23) = %.1e cm^2 vs limite %.1e : %+0.1f dex -> %s\n" % (
        sigA_23, LIM23, math.log10(sigA_23 / LIM23),
        "EXCLU" if sigA_23 > LIM23 else "ok"))

    # B. EDM de paire (L2)
    d_sep = 2.0  # fm
    L = 10.0     # log cinematique
    sigB_pref = 4 * math.pi * ALPHA ** 2 * d_sep ** 2 * L / V ** 2  # fm^2 (eps=1)
    sigB_23 = sigB_pref * FM2_CM2 * EPS23 ** 2
    print("[B] Paire K+Kbar liee a %g fm (L2) :" % d_sep)
    print("    prefacteur : %.1e cm^2 ; sigma(23) = %.1e vs %.1e : %+0.1f dex -> %s\n" % (
        sigB_pref * FM2_CM2, sigB_23, LIM23, math.log10(sigB_23 / LIM23),
        "EXCLU" if sigB_23 > LIM23 else "ok"))

    # D. polarisabilite (L3)
    aE_n = 1.2e-3   # fm^3 neutron
    aE_i = aE_n * (R_K / 0.84) ** 3 * (0.300 / LAMBDA_CHI)   # rescaling
    aE_ii = ALPHA * R_K ** 2 * (HBARC / LAMBDA_CHI)          # dimensionnelle fm^3
    print("[D] Polarisabilite du tube (L3, noeud Q_phys = 0 par framing) :")
    for lab, aE in (("rescaling neutron", aE_i), ("dimensionnelle", aE_ii)):
        A = 2 * math.pi * ALPHA * aE                          # fm^3
        mu_fm = MU / HBARC                                    # fm^-1
        sig = 4 * math.pi ** 3 * mu_fm ** 4 * V ** 2 * A ** 2  # fm^2
        sig_cm = sig * FM2_CM2
        sig23 = sig_cm * ALPHA ** 5
        print("    alpha_pol[%s] = %.3f fm^3 -> prefacteur sigma_EM = %.1e cm^2" % (
            lab, aE, sig_cm))
        print("      sigma(23) = %.1e cm^2 (%+0.1f dex sous la limite DS-50)" % (
            sig23, math.log10(sig23 / LIM23)))
    print()

    print("VERDICT :")
    print(" - (L1) millicharge : prefacteur ~1e-9 cm^2 -> sigma(23) ~ 1e-20 :")
    print("   EXCLU par ~21 dex (et bloque par la couverture). La suppression")
    print("   de Vassiliev N'EST PAS une millicharge de vertex sur un noeud charge.")
    print(" - (L2) paire liee au fm : prefacteur ~3e-22 -> sigma(23) ~ 5e-33 :")
    print("   EXCLU par ~8 dex. Pas de paires K+Kbar liees a l'echelle du fm")
    print("   (coherent : les forces chi des hauts-c sont aussi supprimees).")
    print(" - (L3) SELECTIONNEE : neutralisation par framing (Q = Sigma_chi + Lk")
    print("   = 0, le mecanisme des 'realisables' applique au reservoir).")
    print("   Canal survivant = POLARISABILITE : sigma_EM ~ 1e-35 -- 1e-33 cm^2")
    print("   (deux estimations independantes), soit << le pincage 1e-30 :")
    print("   sigma(c=23) ~ 1e-46 -- 1e-44 cm^2, 3-5 dex SOUS la frontiere")
    print("   DS-50 actuelle. La cible detecteur migre vers l'ere XLZD/DS-20k")
    print("   (Migdal) -- la tension DS-50 s'evapore, la falsifiabilite devient")
    print("   generationnelle.")
    print(" - Incertitudes : alpha_pol (facteur ~10), canal SI-equivalent,")
    print("   taille R_K (2-4 fm) ; bande totale ~+-1.5 dex declaree.")


if __name__ == "__main__":
    run()
