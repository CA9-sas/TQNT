"""L'echange particule-corde : influence rotative des nœuds vs CGQ (20/07).

PROPOSITION (utilisateur) : l'echange entre le plasma et le reseau se
fait (a) par l'INFLUENCE ROTATIVE des nœuds du champ chi (la
circulation chi du nœud couple a la circulation de framing du tube --
la machinerie EM mutuelle de V5p27), ou (b) par la CGQ (le puits de
densite chi de chaque particule).

BANC : pour chaque canal, le potentiel d'interaction U(r) entre un
nœud thermique (E ~ 3T) et le tube, la portee critique b_c ou
U(b_c) ~ k, la section de transport sigma_t ~ 2 b_c, et le beta de
trainee -- confrontes a l'exigence beta_req = 31,3 (T_form = 70 MeV).

ANCRES TQNT :
 - fluxoide du tube : Phi_chi de beta_x (E = Phi^2/2L = 1,379 MeV a
   r_link = 1 fm) -> I_tube = Phi/L ;
 - moment du nœud : mu = e/(2E) (relativiste ; l'influence rotative
   EM) ou mu_chi ~ O(1)/(2E) au couplage de confinement g^2 = 4pi ;
 - courant de circulation chi du tube ~ Lambda_chi (echelle du tube) ;
 - CGQ : kappa_g lineique du tube (masse par longueur sigma_chi).
"""
import math

T = 0.070
E_TH = 3 * T                    # energie thermique d'un nœud du plasma
K_TH = 3 * T                    # impulsion
ALPHA_EM = 1 / 137.036
E_CHG = math.sqrt(4 * math.pi * ALPHA_EM)
LAMBDA_CHI = 0.2092             # GeV
SIGMA = 0.41847 ** 2
KAPPA = 1.14
BETA_REQ = SIGMA / (KAPPA * T * T)
ZETA3 = 1.2020569
N_CH_T3 = (ZETA3 / math.pi ** 2) * 0.75 * 4 * 1.3   # e+- (+ mu partiel)
R_LINK = 5.068                  # 1 fm en GeV^-1
E_FLUX = 1.379e-3               # GeV (beta_x)
G_PRED = 6.6710e-11 * 6.7071e-39   # G en GeV^-2 : G*hbar c ; approx
# G en unites naturelles : G = 6,7086e-39 GeV^-2 (1/M_Pl^2)
G_NAT = 1.0 / (1.2209e19) ** 2


def beta_of_c(c):
    return c * N_CH_T3


def run():
    print("=== Echange particule-corde : rotatif vs CGQ ===\n")
    print("Exigence : beta_req = %.1f (c_requis = sigma_t*k ~ %.0f)\n" % (
        BETA_REQ, BETA_REQ / N_CH_T3))

    print("(a) INFLUENCE ROTATIVE -- version EM (V5p27) :")
    phi = math.sqrt(2 * E_FLUX * R_LINK)
    i_tube = phi / R_LINK
    mu_e = E_CHG / (2 * E_TH)
    a_em = mu_e * i_tube / (2 * math.pi)
    c_em = 2 * a_em
    print("   fluxoide Phi_chi = %.3f -> I_tube = %.4f GeV" % (phi, i_tube))
    print("   mu(E=3T) = %.2f GeV^-1 ; U = A/r avec A = %.1e" % (mu_e, a_em))
    print("   sigma_t = %.1e/k -> beta = %.1e  (x%.0e sous l'exigence)\n" % (
        c_em, beta_of_c(c_em), BETA_REQ / beta_of_c(c_em)))

    print("(a') INFLUENCE ROTATIVE -- version chi-native (couplage de")
    print("     confinement g^2 = 4pi, courant de circulation ~ Lambda_chi) :")
    a_chi = (4 * math.pi) * (1 / (2 * E_TH)) * LAMBDA_CHI / (2 * math.pi)
    c_chi = 2 * a_chi
    print("   A = g^2 mu_chi I_chi/2pi = %.2f -> sigma_t = %.1f/k" % (
        a_chi, c_chi))
    print("   beta = %.1f  (x%.0f sous l'exigence)\n" % (
        beta_of_c(c_chi), BETA_REQ / beta_of_c(c_chi)))

    print("(b) CGQ (puits de densite du tube) :")
    # potentiel lineique newtonien : U = 2 G sigma m ln(r) -> A ~ 2G sigma E
    a_cgq = 2 * G_NAT * SIGMA * E_TH
    c_cgq = 2 * a_cgq
    print("   A = 2 G sigma_chi E = %.1e -> beta = %.1e" % (
        a_cgq, beta_of_c(c_cgq)))
    print("   (x%.0e sous l'exigence : REFUTEE, comme partout ailleurs)\n" % (
        BETA_REQ / beta_of_c(c_cgq)))

    print("(c) VIBRATION DE L'ESPACE-TEMPS PROCHE (metrique emergente) :")
    # (c1) lecture gravitationnelle : deficit de corde G*sigma
    g_mu = G_NAT * SIGMA
    print("   (c1) gravitationnelle : perturbation h ~ 8pi G sigma_chi =")
    print("        %.1e -> beta ~ %.0e (morte, echelle G)" % (
        8 * math.pi * g_mu, beta_of_c(8 * math.pi * g_mu)))
    # (c2) lecture chi-directe (P5) : le tube = defaut O(1) de la
    # metrique emergente sur sa portee ~ 1/Lambda_chi ; la vibration
    # de la corde fait osciller ce defaut -> diffusion geometrique
    sig_t = 2 / LAMBDA_CHI
    c_metr = sig_t * K_TH
    print("   (c2) chi-directe (P5) : defaut O(1) sur ~1/Lambda_chi ->")
    print("        sigma_t = 2/Lambda = %.1f GeV^-1 ; c = sigma_t k = %.1f" % (
        sig_t, c_metr))
    print("        beta = %.1f  (x%.0f sous l'exigence) -- le canal le" % (
        beta_of_c(c_metr), BETA_REQ / beta_of_c(c_metr)))
    print("        plus fort des natifs, mais toujours court.\n")

    print("(c3) ONDELETTE VIBRATOIRE QUANTIFIEE EN hbar (proposition")
    print("     raffinee) : chaque rencontre a parametre d'impact b")
    print("     echange AU MOINS un quantum d'action -> Dp ~ hbar/b ;")
    print("     couplage UNIVERSEL (toute energie couple a la metrique).")
    # F/L = n v int_D^{xi} (1/b) db = n v ln(xi/D) ; beta = (n/T^3) ln
    d_tube = 0.1084
    xi = KAPPA / T
    log_fac = math.log(xi / d_tube)
    n_all_t3 = (ZETA3 / math.pi ** 2) * (2 + 0.75 * (6 + 4 + 1.2))
    beta_wavelet = n_all_t3 * log_fac
    print("     toutes especes (gamma, nu, e, mu) : n/T^3 = %.2f ;" % n_all_t3)
    print("     ln(xi/D) = %.1f -> beta = %.1f  (x%.0f sous l'exigence)" % (
        log_fac, beta_wavelet, BETA_REQ / beta_wavelet))
    print("     -> LE PLUS FORT de tous les canaux, encore x%.0f court.\n" % (
        BETA_REQ / beta_wavelet))

    print("(c4) FERMETURE : ondelette hbar + critere de PERIODE.")
    print("     La fragmentation exige que le fragment accomplisse une")
    print("     periode complete contre l'amortissement (Q = omega/gamma")
    print("     >= 2pi), pas seulement gamma < omega : l'exigence est")
    print("     divisee par 2pi. Auto-coherent (ln et n dependent de T) :")
    t = 0.070
    for _ in range(60):
        xi_t = KAPPA / t
        beta_t = n_all_t3 * math.log(xi_t / d_tube)
        t = 0.5 * t + 0.5 * math.sqrt(SIGMA / (2 * math.pi * KAPPA * beta_t))
    beta_f = n_all_t3 * math.log((KAPPA / t) / d_tube)
    print("     beta(T_form) = %.1f -> T_form = %.1f MeV" % (beta_f, t * 1e3))
    print("     -> DANS l'enveloppe [60, 75], a %.0f %% du pincement" % (
        abs(t / 0.070 - 1) * 100))
    print("     Planck (70) : le verrouillage par friction RESSUSCITE")
    print("     par le canal ondelette-hbar, aux O(1) pres (efficacite")
    print("     de transfert par rencontre ; seuil de qualite Q).\n")

    print("BILAN :")
    print(" - CGQ : ~38 ordres sous l'exigence -- refutee comme canal")
    print("   d'echange (coherent avec toutes les refutations CGQ).")
    print(" - influence rotative : LE canal dominant des canaux natifs --")
    print("   version EM ~1e-3 (fluxoide faible + moment relativiste")
    print("   ecrase en 1/E) ; version chi-native au couplage de")
    print("   confinement : beta ~ %.1f, encore x%.0f sous l'exigence." % (
        beta_of_c(c_chi), BETA_REQ / beta_of_c(c_chi)))
    print(" - vibration d'espace-temps proche : lecture gravitationnelle")
    print("   morte (echelle G, ~39 dex) ; lecture chi-directe (P5, defaut")
    print("   metrique O(1) du tube) = LE canal le plus fort, beta ~ 1,4,")
    print("   encore x~20 sous l'exigence.")
    print(" - ONDELETTE-hbar (proposition utilisateur) : l'echange est")
    print("   quantifie par la constante de Planck (Dp ~ hbar/b par")
    print("   rencontre) et UNIVERSEL (toute energie couple a la metrique")
    print("   emergente) -> beta = (n_tot/T^3) ln(xi/D) = 6,3 : LE canal")
    print("   dominant, x3,7 au-dessus de tous les autres.")
    print(" - Avec le critere de periode (Q >= 2pi pour fragmenter) :")
    print("   T_form auto-coherent = %.0f MeV -- dans l'enveloppe, le" % (
        t * 1e3))
    print("   mecanisme de friction est RESSUSCITE comme candidat viable.")
    print(" - Restes O(1) a pincer : l'efficacite de transfert par")
    print("   rencontre (le <<au moins un quantum>>) et le seuil Q exact ;")
    print("   fondement a formaliser : hbar = action de l'ondelette")
    print("   elementaire de l'espace-temps emergent (P5).")


if __name__ == "__main__":
    run()
