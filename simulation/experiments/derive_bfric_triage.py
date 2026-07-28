"""Deriver b_fric -- triage sous l'ontologie CORRIGEE (20/07).

CORRECTIONS (utilisateur) : les quanta du plasma ne sont PAS des ondes
chi, et il n'y a PAS de phonons en TQNT. Le plasma a 70-155 MeV est
fait de PARTICULES TQNT : photons (quanta de jauge), neutrinos
(inertes), e+-/mu+- (nœuds 3_1), pions (paires K+Kbar). Les candidats
fondes sur la diffusion d'ondes du condensat (reflecteur sur lambda,
unitarite d'ondes partielles, son-vortex/Iordanskii, << b = 2
polarisations >>) sont TOUS MORTS ontologiquement.

CANAUX TQNT-NATIFS de trainee sur le tube :
 (1) Aharonov-Bohm EM des charges sur le flux du tube (le tube porte
     la circulation de framing) : sigma_t = (2/k) sin^2(pi nu) ;
 (2) reconnexion P15 des nœuds : empreinte = lambda(E) = hbar c/E
     (cadre collision TQNT : << la collision reduit lambda jusqu'a
     hbar c/sqrt(s) >>) -> sigma_t ~ (2/k) P_reco ;
 (3) photons : polarisabilite du tube (Rayleigh-like) ;
 (4) neutrinos : inertes (pas de charge, pas de topologie).

EXIGENCE : beta_req = sigma_chi/(kappa_dom T^2) = 31,3 a T = 70 MeV.
"""
import math

SIGMA = 0.41847 ** 2
KAPPA = 1.14
T = 0.070
ZETA3 = 1.2020569
BETA_REQ = SIGMA / (KAPPA * T * T)


def n_over_t3(g_b, g_f):
    return (ZETA3 / math.pi ** 2) * (g_b + 0.75 * g_f)


def run():
    print("=== b_fric sous l'ontologie corrigee : canaux de particules ===\n")
    print("Exigence a T = 70 MeV : beta_req = %.1f\n" % BETA_REQ)

    print("CANAUX (sigma_t = c/k -> beta = c x n/T^3) :")
    # (1) AB-EM : charges e+- (relativistes), mu/pi supprimes
    n_ch = n_over_t3(0, 4) + 0.3 * n_over_t3(0, 4)   # e+- plein, mu ~30 %
    b_ab = 2.0 * n_ch   # sin^2 maximal
    print(" (1) AB-EM charges (nu maximal)     : beta <= %.1f" % b_ab)
    # (2) reconnexion P15 : e+- (nœuds), empreinte 1/E -> c ~ 2 P_reco
    b_p15 = 2.0 * 1.0 * n_over_t3(0, 4)
    print(" (2) reconnexion P15 (P_reco = 1)   : beta ~  %.1f" % b_p15)
    # (3) photons : polarisabilite -> tres sous-dominant
    print(" (3) photons (polarisabilite)       : beta << 1")
    print(" (4) neutrinos                      : 0")
    b_tot = b_ab + b_p15
    print("   TOTAL beta_TQNT ~ %.1f  ->  x%.0f SOUS l'exigence" % (
        b_tot, BETA_REQ / b_tot))

    t_release = math.sqrt(SIGMA / (b_tot * KAPPA))
    print("\nCONSEQUENCE : liberation a T = %.0f MeV >= T_trans :" % (
        t_release * 1e3))
    print("le reseau n'est JAMAIS verrouille par friction -- le")
    print("mecanisme << verrouillage par friction / liberation a 70 MeV >>")
    print("NE TIENT PAS sous l'ontologie corrigee (les candidats en")
    print("~1/k plafonnent a beta ~ 1-3 ; il faudrait x10-30).\n")

    print("VERDICT :")
    print(" - beta_fric = 2 g_s (2,06) : REFUTE -- il presupposait une")
    print("   trainee d'ondes du condensat qui n'existe pas en TQNT ;")
    print(" - le mecanisme de T_form est ROUVERT : la friction des")
    print("   particules ne verrouille pas le reseau ; T_form retombe")
    print("   vers l'epoque de condensation, et le gel echelonne y")
    print("   donnerait Omega >> Planck -- il manque le vrai mecanisme")
    print("   de retard/verrouillage entre la condensation et ~70 MeV.")
    print(" - Pistes TQNT-natives a examiner (non tranchees ici) :")
    print("   (i) structure hybride de la trialite Z_3 (les trois vides")
    print("       de la porte -> parois/tensions entre domaines tirant")
    print("       les cordes ; verrouillage geometrique, pas cinetique) ;")
    print("   (ii) developpement de sigma(T) sous la transition (le")
    print("       reseau est mou tant que la tension n'est pas etablie) ;")
    print("   (iii) le taux de nucleation des nœuds lui-meme (P22/P15)")
    print("       comme horloge de la fragmentation.")


if __name__ == "__main__":
    run()
