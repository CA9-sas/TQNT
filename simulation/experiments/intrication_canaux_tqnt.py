"""Intrication quantique en TQNT : departage des canaux porteurs (20/07).

INTUITION TESTEE (utilisateur) : l'intrication = mise en phase des
particules par la CGQ ou la structure chi. Trois canaux candidats pour
le VEHICULE 3D de la correlation, chiffres :

 (a) CGQ (puits gravitationnel quantique) : couplage de phase entre les
     deux particules via leurs puits Omega^2-1 ~ kappa_g|Psi|^2.
 (b) TUBE DE FLUX chi tendu entre les particules : porte la tension
     sigma_chi -> cout energetique proportionnel a la distance.
 (c) COHERENCE DE PHASE DU CONDENSAT chi : les deux noeuds nes du meme
     paquet partagent la phase du fond (comme la phase d'un condensat
     BCS sur tout l'echantillon) -> cout NUL, independant de la
     distance ; la structure TOPOLOGIQUE (enlacement des lignes de
     phase) encode le lien discret.

CONTRAINTES a satisfaire : correlations a 1200 km (Micius) ; violation
de Bell (une phase commune CLASSIQUE est une variable cachee locale,
CHSH <= 2 : insuffisante seule) ; monogamie ; decoherence par
environnement.
"""
import math

KAPPA_G = 3.1201e-53        # m (nucleonique, derive)
HBARC = 197.3269804e-15     # MeV.m
M_E = 0.511                 # MeV
SIGMA_CHI = 418.47 ** 2 / 197.3269804   # MeV/fm (tension de corde)
D_BELL = 1.2e6              # m (test satellite, 1200 km)
D_LAB = 1.0                 # m


def run():
    print("=== Intrication : departage des canaux TQNT ===\n")

    # (a) CGQ : profondeur de puits d'un electron a distance d
    print("(a) CGQ (mise en phase gravitationnelle) :")
    for lab, d in (("1 m", D_LAB), ("1200 km", D_BELL)):
        # h = 2 G m/(d c^2) = kappa_g(e) / (4 pi d) x ... ordre : h ~ r_s(e)/d
        G = 6.6710e-11
        m_kg = M_E * 1.7826619e-30
        h = 2 * G * m_kg / (d * 2.99792458e8 ** 2)
        # taux de mise en phase ~ h x omega_Compton ; temps pour dephasage O(1)
        omega_c = M_E * 1e6 * 1.602e-19 / 1.0546e-34   # rad/s
        t_phase = 1.0 / (h * omega_c)
        print("  d = %-8s : h = %.1e ; temps de mise en phase ~ %.1e s (%.0e ages de l'univers)" % (
            lab, h, t_phase, t_phase / 4.4e17))
    print("  -> MORT (~35-40 ordres) : la CGQ ne peut pas etre le vehicule ;")
    print("     et une phase commune CLASSIQUE serait de toute facon bornee")
    print("     par Bell (CHSH <= 2). La moitie << CGQ >> de l'intuition est")
    print("     refutee.\n")

    # (b) tube de flux tendu
    print("(b) Tube de flux chi tendu entre les particules :")
    for lab, d in (("1 m", D_LAB), ("1200 km", D_BELL)):
        E = SIGMA_CHI * (d / 1e-15)      # MeV
        print("  d = %-8s : E_tube = sigma_chi x d = %.1e MeV = %.1e J" % (
            lab, E, E * 1.602e-13))
    print("  -> MORT : un tube CONFINE (tension) sur des distances")
    print("     macroscopiques coute une energie astronomique ; l'intrication")
    print("     mesuree ne transporte aucune energie de ce type.\n")

    # (c) coherence de phase du condensat
    print("(c) Coherence de phase du condensat chi :")
    print("  cout energetique : NUL (la phase globale d'un condensat est un")
    print("  mode de Goldstone -- comme la phase BCS sur tout l'echantillon) ;")
    print("  independance a la distance : STRUCTURELLE (la phase n'est pas")
    print("  un objet local) ; c'est la moitie << structure chi >> de")
    print("  l'intuition, au niveau CONDENSAT (pas CGQ, pas tube tendu).")
    print("  Le lien DISCRET est porte par la topologie : l'enlacement des")
    print("  lignes de phase de la paire nucleee (Sigma chi = 0, porte Z_3).\n")

    # classification topologique <-> classes d'intrication
    print("Classification (litt. Aravind 1997, Kauffman-Lomonaco ; en TQNT")
    print("l'analogie devient ontologie) :")
    print("  paire EPR (2 qubits)     <-> enlacement de Hopf (Lk = 1)")
    print("  GHZ (3 qubits)           <-> anneaux borromeens : Lk par paires")
    print("      = 0, invariant TRIPLE de Milnor non nul -- tracer 1 qubit")
    print("      laisse une paire SEPARABLE = couper 1 anneau libere tout")
    print("  W (3 qubits)             <-> chaine de Hopf : Lk par paires non")
    print("      nuls -- tracer 1 qubit laisse une paire INTRIQUEE")
    print("  monogamie                <-> budget d'enlacement du tube\n")

    print("VERDICT :")
    print(" - vehicule viable = (c) coherence de phase du condensat +")
    print("   enlacement topologique (le lien est une propriete du COUPLE,")
    print("   non factorisable -- l'analogue structurel exact de l'etat")
    print("   intrique) ;")
    print(" - la violation de Bell est HERITEE : la TQNT derive Schrodinger")
    print("   (P1-P10), les correlations cos^2 suivent des que l'etat joint")
    print("   est decrit par la fonction d'onde chi commune ; l'apport TQNT")
    print("   est l'ONTOLOGIE 3D (topologie + phase), pas de nouvelles")
    print("   correlations ;")
    print(" - decoherence = reconnexion du lien avec l'environnement : le")
    print("   MEME Gamma que la dissipation (unification interne, cf. la")
    print("   formule maitresse de la dissipation) ;")
    print(" - swap d'intrication = reconnexion P15 aux noeuds intermediaires")
    print("   (le lien topologique se recolle de bout en bout).")


if __name__ == "__main__":
    run()
