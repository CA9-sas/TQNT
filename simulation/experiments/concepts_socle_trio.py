"""Trois concepts du socle a revoir (brouillon, 21/07).

[1] LE TEMPS = L'INVERSE DE LA FREQUENCE DU SOCLE.
    Definition : l'horloge locale compte les periodes du socle ; pres
    d'une particule les sommets sont PLUS ECARTES (la deplietion S4'
    etire les dents) -> moins de tics par temps de coordonnee -> le
    temps ralentit. TEST QUANTITATIF : l'etirement relatif requis pour
    reproduire la dilatation gravitationnelle est delta-f/f = Phi/c^2 ;
    au sol terrestre ~7e-10, et la difference sol/orbite GPS doit
    redonner les ~45,7 microsecondes/jour (partie gravitationnelle) --
    LE test d'horloge le plus mesure au monde. Coherence S4' : la
    fraction d'etirement = la fraction de deficit (meme source,
    bi-conformite P5 : Omega^2 est le facteur des deux lectures).

[2] LE SPIN PAR LES SOMMETS VOISINS (particule partiellement visible).
    Le socle ne transmet qu'aux sommets : la rotation d'un noeud n'est
    ECHANTILLONNEE qu'aux sommets voisins -> le TAUX de rotation est
    inobservable (exactement le document principal : spin = circulation chi
    quantifiee, taux inobservable -- le socle EXPLIQUE pourquoi).
    Ce qui est lisible dans l'echantillonnage : la CLASSE topologique
    du repere -- le doublet 4pi. << Partiellement visible >> = a 2pi la
    particule est dans le feuillet -1 du double revetement : visible en
    module, INVERSEE en phase (interferometrie neutronique 4pi, document principal).
    TEST : la visibilite d'interference V(theta) = cos(theta/2) -- nulle
    a 2pi (feuillet oppose), retablie a 4pi. HONNETE : la valeur de j
    ne se calcule PAS des sommets (spin = arite de tresse, topologique).

[3] G PAR COMPENSATION AVEC N'IMPORTE QUELLE PARTICULE.
    Oui -- et c'est un THEOREME du socle : G = kappa_g(K) c^2/(8 pi m_K)
    avec kappa_g(K) proportionnel a m_K (document principal : kappa_g varie, G
    invariant) -> G est LE MEME pour toute particule, et se reduit a
    l'identite SANS ANCRE : G = hbar c / M_Pl^2.
    Lecture socle : UN seul substrat, UNE amplitude de reference ->
    UN taux de conversion deficit->metrique pour tous : l'universalite
    de la chute libre (principe d'equivalence) = l'unicite du socle.
    Le nucleon n'est PAS special pour G ; il est special pour la
    DERIVATION de M_Pl : seule particule a position ENTIERE sur
    l'echelle par croisement (17+1 -> m_base/M_Pl = sqrt(12/7) alpha^18).
    Toute particule marche en principe ; il faudrait sa position
    d'echelle derivee (l'electron passe par Koide + sqrt(sigma_chi),
    eux-memes ancres via le nucleon). Verification numerique : G depuis
    e, mu, p identiques (machine) ; G = hbar c/M_Pl^2 ; l'exclusion
    canonique du << G par particule >> (x14 ordres) reste la borne
    observationnelle.
"""
import numpy as np

C = 299792458.0
HBAR = 1.054571817e-34
G_NEWTON = 6.6743e-11        # CODATA (comparaison)
G_PRED = 6.6710e-11          # document principal
M_PL = 2.176434e-8           # kg
M_EARTH = 5.9722e24
R_EARTH = 6.371e6
R_GPS = 2.6571e7             # rayon orbital GPS (m)
KAPPA_G = 3.1201e-53         # m (document principal, ancre nucleon)
M_BASE = 938.240e6 * 1.602176634e-19 / C**2


def run():
    print(__doc__)
    print("=" * 72)
    print("[1] TEMPS = 1/f_socle : le test GPS")
    print("=" * 72)
    phi_surf = G_NEWTON * M_EARTH / (R_EARTH * C * C)
    phi_gps = G_NEWTON * M_EARTH / (R_GPS * C * C)
    print("  etirement des dents requis (delta-f/f = Phi/c^2) :")
    print("    au sol      : %.3e" % phi_surf)
    print("    orbite GPS  : %.3e" % phi_gps)
    dt_day = (phi_surf - phi_gps) * 86400e6
    print("  difference sol/orbite x 1 jour = %.1f microsecondes/jour" % dt_day)
    print("  (valeur standard de la partie gravitationnelle GPS : ~45,7)")
    print("  -> definir le temps comme l'inverse de la frequence du socle,")
    print("     avec l'etirement S4' des dents, REPRODUIT la dilatation")
    print("     gravitationnelle mesuree (l'horloge la plus testee au monde).")

    print("\n" + "=" * 72)
    print("[2] SPIN AUX SOMMETS : visibilite 4pi (le taux reste cache)")
    print("=" * 72)
    for theta in [0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi]:
        v = np.cos(theta / 2)
        print("  rotation %5.2f pi : amplitude relative = %+.2f  %s"
              % (theta / np.pi, v,
                 "(feuillet -1 : partiellement visible)" if v < 0
                 else "(visible)" if abs(v) > 0.5 else "(eteinte)"))
    print("  -> periodicite 4pi (interferometrie neutronique, document principal) ;")
    print("     l'echantillonnage aux sommets rend le TAUX inobservable")
    print("     (document principal : spin = circulation quantifiee) et ne laisse lire")
    print("     que la classe du repere. La VALEUR de j reste topologique")
    print("     (arite de tresse) : elle ne se calcule pas des sommets.")

    print("\n" + "=" * 72)
    print("[2bis] LA VALEUR DE J EST LISIBLE -- par la MASSE (Regge inverse)")
    print("=" * 72)
    # Precision utilisateur : la rotation est calculee pour definir la
    # masse (terme de Regge M^2 = M0^2 + 2 pi sigma_chi J, document principal) -- donc
    # le socle, qui lit les frequences (E = hf), lit J via la masse.
    # Inversion sur la trajectoire du rho (PDG) avec la pente DERIVEE :
    sqrt_sigma = 0.41847                      # GeV (document principal)
    slope = 2 * np.pi * sqrt_sigma**2         # 2 pi sigma_chi = 1.100 GeV^2
    traj = [("rho(775)", 0.77526, 1), ("a2(1318)", 1.3182, 2),
            ("rho3(1689)", 1.6888, 3), ("a4(1967)", 1.967, 4)]
    m0sq = traj[0][1]**2 - slope * traj[0][2]
    print("  pente derivee 2 pi sigma_chi = %.4f GeV^2 ; lecture" % slope)
    print("  j = (M^2 - M0^2)/(2 pi sigma_chi) sur la trajectoire du rho :")
    for name, m, j_true in traj:
        j_read = (m * m - m0sq) / slope
        print("    %-11s M = %.3f GeV -> j lu = %.2f  (vrai : %d)"
              % (name, m, j_read, j_true))
    print("  -> le socle LIT j : pas de la geometrie d'echantillonnage,")
    print("     mais de l'ENERGIE de rotation encodee dans la masse")
    print("     (masse = frequence confinee, rotation comprise). Le taux")
    print("     omega reste inobservable -- la distinction fine du document principal")
    print("     << toutes les lois de masse utilisent J, jamais omega >>")
    print("     devient un theoreme de lecture du socle. (Baryons : meme")
    print("     lecture, + tenseur S(S+1) Lambda_chi/8 a bas J, document principal.)")

    print("\n" + "=" * 72)
    print("[3] G PAR TOUTE PARTICULE : l'identite sans ancre")
    print("=" * 72)
    parts = [("electron", 9.1093837015e-31), ("muon", 1.883531627e-28),
             ("nucleon", 1.67262192e-27)]
    for name, m_k in parts:
        kappa_k = KAPPA_G * (m_k / M_BASE)      # kappa_g(K) prop. m_K (document principal)
        g_k = kappa_k * C * C / (8 * np.pi * m_k)
        print("  %-9s : kappa_g(K) = %.3e m -> G = %.5e" % (name, kappa_k, g_k))
    g_ident = HBAR * C / (M_PL * M_PL * G_NEWTON) * G_NEWTON  # = hbar c/M_Pl^2
    g_ident = HBAR * C / (M_PL * M_PL)
    print("  identite sans ancre : G = hbar c / M_Pl^2 = %.5e" % g_ident)
    print("  (CODATA %.4e ; predit document principal %.4e)" % (G_NEWTON, G_PRED))
    print("  -> G est INDEPENDANT de la particule (kappa_g prop. m_K, document principal) :")
    print("     un seul socle, une seule amplitude de reference, un seul taux")
    print("     de conversion deficit->metrique = principe d'EQUIVALENCE.")
    print("     Le nucleon n'est special que pour DERIVER M_Pl (position")
    print("     ENTIERE 17+1 sur l'echelle par croisement) ; en principe")
    print("     toute particule marche, si sa position d'echelle est derivee.")


def section4():
    print("\n" + "=" * 72)
    print("[4] LA MASSE PAR LA MEME LOGIQUE : ecrasement, torsion, compensation")
    print("=" * 72)
    # (a) GARDE-FOU : h lui-meme est l'INVARIANT du quatuor (jamais
    #     ecrase -- c'est la graduation en action qui interdit l'ether).
    #     Ce qui s'ecrase : l'ENERGIE DE LA DENT E_dent = h f_local.
    #     L'ecrasement relatif = delta-f/f = Phi/c^2 -- VARIABLE (par
    #     particule ET par distance), calculable :
    m_e, m_mu = 9.1093837015e-31, 1.883531627e-28
    m_p, r_p = 1.67262192e-27, 0.841e-15
    lam = lambda m: HBAR / (m * C)
    def ecrasement(m, r):
        return (m / M_PL) * (1.616255e-35 / r)
    rows = [
        ("electron (a lambdabar_C)", ecrasement(m_e, lam(m_e))),
        ("muon (a lambdabar_C)", ecrasement(m_mu, lam(m_mu))),
        ("nucleon (a r_p)", ecrasement(m_p, r_p)),
        ("Terre (surface)", G_NEWTON * M_EARTH / (R_EARTH * C * C)),
        ("etoile a neutrons (1,4 Msol, 12 km)",
         G_NEWTON * 1.4 * 1.989e30 / (1.2e4 * C * C)),
        ("trou noir (horizon)", 0.5),
    ]
    print("  ecrasement des dents (delta-f/f = Phi/c^2), VARIABLE :")
    for name, e in rows:
        print("    %-36s %.2e" % (name, e))
    print("  particule a sa Compton : ecrasement = (m/M_Pl)^2 -- le CARRE")
    print("  de la fraction confisquee ; limite : ecrasement -> 1/2 a")
    print("  l'horizon => f_local -> 0 : les dents s'arretent -- LE TROU")
    print("  NOIR = ecrasement total = << etoile a noeuds GELES >> (document principal).")

    # (b) la torsion des sommets voisins VERS la particule = le gradient
    #     de l'ecrasement = le CGQ ; par dent :
    g_terre = G_NEWTON * M_EARTH / R_EARTH**2
    tilt = g_terre * 1.616255e-35 / C**2
    g_ns = G_NEWTON * 1.4 * 1.989e30 / (1.2e4)**2
    tilt_ns = g_ns * 1.616255e-35 / C**2
    print("\n  torsion differentielle par dent (g l_P/c^2) :")
    print("    surface de la Terre        : %.1e / dent" % tilt)
    print("    surface d'etoile a neutrons: %.1e / dent" % tilt_ns)
    print("  -> la torsion vers la particule = grad(ecrasement) ; g = c^2 x")
    print("     grad(delta-f/f) redonne Newton g = GM/r^2 (identite) ; la")
    print("     chute libre = la somme des micro-torsions sur les ~1e20")
    print("     dents que la particule enjambe.")

    # (c) l'augmentation des sommets autour SELON LES NOEUDS
    print("\n  compensation autour prop. a la masse : bulge(r) = (m/M_Pl)(l_P/r)")
    print("  -- et m = sqrt(sigma_chi) S selon le NOEUD (lois de secteur) :")
    print("  le motif des sommets autour encode QUEL noeud. Avec [2bis] :")
    print("  le socle autour d'une particule porte (i) sa MASSE (amplitude")
    print("  de compensation), (ii) son J (Regge dans la frequence),")
    print("  (iii) sa parite (4pi) -- la carte d'identite complete, lue")
    print("  des sommets. h reste h : seul E_dent = h f_local varie.")


if __name__ == "__main__":
    run()
    section4()
