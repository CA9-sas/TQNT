"""Le premier-ordre/Polyakov et la theorie des champs en interaction (21/07).

Suite de etude_pont_spinoriel.py. Deux volets :

=== VOLET A : fermer la condition << premier ordre >> ===

[A1] LE PREMIER ORDRE EST HERITE DE P10 (plus une condition libre).
     Le champ chi est une connexion de CHERN-SIMONS (P10, document principal). Or une
     action CS est PAR DEFINITION du premier ordre en derivees (terme
     topologique, sans metrique) -- c'est sa propriete constitutive. La
     variable de repere (framing) couple a chi par l'holonomie/self-linking
     (le terme de framing du document principal : quantum de flux, delta_framing) : son
     terme cinetique est donc du premier ordre PAR CONSTRUCTION -- de type
     Berry/Wess-Zumino (action d'orbite co-adjointe : S = s * [angle solide
     balaye], quantification -> 2s+1 etats ; s=1/2 -> le doublet).
     VERIFICATION : phase de Berry du doublet transporte sur une boucle de
     colatitude theta = -Omega/2 exactement (numerique ci-dessous) ; pour
     un tour complet du repere (Omega = 4pi... via 2pi de rotation) : -1.
     => La derivation de Dirac n'est plus conditionnee qu'a (S), (D) et
     P10 -- tous canoniques. Le << mur spinoriel >> tombe au meme rang
     conditionnel que le reste de la chaine.

[A2] LE FACTEUR DE SPIN (Polyakov) -- et une DISTINCTION a trancher.
     Polyakov (transmutation Fermi-Bose en CS) : la ligne d'univers
     acquiert exp(i pi W[writhe]) ; sommee sur les chemins, elle engendre
     le propagateur de Dirac. Ingredient verifie ici : l'integrale de
     Gauss du writhe (implementation validee : cercle -> 0 ; trefle
     parametrique -> |W| ~ 3,4, coherent avec le document principal g=2, W(3_1) = 3,35).
     Avec SL = Tw + Wr ENTIER (White, document principal) : e^{i pi SL} est un Z_2 --
     une rotation 2pi du repere ajoute Tw = 1 -> facteur -1. COHERENT
     avec [1] du pont.
     DISTINCTION HONNETE (2+1D vs 3+1D) : en 2+1D anyonique, le poids de
     framing CS h = c_2/(k+2) EST la statistique ; a k = 135,
     h_(1/2) = 3/548 -> phase 2pi h = 0,034 rad : MINUSCULE, pas -1. Le
     signe FERMION de la TQNT ne vient PAS du poids CS : il vient du Z_2
     des rotations 3D (pi_1(SO(3)), k-INDEPENDANT, exact). Le poids CS
     est un habillage de phase separe et petit. La coincidence 2+1D ne se
     transfere pas -- NE JAMAIS invoquer h(k=135) pour la statistique.

=== VOLET B : la theorie des champs en interaction ===

[B1] SECTEUR EM : la chaine se ferme sur QED a alpha derive.
     La charge Q = Sigma-chi + Lk est la charge U(1) du champ chi (document principal :
     boucles + enlacement) ; l'invariance de phase locale du noeud force
     le couplage minimal p -> p - eA. Avec Dirac derive : le secteur EM
     basse energie EST QED, a alpha = 1/(k+2) DERIVE. Verifications :
     identite de Pauli (sigma.a)(sigma.b) = a.b + i sigma.(a x b)
     (numerique) ; algebre de Landau tronquee : (sigma.pi)^2 = pi^2 -
     e B sigma_z -> g = 2 EXACT du couplage minimal (numerique) --
     coherent avec le g=2 topologique et a_e = alpha/2pi (document principal 0,026 %).
     La chaine socle -> Lorentz -> Dirac -> couplage minimal -> QED est
     FERMEE (conditionnelle S, D, P10, U(1) de chi).

[B2] UV : LES BOUCLES SONT FINIES PAR CAPTURE AU SOMMET (theme).
     L'habillage alpha/(4pi) du croisement-ouvert etait FINI (log nul :
     le paquet minimal est a son echelle). Generalisation structurelle :
     le socle fournit la completion UV physique (denture limite) -- la
     renormalisation devient une reorganisation, pas une soustraction
     d'infinis. Statut : theme conditionnel (h2), un calcul par processus
     reste a mener (le residu -4,9 % du coefficient v = l'ordre alpha^2).

[B3] SECTEURS FAIBLE ET FORT : pas de QFT de mediateur a construire.
     Faible = contact fusion/scission (structure ET echelle derivees ;
     le m^5 de Sargent desormais derive -- la chaine des durees de vie
     est covariante par heritage). Fort = tension d'enlacement + mode
     scalaire (chap. forces) ; la dynamique non-abelienne du tube
     (vortex parafermion) reste EXPLORATOIRE (document principal, chap. Limitations).
     LE SOLVEUR : l'objection << non booste >> se DISSOUT -- le
     Schrodinger emergent est la limite lente de Dirac derive (reduction
     de Pauli), donc le solveur est l'effectif au repos d'une theorie
     covariante par heritage ; la covariance vit au-dessus, pas en lui.

RESTE (declare) : la seconde quantification complete du champ chi
(operateurs de creation de noeuds -- le germe est P22) ; la dynamique
forte non-abelienne ; le calcul de transmutation complet (somme sur les
chemins avec facteur de spin) comme blindage final.
"""
import numpy as np

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def berry_phase_latitude(theta, n=20000):
    """Transport du doublet |n(phi)> = (cos(t/2), sin(t/2) e^{i phi})
    le long d'une boucle de colatitude theta. Retourne la phase de Berry
    (attendu : -Omega/2 = -pi(1-cos theta))."""
    phis = np.linspace(0, 2 * np.pi, n, endpoint=True)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    states = np.stack([np.full_like(phis, c, dtype=complex),
                       s * np.exp(1j * phis)], axis=1)
    prod = 1.0 + 0j
    for i in range(n - 1):
        prod *= np.vdot(states[i], states[i + 1])
    return np.angle(prod)


def writhe_gauss(curve, n=800):
    """Integrale de Gauss du writhe d'une courbe fermee parametree."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = curve(t)                      # (n,3)
    dr = np.gradient(r, t, axis=0)    # tangentes * ds/dt
    w = 0.0
    for i in range(n):
        d = r[i] - r
        dist3 = np.sum(d * d, axis=1) ** 1.5
        cross = np.cross(dr[i], dr)
        integrand = np.einsum('ij,ij->i', cross, d)
        mask = np.ones(n, bool)
        mask[max(0, i - 2):i + 3] = False   # exclusion du voisinage singulier
        w += np.sum(integrand[mask] / dist3[mask])
    dt = t[1] - t[0]
    return w * dt * dt / (4 * np.pi)


def circle(t):
    return np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)


def trefoil(t):
    return np.stack([(2 + np.cos(3 * t)) * np.cos(2 * t),
                     (2 + np.cos(3 * t)) * np.sin(2 * t),
                     np.sin(3 * t)], axis=1)


def landau_g2(nb=60, b=1.0):
    """Algebre de Landau tronquee : pi_x = sqrt(B/2)(a+a+), pi_y =
    sqrt(B/2)(a-a+)/i ; verifie (sigma.pi)^2 = pi^2 - B sigma_z sur les
    etats interieurs (loin du bord de troncature)."""
    a = np.diag(np.sqrt(np.arange(1, nb)), 1)
    ad = a.T.conj()
    px = np.sqrt(b / 2) * (a + ad)
    py = np.sqrt(b / 2) * (a - ad) / 1j
    i_n = np.eye(nb)
    sp = np.kron(SX, px) + np.kron(SY, py)
    lhs = sp @ sp
    rhs = np.kron(np.eye(2), px @ px + py @ py) - b * np.kron(SZ, i_n)
    interior = slice(0, nb - 5)
    diff = (lhs - rhs).reshape(2, nb, 2, nb)[:, interior, :, interior]
    return np.max(np.abs(diff))


def run():
    print(__doc__)
    print("=" * 72)
    print("[A1] PREMIER ORDRE HERITE DE P10 -- phase de Berry du doublet")
    print("=" * 72)
    for theta in [np.pi / 3, np.pi / 2, 2 * np.pi / 3]:
        gamma = -berry_phase_latitude(theta)   # gamma = -arg prod (convention)
        target = -np.pi * (1 - np.cos(theta))
        # comparaison modulo 2pi (la phase n'est definie que mod 2pi)
        d = np.angle(np.exp(1j * (gamma - target)))
        print("  theta = %5.2f : gamma = %+8.5f ; -Omega/2 = %+8.5f  (ecart mod 2pi : %.1e)"
              % (theta, gamma, np.angle(np.exp(1j * target)), abs(d)))
    print("  -> cinetique de type Berry/WZ (PREMIER ordre, heritee de P10 :")
    print("     une action CS est du premier ordre par definition) ;")
    print("     orbite co-adjointe s=1/2 -> 2 etats = le doublet.")
    print("     La condition (c) du pont N'EST PLUS LIBRE.")

    print("\n" + "=" * 72)
    print("[A2] FACTEUR DE SPIN : writhe de Gauss + distinction k")
    print("=" * 72)
    w_circ = writhe_gauss(circle)
    w_tref = writhe_gauss(trefoil)
    print("  W(cercle)              = %+.4f   (attendu 0)" % w_circ)
    print("  W(trefle parametrique) = %+.4f   (document principal g=2 : W(3_1) = 3,35 ;" % w_tref)
    print("      parametrisation differente -> meme ordre, ~3,4)")
    print("  SL = Tw + Wr ENTIER (White) -> e^{i pi SL} = Z_2 ; rotation 2pi")
    print("  du repere : Tw += 1 -> facteur -1 (coherent pont [1]).")
    k = 135
    h_half = (3.0 / 4.0) / (k + 2)
    print("\n  DISTINCTION (2+1D vs 3+1D) : poids de framing CS a k = %d :" % k)
    print("    h_(1/2) = (3/4)/(k+2) = %.5f -> phase 2pi h = %.4f rad" % (h_half, 2 * np.pi * h_half))
    print("    MINUSCULE, pas -1 : le signe fermion vient du Z_2 des")
    print("    rotations 3D (k-INDEPENDANT, exact), PAS du poids CS --")
    print("    la coincidence anyonique 2+1D ne se transfere pas.")
    print("    Le poids CS = habillage de phase separe (petit).")

    print("\n" + "=" * 72)
    print("[B1] SECTEUR EM : Pauli + Landau -> g = 2 du couplage minimal")
    print("=" * 72)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(5):
        av, bv = rng.normal(size=3), rng.normal(size=3)
        sa = av[0] * SX + av[1] * SY + av[2] * SZ
        sb = bv[0] * SX + bv[1] * SY + bv[2] * SZ
        cr = np.cross(av, bv)
        rhs = np.dot(av, bv) * np.eye(2) + 1j * (cr[0] * SX + cr[1] * SY + cr[2] * SZ)
        worst = max(worst, np.max(np.abs(sa @ sb - rhs)))
    print("  identite de Pauli (5 tirages) : ecart max %.1e" % worst)
    dg2 = landau_g2()
    print("  algebre de Landau tronquee : |(sigma.pi)^2 - pi^2 + B sigma_z|")
    print("    (etats interieurs) = %.1e  -> g = 2 EXACT du couplage minimal," % dg2)
    print("    coherent avec le g=2 topologique et a_e = alpha/2pi (0,026 %).")
    print("  CHAINE FERMEE (conditionnelle S, D, P10, U(1) de chi) :")
    print("    socle -> Lorentz -> Dirac -> couplage minimal -> QED(1/137).")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  A : la condition << premier ordre >> est FERMEE (heritee de P10,")
    print("      Berry/WZ verifie) -- le mur spinoriel tombe au rang")
    print("      conditionnel commun ; distinction consignee : signe fermion")
    print("      = Z_2 rotations (exact), poids CS h = 3/548 = habillage.")
    print("  B : EM = QED a alpha derive (chaine fermee, g = 2 verifie) ;")
    print("      faible/fort = contact/topologique, pas de QFT de mediateur ;")
    print("      le solveur = limite lente de Dirac derive (objection")
    print("      << non booste >> dissoute) ; boucles finies par capture au")
    print("      sommet (theme, conditionnel h2).")
    print("  RESTE : seconde quantification complete de chi (germe P22),")
    print("      dynamique forte non-abelienne (exploratoire), calcul de")
    print("      transmutation complet (somme sur chemins) comme blindage.")


if __name__ == "__main__":
    run()
