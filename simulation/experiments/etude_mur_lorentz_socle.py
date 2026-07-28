"""Le mur de Lorentz sur le socle vibrant a f_socle -- etude (2026-07-21).

Le SEUL mur general restant du cadre : la cinematique relativiste est POSEE,
non derivee (chap. Limitations ; le m^5 de Sargent est declare cinematique).
Question : le socle ondulant (engrenage gradue en action, f_socle = c/l_P)
peut-il la DERIVER ?

DISCIPLINE ANTI-CIRCULARITE (lecon du papier socle v1) : on ne postule
aucune transformation de repere. TOUT est calcule dans LE repere du socle,
a partir de deux premisses canoniques :
  (S) tout objet du cadre est fait d'ondes chi (noeuds compris) et TOUTE
      dynamique porte c dans son terme cinetique (document principal : << c = structure
      causale du champ chi >>) ;
  (D) dispersion NULLE structurelle (S3, graduation en action) : tous les
      modes du socle s'engrenent a v = lambda f = c EXACTEMENT, du mode le
      plus long a la denture limite f_socle = c/l_P.

[1] LE COEUR : (S)+(D) => l'equation d'onde unique []chi = 0.
    FAIT ALGEBRIQUE (verifiable) : le d'alembertien est invariant sous les
    transformations qui preservent c -- le groupe de Lorentz est LE groupe
    de symetrie de la dynamique du socle. Ce n'est pas un postulat sur
    l'espace-temps : c'est une propriete de l'equation que (S)+(D) forcent.

[2] LA CINEMATIQUE EN DECOULE, calculee DANS le repere du socle
    (voie constructive, lignee Lorentz-Poincare-Bell -- ascendance declaree ;
    l'apport TQNT est que (S) n'est pas une hypothese sur << toute la
    matiere >> mais l'ontologie meme : il n'existe RIEN d'autre que chi).
    GARDE ANTI-CIRCULARITE, structure exacte de l'argument :
    (a) le THEOREME est en [1] : la symetrie de []chi = 0 (fait algebrique,
        verifie en [1bis]) garantit que toute solution statique (noeud au
        repos, onde stationnaire E0 = hbar w0, m := E0/c^2) possede une
        famille de solutions mobiles -- SANS postulat de repere ;
    (b) la decomposition en modes de la solution mobile a pour facteurs
        D± = sqrt((1±b)/(1∓b)) ; leur RATIO D+/D- = (1+b)/(1-b) est le
        Doppler classique du miroir mobile (non circulaire) ; leur
        NORMALISATION D+D- = 1 (equivalente a L = L0/gamma, compte de
        noeuds du mode invariant) vient de la symetrie (a) -- c'est LA
        ou la dynamique entre, pas d'un postulat ;
    (c) verification de coherence [2] : E(v) = gamma E0, p(v) = gamma b
        E0/c, E^2-(pc)^2 = E0^2 exacts -> la dispersion massive EMERGE ;
        dilatation gamma T0 (toute horloge EST un noeud chi) ; addition
        des vitesses = composition des Doppler.

[3] POURQUOI PAS D'ETHER : parmi les graduations possibles d'un substrat
    (longueur, duree, energie, ACTION), seule l'action ne fait reference ni
    a l'espace ni au temps separement (mesure symplectique) : un substrat
    gradue en h n'offre AUCUNE regle fixe a comparer au passage d'un objet
    -- le repere du socle est dynamiquement INDETECTABLE (Michelson nul par
    construction). Un reseau gradue en LONGUEUR (pas l_P spatial) casserait
    (D) a E ~ E_P : disperssion LIV lineaire exclue par les GRB
    (E_QG > 9,3 E_P). Le nul GRB, deja prediction du socle, devient ici la
    PROTECTION EXPERIMENTALE de la derivation.

[4] ROLE DE f_socle : plafond de la tour de modes. La carte de boost est
    exacte a TOUTE frequence < f_socle ; et la masse d'un noeud est la
    FRACTION de f_socle qu'il confine : m/M_Pl = w0/w_socle -- pour le
    nucleon, cette fraction est fixee par l'echelle par croisement,
    w0/w_socle = sqrt(12/7) alpha^9 (identite du raccord).

[5] CE QUI RESTE (le mur retreci, honnete) : la cinematique SCALAIRE est
    derivee conditionnellement ((S),(D)) ; restent (i) la structure
    SPINORIELLE (Dirac, connexion spin-statistique -- le 4pi du repere de
    ruban est le pont candidat) ; (ii) le boost des solutions du SOLVEUR
    (le Schrodinger emergent est la limite lente, coherent mais non boost) ;
    (iii) la these << rien d'autre que chi >> est l'ontologie, pas un
    theoreme. Le << mur de Lorentz >> devient un MUR SPINORIEL.
"""
import math

C = 299792458.0
HBAR = 1.054571817e-34
L_P = 1.616255e-35
M_PL_GEV = 1.220890e19
M_BASE_GEV = 0.938240
ALPHA = 1.0 / 137.035999


def check_wave_symmetry(beta, w, kx):
    """[1bis] Fait algebrique : x' = g(x-vt), t' = g(t-vx/c^2) envoie toute
    onde plane solution de []chi=0 (w = c|k|) sur une solution (w' = c|k'|).
    Retourne le defaut |w'^2 - c^2 k'^2| (en unites c=1)."""
    g = 1.0 / math.sqrt(1 - beta**2)
    wp = g * (w - beta * kx)
    kp = g * (kx - beta * w)
    return abs(wp * wp - kp * kp - (w * w - kx * kx))


def boost_standing_wave(beta):
    """Onde stationnaire = deux modes contre-propageants d'energie E0/2.
    Doppler d'ONDE CLASSIQUE sur un substrat sans dispersion : un recepteur
    (le confinement mobile) voit D± = sqrt((1±b)/(1∓b)). Retourne
    (E/E0, pc/E0) -- a comparer a (gamma, gamma*beta)."""
    dp = math.sqrt((1 + beta) / (1 - beta))
    dm = math.sqrt((1 - beta) / (1 + beta))
    e = 0.5 * (dp + dm)          # somme des energies des deux modes
    p = 0.5 * (dp - dm)          # somme des impulsions (E/c par mode)
    return e, p


def run():
    print(__doc__)
    print("=" * 72)
    print("[1bis] FAIT ALGEBRIQUE : la carte de boost preserve []chi = 0")
    print("=" * 72)
    worst = 0.0
    for beta in [0.3, 0.9, 0.9999]:
        for w, kx in [(1.0, 1.0), (1.0, -1.0), (5.0, 5.0), (2.5, -2.5)]:
            worst = max(worst, check_wave_symmetry(beta, w, kx))
    print("  defaut max |w'^2 - c^2k'^2| sur le balayage : %.1e" % worst)
    print("  -> toute solution du socle a sa famille mobile : le groupe de")
    print("     Lorentz est LE groupe de symetrie de la dynamique (S)+(D) --")
    print("     aucun postulat de repere n'est entre.")

    print()
    print("=" * 72)
    print("[2] VERIFICATION DE COHERENCE : la cinematique du boost d'onde")
    print("=" * 72)
    print("  %-8s %-14s %-14s %-14s %-12s" % ("beta", "E/E0", "gamma", "pc/E0", "E^2-(pc)^2"))
    for beta in [0.1, 0.5, 0.9, 0.99, 0.999999]:
        e, p = boost_standing_wave(beta)
        g = 1.0 / math.sqrt(1 - beta**2)
        print("  %-8g %-14.9f %-14.9f %-14.9f %-12.3e"
              % (beta, e, g, p, e * e - p * p - 1.0))
    print("  -> E = gamma E0 et p = gamma beta E0/c EXACTS (identite d'onde,")
    print("     machine precision) ; E^2-(pc)^2 = E0^2 : la dispersion massive")
    print("     E^2=(pc)^2+(mc^2)^2 EMERGE -- le m^5 de Sargent en decoule,")
    print("     l'emprunt cinematique de P15 quantitative se referme.")

    print("\n  Composition des vitesses (produit des Doppler) :")
    b1, b2 = 0.8, 0.7
    d1 = math.sqrt((1 + b1) / (1 - b1))
    d2 = math.sqrt((1 + b2) / (1 - b2))
    d12 = d1 * d2
    b12 = (d12**2 - 1) / (d12**2 + 1)
    b_add = (b1 + b2) / (1 + b1 * b2)
    print("    b1=0,8 (+) b2=0,7 -> %.9f ; formule (b1+b2)/(1+b1b2) = %.9f"
          % (b12, b_add))
    print("    ecart %.1e : la loi d'addition relativiste = composition des" % abs(b12 - b_add))
    print("    Doppler du socle. Dilatation gamma T0 et contraction L0/gamma :")
    print("    memes identites (compte de noeuds du mode invariant).")

    print("\n" + "=" * 72)
    print("[3] PROTECTION : action vs longueur (le nul GRB protege (D))")
    print("=" * 72)
    e_grb_gev = 100.0            # photon GRB ~100 GeV, z~8 : bornes E_QG
    liv_lin = e_grb_gev / (M_PL_GEV)   # dispersion lineaire ~E/E_QG
    print("  Reseau gradue en LONGUEUR (pas spatial l_P) : dv/c ~ E/E_QG")
    print("    -> a 100 GeV : dv/c ~ %.1e ; retards GRB mesurables ->" % liv_lin)
    print("    EXCLU (E_QG > 9,3 E_P observe : Fermi-LAT/HESS).")
    print("  Socle gradue en ACTION (pas = h, aucune regle spatiale) :")
    print("    dispersion structurellement NULLE a tout ordre -> le nul GRB")
    print("    est la PREDICTION (deja aux regles etablies) et devient la protection")
    print("    experimentale de la derivation : (D) est teste a >9 E_P.")

    print("\n" + "=" * 72)
    print("[4] f_socle : plafond et fractions de masse")
    print("=" * 72)
    f_socle = C / L_P
    print("  f_socle = c/l_P = %.4e Hz (identite definitionnelle -- jamais" % f_socle)
    print("  << derivee >>) ; E(hbar f_socle) = %.3e GeV = M_Pl."
          % (HBAR * f_socle / 1.602176634e-10))
    frac_nucleon = M_BASE_GEV / M_PL_GEV
    frac_ladder = math.sqrt(12.0 / 7.0) * ALPHA**9
    print("  Masse = fraction confinee du socle : m/M_Pl = w0/w_socle")
    print("    nucleon : %.4e ; echelle par croisement sqrt(12/7) alpha^9 = %.4e"
          % (frac_nucleon, frac_ladder))
    print("    (%+.3f %% -- l'identite du raccord, relue en frequences)"
          % (100 * (frac_ladder / frac_nucleon - 1)))
    print("  La carte de boost [2] est exacte pour tout w0 < w_socle : le")
    print("  plafond ne casse PAS la cinematique (pas de regle spatiale).")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  DERIVE (conditionnel (S)+(D), voie constructive declaree,")
    print("  lignee Lorentz-Poincare-Bell) : la cinematique relativiste")
    print("  SCALAIRE -- gamma, contraction, addition des vitesses,")
    print("  E^2=(pc)^2+(mc^2)^2 (donc le m^5 de Sargent), masse = frequence")
    print("  confinee ; le repere du socle est dynamiquement indetectable")
    print("  (graduation en action = seule graduation sans regle de repere).")
    print("  RESTE (le mur retreci) : structure SPINORIELLE (Dirac,")
    print("  spin-statistique ; pont candidat = le 4pi du repere de ruban),")
    print("  boost du solveur, et le statut d'ontologie de << tout est chi >>.")
    print("  Le << mur de Lorentz >> devient un MUR SPINORIEL -- plus etroit,")
    print("  nomme, et adosse a un programme existant (g=2, ruban, tresse).")


if __name__ == "__main__":
    run()
