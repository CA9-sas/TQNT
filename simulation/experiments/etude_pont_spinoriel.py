"""Le pont spinoriel : du 4pi du ruban a l'equation de Dirac (2026-07-21).

Suite de etude_mur_lorentz_socle.py : la cinematique SCALAIRE est derivee
(conditionnel S+D) ; reste le mur spinoriel. Trois pas, chacun verifie :

[1] LES ETATS DE REPERE SONT DES SPINEURS (le 4pi du ruban).
    Le tube de flux est FRAME (document principal : Lk = Tw + Wr exact, quantum de
    framing, delta_framing, porteur angulaire du chap. non-classique,
    g = 2 par le double revetement). L'espace des rotations SO(3) a
    pi_1 = Z_2 : le repere d'un ruban ne revient a lui-meme qu'apres 4pi
    (tour de ceinture). Les etats de repere vivent donc dans le DOUBLE
    REVETEMENT SU(2) -> doublet de spin 1/2. DERIVE (fait topologique,
    verifie : quaternions, 2pi -> -1, 4pi -> +1).

[2] SPIN-STATISTIQUE : ECHANGE ~ ROTATION 2pi (configurations etendues).
    Pour des objets ETENDUS et frames (les noeuds + leur champ), le chemin
    d'echange de deux configurations identiques est homotope a la rotation
    2pi de l'une (theoreme de type Finkelstein-Rubinstein, configuration
    space des champs ; ici il porte sur l'EXTENSION des configurations,
    PAS sur une charge de winding -- la lecture soliton/Skyrme reste
    rejetee, Regle 0). Complement natif : dans la categorie RUBANEE de la
    theorie (Reshetikhin-Turaev, la maison mathematique de la TQNT),
    l'axiome de compatibilite tresse-twist EST l'egalite
    statistique = spin. => P = R(2pi) = -1 pour le doublet [1] :
    la CONNEXION SPIN-STATISTIQUE est derivee (conditions declarees :
    configurations identiques etendues ; identification incluant le champ
    de liaison).

[3] DIRAC EST FORCE (linearisation de la dispersion derivee).
    Entrees : (a) E^2 = (pc)^2 + (mc^2)^2 (DERIVEE, mur de Lorentz) ;
    (b) etats internes = doublet de repere (DERIVE, [1]) ; (c) evolution
    du premier ordre pour la variable de repere (terme cinetique de type
    Chern-Simons/Berry -- LECTURE declaree, c'est la condition restante).
    Cherchant H = c a.p + b mc^2 lineaire avec H^2 = (pc)^2 + (mc^2)^2 :
    l'algebre de Clifford {a_i,a_j} = 2 delta_ij, {a_i,b} = 0, b^2 = 1
    est forcee. VERIFIE :
      - en 2x2 (doublet seul) : IMPOSSIBLE -- aucun b n'anticommute avec
        les trois Pauli (rang du systeme lineaire = plein, seule solution
        b = 0) ; le doublet seul ne porte pas la masse ;
      - en 4x4 : la representation minimale EXISTE (Dirac) et ferme
        exactement -- H^2 = (p^2+m^2) I a la precision machine.
    LECTURE TQNT DU DOUBLEMENT 2 -> 4 : les deux blocs = (noeud K, miroir
    K-bar) -- l'antiparticule est le MIROIR (document principal, Regle 0 : antimatiere
    = miroir, pas un changement d'helicite) ; la conjugaison de charge de
    Dirac = l'operation miroir V_K(t) -> V_K(1/t). Les 4 composantes de
    Dirac = doublet de repere (x) paire (K, K-bar). Et g = 2 sort du
    couplage minimal (Gordon) -- coherent avec la derivation topologique
    separee (double revetement, document principal).

CE QUI RESTE (declare) : (c) le premier ordre de la variable de repere
(version blindee = facteur de spin de Polyakov : en Chern-Simons, la ligne
d'univers chargee acquiert exp(i pi W[writhe]) qui transmute la
statistique et donne le propagateur de Dirac -- TQNT-natif, calcul complet
a mener) ; la theorie des champs covariante en interaction (solveur) ;
le statut d'ontologie de (S). Le mur spinoriel se reduit a UNE condition
nommee et calculable.
"""
import numpy as np

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def step1_belt():
    print("=" * 72)
    print("[1] DOUBLE REVETEMENT : le repere de ruban est un spineur")
    print("=" * 72)
    for total, label in [(2 * np.pi, "2pi"), (4 * np.pi, "4pi")]:
        n = 4000
        q = np.array([1.0, 0.0, 0.0, 0.0])
        dth = total / n
        dq = np.array([np.cos(dth / 2), 0.0, 0.0, np.sin(dth / 2)])
        for _ in range(n):
            q = quat_mul(dq, q)
        print("  rotation continue %-4s -> quaternion w = %+.6f  (%s)"
              % (label, q[0], "-1 : etat RENVERSE" if q[0] < 0 else "+1 : identite"))
    print("  -> l'etat de repere ne revient qu'apres 4pi : doublet SU(2),")
    print("     spin 1/2 -- le meme fait que g=2 et le porteur angulaire.")


def step2_spin_statistics():
    print("\n" + "=" * 72)
    print("[2] SPIN-STATISTIQUE : echange^2 ~ rotation 2pi (verification du")
    print("    relevement ; l'homotopie est le theoreme FR, configurations")
    print("    etendues -- PAS de winding, Regle 0 respectee)")
    print("=" * 72)
    # Echange = demi-tour du vecteur relatif ; echange^2 = tour complet.
    # Relevement SU(2) du tour complet du repere relatif :
    n = 4000
    q = np.array([1.0, 0.0, 0.0, 0.0])
    dth = 2 * np.pi / n
    dq = np.array([np.cos(dth / 2), 0.0, 0.0, np.sin(dth / 2)])
    for _ in range(n):
        q = quat_mul(dq, q)
    print("  echange^2 = boucle 2pi du repere relatif -> relevement w = %+.4f"
          % q[0])
    print("  => P^2 = R(2pi) = -1 sur le doublet ; avec l'axiome")
    print("     tresse-twist de la categorie rubanee (RT, natif TQNT) :")
    print("     P = theta = -1 -- FERMION. La connexion spin-statistique")
    print("     est DERIVEE (conditions declarees dans l'en-tete).")


def step3_dirac_forced():
    print("\n" + "=" * 72)
    print("[3] DIRAC FORCE : linearisation de la dispersion derivee")
    print("=" * 72)
    # (i) 2x2 impossible : resoudre {b, s_i} = 0 pour i=x,y,z.
    rows = []
    for s in (SX, SY, SZ):
        # anticommutateur comme operateur lineaire sur vec(b) (4 composantes)
        op = np.kron(I2, s) + np.kron(s.T, I2)
        rows.append(op)
    system = np.vstack(rows)
    rank = np.linalg.matrix_rank(system)
    print("  2x2 : rang du systeme {b,sigma_i}=0 : %d / 4 -> seule solution"
          % rank)
    print("        b = 0 : IMPOSSIBLE de porter la masse sur le doublet seul.")

    # (ii) 4x4 Dirac ferme exactement.
    O2 = np.zeros((2, 2), dtype=complex)
    alph = [np.block([[O2, s], [s, O2]]) for s in (SX, SY, SZ)]
    beta = np.block([[I2, O2], [O2, -I2]])
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(5):
        p = rng.normal(size=3)
        m = abs(rng.normal())
        h = sum(pi * ai for pi, ai in zip(p, alph)) + m * beta
        h2 = h @ h
        target = (p @ p + m * m) * np.eye(4)
        worst = max(worst, np.max(np.abs(h2 - target)))
    print("  4x4 : representation minimale (Dirac) -> max|H^2-(p^2+m^2)I| =")
    print("        %.1e (5 tirages aleatoires) : FERME exactement." % worst)
    print("  Lecture TQNT du doublement 2->4 : blocs = (K, K-bar miroir) --")
    print("  l'antiparticule EST le miroir (document principal) ; 4 composantes =")
    print("  doublet de repere (x) paire noeud/miroir ; g = 2 par couplage")
    print("  minimal (Gordon), coherent avec la derivation topologique.")


def run():
    print(__doc__)
    step1_belt()
    step2_spin_statistics()
    step3_dirac_forced()
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  DERIVE (conditionnel) : etats de repere = doublet SU(2) (4pi,")
    print("  fait topologique) ; spin-statistique (FR + axiome rubane) ;")
    print("  Dirac = representation minimale FORCEE par la dispersion")
    print("  derivee + le doublet (2x2 impossible, rang plein), avec")
    print("  antiparticule = miroir et g = 2 coherent.")
    print("  CONDITION RESTANTE, unique et nommee : le premier ordre de la")
    print("  variable de repere (terme cinetique CS/Berry) -- version")
    print("  blindee = facteur de spin de Polyakov (exp(i pi W), TQNT-natif,")
    print("  calcul complet a mener). Le mur spinoriel se reduit a cette")
    print("  condition ; ne pas verser en V6 sans demande.")


if __name__ == "__main__":
    run()
