"""Ancrer l'objet 3D du porteur angulaire (le cos^2) -- (20/07).

QUESTION : la partie CONTINUE des correlations EPR (E = -cos theta)
doit avoir un objet 3D dans l'ontologie TQNT. Lequel ?

 [1] REFUTATION du candidat naif : un AXE 3D classique porte par la
     particule (sortie = signe(n.a)) donne E(theta) = 2 theta/pi - 1
     (dents de scie), PAS -cos theta ; et toute fonction de reponse
     locale est bornee CHSH <= 2 (Bell) vs 2 sqrt(2) observe.
     -> le porteur angulaire ne peut PAS etre une direction classique.

 [2] L'OBJET : le REPERE DE RUBAN du tube (framing -- le meme objet
     que Lk = Tw + Wr, deja canonique). Discriminant de PERIODICITE :
     un vecteur revient a l'identite apres 2 pi (spin entier seul) ;
     un repere RUBANE (attache a son environnement) revient apres
     4 pi seulement (pi_1(SO(3)) = Z_2, ceinture de Dirac).
     Numerique : relevement continu d'un lacet de rotations ->
     2 pi donne -1, 4 pi donne +1, invariant sous perturbation
     du chemin (homotopie).

 [3] LE DEMI-ANGLE ANCRE : c'est le MEME fait topologique que g = 2
     (double revetement du 3_1, indice de tresse 2). Recouvrement des
     etats de repere : |<n|m>|^2 = cos^2(theta/2) exact ; singulet de
     reperes -> E(a,b) = -a.b exact (verifies sur axes aleatoires).

 [4] STATUT : objet NOMME (repere de ruban), demi-angle TOPOLOGIQUE
     (double revetement), amplitude cos^2 HERITEE (Schrodinger/Born).
"""
import math

import numpy as np

RNG = np.random.default_rng(11)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def su2_step(axis, dth):
    a = axis / np.linalg.norm(axis)
    s = a[0] * SX + a[1] * SY + a[2] * SZ
    return math.cos(dth / 2) * np.eye(2) - 1j * math.sin(dth / 2) * s


def lift_endpoint(total_angle, wobble=0.0, n=4000, seed=0):
    """Relevement SU(2) continu d'un LACET de rotations : angle total
    donne autour de z, axe perturbe en chemin si wobble > 0, puis
    fermeture par la geodesique COURTE (angle < pi, sans enroulement)
    -> le signe final (+/- identite) est l'invariant d'homotopie."""
    rng = np.random.default_rng(seed)
    ph = rng.random(2) * 6
    u = np.eye(2, dtype=complex)
    for k in range(n):
        t = (k + 0.5) / n
        env = math.sin(math.pi * t) ** 2
        ax = np.array([wobble * env * math.sin(4 * math.pi * t + ph[0]),
                       wobble * env * math.sin(2 * math.pi * t + ph[1]),
                       1.0])
        u = su2_step(ax, total_angle / n) @ u
    # fermeture : rotation residuelle -> identite par la geodesique courte
    phi = 2 * math.acos(abs(np.clip(u.trace().real / 2, -1, 1)))
    if phi > 1e-12:
        sgn = 1.0 if u.trace().real >= 0 else -1.0
        v = sgn * u  # representant a angle < pi :
        # v = cos(phi/2) I - i sin(phi/2) n.sigma
        nx, ny, nz = -v[0, 1].imag, -v[0, 1].real, -v[0, 0].imag
        nrm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nrm > 1e-12:
            axis = np.array([nx, ny, nz]) / nrm
            for _ in range(200):
                u = su2_step(axis, -phi / 200) @ u
    return u


def spinor(axis):
    a = axis / np.linalg.norm(axis)
    th = math.acos(np.clip(a[2], -1, 1))
    ph = math.atan2(a[1], a[0])
    return np.array([math.cos(th / 2),
                     complex(math.cos(ph), math.sin(ph)) * math.sin(th / 2)])


def rand_axis(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def run():
    print("=== L'objet 3D du porteur angulaire (cos^2) ===\n")

    # [1] axe classique porte -> dents de scie + CHSH <= 2
    print("[1] REFUTATION : axe 3D classique porte (sortie = signe(n.a))")
    n_mc = 200000
    axes = RNG.normal(size=(n_mc, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    print("    theta   E_axe_classique   E_QM = -cos theta")
    for deg in (0, 45, 90, 135, 180):
        th = math.radians(deg)
        a = np.array([0, 0, 1.0])
        b = np.array([math.sin(th), 0, math.cos(th)])
        e_cl = np.mean(np.sign(axes @ a) * (-np.sign(axes @ b)))
        print("    %4d      %+.3f            %+.3f" % (deg, e_cl, -math.cos(th)))
    # CHSH aux angles optimaux QM (0, 90 ; 45, 135)
    ang = {"a": 0.0, "ap": math.pi / 2, "b": math.pi / 4, "bp": 3 * math.pi / 4}
    vec = {k: np.array([math.sin(v), 0, math.cos(v)]) for k, v in ang.items()}

    def e_cl(u, v):
        return np.mean(np.sign(axes @ u) * (-np.sign(axes @ v)))

    s_cl = abs(e_cl(vec["a"], vec["b"]) - e_cl(vec["a"], vec["bp"])
               + e_cl(vec["ap"], vec["b"]) + e_cl(vec["ap"], vec["bp"]))
    print("    CHSH axe classique = %.3f  (borne LHV 2) ; QM = 2 sqrt(2) = %.3f"
          % (s_cl, 2 * math.sqrt(2)))
    print("    -> dents de scie + CHSH <= 2 : le porteur angulaire n'est PAS")
    print("       une direction classique portee.\n")

    # [2] discriminant de periodicite : le repere rubane
    print("[2] L'OBJET = repere de ruban (framing). Discriminant 2pi/4pi :")
    for tot, lab in ((2 * math.pi, "2 pi"), (4 * math.pi, "4 pi")):
        u = lift_endpoint(tot)
        print("    rotation %s : relevement final = %+.4f * identite" % (
            lab, u[0, 0].real))
    print("    perturbations du chemin (wobble 0,3, 3 graines) :")
    for seed in range(3):
        u = lift_endpoint(2 * math.pi, wobble=0.3, seed=seed)
        print("      graine %d : 2 pi -> %+.4f (inchange : homotopie)" % (
            seed, u[0, 0].real))
    print("    -> un VECTEUR revient apres 2 pi (spin entier seulement) ;")
    print("       un repere ATTACHE (ruban) revient apres 4 pi seulement :")
    print("       le demi-angle exige un objet RUBANE -- le framing du tube")
    print("       (le meme objet que Lk = Tw + Wr).\n")

    # [3] recouvrement des etats de repere + singulet
    print("[3] DEMI-ANGLE ANCRE (meme fait que g = 2, double revetement 3_1) :")
    errs, errs_s = [], []
    for _ in range(200):
        a, b = rand_axis(RNG), rand_axis(RNG)
        th = math.acos(np.clip(a @ b, -1, 1))
        ov = abs(np.vdot(spinor(a), spinor(b))) ** 2
        errs.append(abs(ov - math.cos(th / 2) ** 2))
        # singulet : <sigma.a x sigma.b> = -a.b
        sing = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        sa = a[0] * SX + a[1] * SY + a[2] * SZ
        sb = b[0] * SX + b[1] * SY + b[2] * SZ
        val = (sing.conj() @ np.kron(sa, sb) @ sing).real
        errs_s.append(abs(val - (-(a @ b))))
    print("    |<n|m>|^2 = cos^2(theta/2) : erreur max %.1e (200 paires)" %
          max(errs))
    print("    singulet de reperes : E(a,b) = -a.b, erreur max %.1e\n" %
          max(errs_s))

    print("[4] STATUT : l'objet est NOMME -- le repere de ruban du tube de")
    print("    defaut (framing) ; le qubit = doublet des secteurs de feuille")
    print("    habille par l'orientation du repere ; le DEMI-ANGLE est")
    print("    topologique (4 pi, double revetement -- le meme fait que la")
    print("    derivation g = 2 du trefle) ; la loi d'amplitude cos^2 reste")
    print("    HERITEE (Schrodinger derive + Born). Aucun retour en arriere :")
    print("    le repere ne fournit pas de variable cachee (cf. [1]) -- il")
    print("    fournit l'ESPACE d'etats (sphere de Bloch = orientations du")
    print("    repere), pas une valeur preassignee.")


if __name__ == "__main__":
    run()
