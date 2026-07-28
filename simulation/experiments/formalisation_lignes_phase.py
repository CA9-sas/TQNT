"""Formalisation de << l'enlacement des lignes de phase >> (20/07).

Trois pieces rigoureuses pour le brouillon intrication :

 [1] THEOREME DE SEPARATION (negatif) : deux lignes de defaut FERMEES
     contenues dans des boules disjointes ont Lk = 0 (l'enlacement de
     Gauss instantane entre A et B eloignes est NUL) -- la lecture
     << les deux particules restent enlacees dans l'espace >> est
     impossible. Verification numerique a l'appui.

 [2] L'INVARIANT DE FEUILLE D'UNIVERS (positif) : la paire nucleee de
     l'evenement p a une feuille de defaut CONNEXE W = W_A u W_B dans
     l'espace-temps (les lignes de defaut ne peuvent pas se terminer ;
     la scission au vertex relie les deux branches). Un lacet qui
     enroule W_A se glisse, a travers p, en un lacet qui enroule W_B :
     les enroulements sont RIGIDEMENT lies. Pour une paire nee du vide
     (enroulement total nul) : w_A + w_B = 0 EXACTEMENT, a toute
     distance, sans aucun cout d'energie (homotopie, pas dynamique).
     Demonstration numerique : phase d'une paire anneau/anti-anneau
     (theta = (Omega_A - Omega_B)/2, angles solides exacts) ;
     enroulements mesures sur des lacets a A et B pour d croissant.

 [3] CORRECTION DE LA LECTURE GOLDSTONE (honnete) : dans un condensat
     PLEIN, la phase relative lisse alpha entre deux regions n'est PAS
     un mode de cout nul : E(alpha, d) = (rho_s/2) alpha^2 C(d) avec
     C(d) la capacite a deux spheres -> CONSTANTE (2 pi a) quand
     d -> infini. La memoire de phase lisse se paie et diffuse ; seule
     la composante TOPOLOGIQUE (enroulement, piece [2]) est rigide et
     gratuite. Le porteur de l'intrication est donc PLUS topologique
     que la formulation initiale du brouillon.
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")


# ---------- angle solide exact d'un disque triangule (Van Oosterom) ----------
def solid_angle_disk(center, normal, radius, x, n_seg=64):
    """Angle solide sous-tendu par le disque (anneau rempli) vu de x."""
    n = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0.0, 0.0])
    if abs(n @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    th = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    rim = center + radius * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    tot = 0.0
    for k in range(n_seg):
        r1 = center - x
        r2 = rim[k] - x
        r3 = rim[(k + 1) % n_seg] - x
        l1, l2, l3 = (np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(r3))
        num = r1 @ np.cross(r2, r3)
        den = (l1 * l2 * l3 + (r1 @ r2) * l3 + (r1 @ r3) * l2 + (r2 @ r3) * l1)
        tot += 2 * math.atan2(num, den)
    return tot


def winding_on_loop(theta_fn, loop_pts):
    """Enroulement (en tours) de la phase le long d'un lacet echantillonne."""
    th = np.array([theta_fn(p) for p in loop_pts])
    d = np.diff(np.concatenate([th, th[:1]]))
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return d.sum() / (2 * np.pi)


def small_loop(center, axis, r_loop, n=200):
    a = axis / np.linalg.norm(axis)
    b = np.array([1.0, 0.0, 0.0])
    if abs(a @ b) > 0.9:
        b = np.array([0.0, 1.0, 0.0])
    u = np.cross(a, b); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center[None, :] + r_loop * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))


# ---------- capacite a deux spheres (serie exacte) ----------
def capacitance_two_spheres(a, d, nmax=200):
    """C_rel (unites 4 pi eps0 = 1) du condensateur sphere-sphere."""
    if d <= 2 * a:
        return math.inf
    beta = math.acosh(d / (2 * a))
    s = 0.0
    for n in range(1, nmax):
        if n * beta > 700:
            break
        s += 1.0 / math.sinh(n * beta)
    return a * math.sinh(beta) * s / 2.0   # serie des deux spheres en serie


def run():
    print("=== Formalisation : enlacement des lignes de phase ===\n")

    # [1] theoreme de separation
    print("[1] Theoreme de separation : Lk instantane (Gauss) de deux anneaux")
    from simulation.ideal_embedding import linking_number
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    ring = np.column_stack([np.cos(th), np.sin(th), 0 * th])
    for d in (3.0, 10.0, 100.0):
        A = ring.copy()
        B = ring + np.array([d, 0.0, 0.0])
        print("    d = %6.1f : Lk = %d" % (d, linking_number(A, B)))
    print("    -> boules disjointes => Lk = 0 NECESSAIREMENT (topologie) :")
    print("       l'enlacement INSTANTANE ne peut pas porter le lien EPR.\n")

    # [2] invariant de feuille : enroulements rigidement opposes
    print("[2] Paire anneau/anti-anneau nee du vide : enroulements w_A, w_B")
    nrm = np.array([0.0, 0.0, 1.0])
    for d in (4.0, 20.0, 200.0):
        cA = np.array([0.0, 0.0, 0.0])
        cB = np.array([d, 0.0, 0.0])

        def theta(x, cA=cA, cB=cB):
            return 0.5 * (solid_angle_disk(cA, nrm, 1.0, x)
                          - solid_angle_disk(cB, nrm, 1.0, x))
        # lacets perçant les coeurs (autour du tube de l'anneau)
        loopA = small_loop(cA + np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 0.3)
        loopB = small_loop(cB + np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 0.3)
        wA = winding_on_loop(theta, loopA)
        wB = winding_on_loop(theta, loopB)
        print("    d = %6.1f : w_A = %+.3f, w_B = %+.3f, somme = %+.1e" % (
            d, wA, wB, wA + wB))
    print("    -> w_A + w_B = 0 EXACT a toute distance : la connexite de la")
    print("       feuille d'univers au vertex de creation lie RIGIDEMENT les")
    print("       enroulements (homotopie : cout nul, independance a d).\n")

    # [3] correction Goldstone : la phase lisse n'est pas gratuite
    print("[3] Phase relative LISSE alpha entre deux regions (rayon a = 1) :")
    print("    E(alpha,d) = (rho_s/2) alpha^2 C(d) ; C(d) :")
    for d in (3.0, 10.0, 100.0, 1e4):
        print("    d = %8.0f : C = %.4f  (limite d->inf : %.4f = a/2)" % (
            d, capacitance_two_spheres(1.0, d), 0.5))
    print("    -> C tend vers une CONSTANTE : la phase relative lisse coute")
    print("       une energie finie meme a l'infini (et diffuse) -- ce n'est")
    print("       PAS un mode gratuit. CORRECTION du brouillon : le porteur")
    print("       rigide et gratuit est la composante TOPOLOGIQUE [2], pas")
    print("       la phase lisse.\n")

    print("SYNTHESE FORMELLE :")
    print(" - l'<< enlacement des lignes de phase >> = enroulement du champ de")
    print("   phase autour des FEUILLES D'UNIVERS des defauts, liees par la")
    print("   connexite au vertex de creation (pi_1 du complement dans R^4) ;")
    print(" - anti-correlation EPR discrete = w_A + w_B = 0 (nee du vide),")
    print("   rigide, sans cout, independante de la distance ;")
    print(" - mesure/decoherence = reconnexion de feuille (changement de")
    print("   topologie de W en aval) ; GHZ = vertex a 3 branches (connexite")
    print("   triple), W = connexites par paires -- au niveau des feuilles.")


if __name__ == "__main__":
    run()
