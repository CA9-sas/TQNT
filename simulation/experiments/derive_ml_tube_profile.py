"""Le chantier m_l : profil du tube par le solveur -- verdict structurel (17/07).

QUESTION : m_l = 310 MeV (normalisation du rayon de tube de la loi
geometrique) est-il derivable de la couche microscopique du solveur chi
(E_0, gamma_J, lambda_V ; potentiel V = -Lambda rho_J / d^2) ?

REPONSE : NON -- et c'est un THEOREME de structure, pas un echec de calcul.

1. INVARIANCE D'ECHELLE : V ~ 1/d^2 et le laplacien portent la MEME
   dimension (1/longueur^2) : le probleme continu n'a AUCUNE longueur
   propre (x -> s x => E -> E/s^2 exactement). Les couplages de la
   couche (Lambda rho_J, E_0, gamma_J) sont sans dimension ou de pure
   normalisation : aucune combinaison ne fabrique une longueur.

2. SURCRITICITE : 2m Lambda rho_J / hbar^2 = 6.6 >> 1/4 (3D) ; dans le
   plan transverse au tube (2D), TOUTE attraction en 1/d^2 est
   surcritique : chute au centre. Le spectre est alors fixe par la
   coupure UV (le rayon d'exclusion epsilon du tube) -- transmutation
   dimensionnelle, structurellement identique a Lambda_QCD.

3. DEMONSTRATION NUMERIQUE (solve_knot('3_1'), n_coarse=24) :
       eps = 0.30 : E1 =  -2.16
       eps = 0.20 : E1 =  -9.39
       eps = 0.10 : E1 = -77.9
   Pas de limite eps -> 0 : E est domine par la coupure (croissance
   plus rapide que 1/eps^2 -- geometrie 2D transverse + raffinement).

CONSEQUENCES :
 - m_l N'EST PAS derivable de la couche microscopique actuelle : la
   calibration m_l = 310 MeV EST la mesure de la longueur UV du tube,
   D = 0.0214 fm (= la constante de transmutation de la TQNT).
 - La cible de derivation MIGRE : de « profil du tube par le solveur »
   (ferme : invariant d'echelle) vers la COMPLETION UV -- quelle
   physique fixe D ? Identification candidate (la seule qui relie le
   coeur au niveau CS) :
       D = 2 pi hbar c / ((k+2) sqrt(sigma_chi))   (+1.2 %)
   soit m_l = 2 pi l(3_1) sqrt(sigma_chi)/(k+2) = 313.6 MeV --
   coherente avec alpha = 1/(k+2) emergeant au coeur du vortex, MAIS au
   niveau coincidence (null des formes, scan_ml_sigma_forms) tant
   qu'un mecanisme de flux CS ne la porte pas.
 - Curiosite structurelle : la surcriticite implique une invariance
   d'echelle DISCRETE (tour d'Efimov, s_0 = sqrt(6.6 - 1/4) = 2.52,
   ratio e^{pi/s_0} = 3.48) -- piste non exploree pour les hierarchies.
"""
import math
import sys

sys.path.insert(0, ".")


def run(numeric=False):
    from theory.postulates import LAMBDA
    from theory.chi_field import jones_rho
    rho = jones_rho("3_1")
    lam = 2 * LAMBDA * rho          # unites 2m/hbar^2 = 1
    s0 = math.sqrt(lam - 0.25)
    print("=== m_l et le profil du tube : verdict structurel ===\n")
    print("couplage 2m Lambda rho/hbar^2 = %.1f (critique 3D : 0.25) ->" % lam)
    print("SURCRITIQUE ; transverse 2D : toute attraction 1/d^2 surcritique.")
    print("Invariance d'echelle du continu : E(s x) = E(x)/s^2 EXACTE.")
    print("=> le spectre est fixe par la coupure epsilon : TRANSMUTATION.")
    print("Efimov : s_0 = %.2f, ratio de tour e^(pi/s0) = %.2f\n" % (
        s0, math.exp(math.pi / s0)))
    if numeric:
        from simulation.solver import solve_knot
        for eps in (0.3, 0.2, 0.1):
            r = solve_knot("3_1", n_coarse=24, epsilon=eps, n_eigenvalues=1)
            print("  eps = %.2f : E1 = %.2f (E1*eps^2 = %.3f)" % (
                eps, r["eigenvalues"][0], r["eigenvalues"][0] * eps ** 2))
    K = 135
    ml = 2 * math.pi * 16.372 * 417.65 / (K + 2)
    print("Candidat UV (a porter par un mecanisme de flux CS, sinon")
    print("coincidence au null des formes) :")
    print("  D = 2 pi hbarc/((k+2) sqrt(sigma)) -> m_l = %.1f MeV (vs 310 : %+.1f %%)" % (
        ml, 100 * (ml / 310 - 1)))
    print("\nVERDICT : m_l = constante de transmutation dimensionnelle de la")
    print("couche microscopique (analogue Lambda_QCD) ; PAS derivable des")
    print("couplages sans dimension ; la cible migre vers la completion UV.")


if __name__ == "__main__":
    run(numeric="--numeric" in sys.argv)
