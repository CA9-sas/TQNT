"""Constitution d'une perle de Classe A : contrôles et test de Liouville."""
import math

Msun, G, hbar, h_p, c = 1.989e30, 6.6710e-11, 1.0546e-34, 6.626e-34, 2.998e8
GeV, AU, pc, kB = 1.7827e-27, 1.496e11, 3.086e16, 1.381e-23
M_PL = math.sqrt(hbar * c / G)

M_TYP, N_KNOTS, R_CAT, M_CH_CAT = 2.03e-3 * Msun, 1.3e54, 21e3, 0.50 * Msun
M_ER = 1.8 * GeV


def rayon_degenerescence(M, m):
    return (9 * math.pi / 4) ** (2 / 3) * hbar ** 2 / (G * m ** (8 / 3) * M ** (1 / 3))


def Q_gel(m, T_c_MeV=62.0):
    """Densité d'espace des phases au gel : rho / sigma_v^3."""
    T = T_c_MeV * 1e6 * 1.602e-19
    z = T / kB / 2.725
    n = 0.265 * 9.47e-27 * (1 + z) ** 3 / m
    return n * m / math.sqrt(3 * T / m) ** 3


def Q_objet(M, R):
    rho = M / ((4 * math.pi / 3) * R ** 3)
    return rho / math.sqrt(G * M / R) ** 3


def R_min_liouville(M, Q):
    return (3 / (4 * math.pi * Q) * M ** -0.5 * G ** -1.5) ** (2 / 3)


def rayon_einstein(M, D=8e3 * pc, x=0.5):
    return math.sqrt(4 * G * M / c ** 2 * D * x * (1 - x))


def statistique(famille, q):
    """Parité de tresse : T(2,q) -> mot de longueur q ; T(3,q) -> 2q."""
    longueur = q if famille == 2 else 2 * q
    return "fermion" if longueur % 2 else "boson"


if __name__ == "__main__":
    print("=== CONTRÔLES DU CATALOGUE ===")
    print(f"  N x m        = {N_KNOTS*M_ER/Msun:.3e} M_sun   (catalogue 2,03e-3)")
    Rd = rayon_degenerescence(M_TYP, M_ER)
    print(f"  R_degen      = {Rd/1e3:.1f} km          (catalogue 21 km, {Rd/R_CAT-1:+.0%})")
    print(f"  M_Pl^3/m^2   = {M_PL**3/M_ER**2/Msun:.3f} M_sun  <- le 0,50 du catalogue")
    print(f"     Chandrasekhar newtonien x3,1 : {3.1*M_PL**3/M_ER**2/Msun:.2f} M_sun")
    print(f"     Oppenheimer-Volkoff   x0,384 : {0.384*M_PL**3/M_ER**2/Msun:.2f} M_sun")

    Qg = Q_gel(M_ER)
    Qp = Q_objet(M_TYP, R_CAT)
    print(f"\n=== TEST DE LIOUVILLE ===")
    print(f"  Q_gel  = {Qg:.3e}    Q_perle = {Qp:.3e}")
    print(f"  rapport exigé = {Qp/Qg:.2e}  -> INTERDIT sans dissipation")
    Rm = R_min_liouville(M_TYP, Qg)
    print(f"  R_min  = {Rm/AU:.2f} UA ({Rm/1e3:.2e} km), soit {Rm/R_CAT:.2e} fois R_catalogue")

    print(f"\n=== RÉGIME DE LENTILLE ===")
    for Ms in (5.3e-11, 1.4e-8, 2.03e-3, 0.5):
        M = Ms * Msun
        rm, re = R_min_liouville(M, Qg), rayon_einstein(M)
        print(f"  M={Ms:9.2e} M_sun : R_min/R_E = {rm/re:9.2e}"
              f"  {'ÉTENDUE' if rm > re else 'ponctuelle'}")

    print(f"\n=== STATISTIQUE DU RÉSERVOIR (parité de tresse) ===")
    print(f"  contrôle trèfle T(2,3) : {statistique(2,3)}  (l'électron : attendu fermion)")
    for q in (23, 25, 27):
        print(f"  T(2,{q:2d}) c={q:2d} : {statistique(2,q)}")
    for q in (13, 14, 16):
        print(f"  T(3,{q:2d}) c={2*q:2d} : {statistique(3,q)}")


def rayon_melange(M, m, x_boson):
    """R(x) = R(0) (1-x)^(5/3) : seuls les fermions fournissent la pression."""
    return rayon_degenerescence(M, m) * (1.0 - x_boson) ** (5.0 / 3.0)


def fraction_bosonique_critique(M, m):
    """x au-delà duquel le rayon tombe sous celui de Schwarzschild."""
    Rs = 2 * G * M / c ** 2
    return 1.0 - (Rs / rayon_degenerescence(M, m)) ** 0.6
