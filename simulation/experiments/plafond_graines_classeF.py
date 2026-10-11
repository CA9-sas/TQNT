"""Plafond sur la fraction de graines Classe F imposé par le théorème de croissance.

Le théorème (V6, volet B.6) fait de l'indiscernabilité d'avec LambdaCDM une
PRÉDICTION : une population à z>10 au-delà de LambdaCDM le réfute. Des graines
massives ajoutent de la puissance d'isocourbure et habillent chacune un halo.
Le plafond est atteint quand la densité de graines égale l'abondance LambdaCDM
de halos de la même masse au même redshift : au-delà, le cadre fabrique un
excès qu'il s'interdit lui-même.

Méthode déclarée : Press-Schechter, sigma(M) ancré sur des valeurs standard,
facteur de croissance D(z)/D(0) ~ 1,3/(1+z) à grand z. Ordre de grandeur.
"""
import math

RHO_C = 1.27e11          # M_sun / Mpc^3 (h = 0,674)
OM_M, OM_DM = 0.315, 0.265
RHO_M, RHO_DM = OM_M * RHO_C, OM_DM * RHO_C
Z_EQ, DELTA_C = 3400.0, 1.686


def sigma_M(M):
    """sigma(M, z=0), interpolation log-linéaire sur des ancres LambdaCDM."""
    anchors = [(1e14, 1.8), (1e12, 3.0), (1e10, 5.0), (1e8, 7.0), (1e6, 9.0)]
    lm = math.log10(M)
    for (m1, s1), (m2, s2) in zip(anchors, anchors[1:]):
        if math.log10(m2) <= lm <= math.log10(m1):
            t = (lm - math.log10(m2)) / (math.log10(m1) - math.log10(m2))
            return s2 + t * (s1 - s2)
    return anchors[-1][1] if lm < 6 else anchors[0][1]


def growth(z):
    return 1.3 / (1.0 + z)


def f_coll(M, z):
    """Fraction de masse effondrée en halos > M (Press-Schechter)."""
    s = sigma_M(M) * growth(z)
    return math.erfc(DELTA_C / (math.sqrt(2.0) * s))


def plafond(M_seed, z):
    """f_F maximal : densité de graines = abondance LambdaCDM de leurs halos."""
    M_halo = M_seed * (1.0 + Z_EQ) / (1.0 + z)
    n_lcdm = f_coll(M_halo, z) * RHO_M / M_halo
    return n_lcdm * M_seed / RHO_DM, M_halo, n_lcdm


if __name__ == "__main__":
    print("=== PLAFOND SUR f_F (théorème de croissance) ===")
    print("  fenêtre actuellement permise par le cadre : 1e-5 <= f_F <= 1e-4\n")
    print(f"  {'M_graine':>10s} {'z':>4s} {'M_halo':>10s} {'n_LCDM':>11s} {'f_F max':>10s}  verdict")
    for M_seed in (1.6e4, 1e5):
        for z in (10, 15, 20, 30):
            fmax, M_halo, n = plafond(M_seed, z)
            v = ("ferme la fenêtre" if fmax < 1e-5 else
                 "coupe le haut" if fmax < 1e-4 else "sans effet")
            print(f"  {M_seed:10.1e} {z:4d} {M_halo:10.2e} {n:11.2e} {fmax:10.2e}  {v}")

    print("\n=== SENSIBILITÉ À LA NORMALISATION DE sigma ===")
    print("  (sigma renormalisé globalement d'un facteur k ; le plafond le plus")
    print("   contraignant reste celui de z=20, M_graine=1,6e4)")
    base = sigma_M
    for k in (0.8, 0.9, 1.0, 1.1, 1.2):
        globals()["sigma_M"] = lambda M, k=k, b=base: k * b(M)
        f, _, _ = plafond(1.6e4, 20)
        print(f"  sigma x {k:.1f}  ->  f_F max = {f:.2e}")
    globals()["sigma_M"] = base

    print("\n=== LECTURE ===")
    print("  Le plafond dépend surtout du redshift auquel on exige que les halos")
    print("  ensemencés ne dépassent pas LambdaCDM : sans effet à z=10, il ferme")
    print("  la fenêtre à z=20-30. C'est donc une contrainte VIVE, pas académique.")
