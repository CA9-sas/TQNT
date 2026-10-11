"""La matière noire comme échafaudage de l'agrégation baryonique dans les
40 premiers millions d'années : ce que les éremnions font, quand, et ce
qui casserait le mécanisme.

Entrées du cadre : éremnions m = 1,8 GeV, gel T_c = 62 MeV, sigma/m ~
3e-25 cm2/g, M_fs = 1e-5 M_sun, sigma(M) via plafond_graines_cadre
(amplitude A_s du cadre, N_* = 63 : lecture cohérente du 15/09).
"""
from __future__ import annotations
import math, sys
sys.path.insert(0, "simulation/experiments")
from plafond_graines_cadre import sigma_M, REF, DELTA_C  # sigma calée, forme EH98

Msun, G, c, kB, mp = 1.989e30, 6.6710e-11, 2.998e8, 1.381e-23, 1.673e-27
Mpc, pc, Myr, GeV = 3.086e22, 3.086e16, 3.156e13, 1.7827e-27
H0 = 67.2e3 / Mpc
OM_M, OM_B = 0.315, 0.0493
Z_EQ, Z_REC, Z_THERM = 3400.0, 1100.0, 150.0

# lecture cohérente : A_s = 2,10e-9 (N_* = 63,3), donc le jeu "REF"
P = REF


def t_of_z(z):
    return 2.0 / (3.0 * H0 * math.sqrt(OM_M)) * (1.0 + z) ** -1.5


def growth(z):
    return 1.3 / (1.0 + z)


def f_coll(M, z):
    s = sigma_M(M, P) * growth(z)
    return math.erfc(DELTA_C / (math.sqrt(2.0) * s))


def z_first(M, frac=1e-6):
    """Redshift où une fraction `frac` de la masse est effondrée en halos > M."""
    lo, hi = 0.0, 3000.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f_coll(M, mid) > frac:
            lo = mid
        else:
            hi = mid
    return lo


def T_baryon(z):
    """Température du gaz : couplée au CMB jusqu'à z~150, adiabatique ensuite."""
    Tcmb = 2.725 * (1.0 + z)
    return Tcmb if z > Z_THERM else 2.725 * (1.0 + z) ** 2 / (1.0 + Z_THERM)


def jeans_mass_baryon(z):
    """Masse de Jeans du gaz primordial (mu = 1,22, gaz neutre)."""
    T = T_baryon(z)
    rho_b = OM_B * 3 * H0 ** 2 / (8 * math.pi * G) * (1.0 + z) ** 3
    rho_m = OM_M * 3 * H0 ** 2 / (8 * math.pi * G) * (1.0 + z) ** 3
    cs = math.sqrt(1.4 * kB * T / (1.22 * mp))
    lam_J = cs * math.sqrt(math.pi / (G * rho_m))
    return (4 * math.pi / 3) * rho_b * (lam_J / 2) ** 3 / Msun


def v_eremnion(z, m=1.8 * GeV, T_c_MeV=62.0):
    """Dispersion de vitesse des éremnions : v ~ v(T_c) (1+z)/(1+z_c)."""
    T = T_c_MeV * 1e6 * 1.602e-19
    z_c = T / kB / 2.725
    return math.sqrt(3 * T / m) * (1.0 + z) / (1.0 + z_c)


if __name__ == "__main__":
    print("=== 1. L'AVANCE PRISE PAR LA MATIÈRE NOIRE ===")
    print("  Entre l'égalité (z=3400) et la recombinaison (z=1100), delta_DM croît")
    print("  pendant que les baryons, couplés aux photons, oscillent sans croître.")
    head = growth(Z_REC) / growth(Z_EQ)
    print(f"  facteur d'avance = D(1100)/D(3400) = {head:.2f}")
    d_cmb = 1e-5
    print(f"\n  Univers SANS matière noire : delta_b part de {d_cmb:.0e} à z=1100")
    print(f"     -> aujourd'hui delta_b ~ {d_cmb*(1+Z_REC)*0.78:.1e}  : jamais non linéaire, AUCUNE galaxie.")
    print(f"  Avec la matière noire : les puits existent déjà, les baryons y tombent")
    print(f"     et rattrapent delta_DM en ~ une décade de facteur d'échelle.\n")

    print("=== 2. CHRONOLOGIE DES ÉREMNIONS DANS LES 40 PREMIERS MYR ===")
    print(f"  {'M (M_sun)':>10s} {'z (1e-6)':>9s} {'t (Myr)':>8s} {'z (1e-3)':>9s} {'t (Myr)':>8s} {'z (1 %)':>8s} {'t (Myr)':>8s}")
    for M in (1e-5, 1e-3, 1.0, 1e3, 1e5, 1e6, 1e8):
        zs = [z_first(M, f) for f in (1e-6, 1e-3, 1e-2)]
        print(f"  {M:10.0e}" + "".join(f" {z:9.0f} {t_of_z(z)/Myr:8.1f}" for z in zs))
    print(f"  ({'M_fs = 1e-5 M_sun : les PREMIERS halos possibles'})")

    print("\n=== 3. LES BARYONS NE PEUVENT PAS SUIVRE AVANT z ~ 20-30 ===")
    print(f"  {'z':>5s} {'t (Myr)':>8s} {'T_gaz (K)':>10s} {'M_Jeans,b (M_sun)':>18s}")
    for z in (200, 100, 56, 30, 20, 15):
        print(f"  {z:5d} {t_of_z(z)/Myr:8.1f} {T_baryon(z):10.1f} {jeans_mass_baryon(z):18.2e}")
    print("  -> à 40 Myr (z=56) le gaz ne s'effondre dans rien de moins de ~1e5 M_sun.")
    print("     Les halos d'éremnions existants font 1e-5 à 1 M_sun : purement sombres.")

    print("\n=== 4. LES ÉREMNIONS SONT FROIDS AU-DELÀ DU NÉCESSAIRE ===")
    for z in (1100, 56, 20):
        v = v_eremnion(z)
        print(f"  z={z:5d} : v_eremnion = {v:.2e} m/s = {v/c:.1e} c")
    print("  -> dispersion négligeable : pas de free-streaming au-dessus de 1e-5 M_sun.")

    print("\n=== 5. CE QUI CASSERAIT L'ÉCHAFAUDAGE ===")
    print("  (a) matière noire CHAUDE (m ~ keV) : M_fs ~ 1e9 M_sun, aucun mini-halo,")
    print("      premières structures retardées de z~30 à z~10. Éremnions : 1,8 GeV -> exclu.")
    print("  (b) matière noire DISSIPATIVE : elle se condenserait AVEC les baryons,")
    print("      formerait des disques sombres et disputerait le centre. Éremnions :")
    print("      sigma/m = 3e-25 cm2/g, 25 ordres sous SIDM -> exclu.")
    print("  (c) matière noire trop PEU abondante : Omega_DM/Omega_b = 5,4 est ce qui")
    print("      donne l'avance ; à Omega_DM ~ Omega_b la croissance s'effondrerait.")
    print("\n  => rien à AJOUTER : les trois propriétés (froide, sans collision,")
    print("     abondante) sont déjà celles des éremnions. Et la propriété qui interdit")
    print("     les perles (Liouville, §P-3) est CELLE qui fait de la matière noire un")
    print("     échafaudage plutôt qu'un concurrent : elle ne peut pas rejoindre le")
    print("     coeur dense, donc elle le laisse aux baryons.")


def survie_halo(M_h=1e12 * Msun, R_h=200 * 3.086e19, N_SN=1e9, E_SN=1e51 * 1e-7):
    """Énergie de rétroaction transmise vs énergie de liaison du halo."""
    E_bind = 3 / 5 * G * M_h ** 2 / R_h
    return {eps: eps * N_SN * E_SN / E_bind for eps in (1.0, 0.05, 0.01)}


def fraction_nfw(r, rs=20 * 3.086e19, R_h=200 * 3.086e19):
    """Fraction de la masse d'un halo NFW à l'intérieur de r."""
    m = lambda x: math.log(1 + x / rs) - (x / rs) / (1 + x / rs)
    return m(r) / m(R_h)
