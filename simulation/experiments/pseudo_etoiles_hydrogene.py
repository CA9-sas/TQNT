"""Pseudo-étoiles d'hydrogène dans les 67 premiers Myr : quels objets,
quelles masses, quelles dates. Physique du gaz primordial standard sur
la hiérarchie sombre du cadre (sigma(M) du cadre, N_* = 63)."""
from __future__ import annotations
import math, sys
sys.path.insert(0, "simulation/experiments")
from echafaudage_sombre_40Myr import t_of_z, f_coll, z_first, T_baryon, jeans_mass_baryon, Myr, Msun
from plafond_graines_cadre import REF, rho_m

G, kB, mp, pc = 6.6710e-11, 1.381e-23, 1.673e-27, 3.086e16
OM_M, OM_B = 0.315, 0.0493
F_B = OM_B / OM_M
MU = 1.22


def R_vir(M, z, delta=200.0):
    rho = rho_m(REF) * Msun / (3.086e22) ** 3 * (1 + z) ** 3
    return (3 * M / (4 * math.pi * delta * rho)) ** (1 / 3)


def T_vir(M, z):
    return MU * mp * G * M / (2 * kB * R_vir(M, z))


def n_vir(z, delta=200.0):
    """Densité baryonique au viriel, cm^-3."""
    rho_b = OM_B * 3 * (67.2e3 / 3.086e22) ** 2 / (8 * math.pi * G) * (1 + z) ** 3
    return delta * rho_b / (MU * mp) * 1e-6


def M_jeans(T, n_cm3):
    """Masse de Jeans du gaz à (T, n), en M_sun."""
    rho = n_cm3 * 1e6 * MU * mp
    cs = math.sqrt(1.4 * kB * T / (MU * mp))
    lam = cs * math.sqrt(math.pi / (G * rho))
    return (4 * math.pi / 3) * rho * (lam / 2) ** 3 / Msun


def M_halo_for_Tvir(T, z):
    """Masse de halo dont la température de viriel vaut T."""
    lo, hi = 1e-3, 1e12
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if T_vir(mid * Msun, z) < T:
            lo = mid
        else:
            hi = mid
    return lo


if __name__ == "__main__":
    print("=== 67 Myr = z ~ 40. La fenêtre s'étend de la recombinaison à z = 40. ===\n")

    print("=== OBJET I : le nuage d'hydrogène retenu par un mini-halo (ne refroidit pas) ===")
    print("  Condition d'entrée du gaz : T_vir(halo) > T_gaz. Condition de refroidissement :")
    print("  H2 (via H-) efficace pour T_vir > ~1000 K ; H atomique pour T_vir > 1e4 K.\n")
    print(f"  {'z':>4s} {'t(Myr)':>7s} {'T_gaz':>6s} {'M_h(T_vir=T_gaz)':>17s} {'M_h(1000 K)':>12s} {'M_h(1e4 K)':>11s}")
    for z in (100, 60, 56, 50, 45, 40):
        Tg = T_baryon(z)
        print(f"  {z:4d} {t_of_z(z)/Myr:7.1f} {Tg:6.0f} {M_halo_for_Tvir(Tg,z):17.1e}"
              f" {M_halo_for_Tvir(1000,z):12.1e} {M_halo_for_Tvir(1e4,z):11.1e}")

    print("\n  Quand ces halos existent-ils (fraction 1e-6, la queue extrême) ?")
    for M in (1e3, 1e4, 1e5, 1e6):
        z6 = z_first(M, 1e-6); z3 = z_first(M, 1e-3)
        print(f"   M_h = {M:7.0e} M_sun : 1e-6 à z={z6:3.0f} ({t_of_z(z6)/Myr:5.1f} Myr) ; 1e-3 à z={z3:3.0f} ({t_of_z(z3)/Myr:5.1f} Myr)")

    print("\n  Portrait de l'objet I à z=50 dans un halo de 1e4 M_sun :")
    z, Mh = 50, 1e4 * Msun
    Tv = T_vir(Mh, z); nv = n_vir(z)
    print(f"   T_vir = {Tv:.0f} K, n_gaz ~ {nv:.1f} cm^-3, R_vir = {R_vir(Mh,z)/pc:.1f} pc")
    print(f"   masse de gaz = f_b M_h = {F_B*1e4:.0f} M_sun")
    print(f"   masse de Jeans interne à (T_vir, n_vir) = {M_jeans(Tv,nv):.0e} M_sun  -> {'> M_gaz : STABLE, ne fragmente pas' if M_jeans(Tv,nv)>F_B*1e4 else 'instable'}")
    print("   -> une sphère de gaz tiède, soutenue par sa pression, SANS refroidissement.")
    print("      Ce n'est pas une étoile : rien ne s'y contracte. C'est la « pseudo-étoile »")
    print("      au sens strict : de l'hydrogène tenu en boule par un puits sombre.")

    print("\n=== OBJET II : le coeur refroidi par H2 (proto-étoile de population III) ===")
    print("  Le refroidissement H2 amène le gaz au « palier » T ~ 200 K, n ~ 1e4 cm^-3.")
    MJ = M_jeans(200, 1e4)
    print(f"  Masse de Jeans au palier : {MJ:.0f} M_sun  -> masse du coeur qui se détache.")
    print("  Il faut un halo à T_vir > ~1000 K, soit M_h > ~1e5-1e6 M_sun (table ci-dessus).")
    print(f"  Premiers tels halos (1e-6) : 1e5 à z={z_first(1e5,1e-6):.0f} ({t_of_z(z_first(1e5,1e-6))/Myr:.0f} Myr),"
          f" 1e6 à z={z_first(1e6,1e-6):.0f} ({t_of_z(z_first(1e6,1e-6))/Myr:.0f} Myr)")
    print("  -> les PREMIERS coeurs H2 apparaissent à z ~ 40-45, soit 57-67 Myr :")
    print("     tout au bord de la fenêtre, et seulement dans les pics à 1e-6.")

    print("\n=== OBJET III : l'étoile de population III allumée ===")
    print("  Le coeur de ~500 M_sun accrète à ~1e-3 M_sun/an ; l'allumage suit en ~1e5 ans.")
    print("  Masse finale : 10 à 300 M_sun (fragmentation limitée, pas de métaux), typique 30-100.")
    print("  -> première lumière stellaire : z ~ 38-42, soit 65-75 Myr. HORS fenêtre ou à sa lisière.")

    print("\n=== ET LES GRAINES DE CLASSE F ? ===")
    Z_EQ = 3400.0
    for z in (40, 50):
        Mh_F = 1.6e4 * (1 + Z_EQ) / (1 + z)
        print(f"  à z={z} : une graine de 1,6e4 M_sun porte déjà un halo de {Mh_F:.1e} M_sun, T_vir = {T_vir(Mh_F*Msun,z):.0f} K")
    print("  -> une graine F franchit le seuil H2 dès z ~ 50 (47 Myr), 15-20 Myr AVANT")
    print("     les pics LambdaCDM. Si f_F approche son plafond, les premières étoiles")
    print("     naissent autour des graines F. Mais le théorème de croissance borne f_F")
    print("     par l'abondance LambdaCDM au z considéré : à z=40-50 le plafond est ~1e-8.")
