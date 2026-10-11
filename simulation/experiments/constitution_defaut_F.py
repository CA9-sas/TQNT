"""Dette 1 : la constitution du défaut de Classe F — un amas d'éremnions de
~3e5 Msun dans 10-30 UA. Permis par Liouville ? Atteignable par quelle route ?"""
from __future__ import annotations
import math

G, c, hbar, kB = 6.6710e-11, 2.998e8, 1.0546e-34, 1.381e-23
Msun, AU, pc, yr, Rsun = 1.989e30, 1.496e11, 3.086e16, 3.156e7, 6.96e8
GeV, m_er = 1.7827e-27, 1.8 * 1.7827e-27
Q_GEL = 5.156e-17                 # kg m^-6 s^3, densité d'espace des phases au gel
SIGMA_M_ER = 3e-25                # cm2/g, éremnion (polarisabilité)
H0, OM_M = 67.2e3 / 3.086e22, 0.315


def Q_objet(M, R):
    rho = M / (4 * math.pi / 3 * R ** 3)
    return rho / math.sqrt(G * M / R) ** 3


def R_min_liouville(M):
    return (3 / (4 * math.pi * Q_GEL) * M ** -0.5 * G ** -1.5) ** (2 / 3)


def t_relax(M, R):
    """Temps de relaxation à deux corps d'un système collisionnel de N particules."""
    N = M / m_er
    t_cross = R / math.sqrt(G * M / R)
    return 0.1 * N / math.log(N) * t_cross


def R_vir(M, z, delta=200):
    rho_m = OM_M * 3 * H0 ** 2 / (8 * math.pi * G) * (1 + z) ** 3
    return (3 * M / (4 * math.pi * delta * rho_m)) ** (1 / 3)


def M_horizon(T_MeV, gstar=10.75):
    t = 2.4 / math.sqrt(gstar) * (1 / T_MeV) ** 2
    return c ** 3 * t / G


if __name__ == "__main__":
    M = 3e5 * Msun
    print("=== 1. L'OBJET EST-IL PERMIS ? (espace des phases) ===")
    print(f"  R_min de Liouville pour 3e5 Msun : {R_min_liouville(M)/Rsun:.2f} R_sun = {R_min_liouville(M)/AU:.1e} UA")
    for R_AU in (10, 20, 30):
        R = R_AU * AU
        Q = Q_objet(M, R)
        print(f"  R = {R_AU:2d} UA : Q_objet/Q_gel = {Q/Q_GEL:.1e}  -> {'PERMIS' if Q < Q_GEL else 'interdit'}"
              f" | sigma_v = {math.sqrt(G*M/R)/1e3:.0f} km/s | compacité 2GM/Rc2 = {2*G*M/(R*c**2):.1e}")
    print(f"  temps de relaxation à 20 UA : {t_relax(M, 20*AU)/yr:.1e} ans (N = {M/m_er:.1e})")
    print("  -> amas SANS collision, kinétiquement soutenu, non relativiste, jamais évaporé :")
    print("     un état final parfaitement cohérent. Ce n'est PAS le cas des perles.\n")

    print("=== 2. L'OBJET EST-IL ATTEIGNABLE ? (énergie) ===")
    for z in (20, 10):
        Rv = R_vir(M, z)
        print(f"  halo de 3e5 Msun virialisé à z={z} : R_vir = {Rv/pc:.1f} pc = {Rv/AU:.1e} UA"
              f" -> contraction x{Rv/(20*AU):.1e} en rayon, énergie de liaison x{Rv/(20*AU):.1e}")
    print("  Sans dissipation, cette énergie ne peut pas être rayonnée. Routes sans collision :\n")

    Rv = R_vir(M, 10); Rf = 20 * AU
    print("  (a) Relaxation violente : le coeur se resserre en éjectant une enveloppe.")
    print("      Conservation d'énergie : M_c^2/R_c = M^2/R_0 -> M_c/M = sqrt(R_c/R_0)")
    print(f"      = {math.sqrt(Rf/Rv):.1e} : il faudrait éjecter {100*(1-math.sqrt(Rf/Rv)):.3f} % de la masse")
    print("      (la relaxation violente en retient > 50 %). NON.")
    print("  (b) Contraction adiabatique par les baryons : facteur ~10 en rayon (brouillon R-4). NON.")
    print("  (c) Fusions hiérarchiques : produisent des halos NFW, concentration ~10-20 ;")
    print(f"      il faudrait un contraste de densité x{(Rv/Rf)**3:.0e} au-dessus du viriel. NON.")

    print("\n  (d) Primordial, effondrement à l'entrée dans l'horizon :")
    for T in (10, 1, 0.1):
        print(f"      T = {T:4.1f} MeV : masse d'horizon = {M_horizon(T)/Msun:.1e} Msun")
    print("      3e5 Msun entre dans l'horizon vers T ~ 1 MeV (t ~ 1 s), APRÈS le gel (62 MeV) :")
    print("      les éremnions existent. Mais l'effondrement à l'horizon exige delta ~ 0,3-0,5 ;")
    print("      le spectre de Starobinsky du cadre donne delta ~ sqrt(A_s) ~ 5e-5 : facteur 1e4.")
    print("      Et un tel effondrement est RELATIVISTE : il donne un trou noir, pas un défaut sans horizon.")
    print(f"      (limite OV d'un gaz d'éremnions dégénéré : 0,19 Msun ; 3e5 Msun n'est soutenu que par la")
    print("       dispersion de vitesse -- impossible à conserver dans un effondrement d'horizon). NON.")

    sigma_chi = (0.418 * GeV) ** 2 / (1.0546e-34 * c) ** 0  # tension en J^2 ... on passe direct en J/m
    mu_J_per_m = 1.4e5
    L_loop = M * c ** 2 / mu_J_per_m
    print("\n  (e) Boucle de corde chi au gel se fragmentant en un amas :")
    print(f"      tension sigma_chi = 0,175 GeV^2 = {mu_J_per_m:.1e} J/m ; pour 3e5 Msun il faut L = {L_loop:.1e} m")
    print(f"      contre un horizon de 4,8e4 m au gel : facteur {L_loop/4.8e4:.0e}. ABSURDE.")

    print("\n=== 3. CE QU'IL FAUDRAIT : UN CANAL DISSIPATIF ===")
    print("  Pour refroidir, il faut ~1 collision par particule par temps dynamique quelque part")
    print("  sur le chemin. sigma/m requis ~ 1/(rho R) à l'échelle considérée :")
    for lab, R in (("au viriel (z=10)", R_vir(M, 10)), ("à 1000 UA", 1000 * AU), ("à 20 UA", 20 * AU)):
        rho = M / (4 * math.pi / 3 * R ** 3)
        sm = 1 / (rho * R) * 10        # m2/kg -> cm2/g
        print(f"  {lab:18s} : sigma/m requis ~ {sm:.1e} cm2/g   (éremnion : {SIGMA_M_ER:.0e}, écart {sm/SIGMA_M_ER:.0e})")
    print("  -> le seuil le plus doux (à 20 UA, état final) est à 15 ordres du cadre ; celui qui")
    print("     compte, au DÉPART de la contraction (viriel), à 28. Et 2e3 cm2/g excède de 3 ordres")
    print("     la borne SIDM des amas : AUCUNE matière noire astrophysiquement permise ne le peut,")
    print("     pas seulement les éremnions. Un tel canal ferait s'effondrer les halos ordinaires")
    print("     et rendrait les éremnions détectables : il casserait l'échafaudage ET l'invisibilité.")
