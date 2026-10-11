"""Définir le défaut primordial de la trame candidat à la Classe F.
Cible : M ~ 3e5 Msun, R ~ 10-30 UA, f_F ~ 1e-5, primordial, sans horizon, survit."""
from __future__ import annotations
import math

G, c, hbar = 6.6710e-11, 2.998e8, 1.0546e-34
Msun, AU, pc, yr = 1.989e30, 1.496e11, 3.086e16, 3.156e7
MeV_J = 1.602e-13
HBARC_MeV_m = 197.327e-15          # MeV.m
LAMBDA_CHI, SQRT_SIGMA_CHI = 209.2, 418.47   # MeV
ALPHA = 1 / 137.036
M_F, R_F = 3e5 * Msun, 20 * AU
RHO_DM_MPC3 = 0.265 * 1.27e11      # Msun / Mpc3


def energie_par_surface(scale_MeV):
    """sigma_w = scale^3 / (hbar c)^2  ->  J/m2."""
    return scale_MeV ** 3 / HBARC_MeV_m ** 2 * MeV_J


def tension_corde():
    """mu = sigma_chi = (sqrt sigma_chi)^2 / (hbar c)  ->  J/m."""
    return SQRT_SIGMA_CHI ** 2 / HBARC_MeV_m * MeV_J


def t_domination_parois(sigma_w):
    """rho_wall ~ sigma_w/(c^2 t) égale rho_crit ~ 1/(G t^2)  ->  t ~ c^2/(G sigma_w)."""
    return c ** 2 / (G * sigma_w)


def horizon_comobile(T_MeV, gstar=10.75):
    t = 2.4 / math.sqrt(gstar) * (1 / T_MeV) ** 2
    z1 = T_MeV * 1e6 / 2.35e-4
    return c * t * z1, c ** 3 * t / G


if __name__ == "__main__":
    print("=== 1. LE ZOO : ce que la brisure du cadre AUTORISE (Kibble) ===")
    print("  Champ chi : SU(2), condensation au gel vers un vide à structure DISCRÈTE (Z3).")
    print("  Variété du vide SU(2)/H avec H discret :")
    print("    pi_0 != 0 -> PAROIS Z3           : oui (« coût de paroi Z3 », parois sur les surfaces nodales)")
    print("    pi_1 != 0 -> CORDES              : oui (le réseau de cordes chi du gel)")
    print("    pi_2(SU(2)/H) = pi_2(S^3) = 0  -> MONOPÔLES : NON (il faut un U(1) continu non brisé)")
    print("    pi_3(S^3) = Z  -> TEXTURES        : oui — ce sont les NŒUDS, les particules elles-mêmes")
    print("  Cohérent avec la table V7 : « n_M/n_gamma ~ 0 (pas de défaut 0D) ».")
    print("  Le monopôle n'est donc pas un candidat : le cadre ne le produit pas.\n")

    print("=== 2. CHAQUE DÉFAUT CONTRE LA CIBLE (3e5 Msun, 20 UA, f_F ~ 1e-5) ===")
    sw = energie_par_surface(LAMBDA_CHI)
    R_bulle = math.sqrt(M_F * c ** 2 / (4 * math.pi * sw))
    print(f"  (a) PAROI Z3, tension ~ Lambda_chi^3 = {sw:.2e} J/m2")
    print(f"      bulle fermée de 3e5 Msun : R = {R_bulle/pc:.2f} pc = {R_bulle/AU:.1e} UA  (cible 20 UA : x{R_bulle/R_F:.0e})")
    print(f"      une bulle de paroi n'a rien qui la soutienne : elle s'effondre sous sa tension")
    print(f"      -> rayonne ou forme un trou noir (R_S = {2*G*M_F/c**2/1e3:.0f} km).")
    print(f"      réseau de parois : domine l'univers à t ~ c^2/(G sigma_w) = {t_domination_parois(sw)/yr:.1f} an")
    print(f"      -> catastrophe cosmologique ; le cadre DOIT les annihiler au gel (parois liées aux cordes).")
    print("      EXCLU : ne survit pas, et si elle survivait, ni la taille ni la stabilité.\n")

    mu = tension_corde()
    Gmu = G * mu / c ** 2
    L_loop = M_F * c ** 2 / mu
    print(f"  (b) CORDE chi, tension sigma_chi = {mu:.2e} J/m, G mu/c^2 = {Gmu:.1e}")
    print(f"      boucle d'horizon à t = 1 s : {mu*c*1/c**2:.1e} kg ; pour 3e5 Msun : L = {L_loop:.1e} m")
    print("      EXCLU : trop légère de 39 ordres — échelle QCD, pas GUT.\n")

    M_mono = LAMBDA_CHI / ALPHA
    print(f"  (c) MONOPÔLE : n'existe pas (pi_2 = 0). S'il existait : M ~ Lambda_chi/alpha = {M_mono/1e3:.0f} GeV,")
    print("      une particule — et des particules massives sans collision ramènent à la dette 1. EXCLU.\n")

    L_knot = M_F * c ** 2 / mu
    print(f"  (d) NŒUD GÉANT (texture pi_3) : masse = sigma_chi x longueur -> L = {L_knot:.1e} m,")
    print(f"      croisements ~ 1e64. Même impasse que la corde. EXCLU.\n")
    print("  => AUCUN défaut de la variété du vide de chi ne peut être la Classe F.\n")

    print("=== 3. LE SEUL SENS RESTANT : UN DÉFAUT DU RÉSEAU DE RAILS LUI-MÊME ===")
    print("  Non un défaut du champ dans la trame, mais de la TRAME : dislocation/disclinaison du")
    print("  réseau cubique de rails. Ce qu'un tel objet aurait pour lui :")
    print("   - il PRÉEXISTE à t_c (la trame préexiste) et gagne sa masse gravitante à la marche de G :")
    print("     jamais assemblé -> dispensé de dissiper. C'est la sortie de la dette 1.")
    print("   - il est GELÉ par le réseau, comme une dislocation dans un cristal : pas d'effondrement.")
    print("   - il est TRANSPARENT par construction : une onde de matière traverse un défaut de")
    print("     réseau avec une diffusion faible, comme un phonon une dislocation.")
    print("  Ce qui lui manque, et que le cadre ne fournit pas :")
    print("   - une LOI DE DÉPLÉTION : masse = déficit d'ondulation intégré ; la trame nue a une")
    print("     réserve planckienne, un nucléon la déplète de 1,5e-39 sur son tube — quelle déplétion")
    print("     porte un défaut de réseau, sur quel volume ? Sans cette loi, pas de masse.")
    print("   - une ÉCHELLE DE TAILLE : la trame n'a que l_Pl et, au gel, xi. 20 UA = 3e12 m n'est ni l'un ni l'autre.\n")

    print("=== 4. LE COMPTAGE DE KIBBLE, POUR SITUER ===")
    print("  Si les défauts de réseau naissent à raison de ~1 par volume de corrélation :")
    n_cible = 1e-5 * RHO_DM_MPC3 / 3e5
    print(f"  cible : n_F = f_F rho_DM / M_F = {n_cible:.0f} par Mpc3 comobile")
    for lab, T in (("au gel, T = 62 MeV", 62.0), ("à T = 1 MeV (M_hor = 1,5e5 Msun)", 1.0)):
        Lc, Mh = horizon_comobile(T)
        n_hor = 1 / (Lc / 3.086e22) ** 3
        print(f"  {lab:34s} : horizon comobile {Lc/pc:9.2e} pc, M_hor = {Mh/Msun:8.1e} Msun,"
              f" 1/horizon^3 = {n_hor:9.1e} /Mpc3  (cible : x{n_hor/n_cible:.0e})")
    # température à laquelle « un défaut par horizon » donne exactement n_cible (n ∝ T^3)
    Lc62, _ = horizon_comobile(62.0)
    n62 = 1 / (Lc62 / 3.086e22) ** 3
    T_ok = 62.0 * (n_cible / n62) ** (1 / 3)
    Lc_ok, Mh_ok = horizon_comobile(T_ok)
    print(f"  {'pour tomber sur la cible':34s} : T = {T_ok*1e6:.0f} eV, M_hor = {Mh_ok/Msun:.1e} Msun,"
          f" M_F/M_hor = {M_F/Mh_ok:.1e}")
    print("  -> un défaut par horizon au gel en donne 1e19 de trop ; la cible n'est atteinte qu'à")
    print(f"     T ~ {T_ok*1e6:.0f} eV, où la masse de l'horizon vaut {Mh_ok/Msun:.0e} Msun : le défaut pèserait")
    print(f"     {M_F/Mh_ok:.0e} de son horizon. Aucun mécanisme ne produit ce rapport.")
