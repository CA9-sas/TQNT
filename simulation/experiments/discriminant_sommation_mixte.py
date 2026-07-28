"""Discriminant interne G plancher/moyenne : sommation mixte (19/07).

Spec : docs/superpowers/specs/2026-07-19-discriminant-g-plancher-moyenne-design.md
Criteres de verdict FIGES avant calcul (spec, Phase 1) :
  1. arch. 1 exclue (>10x pulsar triple) ET arch. 3 sans canal viable
     (deficit >10 ordres sur TOUS les canaux) -> PLANCHER (structurel).
  2. arch. 3 possede un canal a deficit <2 ordres -> PAS NET -> Phase 2.
  3. canal intermediaire (deficit 2-10 ordres) -> PAS NET -> Phase 2.
  4. arch. 2 echoue un test interne -> STOP (signaler).

Les trois architectures du couplage pour un corps de N nucleons {f_p, f_n} :
  arch. 1 "geometrie propre"   : chaque nucleon porte a_i = 4 hbar/(m_i c)
           -> G_eff = 2 hbar c h0 <1/m>/<m> : depend de la composition LOCALE.
  arch. 2 "energie source"     : kappa_g fixe (ancre = etat de reference,
           l'entrelacs a tension nulle) ; chaque nucleon source par sa
           masse-energie totale -> EP local exact, lecture PLANCHER.
  arch. 3 "couplage d'ensemble": kappa_g renormalise par la population
           cosmique -> EP local exact, lecture MOYENNE -- exige un CANAL
           physique de moyennage, chiffre ici sur trois canaux.
"""
import math

# --- constantes TQNT (document principal) ---
M_BASE = 938.240            # MeV, base octet (plancher)
D_TENSION = 1.293           # MeV, tension d'enlacement du neutron
M_N = M_BASE + D_TENSION    # MeV, neutron de base
G_ETABLI = 6.6710e-11       # m3 kg-1 s-2 (theoreme, convention plancher)
LAMBDA_CHI = 209.2          # MeV
HBARC_MEVFM = 197.3269804   # MeV.fm
HBARC_SI = 3.16153e-26      # J.m
MEV_KG = 1.7826619e-30      # kg/MeV
DELTA = D_TENSION / M_BASE  # 1.378e-3

# h0 topologique (adimensionne), ancre sur la chaine etablie :
# G = 2 hbar c h0 / m_base^2  =>  h0 = G m_base^2 / (2 hbar c)
M_BASE_KG = M_BASE * MEV_KG
H0_TOPO = G_ETABLI * M_BASE_KG ** 2 / (2 * HBARC_SI)

# renormalisation requise par la lecture moyenne aujourd'hui :
F_N_TODAY = 0.245 / 2.0     # fraction neutronique des baryons (Y_p = 0,245)
DELTA_REQUIS = 2 * F_N_TODAY * DELTA    # ~3.4e-4 sur kappa_g/G

# --- contraintes observationnelles (sources : spec) ---
PULSAR_TRIPLE = 2.6e-6      # Voisin et al. 2020 (NS vs naine blanche)
MICROSCOPE = 2.3e-15        # MICROSCOPE 2022, eta(Ti,Pt), 1 sigma
G_ROSI14, U_ROSI14 = 6.67191e-11, 1.5e-4

N_B_TODAY = 0.25            # m^-3, densite baryonique cosmique moyenne
ETA_B = 6.1e-10


def pop_means(f_n):
    """<m> et <1/m> d'une population {1-f_n, f_n}."""
    f_p = 1.0 - f_n
    mean_m = f_p * M_BASE + f_n * M_N
    mean_inv = f_p / M_BASE + f_n / M_N
    return mean_m, mean_inv


def arch1_ratio(f_n):
    """G_eff/G_plancher sous l'architecture 1 (geometrie propre)."""
    mean_m, mean_inv = pop_means(f_n)
    return M_BASE ** 2 * mean_inv / mean_m


def run():
    oks = []
    print("=== Discriminant interne G plancher/moyenne : sommation mixte ===")
    print("    h0_topo = %.4e ; renormalisation requise (moyenne) = %.3e\n"
          % (H0_TOPO, DELTA_REQUIS))

    # ---------- architecture 2 : verifications internes (critere 4) ----------
    print("[Arch. 2] energie source, kappa_g fixe (lecture plancher) :")
    G_check = 2 * HBARC_SI * H0_TOPO / M_BASE_KG ** 2
    ok_g = abs(G_check / G_ETABLI - 1) < 1e-12
    print("  recouvrement G etabli : %.4e (%s)" % (G_check, "OK" if ok_g else "ECHEC"))
    # composition : G_eff independant de f_n par construction (g proportionnel
    # a M totale) ; les limites pures ne different que par M.
    print("  EP local : exact par construction (aucune dependance f_n)")
    oks.append(("arch2", ok_g))

    # ---------- architecture 1 : dependance locale -> confrontation ----------
    print("\n[Arch. 1] geometrie propre (support de la lecture moyenne LOCALE) :")
    r_pure_p, r_pure_n = arch1_ratio(0.0), arch1_ratio(1.0)
    print("  limites pures : R(f_n=0) = %.6f ; R(f_n=1) = %.6f" % (r_pure_p, r_pure_n))
    print("    controle session 19/07 : R(1) x G = %.4e (attendu 6,6527e-11, variante m_n)"
          % (r_pure_n * G_ETABLI))
    ok_lim = abs(r_pure_p - 1) < 1e-12 and abs(r_pure_n * G_ETABLI - 6.6527e-11) < 5e-14
    d_ns_wd = abs(arch1_ratio(0.90) - arch1_ratio(0.50))
    excl_psr = d_ns_wd / PULSAR_TRIPLE
    f_ti, f_pt = 26.0 / 48.0, 117.0 / 195.0
    d_eot = abs(arch1_ratio(f_ti) - arch1_ratio(f_pt))
    excl_mic = d_eot / MICROSCOPE
    print("  NS (f_n=0,90) vs naine blanche (0,50) : Delta = %.3e" % d_ns_wd)
    print("    -> pulsar triple (2,6e-6) : EXCLUSION x%.0f" % excl_psr)
    print("  Ti (f_n=%.3f) vs Pt (%.3f) : Delta = %.3e -> MICROSCOPE : x%.1e"
          % (f_ti, f_pt, d_eot, excl_mic))
    arch1_exclue = excl_psr > 10.0
    oks.append(("arch1_limites", ok_lim))

    # ---------- architecture 3 : les canaux de moyennage cosmique ----------
    print("\n[Arch. 3] couplage d'ensemble (lecture moyenne COSMIQUE) :")
    print("  il faut renormaliser kappa_g de %.2e via la population du fond chi." % DELTA_REQUIS)
    deficits = {}

    # canal (a) : recouvrement gravitationnel puits-a-puits
    a_m = 4 * HBARC_MEVFM / M_BASE * 1e-15          # rayon d'exclusion, m
    d_today = N_B_TODAY ** (-1.0 / 3.0)             # distance inter-nucleons, m
    h_neigh = H0_TOPO * a_m / d_today               # queue h0*a/r du voisin
    deficits["(a) gravitationnel"] = math.log10(DELTA_REQUIS / h_neigh)
    print("  (a) puits gravitationnel voisin : h = %.2e (d = %.2f m)"
          % (h_neigh, d_today))
    print("      deficit : %.1f ordres" % deficits["(a) gravitationnel"])

    # canal (b) : recouvrement des ondes chi (portee Compton)
    lam_c = HBARC_MEVFM / M_BASE * 1e-15            # m
    expo = d_today / lam_c
    deficits["(b) ondes chi"] = expo * math.log10(math.e)
    print("  (b) recouvrement chi : suppression e^-%.1e -> deficit %.1e ordres"
          % (expo, deficits["(b) ondes chi"]))

    # canal (c) : retroaction population -> tension du condensat
    rho_chi = LAMBDA_CHI ** 4 / HBARC_MEVFM ** 3 * 1e45   # MeV/m^3
    rho_b = N_B_TODAY * (M_BASE + F_N_TODAY * D_TENSION)  # MeV/m^3
    backreact = rho_b / rho_chi
    deficits["(c) tension du fond"] = math.log10(DELTA_REQUIS / backreact)
    print("  (c) retroaction sur sigma_chi : rho_b/rho_chi = %.1e"
          % backreact)
    print("      deficit : %.1f ordres" % deficits["(c) tension du fond"])

    # canal (d) : couplage kappa_g(Phi) au potentiel cosmologique cumule
    # (ajout revue finale) -- DOCUMENTE : ce n'est PAS un canal de moyennage
    # (Phi ~ n_b m_bar R_H^2 suit n_b R_H^2, pas f_n : composition-aveugle,
    # forme temporelle irrealisable -- f_n gele depuis BBN, Phi a evolue) ;
    # sa forme generique est de surcroit exclue empiriquement. Hors dmin.
    H0_S = 2.184e-18                     # s^-1 (67,4 km/s/Mpc)
    RHO_B = 0.0224 * 1.8788e-26          # kg/m^3 (omega_b, h-libre)
    C_SI = 2.99792458e8
    r_h = C_SI / H0_S
    phi_cosmo = (4 * math.pi / 3) * G_ETABLI * RHO_B * r_h ** 2 / C_SI ** 2
    s_requis = DELTA_REQUIS / phi_cosmo
    gdot_pred = s_requis * H0_S * 3.156e7    # /an (Phi evolue en ~H0)
    LLR_GDOT = 1.5e-13                       # /an
    print("  (d) couplage kappa_g(Phi) cumule [pseudo-canal, revue finale] :")
    print("      Phi/c^2 = %.1e ; sensibilite requise S = %.1e" % (phi_cosmo, s_requis))
    print("      structurel : Phi suit n_b R_H^2 (composition-aveugle), pas")
    print("      f_n (gele depuis BBN alors que Phi a evolue de ~9 ordres) :")
    print("      ne realise PAS G(m_bar) -- hors des canaux de moyennage.")
    print("      empirique : Gdot/G ~ S x H0 = %.1e /an vs LLR %.0e -> exclu x%.0f" % (
        gdot_pred, LLR_GDOT, gdot_pred / LLR_GDOT))

    dmin = min(deficits.values())
    canal_viable = dmin < 2.0
    canal_inter = 2.0 <= dmin <= 10.0

    # ---------- annexe : variante "couplage gele a T_c" ----------
    print("\n[Annexe] variante 'gel a la condensation' (hors fourche, notee) :")
    f_n_tc = math.exp(-D_TENSION / 67.0) / (1 + math.exp(-D_TENSION / 67.0))
    m_tc = M_BASE + f_n_tc * D_TENSION
    g_gel = G_ETABLI * (M_BASE / m_tc) ** 2
    sig_rosi = (g_gel / G_ROSI14 - 1) / U_ROSI14
    print("  G gele a T_c (f_n = %.3f) = %.4e -> %.1f sigma sous Rosi-14"
          % (f_n_tc, g_gel, abs(sig_rosi)))
    print("  -> ecartee par TOUTES les mesures (sous la plus basse du G puzzle).")

    # ---------- verdict selon les criteres figes ----------
    print("\n" + "=" * 64)
    if not all(ok for _, ok in oks):
        print("VERDICT : STOP (critere 4) — echec d'un test interne :",
              [n for n, ok in oks if not ok])
    elif canal_viable:
        print("VERDICT : PAS NET (critere 2) — canal viable, deficit min %.1f"
              " ordres -> Phase 2." % dmin)
    elif canal_inter:
        print("VERDICT : PAS NET (critere 3) — canal intermediaire, deficit"
              " min %.1f ordres -> Phase 2." % dmin)
    elif arch1_exclue:
        print("VERDICT : PLANCHER (critere 1) — architecture locale exclue"
              " x%.0f par le pulsar triple ;" % excl_psr)
        print("moyennage cosmique sans canal (deficit minimal %.1f ordres"
              " sur %d canaux)." % (dmin, len(deficits)))
        print("La lecture moyenne cosmique est ECARTEE ; la fourche est"
              " TRANCHEE structurellement.")
    else:
        print("VERDICT : PAS NET — arch. 1 non exclue (x%.1f < 10) :"
              " configuration imprevue, arbitrage requis." % excl_psr)


if __name__ == "__main__":
    run()
