"""Tentative de derivation de beta_otimes (tension d'enlacement, 0.210 MeV).

CIBLE PHYSIQUE : le differentiel E(dLk=1) = beta_otimes * <sqrt(cJ cJ)> = 0.210*6.15
  = 1.293 MeV = Delta m(n-p). beta_otimes est le SEUL calibre du cadre (audit 2026-07-13),
  et il est calibre SUR Delta m(n-p) : l'ecart n-p n'est "derive" que si beta_otimes l'est.

HYPOTHESES (primitives TQNT uniquement : sigma_chi, flux quantum chi, r_link, d_tube) :
  H1 - Elasticite de twist : dLk absorbe en torsion du tube (Calugareanu),
       E = 2 pi^2 sigma a^2 dLk^2 / L. Donne la forme harmonique mais ~100 MeV : REFUTE
       comme canal dominant -> le tube prefere writhe/geometrie ; il reste le canal EM.
  H2 - Piegeage de fluxoide (boucle bloquee, cf. three_case_circulation) : chaque unite
       de Lk enlace un quantum de flux chi Phi_chi = alpha*h/e (DERIVE, chaine torsion) ;
       un ecart dLk a l'optimum = mismatch de fluxoide -> E = (dLk*Phi_chi)^2 / (2 L_ind),
       L_ind = mu0 r (ln(8r/a) - c) auto-inductance de la boucle (r = r_link = 0.99 fm,
       a = d_tube = 0.209 fm = lambdabar_C(p), tous deux issus de chaines anterieures).
       QUADRATIQUE en dLk (forme harmonique) et centre sur l'entier optimal (Lk_opt).
  H3 - Echelle nue : alpha hbar c / r_link (l'ordre de grandeur sans le log inductif).

Verdict attendu : H2 encadre la cible (surface -2 : 1.40 ; uniforme -7/4 : 1.21 ;
cible 1.293 ENTRE les deux, ~8%/-6%). Pas exact -> hint fort, pas une derivation fermee.
"""
import math

# --- constantes SI ---
HBARC_MEVFM = 197.3269804
ALPHA = 1.0 / 137.0            # TQNT : 1/(k+2), k=135
H_OVER_E = 4.135667696e-15     # h/e  [Wb]
MU0 = 4e-7 * math.pi           # [H/m]
J_PER_MEV = 1.602176634e-13

# --- entrees TQNT (chaines anterieures, aucune recalibration ici) ---
SQSIG = 417.65                 # MeV ; sigma_chi = SQSIG^2
R_LINK = 0.99e-15              # m ; peripherie d'enlacement (mu_p - mu_n, ~1%)
D_TUBE = 0.209e-15             # m ; rayon du tube = lambdabar_C(p) (moment neutron, 1%)
ROPELENGTH_72 = 32.96          # longueur reduite du 7_2 (unites de rayon de tube)

TARGET = 1.293                 # MeV = Delta m(n-p) = beta_otimes * 6.15
CJ_MEAN = 6.15                 # <sqrt(cJ_i cJ_j)> nucleon (parametrisation catalogue)


def h1_twist_elasticity():
    """E = 2 pi^2 sigma a^2 / L pour dTw = dLk = 1 (torsion uniforme du tube 7_2)."""
    sigma_per_fm = SQSIG**2 / 1e6 / (HBARC_MEVFM / 1e3) * 1e3  # MeV/fm
    a_fm = D_TUBE * 1e15
    L_fm = ROPELENGTH_72 * a_fm
    return 2 * math.pi**2 * sigma_per_fm * a_fm**2 / L_fm


def h2_fluxoid(c_geom):
    """E = Phi_chi^2 / (2 mu0 r (ln(8r/a) - c)) pour dLk = 1."""
    phi_chi = ALPHA * H_OVER_E
    l_ind = MU0 * R_LINK * (math.log(8 * R_LINK / D_TUBE) - c_geom)
    return phi_chi**2 / (2 * l_ind) / J_PER_MEV


def h3_bare():
    return ALPHA * HBARC_MEVFM / (R_LINK * 1e15)


def run():
    print("=== Cible : E(dLk=1) = %.3f MeV (= beta_otimes x %.2f) ===\n" % (TARGET, CJ_MEAN))

    e1 = h1_twist_elasticity()
    print("H1 twist elastique  : %8.1f MeV  (x%.0f trop grand) -> REFUTE comme canal" % (e1, e1 / TARGET))
    print("   (le tube absorbe dLk en writhe/geometrie, pas en twist ; reste le canal EM)\n")

    e3 = h3_bare()
    print("H3 echelle nue alpha*hbar*c/r_link : %.3f MeV (%+.0f%%)\n" % (e3, (e3 / TARGET - 1) * 100))

    print("H2 fluxoide E = (Phi_chi)^2 / (2 L_ind), Phi_chi = alpha h/e (derive) :")
    for label, c in (("courant de surface (c=2)   ", 2.0), ("courant uniforme  (c=7/4)  ", 1.75)):
        e2 = h2_fluxoid(c)
        beta = e2 / CJ_MEAN
        print("   %s: E = %.3f MeV (%+.1f%%) -> beta_otimes = %.3f MeV (vs 0.210)"
              % (label, e2, (e2 / TARGET - 1) * 100, beta))

    # sensibilites : quel (r, a) rendrait la cible exacte ?
    print("\n=== Sensibilite (c=2) ===")
    # r pour cible exacte, a fixe
    phi = ALPHA * H_OVER_E
    for a in (D_TUBE,):
        # resoudre E(r)=TARGET par bissection
        lo, hi = 0.5e-15, 3e-15
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            e = phi**2 / (2 * MU0 * mid * (math.log(8 * mid / a) - 2)) / J_PER_MEV
            if e > TARGET:
                lo = mid
            else:
                hi = mid
        print("   a = d_tube fixe : cible exacte pour r = %.3f fm (vs r_link = %.2f ; +%.0f%%)"
              % (lo * 1e15, R_LINK * 1e15, (lo / R_LINK - 1) * 100))
    # forme harmonique : E(dLk) / E(1) = dLk^2 par construction (fluxoide) -> OK
    print("\nForme harmonique : E proportionnel a (Lk - Lk_opt)^2 par construction")
    print("(mismatch de fluxoide au carre), centre sur l'entier optimal = Lk_opt. OK.")

    print("\n=== VERDICT ===")
    print("H2 (fluxoide) : la cible 1.293 MeV tombe ENTRE les deux conventions d'inductance")
    print("(1.21 / 1.40 MeV, soit -6% / +8%) avec des entrees toutes issues de chaines")
    print("anterieures (Phi_chi derive ; r_link, d_tube identifies). FORME derivee ;")
    print("VALEUR encadree a ~8% -> hint fort, PAS une derivation fermee (le facteur")
    print("geometrique de l'inductance n'est pas pince par la theorie).")


if __name__ == "__main__":
    run()
