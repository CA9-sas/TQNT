"""Refaire le scan alpha -> Omega_DM/Omega_b sous les ENTREES CANONIQUES (15/07).

Remplace le scan historique du chapitre Big Bang (masses c^6.65 + comptage
2.68^c/c + serie convergente -> boite verte 5.5), invalide par les re-audits :
  - masses GEOMETRIQUES canoniques (M_topo = scale*l(K), m_s derive) ;
  - comptages VERIFIES (Table 3 Thistlethwaite 2025 : 17-20 ; r_inf 6.5-10.4) ;
  - c_min = 23 DERIVE (decouplage/survie, derive_cmin_reservoir.py) ;
  - c_max ~ 92 (mur geometrique + verrouillage reseau kappa_dom = 1.14,
    source_xi_caselle_nada.py) ;
  - alpha_cond derive sous masses geometriques : 1.214-1.220 (succes d'ordre).

QUESTIONS :
  [1] que donne le scan alpha sous ces entrees (l'analogue honnete de la
      table historique) ?
  [2] quel alpha fermerait Omega = 5.45 a c_max = 92 (par r_inf) ?
  [3] sensibilite a c_max (la coupure sourcee est-elle le noeud du probleme ?).
"""
import math
from simulation.experiments.omega_true_Nc import (N_true, m_geo, omega_ratio)

R_INFS = (6.5, 8.0, 10.4)   # plancher releve : ratio non-alt 6.50 a c=20, croissant
C_MAX = 92
ALPHA_DERIVE = (1.214, 1.220)


def closure_alpha(r_inf, c_max):
    lo, hi = 1.0, 3.0
    for _ in range(80):
        a = 0.5 * (lo + hi)
        if omega_ratio(a, r_inf, c_max) > 5.45:
            lo = a
        else:
            hi = a
    return 0.5 * (lo + hi)


def run():
    print("=== Scan alpha sous entrees canoniques (masses geo, vrai N, c_min=23) ===\n")

    print("[1] Omega_DM/Omega_b (c_max = %d) :" % C_MAX)
    print("   alpha    " + "".join("  r_inf=%-5.2f " % r for r in R_INFS))
    for a in (1.20, 1.22, 1.40, 1.60, 1.75, 1.94, 2.13, 2.30):
        row = [omega_ratio(a, r, C_MAX) for r in R_INFS]
        print("   %4.2f    %10.2e   %10.2e   %10.2e" % (a, *row))
    print("   (cible Planck : 5.45 +- 0.28 ; alpha derive : 1.214-1.220)\n")

    print("[2] alpha de fermeture (Omega = 5.45 a c_max = %d) :" % C_MAX)
    for r in R_INFS:
        a = closure_alpha(r, C_MAX)
        print("   r_inf = %5.2f : alpha_ferm = %.3f  (taux net r*e^-a = %.3f ;"
              " ecart au derive : %+.0f%%)" % (
                  r, a, r * math.exp(-a), 100 * (a / 1.217 - 1)))
    print()

    print("[3] Sensibilite a c_max (r_inf = %.2f, alpha = 1.217 derive) :" % R_INFS[0])
    for cm in (23, 30, 40, 60, 92, 120):
        om = omega_ratio(1.217, R_INFS[0], cm)
        print("   c_max = %3d : Omega = %10.3e %s" % (
            cm, om, "<- 5.45 ~ c_max 24-25" if om < 6 and om > 1 else ""))
    om23 = omega_ratio(1.217, R_INFS[0], 23)
    print("   -> a alpha DERIVE, MEME la coupure minimale c_max = c_min = 23")
    print("      (reservoir = une seule tranche) donne Omega = %.2f (+%.0f%%) :" % (
        om23, 100 * (om23 / 5.45 - 1)))
    print("      AUCUN c_max coherent n'existe ; le c_max source (~92) donne 1e20.\n")

    print("VERDICT (refaire les analyses, 15/07) :")
    print(" - Le tableau historique (5.5 a alpha=1.20, serie convergente) etait")
    print("   DOUBLEMENT conditionne : loi de masse V5 (boost c^6.65 du cote")
    print("   sombre) ET sous-comptage 2.68^c/c (facteur ~45 a c=19).")
    print(" - Sous les entrees canoniques : serie DIVERGENTE (net ~1.8),")
    print("   Omega(alpha derive, c_max source) ~ 1e20 -- ECHEC franc.")
    print(" - Fermeture : alpha 1.79-2.13 requis vs 1.21-1.22 derive")
    print("   (+47 a +75%) ; a alpha derive AUCUN c_max coherent (Omega(23)=9.0).")
    print(" - COMPTAGES VERIFIES (15/07, Thistlethwaite AGT 2025, Table 3) :")
    print("   N(17)=8 053 393 (memoire corrigee, -15), N(18/19) exacts,")
    print("   N(20)=1 847 319 428 NOUVEAU ; ratio 6.28 a c=20, ENCORE croissant,")
    print("   > lambda_alt = (101+sqrt(21001))/40 = 6.1479 EXACT (Sundberg-")
    print("   Thistlethwaite 1998) ; ratio non-alt 6.50 croissant => r_inf > 6.5.")
    print(" - Statut : reconstruction OUVERTE (voies : loi de masse sombre")
    print("   justifiee ; budget de condensation ; r_inf au-dela de c=20,")
    print("   borne inferieure rigoureuse lambda_alt). Acquis robustes : ordre")
    print("   d'alpha derive, c_min=23, branche invisible, evaporation SIMP.")


if __name__ == "__main__":
    run()
