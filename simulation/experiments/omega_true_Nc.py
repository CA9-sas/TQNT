"""Voie de reconstruction : le VRAI N(c) dans la cinetique de condensation.

FAITS MATHEMATIQUES :
  - Comptages EXACTS des noeuds premiers jusqu'a c=19 (Hoste-Thistlethwaite-
    Weeks 1998 pour c<=16 ; Burton 2020 pour 17-19 -- valeurs de memoire, A
    VERIFIER contre Burton, 'The Next 350 Million Knots').
  - Ratios observes : x4.70 (14), x5.39 (15), x5.48 (16), x5.80 (17),
    x5.99 (18), x6.09 (19) -- ENCORE CROISSANTS a c=19.
  - Asymptotique : inconnue ; borne inf ~6.15 (croissance des alternes,
    Sundberg-Thistlethwaite -- les non-alternes dominent a grand c, donc la
    vraie constante est >= 6.15 ; bornes sup ~10.4).
  => le 2.68^c/c canonique SOUS-COMPTE massivement (facteur ~45 des c=19).

CONSEQUENCE STRUCTURELLE : avec alpha_cond = 1.20, le taux net devient
ratio x e^-1.20 ~ 6 x 0.30 = 1.8 > 1 : LA SERIE DIVERGE -- la cinetique
canonique n'etait convergente que par le sous-comptage. Deux issues :
  (I)  alpha = 1.20 fixe + COUPURE c_max (physique nouvelle a deriver) :
       le ratio Omega devient une fonction de c_max -> quelle coupure
       faudrait-il pour retrouver 5.45 (masses geometriques) ?
  (II) penalite premiers-principes par croisement d^3 e^{sqrt-h}
       (la structure meme de la formule d'alpha_cond) : alpha_CS = 2.153
       -> convergent, mais ou tombe Omega ?

SCENARIOS r_inf (extrapolation du ratio au-dela de 19, relaxation
exponentielle depuis 6.09) : 6.15 / 8.0 / 10.4.
Masses : geometriques (document principal V6) ; c_min sombre = 23 (derive).
"""
import math

# VERIFIES 2026-07-15 sur Thistlethwaite, "The enumeration and classification
# of prime 20-crossing knots" (AGT 25 (2025), Table 3 ; 17-19 = Burton 2020,
# 20 = Thistlethwaite+Burton, tabulations independantes concordantes).
# Correction : N(17) = 8 053 393 (la valeur de memoire 8 053 378 etait fausse
# de 15). NOUVEAU : N(20). Ratios : 5.80, 5.99, 6.09, 6.28 -- ENCORE
# croissants a c=20, et desormais > lambda_alt = (101+sqrt(21001))/40 = 6.1479
# (Sundberg-Thistlethwaite 1998, EXACT) : r_inf > 6.28 observe.
HT = {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165, 11: 552,
      12: 2176, 13: 9988, 14: 46972, 15: 253293, 16: 1388705,
      17: 8053393, 18: 48266466, 19: 294130458, 20: 1847319428}
SCALE = 310.0 / 16.372
SLOPE = (38.62 - 16.372) / 6
L31 = 16.372
K = 135
D3 = (2 * math.cos(math.pi / (K + 2))) ** 3
SQRT_H = math.sqrt(0.75 / (K + 2))
ALPHA_CS = 3 * math.log(2 * math.cos(math.pi / (K + 2))) + SQRT_H


def m_geo(c):
    return SCALE * (L31 + SLOPE * (c - 3)) / 1000.0


def N_true(c, r_inf, tau=8.0):
    """Extrapolation au-dela de c=20 : relaxation exponentielle du dernier
    ratio observe (6.28 a c=20) vers r_inf. Toute asymptote < 6.28 est
    exclue par les donnees (ratio encore croissant a c=20)."""
    if c in HT:
        return float(HT[c])
    N = float(HT[20])
    for cc in range(21, c + 1):
        r = r_inf - (r_inf - 6.28) * math.exp(-(cc - 20) / tau)
        N *= r
    return N


def omega_ratio(alpha, r_inf, c_max, c_dark=23):
    w = lambda c: m_geo(c) * N_true(c, r_inf) * math.exp(-alpha * c)
    vis = sum(w(c) for c in range(3, 19))
    dark = sum(w(c) for c in range(c_dark, c_max + 1))
    return dark / vis


def run():
    print("=== Le vrai N(c) dans la cinetique ===\n")
    print("alpha_CS premiers-principes = 3 ln d + sqrt(h) = %.4f "
          "(e^-a = %.4f ; d^3 = %.4f)\n" % (ALPHA_CS, math.exp(-ALPHA_CS), D3))

    print("[1] Taux nets par croisement (asymptotique r_inf x e^-alpha) :")
    for r_inf in (6.15, 8.0, 10.4):
        print("   r_inf = %5.2f : alpha=1.20 -> %.2f %s ; alpha_CS -> %.2f %s" % (
            r_inf, r_inf * math.exp(-1.20),
            "DIVERGE" if r_inf * math.exp(-1.20) > 1 else "converge",
            r_inf * math.exp(-ALPHA_CS),
            "DIVERGE" if r_inf * math.exp(-ALPHA_CS) > 1 else "converge"))
    print()

    print("[2] Issue (II) -- penalite CS pure (convergente), c_max = 80 :")
    for r_inf in (6.15, 8.0, 10.4):
        om = omega_ratio(ALPHA_CS, r_inf, 80)
        print("   r_inf = %5.2f : Omega_DM/Omega_b = %.2e  (cible 5.45)" % (r_inf, om))
    print("   -> penalite premiers-principes : MORT (ratio ~ 1e-5..1e-4).\n")

    print("[3] Issue (I) -- alpha = 1.20, ratio en fonction de la coupure c_max :")
    print("    (divergent : chaque c_max donne un ratio ; 5.45 PINCE c_max)")
    for r_inf in (6.15, 8.0, 10.4):
        prev = None
        pinned = None
        for cmx in range(23, 61):
            om = omega_ratio(1.20, r_inf, cmx)
            if prev is not None and prev < 5.45 <= om:
                pinned = cmx
            prev = om
        oms = {cmx: omega_ratio(1.20, r_inf, cmx) for cmx in (25, 30, 35, 40)}
        print("   r_inf = %5.2f : Omega(25)=%.2f  Omega(30)=%.2f  Omega(35)=%.2f  "
              "Omega(40)=%.2f  -> c_max requis pour 5.45 : %s" % (
                  r_inf, oms[25], oms[30], oms[35], oms[40],
                  str(pinned) if pinned else ">60/déjà passé"))
    print()

    print("[4] Distribution dans la famille sombre (alpha=1.20, r_inf=6.15,")
    print("    c_max au pincement) -- implications SIMP :")
    r_inf = 6.15
    # retrouver le c_max pince
    prev, pinned = None, None
    for cmx in range(23, 61):
        om = omega_ratio(1.20, r_inf, cmx)
        if prev is not None and prev < 5.45 <= om:
            pinned = cmx
        prev = om
    if pinned:
        w = [m_geo(c) * N_true(c, r_inf) * math.exp(-1.20 * c)
             for c in range(23, pinned + 1)]
        tot = sum(w)
        print("   c_max pince = %d ; fractions : c=23 -> %.1f%% ; c=%d -> %.1f%% ;" % (
            pinned, 100 * w[0] / tot, pinned, 100 * w[-1] / tot))
        print("   la distribution est DOMINEE PAR c_max (net %.2f > 1) :" % (
            6.09 * math.exp(-1.20)))
        print("   le gros du reservoir est a c ~ %d (M = %.2f GeV, sigma ~ %.0e) --" % (
            pinned, m_geo(pinned), 1e-25 * (1 / 137.036) ** (pinned - 18)))
        print("   INVISIBLE a toute detection ; le bord Migdal c=23-25 devient")
        print("   une fraction %.1e : la tension SIMP s'EVAPORE aussi." % (
            sum(w[:3]) / tot))

    print("\nVERDICT :")
    print(" - Le vrai N(c) CASSE la convergence de la cinetique canonique :")
    print("   le 5.45 historique reposait sur le sous-comptage 2.68^c/c EN PLUS")
    print("   de la loi de masse V5 (double conditionnement).")
    print(" - Penalite premiers-principes (d^3) : convergente mais Omega ~ 1e-4.")
    print(" - alpha = 1.20 + coupure : Omega = 5.45 PINCE c_max (par r_inf) --")
    print("   la voie de reconstruction devient : DERIVER c_max (taille de")
    print("   domaine de correlation chi a T_c ? temps d'assemblage ?). Si")
    print("   c_max derive coincide avec le pincement, la boucle se referme")
    print("   ET la tension SIMP s'evapore (reservoir concentre a c_max).")
    print(" - A VERIFIER : comptages Burton 17-19 (cites de memoire).")


if __name__ == "__main__":
    run()
