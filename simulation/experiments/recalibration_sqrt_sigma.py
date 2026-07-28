"""Recalibration de sqrt(sigma_chi) : 417.65 -> 418.47 MeV (18/07).

DIAGNOSTIC (pre-publication) : les residus leptoniques sont UNIFORMES
(e -0.198, mu -0.197, tau -0.191 %) et le nucleon (base octet) porte le
meme -0.199 % -- les deux seuls secteurs « tout derive » de la loi
maitresse (Koide : mu = (3/4) sqrt(sigma) ; octet : base = sqrt(sigma) x
2.242006) demandent le MEME decalage :
    leptons : sqrt(sigma) = 418.455 MeV   (+0.193 %)
    nucleon : sqrt(sigma) = 418.483 MeV   (+0.200 %)
accord mutuel a 7e-5. L'ancienne valeur venait de la pente de Regge
(sigma = sigma_Regge/2pi, alpha' = 0.912 GeV^-2), connue a ~+-2 % :
les secteurs derives la pincent 30x mieux.

RECALIBRATION (moyenne des deux secteurs) :
    sqrt(sigma_chi) = 418.47 +- 0.02 MeV
    <=> alpha' = 0.9088 GeV^-2 (dans la fourchette Regge 0.88-0.93).

PROPAGATION : tout le reste est sans dimension ou ancre ailleurs
(m_l, m_c, m_b, T_c) ; les corrections calibrees (Delta_s, tau_perp,
kappa hyperfin, bases decuplet/vecteur, B de GMOR) se recalibrent et
reabsorbent ; la chaine Omega (unites xi, adimensionnee) bouge de <1 %.
Les scripts anterieurs gardent 417.65 (resultats historiques, ecarts
< 0.5 %, sans effet -- declare).
"""
import math

OLD, NEW = 417.65, 418.47
R = NEW / OLD


def run():
    print("=== Recalibration sqrt(sigma_chi) = %.2f MeV (x%.6f) ===\n" % (NEW, R))
    print("Constantes derivees :")
    print("  sigma_chi   = %.6f GeV^2" % ((NEW / 1000) ** 2))
    print("  Lambda_chi  = %.2f MeV" % (NEW / 2))
    print("  E_Z3        = %.2f MeV  (m_pi = 139.57 : %+.2f %%)" % (
        NEW / 3, 100 * (NEW / 3 / 139.57 - 1)))
    print("  alpha'      = %.4f GeV^-2" % (1 / (2 * math.pi * (NEW / 1000) ** 2)))
    print("  base octet  = %.2f MeV" % (936.4 * R))
    print()
    print("Leptons (S_k = 3/4 (1+sqrt2 cos(2/9+2pi k/3))^2) :")
    for k, (nm, me) in enumerate([("tau", 1776.86), ("e", 0.51100), ("mu", 105.658)]):
        S = 0.75 * (1 + math.sqrt(2) * math.cos(2 / 9 + 2 * math.pi * k / 3)) ** 2
        m = NEW * S
        print("  %-3s : %10.4f vs %10.4f  (%+.3f %%)" % (nm, m, me, 100 * (m / me - 1)))
    print("Nucleon : base %.2f vs p 938.27 (%+.3f %%) ; n 939.57 (%+.3f %%, avant DLk)" % (
        936.4 * R, 100 * (936.4 * R / 938.27 - 1), 100 * (936.4 * R / 939.57 - 1)))
    print()
    print("Propagation chaine Omega :")
    sphys = (NEW / 1000) ** 2 / 0.19733
    xi = (1.14 / 0.067) * 0.19733
    q = sphys * xi
    floor = 1.7143 / q
    print("  quantum sigma_chi*xi = %.3f GeV (etait 2.97) ; plancher = %.3f xi" % (q, floor))
    print("  (etait 0.577) ; ratio corde/masse du bord = x%.2f (inchange) ;" % (q / 1.7143))
    print("  Omega direct = 14.3 x (8/9) x n/L : INCHANGE (mesure en unites xi).")
    print()
    print("VERDICT : recalibration sans re-derivation -- 5 lignes du CSV")
    print("(e, mu, tau, p, n), ~35 sites de texte, corrections calibrees")
    print("reabsorbees ; les quatre etats-phares tombent a <= 0.05 %.")


if __name__ == "__main__":
    run()
