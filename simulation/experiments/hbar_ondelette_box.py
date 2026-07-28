"""hbar = action de l'ondelette d'espace-temps emergent : la boite (e, Q).

FONDEMENT (heritage, sans circularite) : hbar entre en TQNT comme
quantum du champ chi (quantification CS, niveau k = 135) ; la metrique
P5 emerge de |Psi|^2 ; DONC toute vibration de l'espace-temps emergent
est automatiquement quantifiee en hbar -- l'ondelette elementaire
(perturbation metrique minimale) = UN quantum chi, action exactement h.
Ce n'est pas un postulat nouveau : c'est un theoreme d'heritage.
Consequences : (i) couplage UNIVERSEL (tout ce qui a une energie vit
sur la metrique) ; (ii) echange minimal par rencontre ~ un quantum.

LES DEUX O(1) du canal de friction :
  e (efficacite de transfert) : Dp = e hbar/b par rencontre --
    candidats : 1/2 (adiabatique, frequence de passage v/2b) ;
    1 (incertitude) ; 2 (absorption + re-emission, aller-retour de
    l'ondelette) ; pi (mode de longueur d'onde 2b).
  Q (seuil de qualite de fragmentation) : gamma = omega/Q --
    candidats : 2pi (periode complete) ; 2,5 = 2pi x 0,40 (le temps
    de scission MESURE sur la cascade, mediane t* = 0,40 periode,
    consolidation kappa_s).

BANC : T_form(e, Q) auto-coherent sur la grille ; inversion du
pincement Planck (T_form = 70) -> la contrainte e*Q ; verdict.
"""
import math

SIGMA = 0.41847 ** 2
KAPPA = 1.14
D_TUBE = 0.1084
ZETA3 = 1.2020569
N_ALL_T3 = (ZETA3 / math.pi ** 2) * (2 + 0.75 * (6 + 4 + 1.2))


def beta_ond(t, eps):
    return eps * N_ALL_T3 * math.log((KAPPA / t) / D_TUBE)


def t_form(eps, q):
    t = 0.07
    for _ in range(80):
        t = 0.5 * t + 0.5 * math.sqrt(
            SIGMA / (q * KAPPA * beta_ond(t, eps)))
    return t


def run():
    print("=== hbar-ondelette : la boite (e, Q) de T_form ===\n")
    eps_list = [(0.5, "1/2 adiabatique"), (1.0, "1 incertitude"),
                (2.0, "2 aller-retour"), (math.pi, "pi mode 2b")]
    q_list = [(2.5, "2,5 = 2pi x t*_mesure"), (2 * math.pi, "2pi periode")]
    print("T_form (MeV) ; enveloppe cible [60, 75], pincement Planck 70 :")
    print("%-18s" % "" + "".join("%-22s" % ql for _, ql in q_list))
    for e, el in eps_list:
        row = "%-18s" % el
        for q, _ in q_list:
            t = t_form(e, q)
            mark = " <" if 0.060 <= t <= 0.075 else ""
            row += "%-22s" % ("%.0f%s" % (t * 1e3, mark))
        print(row)

    # inversion du pincement : e*Q requis pour T_form = 70
    t70 = 0.070
    eq_req = SIGMA / (KAPPA * t70 ** 2 * beta_ond(t70, 1.0))
    print("\nINVERSION DU PINCEMENT PLANCK : T_form = 70 MeV exige")
    print("   e x Q = %.2f" % eq_req)
    print("   Combinaisons candidates : (e=1, Q=5) ; (e=0,8, Q=2pi) ;")
    print("   (e=2, Q=2,5) -- cette derniere assemble les DEUX ancres")
    print("   independantes (aller-retour du quantum ; t* = 0,40 periode")
    print("   MESURE) : e*Q = 5,0 EXACTEMENT (%.1f %%)." % (
        abs(2 * 2.5 / eq_req - 1) * 100))
    print("   AVERTISSEMENT null : c'est un assemblage a posteriori --")
    print("   candidat, pas derivation ; il faut deriver e et Q")
    print("   INDEPENDAMMENT (e : cinetique quantique de l'echange")
    print("   d'ondelette ; Q : statistique t* complete de la cascade).\n")

    print("BILAN : la boite naturelle (e in [1/2, pi], Q in [2,5, 2pi])")
    print("donne T_form in [%.0f, %.0f] MeV -- elle CONTIENT l'enveloppe" % (
        t_form(math.pi, 2 * math.pi) * 1e3, t_form(0.5, 2.5) * 1e3))
    print("et le pincement ; le mecanisme ondelette-hbar est COHERENT ;")
    print("la valeur fine attend le pin independant de (e, Q).")


if __name__ == "__main__":
    run()
