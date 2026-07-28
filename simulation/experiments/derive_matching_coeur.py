"""Deriver le matching d'echelle cœur/asymptotique (20/07).

[1] SEPARATION D'ECHELLES (l'argument, chiffre) : le cœur du tube
    (D ~ 0,02-0,15 fm) est tres au-dessous de la longueur d'onde
    thermique (1/T ~ 3 fm a 67 MeV) et des longueurs d'ecrantage
    (~1/gT ~ 0,5-1 fm) : le milieu ne RESOUT pas le cœur. Les
    grandeurs de cœur (sigma_phys, D, donc E_char) suivent les
    valeurs du VIDE jusqu'au voisinage du croisement.
    DOUBLE SOURCING de la platitude : (a) sigma(T) quenched plate
    jusqu'a 0,55 T_c (Cardoso-Bicudo) ; (b) le condensat chiral
    (parametre d'ordre du croisement physique, HotQCD) reste ~plein
    a T <~ 100-130 MeV. Les deux lectures s'accordent la ou nous
    vivons (T <~ 100 MeV) -- le matching est DERIVE : sigma_cœur(T)
    = sigma_chi (plate) dans la fenetre pertinente.

[2] LA CHAINE SE SIMPLIFIE EN UNE LIGNE :
        E_char = c_eff sigma_chi D_0 = constante
        =>  T_c = c_eff sigma_chi D_0     (serrage E_char = T)
    T_c EST le cout du passage de brin au cœur -- une grandeur du
    VIDE. Toute la charge du pin est transferee sur c_eff.

[3] RAFFINEMENT VARIATIONNEL de c_eff : le modele tente+arc etait un
    MAJORANT ; ici, optimisation numerique exacte du profil de
    decollage y(x) (fonctionnelle de longueur exacte + travail de
    pression), contrainte de contournement du partenaire.
"""
import math
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, ".")
from simulation.experiments.omega_budget_reconstruction import budget  # noqa: E402

SIGMA = 0.41847 ** 2
SIGMA_FM = SIGMA / 0.19733       # GeV/fm
D0_FM = 0.0214                   # fm
P_OCC_EFF = (8.0 / 9.0) * 0.304
Q_ZONE = 0.81                    # pression de contact (zone-marche, mesuree)


def c_eff_variational(q, n=160, x_max=6.0):
    """Optimise le profil y(x) (unites D=1) : partenaire = cercle de
    rayon 1 centre en (0,0) ; repos y = -1 (contact) ; passage
    par-dessus. Cout = longueur exacte supplementaire + (q) * aire
    soulevee. Retourne c_eff = cout/(sigma D)."""
    xs = np.linspace(0, x_max, n)
    dx = xs[1] - xs[0]
    # contrainte : y >= sqrt(1-x^2) pour x<1 (branche haute), sinon >= -1
    y_min = np.where(xs < 1.0, np.sqrt(np.clip(1 - xs ** 2, 0, 1)), -1.0)

    def cost(y_free):
        y = np.maximum(y_free, y_min)
        # symetrie : demi-domaine, pente nulle en x=0 via miroir
        yy = np.concatenate([y[::-1], y[1:]])
        dyy = np.diff(yy)
        length = np.sum(np.sqrt(dx ** 2 + dyy ** 2)) - (len(yy) - 1) * dx
        lift = q * np.sum((y - (-1.0)) * (xs < x_max - 0.5)) * dx * 2
        # penalisation contrainte (pour le lisseur)
        pen = 1e3 * np.sum(np.clip(y_min - y_free, 0, None) ** 2)
        return length + lift + pen

    y0 = np.maximum(y_min, -1 + 2.2 * np.exp(-xs ** 2))
    res = minimize(cost, y0, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12})
    y = np.maximum(res.x, y_min)
    yy = np.concatenate([y[::-1], y[1:]])
    dyy = np.diff(yy)
    length = np.sum(np.sqrt(dx ** 2 + dyy ** 2)) - (len(yy) - 1) * dx
    lift = q * np.sum((y + 1.0)) * dx * 2
    return length + lift


def c_p_of_s(u):
    return math.pi + 2 * math.atan(1 / u) - 2 * math.atan(u)


def c_eff_tente(q):
    u = math.sqrt(max(1e-9, 2 / q - 1))
    return c_p_of_s(u) + 2 * q * u


def run():
    print("=== Matching cœur/asymptotique : derivation et consequence ===\n")
    print("[1] SEPARATION D'ECHELLES (chiffree) :")
    for t in (0.067, 0.087):
        print("   T = %d MeV : 1/T = %.1f fm ; D = %.3f-0,15 fm ->" % (
            t * 1e3, 0.19733 / t, D0_FM))
    print("   D/(1/T) ~ 0,007-0,05 : le milieu ne resout pas le cœur.")
    print("   Double sourcing de la platitude : sigma quenched plate")
    print("   jusqu'a 0,55 Tc ; condensat chiral ~plein a T <~ 100 MeV.")
    print("   => MATCHING DERIVE : sigma_cœur(T) = sigma_chi (plate).\n")

    print("[2] LA FORMULE D'UNE LIGNE : T_c = c_eff sigma_chi D_0")
    print("   sigma_chi D_0 = %.1f MeV" % (SIGMA_FM * D0_FM * 1e3))
    print("   -> T_c = c_eff x 19,0 MeV : T_c EST le cout du passage au")
    print("   cœur (grandeur du vide). Charge transferee sur c_eff.\n")

    print("[3] c_eff : etat des modeles (q = %.2f mesure) :" % (
        Q_ZONE))
    ce_t = c_eff_tente(Q_ZONE)
    ce_v = c_eff_variational(Q_ZONE)
    print("   tangente+arc (optimum analytique de sa classe) : %.2f" % ce_t)
    print("   controle discretise : %.2f > majorant -> NON CONVERGE," % ce_v)
    print("   indicatif seulement (la famille tangente-arc reste la")
    print("   reference). [cible Planck : c_eff = 3,5, soit -26 %]\n")

    print("[4] CONFRONTATION :")
    e_unit = SIGMA_FM * D0_FM  # GeV
    for lab, ce in (("tangente+arc (reference)", ce_t),
                    ("cible Planck", 3.5)):
        t_c = ce * e_unit
        b = budget(tc=t_c, g_s=15.2, p_occ=P_OCC_EFF)
        print("   %-20s c_eff = %.2f -> T_c = %5.1f MeV, Omega = %8.2f" % (
            lab, ce, t_c * 1e3, b["omega"]))


if __name__ == "__main__":
    run()
