"""Pincer le x1,7 : energie MESUREE des fragments porteurs de nœuds (20/07).

HONNETETE PREALABLE : le << x1,7 >> documente (V6 : << chaque
domaine-longueur porte sigma_chi*xi = 2,98 GeV, x1,7 la masse du
bord >>) n'est PAS une mesure independante -- c'est le ratio
sigma*xi/m(23) lui-meme. La << degenerescence a 1,7 % >> de l'etape 3
etait donc quasi tautologique. La VRAIE mesure independante : la
LONGUEUR des fragments porteurs de nœuds dans la cascade branchee
(departage_f_frac, machinerie canonique) -- avec le biais de taille
attendu (les passages tombent sur les fragments a proportion de leur
longueur d'histoire).

MESURE : distribution de L sur les fragments finals a >= 1 passage ->
E_frag = 2,98 GeV x L(xi) -> T_gel par fragment (Gamma = H, barriere
fixe) -> bande de T_c de la population + consequence Omega.
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.departage_f_frac import cascade_branchee  # noqa: E402
from simulation.experiments.derive_lambda_kinks import embedded_fragments  # noqa: E402

M_PL = 1.2209e19
G_S = 15.2
E_PER_XI = 2.98      # GeV par xi de corde (sigma_chi * xi)
M_EDGE = 1.73
T_NOM = 0.067


def t_gel(e_b):
    t = 0.05
    for _ in range(100):
        t = 0.5 * t + 0.5 * e_b / math.log(M_PL / (1.66 * math.sqrt(G_S) * t))
    return t


def batch(seeds, rng_seed):
    rng = np.random.default_rng(rng_seed)
    parents = []
    for seed in seeds:
        r2 = np.random.default_rng(7 + seed)
        ps = embedded_fragments(N=16, seed=seed, n_sample=30, rng=r2)
        parents += [p for p in ps if 6 <= len(p) <= 80]
    parents = parents[:24]
    finals = []
    for p in parents:
        finals += cascade_branchee(p, rng)
    return finals


def run():
    print("=== Pincer le x1,7 : fragments porteurs mesures ===\n")
    res = {}
    for name, seeds, rs in (("A (canonique 42/43)", (42, 43), 2026),
                            ("B (graines neuves 44/45)", (44, 45), 3026)):
        finals = batch(seeds, rs)
        L_all = np.array([L for L, n in finals])
        L_bear = np.array([L for L, n in finals if n >= 1])
        res[name] = (L_all, L_bear)
        print("lot %s : %d fragments, %d porteurs" % (
            name, len(L_all), len(L_bear)))
        print("   <L> tous = %.3f xi ; <L> porteurs = %.3f xi ;"
              " mediane porteurs = %.3f" % (
                  L_all.mean(), L_bear.mean(), np.median(L_bear)))

    L_bear = np.concatenate([res[k][1] for k in res])
    L_all = np.concatenate([res[k][0] for k in res])
    q1, med, q3 = np.percentile(L_bear, (25, 50, 75))
    mean_b = L_bear.mean()
    biais = mean_b / L_all.mean()
    print("\nCOMBINE : %d porteurs ; <L> = %.3f xi (biais de taille"
          " x%.2f vs tous %.3f)" % (
              len(L_bear), mean_b, biais, L_all.mean()))
    print("   quartiles : [%.2f, %.2f, %.2f] xi" % (q1, med, q3))
    err_mean = L_bear.std() / math.sqrt(len(L_bear))
    print("   erreur standard sur <L> : %.3f xi (%.1f %%)" % (
        err_mean, 100 * err_mean / mean_b))

    print("\nBARRIERE ET T_c (E_frag = 2,98 GeV x L) :")
    for lab, Lv in (("mediane porteurs", med),
                    ("moyenne porteurs", mean_b),
                    ("moyenne - 1 e.s.", mean_b - err_mean),
                    ("moyenne + 1 e.s.", mean_b + err_mean),
                    ("Q1", q1), ("Q3", q3)):
        e = E_PER_XI * Lv
        t = t_gel(e)
        om = 3.9 * math.exp(-M_EDGE * (1 / t - 1 / T_NOM))
        print("   %-18s L = %.2f xi -> E = %.2f GeV -> T_c = %5.1f MeV"
              " ; Omega = %6.2f" % (lab, Lv, e, t * 1e3, om))

    e_c = E_PER_XI * mean_b
    x_eff = e_c / M_EDGE
    print("\nLE PIN : E_porteur/m(23) = %.2f (l'ex-<<x1,7>>, MESURE" % x_eff)
    print("   independamment ; e.s. %.1f %%)" % (100 * err_mean / mean_b))
    t_c = t_gel(e_c)
    print("   T_c(moyenne porteurs) = %.1f MeV" % (t_c * 1e3))

    print("\nGEL ECHELONNE (la dispersion est une physique, pas une erreur) :")
    print("   chaque nœud gele a T_gel(L) de SON fragment ; s'il nait deja")
    print("   gele (T_gel > T_form), il garde le poids de formation a")
    print("   T_form. Poids effectif <e^{-m/min(T_form, T_gel(L))}> sur la")
    print("   distribution MESUREE des porteurs :")
    for t_form in (0.070, 0.100, 0.155):
        ws = []
        for L in L_bear:
            t_eff = min(t_form, t_gel(E_PER_XI * L))
            ws.append(math.exp(-M_EDGE / t_eff))
        w_mean = float(np.mean(ws))
        t_pop = -M_EDGE / math.log(w_mean)
        om = 3.9 * w_mean / math.exp(-M_EDGE / T_NOM)
        print("   T_form = %3.0f MeV : T_c effectif = %5.1f MeV ;"
              " Omega = %5.2f" % (t_form * 1e3, t_pop * 1e3, om))
    print("   -> le T_c unique de la chaine est REMPLACE par (T_form,")
    print("      distribution L) ; l'inconnue restante est T_form (la")
    print("      temperature de verrouillage/fragmentation absolue).")


if __name__ == "__main__":
    run()
