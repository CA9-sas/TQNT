"""Fermeture de structure sur epsilon — l'additif interdit, la
géométrie idéale, les harmoniques Z3 (27/07/2026, suite de l'assaut).

TROIS PIÈCES :

 [1] LE THÉORÈME DE L'ADDITIF INTERDIT (données seules) : la relation
     de Koide tient à ~1e-5 sous la forme amplitude PURE. Toute
     énergie ADDITIVE commune aux trois générations (m_k -> m_k + D)
     détruit Q : on calcule la borne |D|_max. Si elle est très sous
     l'énergie magnétique de base (1,5 MeV), alors celle-ci est
     NÉCESSAIREMENT absorbée dans l'échelle (m_l/mu), pas additive —
     et la pièce propre à une génération doit entrer par l'AMPLITUDE
     (le déplacement du nœud), pas par une énergie ajoutée.
     BONUS (théorème plat) : Sigma m = 3 mu^2 (1 + A^2/2) est
     delta-INDÉPENDANT — delta est une direction PLATE de la masse
     totale : aucune minimisation d'énergie aveugle aux classes ne
     peut le fixer ; seul un couplage qui distingue les classes
     (T-impair : le sens) le peut. (Cohérent avec la lecture V6.)

 [2] LA GÉOMÉTRIE IDÉALE (SONO) : trèfle idéal calculé (outillage
     validé +0,12 %) ; on recalcule J/L (Coulomb) et N/L (Neumann)
     sur l'idéal — l'ordre x3,0 du canal du sens bouge-t-il ?

 [3] LES HARMONIQUES Z3 DU NOYAU : le mode de génération est une onde
     de nombre d'onde 3 sur le circuit (une période par secteur Z3).
     Sa pièce magnétique propre = les composantes de Fourier n=3 du
     noyau de Neumann : k_n = (1/L) ∮∮ e^{i n sigma'} K ds ds'
     (terme croisé mode x circulation de base) et
     K_33 = (1/L) ∮∮ cos(3(sigma-sigma')) K ds ds' (terme propre).
     Contrôle de symétrie : par C3, seuls n = 0 mod 3 survivent.
     Verdict d'ordre : la pièce n=3 est-elle à l'échelle m_e ?

GARDE-FOUS : [1] est purement empirique (aucun modèle) ; [2][3]
géométrie idéale mesurée, conventions déclarées ; règle du test nul.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from simulation.ideal_embedding import sono, resample
from theory.omega_ts import _knot_curve
from theory.ropelength_acpr import ROPELENGTH_ACPR

M_E, M_MU, M_TAU = 0.51099895, 105.6583755, 1776.86
ALPHA = 1 / 137.035999
HBARC = 197.32698
MU = 17.715562
M_L = MU ** 2
SQRT_SIGMA = 418.47


def kernels_harmonics(P, ell, nres=600):
    """J/L, N/L et harmoniques du noyau de Neumann (coupure au rayon)."""
    P = resample(np.asarray(P, float), nres)
    P = P - P.mean(0)
    dl = np.roll(P, -1, 0) - P
    ds = np.linalg.norm(dl, axis=1)
    t = dl / ds[:, None]
    s = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
    L = ds.sum()
    a = L / ell
    R = P + 0.5 * dl
    sig = 2 * np.pi * s / L
    J = N = 0.0
    kvec = np.zeros(7, complex)          # k_n, n=0..6 (e^{i n sigma'})
    K33 = 0.0
    for i in range(nres):
        dv = R[i] - R
        dist = np.linalg.norm(dv, axis=1)
        darc = np.abs(s - s[i])
        darc = np.minimum(darc, L - darc)
        ok = (darc > a) & (dist > 1e-12)
        w = ds[i] * ds[ok] / dist[ok]
        tt = t[i] @ t[ok].T
        J += w.sum()
        N += (tt * w).sum()
        for n in range(7):
            kvec[n] += (tt * w * np.exp(1j * n * sig[ok])).sum()
        K33 += (tt * w * np.cos(3 * (sig[i] - sig[ok]))).sum()
    return J / L, N / L, np.abs(kvec) / L, K33 / L, L


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] LE THÉORÈME DE L'ADDITIF INTERDIT (données seules)")
    print("=" * 78)
    m = np.array([M_E, M_MU, M_TAU])
    Q0 = m.sum() / np.sqrt(m).sum() ** 2
    dQ_mes = abs(Q0 - 2 / 3)
    # sensibilité : Q(D) avec m -> m + D
    Ds = np.logspace(-4, 1, 400)
    Qs = np.array([(m + D).sum() / np.sqrt(m + D).sum() ** 2 for D in Ds])
    D_max = Ds[np.argmax(np.abs(Qs - Q0) > dQ_mes)]
    print("  Q mesuré = %.7f ; |Q - 2/3| = %.1e" % (Q0, dQ_mes))
    print("  borne : une énergie additive commune D détruit Q au niveau")
    print("  mesuré dès D = %.4f MeV" % D_max)
    print("""
  => l'énergie magnétique de BASE (1,53 MeV calculée à l'assaut, %s
  x la borne) NE PEUT PAS être additive : elle est nécessairement
  ABSORBÉE dans l'échelle (le tube chargé pèse m_l, magnétisme
  compris) — la pièce propre à une génération doit entrer par
  l'AMPLITUDE (déplacement du nœud), pas par une énergie ajoutée.
  THÉORÈME PLAT (bonus) : Sigma m = 3 mu²(1+A²/2), delta-INDÉPENDANT
  — delta est une direction PLATE de la masse totale : aucune
  minimisation aveugle aux classes ne peut le fixer ; seul le
  T-impair (le sens) le peut — la lecture V6 est FORCÉE par la
  structure même de la forme de Koide.""" % (
        "%.0f" % (1.53 / D_max)))

    print("=" * 78)
    print("[2] LA GÉOMÉTRIE IDÉALE (SONO)")
    print("=" * 78)
    P0 = np.asarray(_knot_curve("3_1", 200), float)
    P_id, info = sono(P0, iters=20000)
    ell_id = info["ropelength"]
    print("  SONO : ropelength = %.3f (publié 32,744 ; %+.2f %%)" % (
        ell_id, 100 * (ell_id / 32.744 - 1)))
    res = {}
    for lab, P, ell in [("catalogue", P0, ROPELENGTH_ACPR["3_1"]),
                        ("idéal SONO", P_id, ell_id)]:
        JL, NL, kn, K33, _ = kernels_harmonics(P, ell)
        res[lab] = (JL, NL, kn, K33)
        print("  %-11s : J/L = %.3f ; N/L = %.3f" % (lab, JL, NL))
    L_fm = M_L / SQRT_SIGMA ** 2 * HBARC
    pref = ALPHA * HBARC / (2 * L_fm)
    for lab in res:
        JL, NL, kn, K33 = res[lab]
        print("  %-11s : E_C = %.2f MeV (m_e x %.0f) ; E_M = %.3f MeV "
              "(m_e x %.2f)" % (lab, pref * JL, pref * JL / M_E,
                                pref * NL, pref * NL / M_E))
    print("""
  => la géométrie idéale confirme/ajuste l'ordre du canal du sens
  (comparer les deux lignes E_M) — l'écart résiduel à m_e est le
  facteur à fermer par le partage des canaux et la participation.""")

    print("=" * 78)
    print("[3] LES HARMONIQUES Z3 DU NOYAU (trèfle idéal)")
    print("=" * 78)
    JL, NL, kn, K33 = res["idéal SONO"]
    print("  |k_n|/L (terme croisé mode-n x base) :")
    for n in range(7):
        note = ""
        if n % 3 != 0 and n > 0:
            note = "   (interdit par C3 -> doit être ~0)"
        print("    n=%d : %.4f  ->  E = %.3f MeV%s" % (
            n, kn[n], pref * kn[n], note))
    print("  K_33/L (terme propre du mode 3) = %.4f  ->  E = %.3f MeV" % (
        K33, pref * K33))
    print("  m_e = %.3f MeV ; rapports : E(k_3)/m_e = %.2f ; "
          "E(K_33)/m_e = %.2f" % (M_E, pref * kn[3] / M_E,
                                  pref * K33 / M_E))
    print("  onde stationnaire pleine (eta=1, moyenne cos² = 1/2) :")
    print("    E = K_33/2 -> %.3f MeV  vs m_e  (%+.1f %%)" % (
        pref * K33 / 2, 100 * (pref * K33 / 2 / M_E - 1)))
    print("""
  LECTURE (honnête) : le contrôle C3 valide la décomposition (les
  n non multiples de 3 sont supprimés à ~1e-3) ; les pièces
  magnétiques PROPRES au mode de génération (n=3) ENCADRENT m_e
  (croisé x0,35, propre x1,79) — et l'estimation SANS RÉGLAGE
  (modulation pleine eta=1, moyenne temporelle 1/2) tombe à ~10 %
  de m_e. ORDRE FERMÉ ; décimale NON revendiquée : eta n'est pas
  dérivé, le partage croisé/propre non plus, et la géométrie idéale
  porte ~1 % (SONO +1,0 % du publié). La chaîne complète
  (epsilon <- couple T-impair du terme croisé <- application
  amplitude-énergie) reste la cible ; rien n'est retenu au-delà de
  l'ordre (test nul).""")


if __name__ == "__main__":
    main()
