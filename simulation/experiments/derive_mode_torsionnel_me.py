"""Le mode torsionnel et m_e — trois maillons de fermeture
(27/07/2026, suite de la fermeture de structure).

ÉTAT : m_e-candidat = (1/2) K_33 · alpha hbar c/(2L) = 0,457 MeV
(-10,5 %), avec deux hypothèses non justifiées (eta = 1 ; canal
charge absent) et un terme croisé non traité. TROIS MAILLONS :

 [A] LE CANAL CHARGE n=3 EST EXCLU PAR MÉCANISME : la charge est
     TOPOLOGIQUE (Q = Sigma chi + Lk, verrouillée par le framing —
     le mécanisme même de la neutralisation du réservoir sombre) :
     elle ne peut pas se bosseler le long du tube (incompressible).
     Le mode de génération est donc TORSIONNEL (il module la
     circulation interne — le canal du sens), pas une onde de
     charge. On chiffre J_33 (harmonique n=3 du noyau de Coulomb)
     pour information — l'exclusion repose sur le seul mécanisme
     (résultat : petit et négatif, pas un sur-tirage).

 [B] eta = 1 FORCÉ PAR LA DÉGÉNÉRESCENCE : l'état massif est la
     combinaison RÉELLE des deux sens (les amplitudes de Koide sont
     réelles) ; la dégénérescence P-paire (théorème vérifié 5e-16)
     force les POIDS ÉGAUX des deux canaux => onde stationnaire à
     CONTRASTE PLEIN : eta = 1 (conditionnel : combinaison réelle).

 [C] THÉORÈME DU CROISÉ NUL : le terme croisé circulation de base
     (STATIONNAIRE) x mode (OSCILLANT à omega) s'annule en moyenne
     temporelle — <cos(omega t)> = 0. La seule pièce de génération
     survivante est le terme PROPRE avec sa moyenne <cos²> = 1/2 :
        m_e-candidat = (1/2) K_33 · alpha hbar c / (2L),
     désormais SANS hypothèse restante au-delà de [A][B] et de la
     régularisation du tube.

 [D] LA BANDE DE COUPURE : le -10,5 % restant est-il dans
     l'incertitude de régularisation ? On calcule K_33 pour une
     coupure au rayon (a = r, convention) et au diamètre (a = D).
     RÉSULTAT (honnête) : la coupure DOMINE (a=D inverse le signe) —
     seule l'ÉCHELLE est robuste ; le profil du tube est LA cible.

GARDE-FOUS : [B] conditionnel (combinaison réelle) ; la décimale
n'est revendiquée nulle part ; règle du test nul.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from simulation.ideal_embedding import sono, resample
from theory.omega_ts import _knot_curve

M_E = 0.51099895
ALPHA = 1 / 137.035999
HBARC = 197.32698
MU = 17.715562
M_L = MU ** 2
SQRT_SIGMA = 418.47


def harmonics(P, ell, cut_mult=1.0, nres=600):
    """K_33, J_33, N/L, J/L — coupure à cut_mult x rayon de tube."""
    P = resample(np.asarray(P, float), nres)
    P = P - P.mean(0)
    dl = np.roll(P, -1, 0) - P
    ds = np.linalg.norm(dl, axis=1)
    t = dl / ds[:, None]
    s = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
    L = ds.sum()
    a = cut_mult * L / ell
    R = P + 0.5 * dl
    sig = 2 * np.pi * s / L
    N = J = K33 = J33 = 0.0
    for i in range(nres):
        dv = R[i] - R
        dist = np.linalg.norm(dv, axis=1)
        darc = np.abs(s - s[i])
        darc = np.minimum(darc, L - darc)
        ok = (darc > a) & (dist > 1e-12)
        w = ds[i] * ds[ok] / dist[ok]
        tt = t[i] @ t[ok].T
        c3 = np.cos(3 * (sig[i] - sig[ok]))
        N += (tt * w).sum()
        J += w.sum()
        K33 += (tt * w * c3).sum()
        J33 += (w * c3).sum()
    return N / L, J / L, K33 / L, J33 / L, L


def main():
    print(__doc__)
    P0 = np.asarray(_knot_curve("3_1", 200), float)
    P_id, info = sono(P0, iters=20000)
    ell = info["ropelength"]
    L_fm = M_L / SQRT_SIGMA ** 2 * HBARC
    pref = ALPHA * HBARC / (2 * L_fm)
    print("=" * 78)
    print("trèfle idéal : ropelength = %.3f ; préfacteur = %.4f MeV" % (
        ell, pref))

    print("=" * 78)
    print("[A] LE CANAL CHARGE n=3 (exclusion par mécanisme)")
    print("=" * 78)
    NL, JL, K33, J33, _ = harmonics(P_id, ell, 1.0)
    E_charge = 0.5 * pref * J33          # onde de charge pleine, <sin²>=1/2
    print("  J_33/L = %.3f  ->  onde de charge pleine : E = %.2f MeV "
          "(m_e x %.1f)" % (J33, E_charge, E_charge / M_E))
    print("""
  MÉCANISME : Q = Sigma chi + Lk est verrouillée par le framing (le
  mécanisme même de la neutralisation du réservoir sombre) — la
  charge ne se bosselle pas : le canal charge du mode est ABSENT par
  topologie. (Numériquement il serait d'ailleurs PETIT et négatif,
  %.2f MeV — l'exclusion repose sur le SEUL mécanisme, pas sur un
  sur-tirage.) Le mode de génération est TORSIONNEL (canal du sens
  seul).""" % E_charge)

    print("=" * 78)
    print("[B][C] eta = 1 (poids égaux forcés) + théorème du croisé nul")
    print("=" * 78)
    print("""  [B] l'état massif = combinaison RÉELLE des deux sens (les
      amplitudes de Koide sont réelles) ; la dégénérescence P-paire
      force les poids égaux => contraste plein, eta = 1
      (conditionnel : combinaison réelle).
  [C] croisé base(stationnaire) x mode(oscillant) : <cos(omega t)>=0
      => NUL en moyenne — la seule pièce de génération est le terme
      propre, avec <cos²> = 1/2. Plus d'hypothèse au-delà de [A][B]
      et de la régularisation.""")

    print("=" * 78)
    print("[D] LE CANDIDAT ET LA BANDE DE COUPURE")
    print("=" * 78)
    for lab, cm in [("coupure au rayon (a=r)", 1.0),
                    ("coupure au diamètre (a=D)", 2.0)]:
        NL_, JL_, K33_, J33_, _ = harmonics(P_id, ell, cm)
        me_c = 0.5 * pref * K33_
        print("  %-26s : K_33/L = %.4f  ->  m_e-candidat = %.3f MeV "
              "(%+.1f %%)" % (lab, K33_, me_c, 100 * (me_c / M_E - 1)))
    print("""
  LECTURE HONNÊTE : la coupure DOMINE — a=r donne -10,5 %, a=D
  INVERSE LE SIGNE : la pièce n=3 est portée par la région
  r < |Delta s| < D (le premier voisinage du tube). Robuste :
  l'ÉCHELLE seule (|E| ~ 0,2-0,5 MeV = l'échelle de m_e). Le
  -10,5 % du premier calcul était CONDITIONNÉ à la convention a=r
  (celle de la normalisation ropelength L/r) — à déclarer, pas à
  revendiquer. Ni la décimale NI même le signe ne sont robustes
  sans le PROFIL physique du tube : c'est LA cible, et elle est
  désormais isolée (tout le reste de la chaîne est dérivé).""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""   epsilon n'est plus un nombre orphelin : c'est une énergie dont
   la CHAÎNE est dérivée ([A] canal charge exclu par topologie,
   [B] eta=1 par la dégénérescence, [C] croisé nul en moyenne) et
   dont l'ÉCHELLE est calculée (~0,2-0,5 MeV = l'échelle de m_e) —
   mais la valeur est DOMINÉE PAR LA RÉGULARISATION (la coupure a=r
   donne -10,5 %, a=D inverse le signe). LA cible restante, isolée :
   le profil physique du tube. Rien au-delà de l'échelle n'est
   revendiqué (règle du test nul).""")


if __name__ == "__main__":
    main()
