"""La matiere noire sous la revision du magnetisme (S50) ? (23/07)

TROIS VOLETS :
[1] COHERENCE ETABLIE : les noeuds de matiere noire SONT des ondes
    entrees dans les rails -> MAGNETISES (courant dirige persistant).
    C'est EXACTEMENT le mecanisme selectionne (2026-07-17) pour
    sigma_EM(neutre) du reservoir : « l'anneau CONDUCTEUR --
    circulation de framing PERSISTANTE, Lk verrouille la circulation »
    (alpha_pol = R_K^3). La revision S50 fournit l'ORIGINE
    microscopique de cette circulation : rien ne change aux nombres
    (sigma_eq(c=23) = 1,1e-48 cm^2, absence predite), la coherence se
    renforce.
[2] QUESTION NEUVE (testable ici) : le poids magnetique favorise les
    courants ALIGNES -> les enroulements COHERENTS -> les familles
    TORIQUES T(2,q) -- precisement les familles du reservoir !
    Jusqu'ici le tri torique etait par SURVIE (stabilite CS) seule ;
    mesurer P(torique)/P(twist) sous e^{kappa U} a c egal (5_1 vs 5_2).
[3] DETECTION : inchangee -- Q = 0 (framing), couplages EM
    Vassiliev-supprimes, le moment magnetique du noeud est un dipole
    CHI dont l'ombre EM passe par le portail supprime ; les forces
    chi-chi entre noeuds hauts-c sont supprimees (corollaire etabli).
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from simulation.experiments.approfondit_rails_e_p_n import (  # noqa: E402
    closed_lattice_walk, identify)
from simulation.experiments.derive_alpha_action_intrachemin import (  # noqa: E402
    neumann)

KAPPAS = [0.0, 0.5, 1.0, 1.5]


def main():
    print(__doc__)
    rng = np.random.default_rng(31)

    print("=" * 78)
    print("[2] TORIQUE vs TWIST SOUS POIDS MAGNETIQUE (5_1 vs 5_2, c = 5)")
    print("=" * 78)
    recs = []
    plan = [(64, 3200), (96, 2400)]
    for n, trials in plan:
        for _ in range(trials):
            w = closed_lattice_walk(n, rng)
            name, _ = identify(w, rng)
            if name in ("5_1", "5_2", "3_1", "0_1", "6_1", "6_2", "6_3"):
                recs.append((name, neumann(w)))
    us = np.array([u for _, u in recs])
    mu, sig = us.mean(), us.std()
    names = np.array([n for n, _ in recs])
    n51 = int((names == "5_1").sum())
    n52 = int((names == "5_2").sum())
    print(f"  evenements : 5_1 (torique) {n51}, 5_2 (twist) {n52}")
    u51 = us[names == "5_1"].mean()
    u52 = us[names == "5_2"].mean()
    print(f"  <U> : 5_1 = {u51:.3f}, 5_2 = {u52:.3f}, ensemble = {mu:.3f}"
          f"  (Delta(5_1 - 5_2) = {u51 - u52:+.3f} en unites brutes,"
          f" {(u51 - u52) / sig:+.2f} sigma_U)")
    for kap in KAPPAS:
        wgt = np.exp(kap * (us - mu) / sig)
        w51 = wgt[names == "5_1"].sum()
        w52 = wgt[names == "5_2"].sum()
        r = w51 / max(w52, 1e-12)
        print(f"  kappa = {kap:3.1f} : P(5_1)/P(5_2) = {r:.3f}")
    print("""  -> lecture : si le ratio torique/twist MONTE avec kappa, le
  magnetisme selectionne les enroulements coherents DES LA FORMATION
  -- les familles du reservoir (T(2,q)) recevraient une DEUXIEME
  selection (formation), en plus du tri de survie (stabilite CS).
  Sinon : le tri reste par survie seule (l'etabli), et le magnetisme
  est neutre pour la composition du reservoir.""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  AU NIVEAU DE LA MATIERE NOIRE :
  (1) RIEN NE CHANGE aux nombres etablis : Q = 0 (framing), c_min =
      23, sigma_eq = 1,1e-48 (absence predite DS-20k/XLZD), Omega
      (alpha = ln(2+sqrt2) intact), 6 raies Fermi-LAT 1,07-1,64 GeV ;
  (2) LA COHERENCE SE RENFORCE : la circulation persistante de
      l'anneau conducteur (le mecanisme selectionne de la
      polarisabilite) recoit son origine microscopique -- c'est le
      courant de rail verrouille dans le tube ferme ; le moment
      magnetique du noeud est un dipole CHI (ombre EM par le portail
      Vassiliev-supprime ; forces chi-chi hauts-c supprimees,
      corollaire etabli) ;
  (3) LE POINT NEUF est la SELECTION DE FAMILLE a la formation
      (torique vs twist sous poids magnetique) -- mesure ci-dessus ;
      dans les deux issues, la composition observable du reservoir
      (bord c = 23-25, masses 1,7-1,9 GeV) est inchangee : c'est un
      raffinement de MECANISME, pas de phenomenologie.""")


if __name__ == "__main__":
    main()
