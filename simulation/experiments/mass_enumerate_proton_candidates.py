r"""
Chantier 1 — énumérer TOUS les entrelacs et chercher ceux qui matchent
proton/neutron par masse + spin.

Avec la nouvelle formule tightness + correction linking :
  m(L) = m_e · [f(n) · R_tot · (1 - α·exp(-D/D₀)) / R(3_1)]^p
  R_tot = 3,85·cr_tot + 5,61·n
  f(1)=1, f(2)=0,86, f(3)=0,77 ; α=0,25, D₀=14, p=7,42
  Lk = Lk_opt = n(n-1)/2  ⇒  D = 0, facteur (1-α) = 0,75

On énumère les 1539 configurations possibles (combinaisons avec
répétition de 1 à 3 nœuds parmi les 19 du catalogue) et on cherche
ceux compatibles avec proton/neutron :
  - m_pred dans [750, 1130] MeV (±20 % autour de 938)
  - parité de spin = demi-entière (1 ou 3 composantes fermioniques)

Question : 3_1+7_2+7_2 est-il toujours le meilleur, ou y a-t-il un
candidat alternatif qui matche mieux la masse expérimentale ?

Cf. docs/brouillon_hamiltonien_multipart.tex.
"""
from __future__ import annotations
from itertools import combinations_with_replacement
import numpy as np

from theory.abraham import REFERENCE_N50K
from theory.jones import is_half_integer_spin, braid_index

KNOTS = list(REFERENCE_N50K.keys())
M_E = 0.51099895
A_R, B_R = 3.85, 5.61
P_EXP = 7.42
ALPHA_LINK = 0.25
D0_LINK = 14.0
F_SCREEN = {1: 1.0, 2: 0.86, 3: 0.77}

R_E = A_R * 3 + B_R     # R(3_1) = 17.16

# Targets : masses expérimentales p, n
TARGETS = {
    "proton":  (938.272, "spin 1/2, Q=+1"),
    "neutron": (939.565, "spin 1/2, Q=0"),
}


def cr(K):  return int(K.split("_")[0])

def R_alg(K): return A_R * cr(K) + B_R

def R_total(components): return float(sum(R_alg(K) for K in components))

def mass_pred(components, Lk: int | None = None) -> tuple[float, float]:
    """Renvoie (m_pred, D) avec Lk_opt = n(n-1)/2 par défaut."""
    n = len(components)
    Lk_opt = n * (n - 1) // 2
    if Lk is None:
        Lk = Lk_opt
    D = (Lk - Lk_opt) ** 2
    R_tot = R_total(components)
    f = F_SCREEN[n]
    R_eff = R_tot * (1 - ALPHA_LINK * np.exp(-D / D0_LINK))
    m = M_E * (f * R_eff / R_E) ** P_EXP
    return m, D


def spin_half_integer(components) -> bool:
    """Vrai si le nombre de composantes à b PAIR (spin demi-entier) est impair."""
    n_half = sum(1 for K in components if is_half_integer_spin(K))
    return n_half % 2 == 1


def total_braid_indices(components):
    return tuple(braid_index(K) for K in components)


def enumerate_all_candidates(n_max: int = 3):
    """Génère tous les entrelacs de 1 à n_max composantes."""
    for n in range(1, n_max + 1):
        for combo in combinations_with_replacement(KNOTS, n):
            yield list(combo)


def main():
    print("=" * 90)
    print("  ÉNUMÉRATION DES ENTRELACS — candidats pour proton/neutron")
    print("=" * 90)
    print(f"  Formule : m = m_e · [f(n)·R_tot·(1-α·exp(-D/D₀))/R(3_1)]^p")
    print(f"  Lk = Lk_opt(n) (compactage maximum, D=0)")
    print(f"  Targets : proton 938.27 ± 20 %, neutron 939.57 ± 20 %")
    print()

    # Tolerance fenêtre
    M_LOW, M_HIGH = 750.0, 1130.0

    candidates = []
    for comps in enumerate_all_candidates(3):
        if not spin_half_integer(comps):
            continue
        m, D = mass_pred(comps)
        if M_LOW <= m <= M_HIGH:
            candidates.append((m, comps, D))

    print(f"  Total : {len(candidates)} entrelacs match (spin demi-entier, "
          f"m ∈ [{M_LOW}, {M_HIGH}]) :\n")

    # Triés par masse croissante
    candidates.sort()
    print(f"  {'rang':>4} {'config':>22} {'n':>2} {'cr_tot':>7} {'R_tot':>8} "
          f"{'b_indices':>14} {'m_pred':>9}")
    print("  " + "-" * 75)
    for rank, (m, comps, D) in enumerate(candidates, 1):
        cfg = "+".join(comps)
        n = len(comps)
        cr_tot = sum(cr(K) for K in comps)
        R_tot = R_total(comps)
        b_idx = total_braid_indices(comps)
        is_pn_target = "p/n" in [t for t, _ in TARGETS.items()
                                  if abs(m - TARGETS[t][0]) / TARGETS[t][0] < 0.10]
        print(f"  {rank:>4} {cfg:>22} {n:>2} {cr_tot:>7} {R_tot:>8.1f} "
              f"{str(b_idx):>14} {m:>9.2f}")

    # Spécifiquement les meilleurs candidats proton et neutron
    print(f"\n  ─── TOP 10 candidats pour PROTON (938,27 MeV) ───")
    by_dev_p = sorted(candidates, key=lambda x: abs(x[0] - TARGETS["proton"][0]))[:10]
    print(f"  {'rang':>4} {'config':>22} {'m_pred':>9} {'écart':>9}")
    for rank, (m, comps, _) in enumerate(by_dev_p, 1):
        cfg = "+".join(comps)
        dev = (m - TARGETS["proton"][0]) / TARGETS["proton"][0] * 100
        print(f"  {rank:>4} {cfg:>22} {m:>9.2f} {dev:>+8.2f}%")

    print(f"\n  ─── TOP 10 candidats pour NEUTRON (939,57 MeV) ───")
    by_dev_n = sorted(candidates, key=lambda x: abs(x[0] - TARGETS["neutron"][0]))[:10]
    print(f"  {'rang':>4} {'config':>22} {'m_pred':>9} {'écart':>9}")
    for rank, (m, comps, _) in enumerate(by_dev_n, 1):
        cfg = "+".join(comps)
        dev = (m - TARGETS["neutron"][0]) / TARGETS["neutron"][0] * 100
        print(f"  {rank:>4} {cfg:>22} {m:>9.2f} {dev:>+8.2f}%")

    # Rang du 3_1+7_2+7_2 (V4 assignment)
    target_cfg = ["3_1", "7_2", "7_2"]
    target_cfg_str = "+".join(sorted(target_cfg))
    for label, (m_exp, descr) in TARGETS.items():
        # Rank in by_dev_<label>
        sorted_by = sorted(candidates, key=lambda x: abs(x[0] - m_exp))
        for rank, (m, comps, _) in enumerate(sorted_by, 1):
            if "+".join(sorted(comps)) == target_cfg_str:
                dev = (m - m_exp) / m_exp * 100
                print(f"\n  Rang du 3_1+7_2+7_2 pour {label} : "
                      f"{rank} / {len(sorted_by)}  (écart {dev:+.2f}%)")
                break
        else:
            print(f"\n  3_1+7_2+7_2 NON trouvé dans les candidats {label}")

    # Lecture
    print(f"\n  LECTURE :")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Si 3_1+7_2+7_2 N'EST PAS dans les top 10 ou qu'il y a")
    print(f"  une configuration plus proche de m_exp avec mêmes contraintes,")
    print(f"  alors la nouvelle formule SUGGÈRE de réviser l'assignment V4.")
    print(f"")
    print(f"  ATTENTION : on n'a contraint que masse + parité de spin, PAS")
    print(f"  la charge (Q=Σχ+Lk). Pour chaque candidat, il faut vérifier")
    print(f"  qu'un Lk physique donne Q entière compatible avec p (+1) ou n (0).")
    print(f"  Avec χ correctement calculée pour chaque composante, on peut")
    print(f"  filtrer davantage.")
    print("=" * 90)


if __name__ == "__main__":
    main()
