r"""
Chantier 1 — sensibilité de U_EM à l'embedding pour les 19 nœuds.

Généralisation de l'investigation 8_3 : on filtre passe-bas en Fourier
les coordonnées (x, y, z) de chaque courbe paramétrique du catalogue,
ne gardant que les modes ≤ k_max, et on recalcule U_EM = ∮(κ²+τ²)ds.

Question : 8_3 est-il SPÉCIFIQUEMENT inflaté par sa paramétrisation,
ou TOUS les nœuds sont-ils similairement affectés ?

* Si tous sont réduits similairement (ratio ~ 2-3×) : la V4 est
  cohérente sur SON catalogue (le choix paramétrique global est une
  convention d'échelle).
* Si 8_3 (et autres) montre un ratio extrême : c'est une fragilité
  spécifique qui crée artificiellement l'échelle EW.

On vérifie ENFIN si la formule de masse V4 (m = α₀·U_EM·c_J^3,55)
appliquée aux U_EM lissés reproduit (ou non) les masses physiques.

CAVEAT : le lissage Fourier peut DÉFAIRE le nœud (perdre la topologie).
Le test mesure la SENSIBILITÉ paramétrique de U_EM dans le catalogue,
pas la valeur U_EM intrinsèque d'un nœud idéal.

Cf. docs/brouillon_hamiltonien_multipart.tex.
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.mass_formula import c_j, ALPHA0, P_EXP
from theory.abraham import REFERENCE_N50K

KNOTS = list(REFERENCE_N50K.keys())


def smooth_curve(pts: np.ndarray, k_max: int) -> np.ndarray:
    """Filtre passe-bas Fourier sur la courbe : ne garde que les modes
    ≤ k_max (en valeur absolue de la fréquence)."""
    pts_no_dup = pts[:-1]
    N = len(pts_no_dup)
    mask = np.zeros(N, dtype=bool)
    mask[:k_max + 1] = True
    if k_max > 0:
        mask[N - k_max:] = True
    smoothed = np.zeros_like(pts_no_dup)
    for i in range(3):
        coef = np.fft.fft(pts_no_dup[:, i])
        coef_filtered = np.where(mask, coef, 0)
        smoothed[:, i] = np.real(np.fft.ifft(coef_filtered))
    return np.vstack([smoothed, smoothed[:1]])


def compute_UEM(pts: np.ndarray) -> float:
    pts = pts[:-1]
    N = len(pts)
    dt = 2 * np.pi / N
    p_next = np.roll(pts, -1, axis=0)
    p_prev = np.roll(pts, 1, axis=0)
    p_next2 = np.roll(pts, -2, axis=0)
    p_prev2 = np.roll(pts, 2, axis=0)
    dr = (p_next - p_prev) / (2 * dt)
    ddr = (p_next - 2 * pts + p_prev) / dt ** 2
    dddr = (p_next2 - 2 * p_next + 2 * p_prev - p_prev2) / (2 * dt ** 3)
    speed = np.linalg.norm(dr, axis=1)
    cross = np.cross(dr, ddr)
    cross_norm = np.linalg.norm(cross, axis=1)
    kappa = cross_norm / (speed ** 3 + 1e-30)
    triple = np.sum(cross * dddr, axis=1)
    tau = triple / (cross_norm ** 2 + 1e-30)
    ds = speed * dt
    return float(np.sum(kappa ** 2 * ds) + np.sum(tau ** 2 * ds))


def main():
    n_points = 5000
    k_max_list = [3, 5, 7, 10, 20]

    print("=" * 90)
    print("  SENSIBILITÉ U_EM À L'EMBEDDING — filtrage Fourier sur les 19 nœuds")
    print("=" * 90)
    print(f"  n_points = {n_points}")
    print(f"  Filtres passe-bas à k_max = {k_max_list}")
    print()
    print(f"  {'K':>5} {'U_EM_cat':>9} | " +
          " ".join(f"{f'k≤{k}':>8}" for k in k_max_list) +
          f" | {'ratio cat/k=3':>14}")
    print("  " + "-" * 80)

    rows = []
    for K in KNOTS:
        pts = knot_curve(K, n_points)
        U_orig = compute_UEM(pts)
        U_smooth = {}
        for k_max in k_max_list:
            pts_s = smooth_curve(pts, k_max)
            U_smooth[k_max] = compute_UEM(pts_s)
        ratio_cat_over_k3 = U_orig / max(U_smooth[3], 1e-30)
        rows.append((K, U_orig, U_smooth, ratio_cat_over_k3))
        print(f"  {K:>5} {U_orig:>9.2f} | " +
              " ".join(f"{U_smooth[k]:>8.2f}" for k in k_max_list) +
              f" | {ratio_cat_over_k3:>14.2f}")

    # Classement
    print("\n  Classement par RATIO cat / k_max=3 (plus grand = plus fragile) :")
    for K, U_orig, U_smooth, ratio in sorted(rows, key=lambda x: -x[3]):
        marker = " ★ ARTEFACT" if ratio > 5 else ""
        print(f"    {K:>5} : ratio = {ratio:>6.2f}×  "
              f"(cat = {U_orig:>8.2f} → k≤3 : {U_smooth[3]:>8.2f}){marker}")

    # Statistiques globales
    ratios = np.array([r[3] for r in rows])
    print(f"\n  Statistiques sur les 19 nœuds :")
    print(f"    moyenne   = {ratios.mean():.2f}")
    print(f"    médiane   = {np.median(ratios):.2f}")
    print(f"    min / max = {ratios.min():.2f} / {ratios.max():.2f}")
    print(f"    écart-type = {ratios.std():.2f}")

    # Test : la formule V4 sur les U_EM lissés reproduit-elle les masses ?
    print(f"\n  Test : formule V4 (m = α₀·U_EM·c_J^3,55) avec U_EM lissé à k≤5 :")
    print(f"  {'K':>5} {'m_V4 cat':>11} {'m_V4 smooth k≤5':>17} {'ratio':>9}")
    for K, U_orig, U_smooth, _ in rows:
        cJ = c_j(K)
        m_cat = ALPHA0 * U_orig * cJ ** P_EXP
        m_smooth = ALPHA0 * U_smooth[5] * cJ ** P_EXP
        ratio_m = m_cat / max(m_smooth, 1e-30)
        print(f"  {K:>5} {m_cat:>11.4f} {m_smooth:>17.4f} {ratio_m:>9.2f}")

    # Lecture
    print(f"\n  LECTURE :")
    print(f"  ──────────────────────────────────────────────────────")
    n_artifact = sum(1 for r in ratios if r > 5)
    if n_artifact == 0:
        print(f"  Tous les nœuds sont raisonnablement stables au filtrage Fourier")
        print(f"  (ratio max = {ratios.max():.1f}×). La V4 est cohérente sur SON")
        print(f"  catalogue — le choix paramétrique est une convention d'échelle.")
    elif n_artifact <= 3:
        print(f"  {n_artifact} nœuds ont un ratio > 5× (FRAGILES) ; les autres sont")
        print(f"  stables. La V4 dépend spécifiquement de ces choix paramétriques.")
    else:
        print(f"  {n_artifact} nœuds sur 19 sont fragiles au filtrage. Les")
        print(f"  paramétrisations Fourier haute fréquence inflatent U_EM massivement.")
        print(f"  La V4 est essentiellement une convention paramétrique.")

    print("=" * 90)


if __name__ == "__main__":
    main()
