r"""
Chantier 1 — Pieranski 1 : minimisation rope-length des courbes individuelles.

Algorithme (Pieranski 1998, version simplifiée) :
Pour chaque nœud K du catalogue, on optimise sa courbe γ(t) ⊂ R³ par
descente de gradient contraint :

  1. SMOOTHING Laplacien : γ_i ← γ_i + α·(γ_{i+1} + γ_{i-1} - 2γ_i)
     (réduit la longueur, lisse les inflexions)
  2. SELF-AVOIDANCE : pour chaque paire (i,j) non-adjacente avec
     |γ_i - γ_j| < D (diamètre de tube), pousser à part :
        push ∝ (D - d) · (γ_i - γ_j)/d
     (maintient le tube épais de diamètre D)
  3. REPARAMÉTRISATION uniforme le long de l'arc (tous les K pas).

Convergence : L plateauesque + d_min ≈ D. La rope-length est alors
R(K) = L_final / D.

Test sur les 19 nœuds catalogue, comparaison à R_publié (Cantarella-
Kusner-Sullivan, Pieranski).

Cf. docs/brouillon_hamiltonien_multipart.tex (chantier 1, J3 amélioré).
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K

KNOTS = list(REFERENCE_N50K.keys())

# Rope-lengths publiées (référence)
PUBLISHED = {
    "3_1":   16.37, "4_1":  20.99, "5_1": 23.55, "5_2": 26.13,
    "6_1":   28.31, "6_2":  28.13, "6_3": 27.21,
    "7_1":   30.69, "7_2":  32.96, "7_3": 33.61, "7_4": 34.97,
    "7_5":   35.16, "7_6":  32.66, "7_7": 31.93,
    "8_1":   37.49, "8_2":  37.32, "8_3": 37.30,
    "9_1":   38.62, "10_1": 42.33,
}


def length(pts: np.ndarray) -> float:
    diff = np.roll(pts, -1, axis=0) - pts
    return float(np.linalg.norm(diff, axis=1).sum())


def d_min_far(pts: np.ndarray, skip: int) -> float:
    """Min distance entre points non-adjacents (arc-distance > skip)."""
    N = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    idx = np.arange(N)
    arc = np.abs(idx[:, None] - idx[None, :])
    arc = np.minimum(arc, N - arc)
    masked = np.where(arc > skip, dist, np.inf)
    return float(masked.min())


def reparametrize(pts: np.ndarray, n_target: int) -> np.ndarray:
    """Re-discrétise uniformément le long de l'arc (préserve la topologie)."""
    diff = np.roll(pts, -1, axis=0) - pts
    seg = np.linalg.norm(diff, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    L = s[-1] + seg[-1]
    s_new = np.linspace(0, L, n_target, endpoint=False)
    s_full = np.concatenate([s, [L]])
    pts_full = np.vstack([pts, pts[:1]])
    new_pts = np.zeros((n_target, 3))
    for i in range(3):
        new_pts[:, i] = np.interp(s_new, s_full, pts_full[:, i])
    return new_pts


def pieranski_iteration(pts: np.ndarray, alpha: float, D: float, beta: float,
                        skip: int) -> np.ndarray:
    """Une itération : smoothing Laplacien + repulsion self-avoidance."""
    N = len(pts)
    # Smoothing
    p_next = np.roll(pts, -1, axis=0)
    p_prev = np.roll(pts, 1, axis=0)
    laplacian = p_next + p_prev - 2 * pts
    pts2 = pts + alpha * laplacian
    # Self-avoidance : vectorisé
    diff = pts2[:, None, :] - pts2[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    idx = np.arange(N)
    arc = np.abs(idx[:, None] - idx[None, :])
    arc = np.minimum(arc, N - arc)
    far_mask = arc > skip
    too_close = (dist < D) & far_mask
    # Repulsion : pour chaque paire violatrice, push proportionnel à (D-d)/d
    safe_dist = np.where(dist > 1e-30, dist, 1e-30)
    push_factor = np.where(too_close, beta * (D - dist) / safe_dist, 0.0)
    # somme des poussées sur chaque point i depuis tous les j
    push = (push_factor[:, :, None] * diff).sum(axis=1)
    return pts2 + push


def run_pieranski(name: str, n_points: int = 200, D: float = 1.0,
                  n_iter: int = 1500, alpha: float = 0.15, beta: float = 0.5,
                  skip_frac: float = 0.03,
                  reparam_every: int = 50) -> dict:
    """Lance Pieranski sur le nœud K."""
    pts = knot_curve(name, n_points + 1)[:-1]
    skip = max(int(n_points * skip_frac), 5)

    # Renormaliser pour que d_min initial ≈ D
    d0 = d_min_far(pts, skip)
    if d0 < 1e-10:
        d0 = 1e-3
    pts = pts * (D / d0)

    L_history, dmin_history = [], []
    L_history.append(length(pts))
    dmin_history.append(d_min_far(pts, skip))

    for it in range(n_iter):
        pts = pieranski_iteration(pts, alpha, D, beta, skip)
        if (it + 1) % reparam_every == 0:
            pts = reparametrize(pts, n_points)
        if (it + 1) % 100 == 0:
            L_history.append(length(pts))
            dmin_history.append(d_min_far(pts, skip))

    L_final = length(pts)
    d_min_final = d_min_far(pts, skip)
    R_final = L_final / d_min_final
    return {
        "L_init": L_history[0],
        "d_min_init": dmin_history[0],
        "L_final": L_final,
        "d_min_final": d_min_final,
        "R_final": R_final,
        "L_history": L_history,
        "dmin_history": dmin_history,
        "pts": pts,
    }


def main():
    print("=" * 84)
    print("  PIERANSKI 1 — minimisation rope-length des courbes individuelles")
    print("=" * 84)
    print("  Paramètres : N=200, D=1.0, n_iter=1500, α=0.15, β=0.5")
    print()
    print(f"  {'K':>5} {'L_init':>8} {'d_init':>8} {'L_final':>9} "
          f"{'d_final':>9} {'R_pieranski':>12} {'R_pub':>9} {'ratio':>8}")
    print("  " + "-" * 75)

    rows = []
    for K in KNOTS:
        try:
            r = run_pieranski(K)
            ratio = r["R_final"] / PUBLISHED[K]
            print(f"  {K:>5} {r['L_init']:>8.2f} {r['d_min_init']:>8.4f} "
                  f"{r['L_final']:>9.2f} {r['d_min_final']:>9.4f} "
                  f"{r['R_final']:>12.2f} {PUBLISHED[K]:>9.2f} {ratio:>8.2f}")
            rows.append((K, r["R_final"], PUBLISHED[K], ratio))
        except Exception as e:
            print(f"  {K:>5}  ERREUR : {type(e).__name__} {e}")

    # Stat
    if rows:
        ratios = np.array([r[3] for r in rows])
        print(f"\n  Statistiques R_pieranski / R_publié :")
        print(f"    moyenne  = {ratios.mean():.2f}")
        print(f"    médiane  = {np.median(ratios):.2f}")
        print(f"    min/max  = {ratios.min():.2f} / {ratios.max():.2f}")
        n_close = (np.abs(ratios - 1) < 0.2).sum()
        print(f"    nb dans 20% de R_pub : {n_close}/{len(rows)}")

    # Lecture
    print(f"\n  LECTURE :")
    print(f"  ─────────────────────────────────────────")
    print(f"  Si R_pieranski ≈ R_pub, l'algorithme converge correctement et")
    print(f"  on peut utiliser ces courbes propres pour le calcul de masse.")
    print(f"  Sinon, l'implémentation simple a des limites (nombreux paramètres")
    print(f"  d'optimisation, convergence locale).")
    print(f"  ")
    print(f"  Une convergence à ~ 20% serait déjà un grand progrès vs catalogue")
    print(f"  actuel (qui a ratio 2 à 1298×).")
    print(f"  ")
    print(f"  Si succès : prochaine étape = Pieranski 2 sur le LIEN nucléon")
    print(f"  (3_1+7_2+7_2 avec Lk=3 imposé) pour le calcul 3D de la masse")
    print(f"  du proton.")
    print("=" * 84)


if __name__ == "__main__":
    main()
