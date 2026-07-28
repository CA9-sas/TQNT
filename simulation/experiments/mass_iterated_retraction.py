r"""
Chantier 1 — pivot β : rétractations imbriquées et émergence d'un exposant.

Hypothèse à tester : la non-linéarité ρ_J^p (p ≈ 3,5) de la formule V4
émerge-t-elle naturellement d'une itération de la rétractation
        ρ(K, E) = ρ_J(K)² / (ρ_J(K) + E/E_0)
jusqu'à saturation, ou d'une auto-cohérence à plusieurs couches ?

On code et compare trois schémas :

* (β1) Itération séquentielle « littérale » : ρ_{n+1} = ρ(K, ρ_n · E_0).
       Limite : point fixe Fibonacci → (1/φ)·ρ_J. LINÉAIRE en ρ_J,
       ne génère AUCUN exposant > 1.

* (β2) Auto-cohérence à profondeur n :  m = U_EM · ρ(K, m)^n,
       avec n entier ou demi-entier (couches de confinement spatial).

* (β3) Profondeur dépendant de la longueur :  m = K_0 · U_EM · ρ_J^{γL(K)}
       avec γ = 0,10 (paramètre TQNT existant, poids du déterminant
       dans c_J — peut-être γ joue aussi le rôle de « densité de couches
       par unité de longueur »).

Mesure : pour chaque schéma, on calibre la seule constante d'échelle sur
m(3_1) = m_e et on regarde la prédiction pour les autres nœuds + le
ratio observé m(π⁰)/m(e⁻) = 264.

Cf. docs/brouillon_hamiltonien_multipart.tex (chantier 1, pivot β).
"""
from __future__ import annotations
import numpy as np

from theory.mass_formula import u_em, c_j
from theory.abraham import REFERENCE_N50K

KNOTS = list(REFERENCE_N50K.keys())
E_0 = 1.0          # MeV (postulates)
GAMMA = 0.10       # poids du déterminant dans c_J (postulates / V5)

M_E   = 0.51099895   # MeV  (électron = 3_1)
M_PI0 = 134.9768     # MeV  (π⁰      = 8_1)


# ============================================================
# β1 — itération séquentielle littérale
# ============================================================
def rho_iter_sequential(K: str, n_iter: int = 50) -> tuple[float, list]:
    """ρ_{n+1} = ρ(K, ρ_n · E_0).  Renvoie (ρ_∞, trajectoire)."""
    rhoJ = c_j(K)
    rho = rhoJ
    traj = [rho]
    for _ in range(n_iter):
        rho_new = rhoJ * rhoJ / (rhoJ + rho)
        traj.append(rho_new)
        if abs(rho_new - rho) < 1e-12:
            break
        rho = rho_new
    return rho, traj


# ============================================================
# β2 — auto-cohérence à profondeur n :  m = U_EM · ρ(K, m)^n
# ============================================================
def mass_sc_depth(K: str, n: float, max_iter: int = 200, tol: float = 1e-10) -> float:
    """m = U_EM · [ρ_J² / (ρ_J + m/E_0)]^n, par itération."""
    rhoJ = c_j(K)
    U = u_em(K)
    m = U * rhoJ ** n             # 0-th guess : m ≈ U · ρ_J^n
    for _ in range(max_iter):
        rho = rhoJ * rhoJ / (rhoJ + m / E_0)
        m_new = U * rho ** n
        if abs(m_new - m) < tol:
            return m_new
        m = m_new
    return m


# ============================================================
# β3 — profondeur dépendant de la longueur
# ============================================================
def mass_length_depth(K: str, alpha: float, K_unit: float) -> float:
    """m = K_unit · U_EM(K) · ρ_J(K)^(alpha · L(K))."""
    rhoJ = c_j(K)
    U = u_em(K)
    L = REFERENCE_N50K[K]["L"]
    return K_unit * U * rhoJ ** (alpha * L)


# ============================================================
# Utilities
# ============================================================
def fit_power_law(rhoJ_list: list, m_list: list) -> tuple[float, float]:
    """Ajuste log m = log A + p · log ρ_J.  Renvoie (p, A)."""
    x = np.log(np.array(rhoJ_list))
    y = np.log(np.array(m_list))
    p, log_A = np.polyfit(x, y, 1)
    return float(p), float(np.exp(log_A))


def main():
    print("=" * 70)
    print("  CHANTIER 1 — PIVOT β : rétractations imbriquées et exposant")
    print("=" * 70)

    # ---- β1 : limite Fibonacci ----------------------------------------
    print("\n  [β1] Itération séquentielle littérale ρ_{n+1} = ρ(K, ρ_n·E_0) :")
    print(f"  {'K':>5} {'ρ_J':>6} {'ρ_∞ (β1)':>10} {'ρ_∞/ρ_J':>10} {'attendu':>10}")
    print("  " + "-" * 50)
    inv_phi = 2.0 / (1.0 + np.sqrt(5.0))
    for K in ("3_1", "5_1", "7_1", "8_1", "8_3", "10_1"):
        rho_inf, _ = rho_iter_sequential(K)
        print(f"  {K:>5} {c_j(K):>6.2f} {rho_inf:>10.4f} "
              f"{rho_inf/c_j(K):>10.4f} {inv_phi:>10.4f}")
    print(f"  ⇒ Confirmé : ρ_∞ = (1/φ)·ρ_J. LINÉAIRE — pas d'exposant. KO.")

    # ---- β2 : auto-cohérence à profondeur n ---------------------------
    print("\n  [β2] Auto-cohérence à profondeur n : m = U_EM · ρ(K, m)^n")
    print("  → calibrer sur (e⁻ = 3_1), prédire (π⁰ = 8_1), pour plusieurs n :")
    print(f"  {'n':>5} {'K_unit (calib.)':>16} {'m(π⁰) prédit':>14} "
          f"{'ratio vs 264':>14}")
    print("  " + "-" * 56)
    best_n, best_ratio = None, np.inf
    for n in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.55, 4.0):
        m31 = mass_sc_depth("3_1", n)
        K_unit = M_E / m31
        m81 = K_unit * mass_sc_depth("8_1", n)
        ratio = m81 / M_E
        err = abs(ratio - 264)
        if err < best_ratio:
            best_ratio, best_n = err, n
        print(f"  {n:>5.2f} {K_unit:>16.4e} {m81:>14.3f} {ratio:>14.1f}")
    n_best = best_n
    print(f"  ⇒ Meilleur n : {n_best} (donne ratio {(264-best_ratio):.0f} "
          f"vs 264 expérimental)")

    # ---- β3 : profondeur dépendant de la longueur (avec γ existant) ---
    print("\n  [β3] Profondeur ∝ longueur :  m = K_unit · U_EM · ρ_J^(γ·L), γ=0,10")
    K_unit = M_E / mass_length_depth("3_1", GAMMA, 1.0)
    m81_pred = mass_length_depth("8_1", GAMMA, K_unit)
    ratio = m81_pred / M_E
    print(f"  γ = {GAMMA:.2f} (poids déjà existant dans c_J)")
    print(f"  exposant effectif 3_1 : γ·L = {GAMMA * REFERENCE_N50K['3_1']['L']:.2f}")
    print(f"  exposant effectif 8_1 : γ·L = {GAMMA * REFERENCE_N50K['8_1']['L']:.2f}")
    print(f"  K_unit calibré sur e⁻ : {K_unit:.4e}")
    print(f"  m(π⁰) prédit          : {m81_pred:.2f} MeV  (vs 134,98 mesuré)")
    print(f"  ratio prédit          : {ratio:.1f}  (vs 264 expérimental)")

    # Balayage de α : quel α donne le meilleur (e⁻, π⁰) ?
    print(f"\n  Balayage de α dans m = K·U·ρ_J^(α·L) pour matcher (e⁻, π⁰) :")
    print(f"  {'α':>7} {'γ·L(3_1)':>10} {'γ·L(8_1)':>10} "
          f"{'m(π⁰) prédit':>14} {'ratio':>8}")
    for alpha in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
        K_unit_a = M_E / mass_length_depth("3_1", alpha, 1.0)
        m81_a = mass_length_depth("8_1", alpha, K_unit_a)
        print(f"  {alpha:>7.3f} {alpha*REFERENCE_N50K['3_1']['L']:>10.2f} "
              f"{alpha*REFERENCE_N50K['8_1']['L']:>10.2f} "
              f"{m81_a:>14.3f} {m81_a/M_E:>8.1f}")

    # ---- Prédictions β2 (n meilleur) sur tous les nœuds ---------------
    print(f"\n  [β2 final] Prédictions m(K) à n = {n_best} pour le catalogue :")
    print(f"  {'K':>5} {'ρ_J':>6} {'U_EM':>8} {'m_pred (MeV)':>14}")
    print("  " + "-" * 42)
    K_unit_final = M_E / mass_sc_depth("3_1", n_best)
    masses_pred = []
    for K in KNOTS:
        m_pred = K_unit_final * mass_sc_depth(K, n_best)
        masses_pred.append(m_pred)
        print(f"  {K:>5} {c_j(K):>6.2f} {u_em(K):>8.2f} {m_pred:>14.4f}")

    # ---- Fit de loi de puissance sur les m prédits par β2 -------------
    rhoJ_list = [c_j(K) for K in KNOTS]
    p_fit, A_fit = fit_power_law(rhoJ_list, masses_pred)
    print(f"\n  Ajustement log m_pred = log A + p · log ρ_J  (sur 19 nœuds)")
    print(f"    p_émergent = {p_fit:.2f}")
    print(f"    A          = {A_fit:.4e}")
    print(f"    (À comparer à l'exposant V4 c_J^3,55)")

    # ---- Lecture honnête ----------------------------------------------
    print("\n  LECTURE :")
    print("  ──────────────────────────────────────────────")
    print(f"  β1 (ρ imbriqué littéral)  : limite linéaire, KO.")
    print(f"  β2 (auto-cohérence profondeur n) :")
    print(f"    le n qui donne le ratio 264 expérimental est ≈ {n_best}")
    print(f"    (exposant émergent sur la loi de puissance : {p_fit:.2f}).")
    print(f"  β3 (profondeur ∝ longueur, γ=0,10) :")
    print(f"    donne ratio {ratio:.1f} (vs 264) — pas assez.")
    print(f"  ")
    print(f"  Si n ≈ 3,5 est l'exposant naturel, il faut justifier")
    print(f"  PHYSIQUEMENT pourquoi le confinement opère sur ~3,5 couches.")
    print(f"  Pistes : 3 dimensions spatiales + dimension anyonique fractionnaire,")
    print(f"  ou nombre effectif de modes de courbure/torsion par tube χ.")
    print("=" * 70)


if __name__ == "__main__":
    main()
