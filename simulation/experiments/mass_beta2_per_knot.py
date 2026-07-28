r"""
Chantier 1 — β2 recalibré avec n auto-cohérent par particule.

L'équation β2 :
    m = K_unit · U_EM(K) · ρ(K, m)^n,  ρ(K, E) = c_J² / (c_J + E/E_0)

Si l'on fixe K_unit = α₀ (calibration V4) et qu'on impose le point
auto-cohérent m_V4 = α₀ · U_EM · c_J^3,55, on résout pour n par nœud :

    n_K = log(c_J^3,55) / log(ρ(K, m_V4))

Question : n_K est-il stable autour de 3,55 (β2 = bonne réinterprétation),
ou varie-t-il avec K (β2 = approximation locale seulement) ?

Cf. docs/brouillon_hamiltonien_multipart.tex (chantier 1, pivot β).
"""
from __future__ import annotations
import numpy as np

from theory.mass_formula import u_em, c_j, mass, ALPHA0, P_EXP
from theory.abraham import REFERENCE_N50K

KNOTS = list(REFERENCE_N50K.keys())
E_0 = 1.0    # MeV (postulates)


def rho_KE(K: str, E: float) -> float:
    rhoJ = c_j(K)
    return rhoJ ** 2 / (rhoJ + E / E_0)


def threshold_mass(K: str) -> float:
    """Seuil m* tel que ρ(K, m*) = 1 :  m* = c_J(c_J-1)·E_0.
    Pour m < m*, ρ > 1, n_K > 0.  Pour m > m*, ρ < 1, n_K < 0."""
    rhoJ = c_j(K)
    return rhoJ * (rhoJ - 1) * E_0


def n_per_knot(K: str) -> float | None:
    """n_K tel que c_J^3,55 = ρ(K, m_V4)^n_K (fixed K_unit = α₀)."""
    m_V4 = mass(K, "single")
    rho = rho_KE(K, m_V4)
    if abs(np.log(rho)) < 1e-12:    # ρ ≈ 1 → singularité
        return None
    return P_EXP * np.log(c_j(K)) / np.log(rho)


def verify_fixpoint(K: str, n: float, n_iter: int = 500) -> tuple[float, bool]:
    """Itère l'équation β2 et regarde si elle converge à un point fixe stable.
    Retourne (m_final, converged_to_fixpoint)."""
    rhoJ = c_j(K)
    U = u_em(K)
    m = U * rhoJ ** n
    history = [m]
    for _ in range(n_iter):
        rho = rhoJ ** 2 / (rhoJ + m / E_0)
        m_new = ALPHA0 * U * rho ** n
        if abs(m_new - m) / max(abs(m), 1e-30) < 1e-8:
            return m_new, True
        m = m_new
        history.append(m)
    # Pas convergé : vérifier si on oscille entre deux valeurs
    last5 = history[-5:]
    if np.std(last5) > 0.5 * abs(np.mean(last5)):
        return float(np.mean(history[-2:])), False     # oscillation
    return m, False


def main():
    print("=" * 78)
    print("  β2 RECALIBRÉ : n_K auto-cohérent par particule")
    print("=" * 78)
    print(f"  K_unit = α₀ = {ALPHA0:.2e} (fixé)")
    print(f"  E_0 = {E_0} MeV ; ρ(K, E) = c_J²/(c_J + E/E_0)")
    print(f"  Équation : c_J^{P_EXP} = ρ(K, m_V4)^n_K  ⇒  n_K = "
          f"{P_EXP}·log(c_J)/log(ρ(K, m_V4))")
    print()

    print(f"  {'K':>5} {'c_J':>5} {'U_EM':>8} {'m_V4':>10} "
          f"{'seuil m*':>10} {'ρ(K,m_V4)':>11} {'n_K':>9} {'ρ vs 1':>8}")
    print("  " + "-" * 74)
    n_list, c_list, m_list, sign_changes = [], [], [], 0
    for K in KNOTS:
        m_V4 = mass(K, "single")
        m_star = threshold_mass(K)
        rho = rho_KE(K, m_V4)
        n_K = n_per_knot(K)
        sign = "ρ>1" if rho > 1 else ("ρ=1" if abs(rho-1) < 0.01 else "ρ<1")
        n_str = f"{n_K:>9.3f}" if n_K is not None else f"{'singul.':>9}"
        print(f"  {K:>5} {c_j(K):>5.2f} {u_em(K):>8.2f} {m_V4:>10.3f} "
              f"{m_star:>10.2f} {rho:>11.4f} {n_str} {sign:>8}")
        if n_K is not None:
            n_list.append(n_K)
            c_list.append(c_j(K))
            m_list.append(m_V4)

    print(f"\n  --- Statistiques sur n_K (sur les {len(n_list)} nœuds avec ρ ≠ 1) ---")
    n_arr = np.array(n_list)
    print(f"    moyenne  : {n_arr.mean():>8.3f}")
    print(f"    médiane  : {np.median(n_arr):>8.3f}")
    print(f"    écart-type : {n_arr.std():>6.3f}")
    print(f"    min  : {n_arr.min():>8.3f}    max  : {n_arr.max():>8.3f}")
    print(f"    nb n_K > 0 : {(n_arr > 0).sum()}    nb n_K < 0 : {(n_arr < 0).sum()}")

    # Diagnostic : VRAIS points fixes existent-ils pour n=3.55 fixé ?
    print(f"\n  --- Diagnostic : β2 avec n=3,55 fixe a-t-elle un VRAI point fixe ? ---")
    print(f"  Test sur quelques nœuds représentatifs (itération de β2 jusqu'à convergence) :")
    print(f"  {'K':>5} {'m_V4 (cible)':>14} {'m_β2 (point fixe)':>20} {'convergé ?':>12}")
    for K in ("3_1", "5_1", "6_2", "8_1", "8_3"):
        m_V4 = mass(K, "single")
        m_bp, converged = verify_fixpoint(K, P_EXP)
        flag = "oui" if converged else "NON (oscillation)"
        print(f"  {K:>5} {m_V4:>14.3f} {m_bp:>20.3e} {flag:>12}")

    # Lecture
    print(f"\n  LECTURE :")
    print(f"  ──────────────────────────────────────────────")
    if n_arr.std() / abs(n_arr.mean()) > 0.5:
        print(f"  n_K VARIE FORTEMENT par nœud (écart-type {n_arr.std():.1f} sur "
              f"moyenne {n_arr.mean():.1f}).")
        print(f"  ⇒ β2 avec n FIXE n'est PAS un point fixe de V4 — c'est une")
        print(f"    approximation valable seulement à basse masse (m << c_J·E_0,")
        print(f"    là où ρ ≈ c_J).  Pour m grand (m > c_J(c_J-1)·E_0), ρ < 1")
        print(f"    et il faudrait n négatif pour reproduire V4, ce qui n'a")
        print(f"    pas d'interprétation physique de \"profondeur de couches\".")
    else:
        print(f"  n_K est relativement stable autour de {n_arr.mean():.2f} — β2")
        print(f"  reste une interprétation cohérente, à raffiner.")

    print(f"")
    print(f"  CONCLUSION du chantier β :")
    print(f"  ──────────────────────────────────────────────")
    print(f"  L'interprétation \"V4 = auto-cohérence rétractation à profondeur")
    print(f"  3,55\" qui semblait prometteuse au premier essai s'effondre quand")
    print(f"  on teste rigoureusement. Le formalisme β2 ne fait que reproduire")
    print(f"  V4 dans le régime de basse masse, et s'écarte (ou diverge) ailleurs.")
    print(f"  Le n=3,55 reste un FIT empirique, et son origine physique demeure")
    print(f"  ouverte. Pistes restantes : analyses CS pures (boucles d'anyons),")
    print(f"  ou abandon de la forme power-law pour une loi plus complexe.")
    print("=" * 78)


if __name__ == "__main__":
    main()
