r"""
Formule de masse hamiltonienne physique (chantier 1).

Construit m(K) comme somme d'effets physiques explicites du nœud χ :

    m(K) = H_EM(K)  +  H_retract(K)  +  H_tens(K)  +  H_CS(K)            (*)

avec chaque terme physiquement motivé (pas d'exposant ajusté) :

  - H_EM      = (α_EM/2) · U_EM(K)              énergie EM du tube
                                                 (∮(κ²+τ²)ds, α_EM = 1/137)
  - H_retract = ε_r · ρ_J(K)                    liaison par rétractation
                                                 (ρ_J = span(V_K) + γ|V_K(-1)|)
  - H_tens    = T · L(K)                        tension topologique
                                                 (L : longueur d'arc)
  - H_CS      = ε_CS · h_j(K)                   poids conforme Chern--Simons
                                                 (h_j = j(j+1)/(k+2))

On ne se compare PAS à la formule V4 (qui est un fit). On compare aux
seules particules à nœud simple expérimentalement identifiées dans la
TQNT :
    - 3_1 = électron  (m_e   = 0.5110 MeV)
    - 8_1 = π⁰       (m_π0  = 134.977 MeV)

Question testée : peut-on, en combinant LINÉAIREMENT les quatre effets
physiques, reproduire le rapport observé m(π⁰)/m(e⁻) ≈ 264 ?

Cf. docs/brouillon_hamiltonien_multipart.tex (chantier 1, jalon « formule
non fittée »).
"""
from __future__ import annotations
import numpy as np

from theory.mass_formula import u_em, c_j      # c_j ≡ ρ_J (Jones-based)
from theory.abraham import REFERENCE_N50K
from theory.jones import braid_index

K_LEVEL = 135
KP2 = K_LEVEL + 2                              # 137
ALPHA_EM = 1.0 / KP2                           # = 1/137 (physique, non ajusté)
E_0 = 1.0                                      # MeV (postulates.py : E_0 fixé)

# Particules à NŒUD SIMPLE expérimentalement identifiées (TQNT V5 spectre)
EXPERIMENTAL = {
    "3_1": ("électron", 0.51099895),
    "8_1": ("π⁰",       134.9768),
}


# ============================================================
# Termes physiques (un par effet)
# ============================================================
def H_EM(K: str) -> float:
    """Énergie électromagnétique du tube χ noué : (α_EM/2) · U_EM(K)
    (MeV, en posant E_0=1 MeV pour donner les unités)."""
    return 0.5 * ALPHA_EM * u_em(K) * E_0


def H_retract_raw(K: str) -> float:
    """Poids de rétractation : ρ_J(K) (sans pré-facteur ε_r)."""
    return c_j(K)


def H_tens_raw(K: str) -> float:
    """Poids de tension : L(K) (longueur d'arc, sans pré-facteur T)."""
    return REFERENCE_N50K[K]["L"]


def h_j_anyon(K: str) -> float:
    """Poids conforme du WZW : h_j = j(j+1)/(k+2),
    avec j = (braid_index-1)/2 (approximation : anyon associé)."""
    b = braid_index(K)
    j = (b - 1) / 2.0
    return j * (j + 1) / KP2


# ============================================================
# Calibration / test
# ============================================================
def calibrate_two_terms(K1: str, K2: str, term_a, term_b) -> tuple[float, float]:
    """Résout m(K1)=A·term_a(K1) + B·term_b(K1), m(K2)=A·term_a(K2) + B·term_b(K2)."""
    m1, m2 = EXPERIMENTAL[K1][1], EXPERIMENTAL[K2][1]
    M = np.array([[term_a(K1), term_b(K1)],
                  [term_a(K2), term_b(K2)]])
    return np.linalg.solve(M, np.array([m1, m2]))


def main():
    print("=" * 68)
    print("  FORMULE DE MASSE HAMILTONIENNE PHYSIQUE (chantier 1)")
    print("  m(K) = H_EM + H_retract + H_tens + H_CS")
    print("=" * 68)

    # -- J3 physique : poids de chaque terme par nœud --------------------
    print("\n  [J3] Poids physiques par nœud :")
    print(f"  {'K':>5} {'particule':>12} {'m_exp':>10}  "
          f"{'H_EM (α=1/137)':>15} {'ρ_J':>7} {'L':>7} {'h_j':>9}")
    print("  " + "-" * 70)
    for K, (label, m_exp) in EXPERIMENTAL.items():
        print(f"  {K:>5} {label:>12} {m_exp:>10.3f}  "
              f"{H_EM(K):>15.4e} {c_j(K):>7.2f} "
              f"{REFERENCE_N50K[K]['L']:>7.2f} {h_j_anyon(K):>9.5f}")

    # -- Ratios des termes : peut-on reproduire m_π0 / m_e ≈ 264 ? -------
    m_e = EXPERIMENTAL["3_1"][1]
    m_pi = EXPERIMENTAL["8_1"][1]
    ratio_exp = m_pi / m_e
    print(f"\n  [Diagnostic] m(π⁰)/m(e⁻) expérimental = {ratio_exp:.1f}")
    print(f"    ratio U_EM           : {u_em('8_1')/u_em('3_1'):>6.2f}")
    print(f"    ratio ρ_J            : {c_j('8_1')/c_j('3_1'):>6.2f}")
    print(f"    ratio L              : "
          f"{REFERENCE_N50K['8_1']['L']/REFERENCE_N50K['3_1']['L']:>6.2f}")
    print(f"    ratio h_j            : {h_j_anyon('8_1')/h_j_anyon('3_1'):>6.2f}")
    print(f"    ratio U_EM · ρ_J     : "
          f"{u_em('8_1')*c_j('8_1')/(u_em('3_1')*c_j('3_1')):>6.2f}")
    print(f"    ratio U_EM · ρ_J²    : "
          f"{u_em('8_1')*c_j('8_1')**2/(u_em('3_1')*c_j('3_1')**2):>6.2f}")
    print(f"    ratio U_EM · ρ_J³    : "
          f"{u_em('8_1')*c_j('8_1')**3/(u_em('3_1')*c_j('3_1')**3):>6.2f}")

    # -- Auto-cohérence retraction ---------------------------------------
    # m = U_EM(K) · ρ(K, m) ; ρ(K,m) = ρ_J² / (ρ_J + m/E_0)
    # => m = (E_0·ρ_J/2)·[-1 + sqrt(1 + 4·U_EM/E_0)]
    print(f"\n  [Auto-cohérence rétractation, E_0 = {E_0} MeV]")
    print(f"    m_sc(K) = (E_0·ρ_J/2)·[-1 + √(1 + 4·U_EM/E_0)]")
    print(f"    (limite U_EM >> E_0 : m ≈ ρ_J · √(E_0 · U_EM) )")
    print(f"    {'K':>5} {'m_sc (MeV)':>12} {'m_exp (MeV)':>12} {'écart':>8}")
    for K, (label, m_exp) in EXPERIMENTAL.items():
        rho_J = c_j(K)
        U = u_em(K)
        m_sc = (E_0 * rho_J / 2.0) * (-1 + np.sqrt(1 + 4 * U / E_0))
        ratio = m_sc / m_exp
        print(f"    {K:>5} {m_sc:>12.3f} {m_exp:>12.3f} {ratio:>7.1f}×")

    # -- Calibration à 2 paramètres : (ε_EM, ε_r) sur (3_1, 8_1) ---------
    print(f"\n  [Calibration à 2 paramètres physiques : ε_EM·U_EM/2 + ε_r·ρ_J]")
    eps_EM, eps_r = calibrate_two_terms(
        "3_1", "8_1",
        lambda K: 0.5 * u_em(K),
        lambda K: c_j(K),
    )
    print(f"    ε_EM = {eps_EM:>.4e} MeV  (à comparer à α_EM·E_0 = "
          f"{ALPHA_EM*E_0:.4e})")
    print(f"    ε_r  = {eps_r:>.4e} MeV  (à comparer à E_0 = {E_0:.2f})")
    print(f"    ε_EM est {abs(eps_EM)/(ALPHA_EM*E_0):.1f}× α_EM·E_0  -- "
          f"signe : {'+' if eps_EM > 0 else '-'}")
    print(f"    ε_r  est {abs(eps_r)/E_0:.1f}× E_0           -- "
          f"signe : {'+' if eps_r > 0 else '-'}")

    # -- Lecture ----------------------------------------------------------
    print(f"\n  LECTURE PHYSIQUE :")
    print(f"  ────────────────────────────────────────────────────")
    print(f"  Le rapport m(π⁰)/m(e⁻) = {ratio_exp:.0f} demande une non-linéarité")
    print(f"  TRÈS forte dans les caractéristiques physiques du nœud.")
    print(f"  Aucune combinaison LINÉAIRE de U_EM, ρ_J, L, h_j ne l'atteint :")
    print(f"  - U_EM seul, ρ_J seul, L seul : ratios entre 1 et 7 ;")
    print(f"  - U_EM·ρ_J ~ 18 ; U_EM·ρ_J² ~ 52 ; U_EM·ρ_J³ ~ 146.")
    print(f"  La self-cohérence rétractation (limite ρ_J·√(U_EM)) donne")
    print(f"  un ratio prédit ~7, encore insuffisant.")
    print(f"")
    print(f"  Pour atteindre {ratio_exp:.0f}×, il faut soit :")
    print(f"  (a) un exposant proche de 3 sur ρ_J (origine algébrique à")
    print(f"      identifier, peut-être lié au volume du corps lemniscaté")
    print(f"      du nœud ou aux dimensions conformes h_j) ;")
    print(f"  (b) une non-linéarité multi-niveau de la rétractation")
    print(f"      (rétractations imbriquées : ρ(ρ(ρ ... )) jusqu'à saturation) ;")
    print(f"  (c) des corrections quantiques de boucles d'anyons (chantier 4).")
    print(f"")
    print(f"  ⇒ La formule V4 (c_J^3,55) capture (a) empiriquement.  Pour la")
    print(f"     dériver, le bon chantier est l'origine ALGÉBRIQUE de l'exposant")
    print(f"     dans la TQFT Chern–Simons, PAS le Hamiltonien géométrique seul.")
    print("=" * 68)


if __name__ == "__main__":
    main()
