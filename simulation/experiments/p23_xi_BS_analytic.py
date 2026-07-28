r"""
CALCUL ANALYTIQUE DE ξ_BS pour le lien de Hopf canonique.

Configuration : deux cercles de rayon R, perpendiculaires, en enlacement
Hopf (Lk = 1).
  K_1 : cercle en plan xy, centré à l'origine, rayon R
    γ_1(s) = (R cos s, R sin s, 0)
    T_1(s) = (-sin s, cos s, 0)
    N_1(s) = (-cos s, -sin s, 0)
  K_2 : cercle en plan yz, centré en (R, 0, 0), rayon R
    γ_2(t) = (R, R cos t, R sin t)
    T_2(t) = (0, -sin t, cos t)

L'intégrale recherchée :
  ξ_BS = ∮_s ∮_t N_1(s) · [T_2(t) × (γ_1(s) - γ_2(t))] / |γ_1(s) - γ_2(t)|^3 dt ds

Après simplification (calcul vectoriel) :
  Numérateur  : sin s cos t - cos s
  Dénominateur : R² · [3 - 2 cos s - 2 sin s cos t]^(3/2)
  Mesure : R² ds dt (les Jacobiens)

Donc :
  ξ_BS = ∫∫ (sin s cos t - cos s) / [3 - 2 cos s - 2 sin s cos t]^(3/2) ds dt

CETTE INTÉGRALE EST DIMENSIONNELLEMENT UNIVERSELLE : R disparaît.
ξ_BS est une CONSTANTE PURE de la topologie de Hopf.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import dblquad
from scipy.special import ellipk, ellipe


def integrand(t, s):
    """Intégrand de ξ_BS au point (s, t)."""
    cos_s = np.cos(s); sin_s = np.sin(s); cos_t = np.cos(t)
    numerator = sin_s * cos_t - cos_s
    denom_sq = 3 - 2*cos_s - 2*sin_s*cos_t
    if denom_sq <= 1e-12:
        return 0.0
    return numerator / denom_sq**1.5


print("═"*90)
print("  Calcul de ξ_BS pour le lien de Hopf canonique (R=1, R=1)")
print("═"*90)

# Méthode 1 : Quadrature 2D
print("\n  Méthode 1 : scipy.integrate.dblquad (haute précision)")
xi_BS_quad, err = dblquad(integrand, 0, 2*np.pi, 0, 2*np.pi, epsabs=1e-8)
print(f"    ξ_BS = {xi_BS_quad:.6f}  (erreur estimée : {err:.2e})")
print(f"    |ξ_BS| = {abs(xi_BS_quad):.6f}")


# Méthode 2 : grille fine pour vérification
print("\n  Méthode 2 : Sommation sur grille fine pour cross-check")
for N in [200, 500, 1000]:
    s = np.linspace(0, 2*np.pi, N, endpoint=False)
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    ds = 2*np.pi/N; dt = 2*np.pi/N
    S, T = np.meshgrid(s, t, indexing='ij')
    cos_S = np.cos(S); sin_S = np.sin(S); cos_T = np.cos(T)
    num = sin_S*cos_T - cos_S
    denom = 3 - 2*cos_S - 2*sin_S*cos_T
    integrand_vals = num / denom**1.5
    integral = np.sum(integrand_vals) * ds * dt
    print(f"    N = {N:>4} : ξ_BS = {integral:+.6f}")


# Méthode 3 : décomposition analytique partielle (intégration sur t)
print("\n" + "═"*90)
print("  Décomposition analytique : intégration interne sur t")
print("═"*90)
print(f"""
  Pour t fixé et s variable, posons :
    a = 3 - 2 cos s,  b = 2 sin s
    denom = (a - b cos t)^(3/2)
    num   = sin s cos t - cos s

  L'intégrale interne :
    I_inner(s) = ∫₀^{{2π}} (sin s · cos t - cos s) / (a - b cos t)^(3/2) dt

  Décomposons :
    I_inner = sin s · ∫ cos t / (a - b cos t)^(3/2) dt
              - cos s · ∫ 1 / (a - b cos t)^(3/2) dt

  Ces deux intégrales sont des intégrales elliptiques complètes :
    ∫₀^{{2π}} dt / (a - b cos t)^(3/2) = 4 / [(a-b)·√(a+b)] · E(k)
    où k² = 2b/(a+b)

  Avec :
    a - b = 3 - 2cos s - 2 sin s = 3 - 2(cos s + sin s) = 3 - 2√2 sin(s+π/4)
    a + b = 3 - 2cos s + 2 sin s = 3 + 2√2 sin(s-π/4)... etc.

  La double intégrale finale s'exprime donc en termes de combinaisons
  d'intégrales elliptiques complètes E(k) et K(k) — une forme close
  qui peut être tabulée numériquement.
""")


# Méthode 4 : forme analytique numérique via intégrales elliptiques
print("\n" + "═"*90)
print("  Méthode 4 : Forme analytique via intégrales elliptiques")
print("═"*90)


def I_inner_analytic(s):
    """Intégrale interne sur t, exprimée via E(k), K(k)."""
    cos_s = np.cos(s); sin_s = np.sin(s)
    a = 3 - 2*cos_s
    b = 2*sin_s
    if abs(b) < 1e-10:
        # Cas b=0 : sin s = 0 → s = 0, π
        # alors denominator = a^(3/2) constant
        return -cos_s * 2*np.pi / a**1.5
    # k² = 2b/(a+b) pour b > 0
    # Mais on doit gérer le signe de b
    if b > 0:
        k2 = 2*b / (a + b)
        if k2 > 1: k2 = 1 - 1e-12  # garde-fou
        k = np.sqrt(max(k2, 0))
        # ∫₀^{2π} dt / (a - b cos t)^(3/2)
        # Identité standard : = 4/((a-b)·(a+b)^(1/2)) · E(k²)
        # Mais avec a-b dans le dénominateur, attention au signe
        if a - b <= 0:
            return 0  # cas dégénéré, ne devrait pas arriver pour notre config
        I_1 = 4*ellipe(k2) / ((a-b) * np.sqrt(a+b))
        # ∫₀^{2π} cos t / (a - b cos t)^(3/2) dt
        # = (2/b)[I_1 · a/2 - K(k²) · 2/√(a+b)]  (formule détaillée)
        # Simplification via intégration par parties
        I_2 = (a/b) * I_1 - 2*ellipk(k2) / (b * np.sqrt(a+b)) * 2
        # NOTE : formule à vérifier, on s'en remet à la quadrature pour l'instant
    else:
        return 0  # cas non analysé pour l'instant
    return sin_s * I_2 - cos_s * I_1


# Comparaison rapide
print("  Comparaison forme analytique vs quadrature à différents s :")
print(f"  {'s':>8} {'I_inner(quad)':>16} {'I_inner(analytic)':>20}")
print("  " + "-" * 50)
for s_val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    # Quadrature
    from scipy.integrate import quad
    I_q, _ = quad(lambda t: integrand(t, s_val), 0, 2*np.pi)
    # Analytique (peut être incomplète)
    try:
        I_a = I_inner_analytic(s_val)
    except Exception:
        I_a = None
    print(f"  {s_val:>8.3f} {I_q:>16.6f} "
          f"{I_a if I_a is not None else '(non implémenté)':>20}")


# ───────── Comparaison à l'empirique ─────────
print("\n" + "═"*90)
print("  COMPARAISON À L'EMPIRIQUE")
print("═"*90)

xi_empirical = 1.73
print(f"""
  ξ_BS analytique (Hopf canonique) : {abs(xi_BS_quad):.4f}
  ξ_BS empirique (extrait par calibration) : {xi_empirical:.4f}

  Rapport : {abs(xi_BS_quad) / xi_empirical:.3f}

  Interprétation :
    • Si |ξ_BS_analytique| ≈ 1.73 ≈ ξ_BS_empirique → validation parfaite
    • Si ratio ≈ 4π ou π → facteur de normalisation manquant
    • Si ratio aléatoire → la configuration Hopf canonique n'est pas
      la bonne référence (entrelacs réels ont géométries différentes)
""")


# ───────── Cas particuliers et limites ─────────
print("\n" + "═"*90)
print("  Cas particuliers et limites")
print("═"*90)

# 1. Limite asymptotique D >> R (boucles éloignées, dipole pur)
print("\n  Limite D >> R (dipole) :")
print("    F^rad ~ m_1 m_2 / D^3 où m_i = πR_i^2")
print("    En unités où R = 1 : m = π, donc F^rad ~ π² / D^3 ≈ 9.87 / D^3")

# 2. Limite D = 0 (configurations dégénérées, intersection)
print("\n  Limite D = 0 (intersection) :")
print("    L'intégrale diverge — singularité de Biot-Savart")

# 3. Notre choix : D = R = 1 (Hopf canonique, lien minimal)
print(f"\n  Notre choix (Hopf canonique, R = 1) : ξ_BS = {abs(xi_BS_quad):.4f}")
print("    Cette valeur est la NORMALISATION GÉOMÉTRIQUE PURE.")
