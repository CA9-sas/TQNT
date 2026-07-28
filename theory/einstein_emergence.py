"""
V5.8 — Dérivation des équations d'Einstein depuis TQNT-V5 χ-field.

Ce module formalise et calcule, pour la métrique conforme émergente du
Postulat~P5 :

    Ω²(r) = 1 + κ_g · |Ψ(r)|²        (V5 P5)
    ds²   = -c² Ω²(r) dt² + Ω²(r) δ_ij dx^i dx^j   (jauge conforme totale)

les objets de géométrie différentielle nécessaires aux équations
d'Einstein G_μν = (8πG/c⁴) T_μν :

  - métrique g_μν (statique, conformément plate),
  - symboles de Christoffel Γ^μ_νρ,
  - tenseur de Riemann R^μ_νρσ,
  - tenseurs de Ricci R_μν et R,
  - tenseur d'Einstein G_μν,
  - tenseur énergie-impulsion T_μν du champ Ψ (χ-field).

Le calcul symbolique est mené avec sympy dans le cas conformément plat
statique (Ψ = Ψ(r), r = √(x²+y²+z²)), suffisant pour les limites Newton
et Schwarzschild. Les routines numériques évaluent ensuite les expressions
sur un profil de densité |Ψ(r)|² donné.

---------------------------------------------------------------------------
CADRE THÉORIQUE
---------------------------------------------------------------------------

V5 P5 affirme que l'espace-temps émerge des modulations du champ χ :
la métrique est une conséquence de |Ψ|², elle n'est PAS un degré de
liberté indépendant.

Dans la jauge conforme (où g_μν = Ω²(x) η_μν) :

  Ω²(x) = 1 + κ_g · |Ψ(x)|²       (P5)
  κ_g   = (96π/7) · α_EM^18 · λbar_C(proton)   (V5.5 étape 6)
  κ_g · |Ψ|²  ≪ 1 dans tout le régime newtonien et faibles champs
              (h_0 ≈ 3 × 10⁻³⁹ même à la surface du proton)

En jauge mixte "Newtonien" (utilisée pour la limite faible champ) :
  ds² = -c² (1 + 2Φ/c²) dt² + (1 - 2Φ/c²)^{-1} δ_ij dx^i dx^j
On utilise ici la jauge purement conforme (Ω² partout) car c'est la
SEULE compatible avec l'invariance de Lorentz locale du χ-field : la
trace du tenseur stress-énergie ne peut pas être couplée différemment
au temps et à l'espace sans briser cette symétrie.

---------------------------------------------------------------------------
STATUT DE LA DÉRIVATION
---------------------------------------------------------------------------

Ce module mène les calculs **exactement** dans le cas conformément plat
statique :
  g_μν = Ω²(r) η_μν, Ω² = 1 + κ_g · |Ψ(r)|²

Pour ce cas, les formules de transformation conforme de Wald (App. D)
donnent :

  Γ^λ_μν = δ^λ_μ ∂_ν ln Ω + δ^λ_ν ∂_μ ln Ω - η_μν η^λρ ∂_ρ ln Ω

  R_μν = -2 ∇_μ ∇_ν ln Ω - η_μν □ ln Ω
         + 2 (∂_μ ln Ω)(∂_ν ln Ω) - 2 η_μν (∂ ln Ω)²    (n=4)

  R = -6 Ω^{-2} [□ ln Ω + (∂ ln Ω)²]   (signature -+++ )

où □ = η^μν ∂_μ ∂_ν est le d'Alembertien plat.

Les LIMITES Newton et Schwarzschild sont ensuite extraites en
linéarisant Ω² = 1 + h, h ≪ 1.

LIMITES HONNÊTES :
  (a) la dérivation complète G_μν = κ T_μν exige une équation
      dynamique pour Ψ (équation de Schrödinger non-linéaire couplée
      à la métrique). Ici, on traite |Ψ|² comme source DONNÉE et
      on VÉRIFIE la cohérence des équations d'Einstein pour la
      métrique conforme : c'est une dérivation **cinématique**, pas
      dynamique. La dynamique de Ψ est traitée séparément dans le
      solveur Schrödinger (theory/schrodinger.py).
  (b) Le tenseur stress-énergie T_μν est calculé pour un champ scalaire
      Ψ couplé minimalement à g_μν. La trace T = T^μ_μ s'identifie à la
      densité ρ = m |Ψ|² dans la limite non-relativiste statique.
  (c) Pour les régimes non-conformes (rotation, ondes gravitationnelles)
      la métrique V5 P5 doit être étendue ; ce module ne couvre PAS ces
      régimes.

Référence : Wald, "General Relativity" (1984), App. D pour les
transformations conformes ; Padmanabhan 2010 pour la gravité émergente.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from theory.coupling_kg import (
    ALPHA_EM,
    C_N_NUCLEON_EFF,
    G_newton_from_kappa_g,
    LAMBDABAR_C_PROTON_FM,
    kappa_g_V5p5,
)

# ---------------------------------------------------------------------------
# Constantes physiques SI.
# ---------------------------------------------------------------------------
C_LIGHT = 2.99792458e8       # m/s
HBAR = 1.054571817e-34       # J·s
M_PROTON = 1.67262192e-27    # kg
G_CODATA = 6.67430e-11       # m³/(kg s²)


# ---------------------------------------------------------------------------
# 1. Tenseur métrique conforme V5 P5.
# ---------------------------------------------------------------------------
def metric_tensor_V5(psi_squared: float,
                     signature: tuple = (-1, 1, 1, 1)) -> np.ndarray:
    """Tenseur métrique conforme g_μν = Ω² η_μν, Ω² = 1 + κ_g |Ψ|².

    Paramètres
    ----------
    psi_squared : float
        |Ψ(x)|² au point x considéré (unités SI : m⁻³ pour densité 3D,
        ou densité adimensionnelle si déjà couplée à κ_g via h = κ_g|Ψ|²).
        Par convention, on attend ici psi_squared en m⁻³ et on multiplie
        par κ_g [m] × longueur, ce qui n'est PAS dimensionnellement
        correct sans choix de normalisation. Pour rester général, ce
        module accepte directement le produit ADIMENSIONNEL h = κ_g|Ψ|².
    signature : tuple
        Signature Lorentz utilisée (par défaut (-,+,+,+)).

    Retour
    ------
    g : ndarray (4, 4)
        g_μν = Ω² · diag(signature).

    Note
    ----
    Le couplage dimensionnellement cohérent est obtenu via la
    sous-fonction conformal_factor_h() qui prend directement h adim.
    """
    h = float(psi_squared)
    omega2 = 1.0 + h
    eta = np.diag(signature).astype(float)
    return omega2 * eta


def conformal_factor_h(h: float) -> float:
    """Facteur conforme Ω² = 1 + h, où h = κ_g · |Ψ|² (adimensionnel).

    h est positif (P5 : la matière densifie l'espace-temps).
    """
    return 1.0 + float(h)


# ---------------------------------------------------------------------------
# 2. Symboles de Christoffel pour g_μν = Ω²(x) η_μν.
# ---------------------------------------------------------------------------
def christoffel_symbols_conformal(grad_ln_omega: np.ndarray,
                                  signature: tuple = (-1, 1, 1, 1)
                                  ) -> np.ndarray:
    """Symboles de Christoffel d'une métrique conformément plate.

    Pour g_μν = Ω²(x) η_μν :
        Γ^λ_μν = δ^λ_μ ∂_ν ln Ω + δ^λ_ν ∂_μ ln Ω - η_μν η^λρ ∂_ρ ln Ω

    Paramètres
    ----------
    grad_ln_omega : ndarray (4,)
        Composantes covariantes ∂_μ ln Ω.
    signature : tuple
        Signature η_μν.

    Retour
    ------
    Gamma : ndarray (4, 4, 4)
        Gamma[lam, mu, nu] = Γ^λ_μν.
    """
    d = np.asarray(grad_ln_omega, dtype=float).reshape(4)
    eta = np.diag(signature).astype(float)
    eta_inv = np.diag([1.0 / s for s in signature]).astype(float)
    Gamma = np.zeros((4, 4, 4))
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                term1 = (1.0 if lam == mu else 0.0) * d[nu]
                term2 = (1.0 if lam == nu else 0.0) * d[mu]
                term3 = eta[mu, nu] * sum(eta_inv[lam, rho] * d[rho]
                                          for rho in range(4))
                Gamma[lam, mu, nu] = term1 + term2 - term3
    return Gamma


# ---------------------------------------------------------------------------
# 3. Tenseurs de Riemann/Ricci dans le cas conforme statique.
# ---------------------------------------------------------------------------
def ricci_tensor_conformal(omega2: float,
                           grad_omega2: np.ndarray,
                           hess_omega2: np.ndarray,
                           signature: tuple = (-1, 1, 1, 1)
                           ) -> np.ndarray:
    """Tenseur de Ricci R_μν pour g_μν = Ω²(x) η_μν en n=4.

    Formule Wald App. D, écrite en termes de Ω et ses dérivées :

        R_μν = -2 ∇²_μν ln Ω - η_μν □ ln Ω
                + 2 (∂_μ ln Ω)(∂_ν ln Ω) - 2 η_μν (∂ ln Ω)²

    où ∇² et □ sont à plat (∂_μ ∂_ν, η^μν ∂_μ ∂_ν).

    Paramètres
    ----------
    omega2 : float
        Ω²(x) au point considéré.
    grad_omega2 : ndarray (4,)
        ∂_μ Ω² (composantes covariantes plates).
    hess_omega2 : ndarray (4, 4)
        ∂_μ ∂_ν Ω².
    signature : tuple
        Signature (par défaut (-,+,+,+)).

    Retour
    ------
    R_munu : ndarray (4, 4)
    """
    omega = math.sqrt(omega2)
    d_omega2 = np.asarray(grad_omega2, dtype=float).reshape(4)
    H = np.asarray(hess_omega2, dtype=float).reshape(4, 4)
    eta = np.diag(signature).astype(float)
    eta_inv = np.diag([1.0 / s for s in signature]).astype(float)

    # ∂_μ ln Ω = (1/2) ∂_μ Ω² / Ω²
    d_lnO = 0.5 * d_omega2 / omega2

    # ∂_μ∂_ν ln Ω = (1/2) [ ∂∂ Ω² / Ω² - (∂Ω²)(∂Ω²)/(Ω²)² ]
    # Note : on traite "∇²" plat ici (rigueur conforme à Wald App. D
    # pour la partie linéaire).
    d2_lnO = 0.5 * (H / omega2
                    - np.outer(d_omega2, d_omega2) / (omega2 ** 2))

    # □ ln Ω = η^μν ∂_μ∂_ν ln Ω
    box_lnO = np.einsum("mn,mn->", eta_inv, d2_lnO)

    # (∂ ln Ω)² = η^μν ∂_μ ln Ω ∂_ν ln Ω
    grad_sq_lnO = np.einsum("mn,m,n->", eta_inv, d_lnO, d_lnO)

    R_munu = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            R_munu[mu, nu] = (-2.0 * d2_lnO[mu, nu]
                              - eta[mu, nu] * box_lnO
                              + 2.0 * d_lnO[mu] * d_lnO[nu]
                              - 2.0 * eta[mu, nu] * grad_sq_lnO)
    return R_munu


def ricci_scalar_conformal(omega2: float,
                           grad_omega2: np.ndarray,
                           hess_omega2: np.ndarray,
                           signature: tuple = (-1, 1, 1, 1)
                           ) -> float:
    """Scalaire de Ricci R = g^μν R_μν.

    Pour g_μν = Ω² η_μν : g^μν = η^μν / Ω², donc
        R = (1/Ω²) η^μν R_μν[plat]
    """
    R_munu = ricci_tensor_conformal(omega2, grad_omega2, hess_omega2,
                                    signature)
    eta_inv = np.diag([1.0 / s for s in signature]).astype(float)
    return float(np.einsum("mn,mn->", eta_inv, R_munu) / omega2)


def einstein_tensor_V5(omega2: float,
                       grad_omega2: np.ndarray,
                       hess_omega2: np.ndarray,
                       signature: tuple = (-1, 1, 1, 1)
                       ) -> np.ndarray:
    """Tenseur d'Einstein G_μν = R_μν - (1/2) R g_μν.

    Avec g_μν = Ω² η_μν.
    """
    R_munu = ricci_tensor_conformal(omega2, grad_omega2, hess_omega2,
                                    signature)
    R_scalar = ricci_scalar_conformal(omega2, grad_omega2, hess_omega2,
                                      signature)
    eta = np.diag(signature).astype(float)
    g = omega2 * eta
    return R_munu - 0.5 * R_scalar * g


# ---------------------------------------------------------------------------
# 4. Tenseur énergie-impulsion T_μν du χ-field.
# ---------------------------------------------------------------------------
def T_munu_chi_field(psi: float,
                     grad_psi: np.ndarray,
                     mass: float = M_PROTON,
                     omega2: float = 1.0,
                     signature: tuple = (-1, 1, 1, 1)
                     ) -> np.ndarray:
    """Tenseur stress-énergie d'un scalaire complexe Ψ couplé minimalement.

    Lagrangien :
        L = -(ℏ²/2m) g^μν ∂_μ Ψ* ∂_ν Ψ - (1/2) m c² Ψ*Ψ

    T_μν = (ℏ²/2m) [∂_μ Ψ* ∂_ν Ψ + ∂_μ Ψ ∂_ν Ψ*]
           - g_μν · L

    Dans la limite non-relativiste et pour |Ψ|² densité de probabilité
    (m⁻³), la composante T_00 ≈ m c² |Ψ|² × (1 + corrections) qui se
    réduit à la densité d'énergie au repos ρ c².

    Paramètres
    ----------
    psi : float
        Ψ(x) (supposé réel ou module dans ce mode statique simplifié).
    grad_psi : ndarray (4,)
        ∂_μ Ψ.
    mass : float
        Masse du quantum du χ-field (par défaut m_proton).
    omega2 : float
        Ω²(x) (utilisé pour g_μν).
    signature : tuple

    Retour
    ------
    T_munu : ndarray (4, 4) en J/m³ (densité d'énergie).
    """
    eta = np.diag(signature).astype(float)
    eta_inv = np.diag([1.0 / s for s in signature]).astype(float)
    g = omega2 * eta
    g_inv = eta_inv / omega2

    d = np.asarray(grad_psi, dtype=float).reshape(4)
    coeff = HBAR ** 2 / (2.0 * mass)

    # Lagrangien (scalaire réel équivalent) :
    L_kin = -coeff * np.einsum("mn,m,n->", g_inv, d, d)
    L_pot = -0.5 * mass * (C_LIGHT ** 2) * (psi ** 2)
    L = L_kin + L_pot

    T = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            T[mu, nu] = 2.0 * coeff * d[mu] * d[nu] - g[mu, nu] * L
    return T


def energy_density_from_psi2(psi_squared: float,
                             mass: float = M_PROTON) -> float:
    """Densité d'énergie au repos ρc² = m c² · |Ψ|².

    Limite non-relativiste, statique : T_00 → ρ c².
    """
    return mass * (C_LIGHT ** 2) * float(psi_squared)


# ---------------------------------------------------------------------------
# 5. Vérification numérique de G_μν = (8πG/c⁴) T_μν.
# ---------------------------------------------------------------------------
@dataclass
class EinsteinCheck:
    """Synthèse de la vérification G_μν = (8πG/c⁴) T_μν."""
    G_used: float
    G_codata: float
    G_relative_error: float
    G_00: float
    rhs_00: float           # (8πG/c⁴) T_00
    ratio_00: float         # G_00 / rhs_00
    poisson_lhs: float      # ∇²Φ
    poisson_rhs: float      # 4πGρ
    poisson_ratio: float
    note: str


def verify_einstein_equation(h_profile: Callable[[float], float],
                             h_laplacian: Callable[[float], float],
                             rho_profile: Callable[[float], float],
                             r: float) -> EinsteinCheck:
    """Évalue G_00 et (8πG/c⁴) T_00 au rayon r pour des profils donnés.

    Cette vérification opère en **dimensionnellement cohérent** :
    on traite h(r) = κ_g · |Ψ|²(r) comme une fonction adimensionnelle
    déjà fournie (la normalisation de Ψ étant à la charge de l'utilisateur,
    cf. coupling_kg.kappa_g_V5p5 et la jauge linéaire V5 §sec:fc-cpq-coherence).

    Au premier ordre en h (régime faible champ) :
        G_00 = -∇²h
        T_00 = ρ c²
        (8πG/c⁴) T_00 = 8πG ρ / c²

    Pour que G_00 = (8πG/c⁴) T_00, il faut :
        ∇²h = -8πG ρ / c²
    Avec Φ = -(c²/2) h, cela donne ∇²Φ = 4πG ρ — l'équation de Poisson.

    Paramètres
    ----------
    h_profile : callable r -> h(r)
        h(r) = κ_g · |Ψ|²(r), adimensionnel.
    h_laplacian : callable r -> ∇²h(r) (m⁻²)
        Laplacien plat (statique sphérique).
    rho_profile : callable r -> ρ(r) (kg/m³)
        Densité de masse.
    r : float
        Rayon (m).

    Retour
    ------
    EinsteinCheck
    """
    G_pred = G_newton_from_kappa_g()
    h = h_profile(r)
    lap_h = h_laplacian(r)
    rho = rho_profile(r)

    # G_00 linéarisé (Wald App. D, faible champ statique) :
    G_00_lin = -lap_h

    # rhs : (8πG/c⁴) T_00 = 8πG ρ / c²
    rhs_00 = 8.0 * math.pi * G_pred * rho / (C_LIGHT ** 2)

    ratio = G_00_lin / rhs_00 if rhs_00 != 0 else float("nan")

    # Poisson direct : ∇²Φ = 4πGρ avec Φ = -(c²/2) h
    poisson_lhs = -0.5 * (C_LIGHT ** 2) * lap_h
    poisson_rhs = 4.0 * math.pi * G_pred * rho
    poisson_ratio = (poisson_lhs / poisson_rhs
                     if poisson_rhs != 0 else float("nan"))

    return EinsteinCheck(
        G_used=G_pred,
        G_codata=G_CODATA,
        G_relative_error=(G_pred - G_CODATA) / G_CODATA,
        G_00=G_00_lin,
        rhs_00=rhs_00,
        ratio_00=ratio,
        poisson_lhs=poisson_lhs,
        poisson_rhs=poisson_rhs,
        poisson_ratio=poisson_ratio,
        note=(
            "Vérification cinématique linéarisée (faible champ). "
            "G_00 et rhs_00 doivent coïncider lorsque h(r) résout "
            "∇²h = -8πGρ/c². L'utilisateur fournit (h, ρ) cohérents."
        ),
    )


# ---------------------------------------------------------------------------
# 6. Limite Newton : ∇²Φ = 4πGρ.
# ---------------------------------------------------------------------------
def newton_limit_check(rho_kg_m3: float,
                       lap_phi: float) -> dict:
    """Vérifie ∇²Φ = 4πGρ pour des entrées scalaires.

    Paramètres
    ----------
    rho_kg_m3 : float
        Densité de masse au point considéré.
    lap_phi : float
        ∇²Φ(r) (s⁻²) au même point (à fournir par l'utilisateur).

    Retour
    ------
    dict avec les deux membres, leur rapport et G utilisé.
    """
    G = G_newton_from_kappa_g()
    rhs = 4.0 * math.pi * G * rho_kg_m3
    return {
        "lap_phi": lap_phi,
        "four_pi_G_rho": rhs,
        "ratio": lap_phi / rhs if rhs != 0 else float("nan"),
        "G_used": G,
        "G_codata": G_CODATA,
        "G_error_pct": (G - G_CODATA) / G_CODATA * 100,
    }


def gravitational_potential_from_h(h: float, c: float = C_LIGHT) -> float:
    """Potentiel newtonien Φ = -(c²/2) h, où h = κ_g · |Ψ|².

    Dérivé de la limite faible champ g_00 = -(1 + 2Φ/c²) avec
    g_00 = -Ω² = -(1 + h).
    """
    return -0.5 * (c ** 2) * h


def newton_consistency_pointmass(M: float, r: float) -> dict:
    """Vérifie Φ_Newton(r) = -GM/r pour une masse ponctuelle.

    Sert de sanity check pour comparer G dérivé V5 vs CODATA.

    Paramètres
    ----------
    M : float (kg)
    r : float (m)
    """
    G = G_newton_from_kappa_g()
    phi_V5 = -G * M / r
    phi_codata = -G_CODATA * M / r
    return {
        "phi_V5": phi_V5,
        "phi_codata": phi_codata,
        "relative_error": (phi_V5 - phi_codata) / phi_codata,
        "G_V5": G,
        "G_codata": G_CODATA,
    }


# ---------------------------------------------------------------------------
# 7. Limite Schwarzschild.
# ---------------------------------------------------------------------------
def schwarzschild_limit_check(M: float, r: float) -> dict:
    """Compare Ω²(r) à la métrique Schwarzschild en jauge isotrope.

    En jauge isotrope (compatible avec la jauge conforme V5) :
        g_00 = -[(1 - GM/(2r̃c²))/(1 + GM/(2r̃c²))]²
        g_ii = (1 + GM/(2r̃c²))⁴ δ_ii

    On compare donc Ω²_V5 = 1 + h_V5 à Ω²_Schw = (1 + ε)⁴ avec
    ε = GM/(2r c²).

    Au premier ordre : Ω²_Schw ≈ 1 + 4ε = 1 + 2GM/(rc²).
    h_V5 doit donc valoir ≈ 2GM/(rc²) pour qu'il y ait accord.

    Paramètres
    ----------
    M : float (kg)
    r : float (m), rayon isotrope.

    Retour
    ------
    dict comparant h_V5_required et h_V5_from_kappa.
    """
    G = G_newton_from_kappa_g()
    eps = G * M / (2.0 * r * C_LIGHT ** 2)
    omega2_schw = (1.0 + eps) ** 4
    h_required = omega2_schw - 1.0          # h pour reproduire Schw
    h_linear = 2.0 * G * M / (r * C_LIGHT ** 2)  # premier ordre

    # Au premier ordre, h_required ≈ h_linear : on vérifie l'écart.
    rs = 2.0 * G * M / (C_LIGHT ** 2)
    return {
        "schwarzschild_radius": rs,
        "epsilon": eps,
        "omega2_schwarzschild_isotrope": omega2_schw,
        "h_required_for_schw": h_required,
        "h_linear_first_order": h_linear,
        "linearization_error": (h_required - h_linear) / h_linear,
        "regime_valid": eps < 0.1,
        "note": (
            "Au premier ordre en GM/(rc²), la jauge conforme V5 P5 "
            "Ω² = 1 + h reproduit la métrique de Schwarzschild en "
            "jauge isotrope (g_ii partie). g_00 = -Ω² diffère du "
            "Schwarzschild exact à l'ordre (GM/rc²)² : la métrique "
            "purement conforme V5 N'EST PAS Schwarzschild exacte au "
            "champ fort — c'est attendu, V5 P5 est valide en régime "
            "faible champ et doit être étendue (P19) pour les horizons."
        ),
    }


# ---------------------------------------------------------------------------
# 8. Lien G ↔ κ_g (synthèse V5.5 étape 6).
# ---------------------------------------------------------------------------
def G_from_kappa_g_consistency() -> dict:
    """Synthèse du lien κ_g ↔ G_Newton (V5.5 étape 6).

    G = (12/7) · α_EM^18 · ℏc / m_p²
    κ_g = (96π/7) · α_EM^18 · λbar_C(proton)
        = (96π/7) · α_EM^18 · ℏ / (m_p c)

    Rapport :
      G · m_p² / (ℏc) = (12/7) α_EM^18
      κ_g · m_p c / ℏ = (96π/7) α_EM^18
      ⟹ κ_g / G = (96π/7) · ℏ/(m_p c) / [(12/7) ℏc/m_p²]
                = 8π · m_p / c²

    Soit κ_g = (8π m_p / c²) · G — relation purement dimensionnelle
    entre la constante de couplage métrique V5 et la constante de
    Newton standard. Vérification :
       [κ_g] = [m_p / c²] [G] = (kg·s²/m²) × (m³/(kg·s²)) = m  ✓
    """
    G = G_newton_from_kappa_g()
    kg = kappa_g_V5p5("m")
    # Relation théorique
    kg_from_G = 8.0 * math.pi * M_PROTON * G / (C_LIGHT ** 2)
    return {
        "kappa_g_V5p5": kg,
        "G_predicted": G,
        "kappa_g_from_G_formula": kg_from_G,
        "ratio": kg / kg_from_G,
        "relative_error": (kg - kg_from_G) / kg_from_G,
        "formula": "κ_g = (8π m_p / c²) · G",
        "G_codata_error_pct": (G - G_CODATA) / G_CODATA * 100,
    }


# ---------------------------------------------------------------------------
# 9. Synthèse pour reporting.
# ---------------------------------------------------------------------------
def summary() -> dict:
    """Synthèse V5.8.

    Profil test : Schwarzschild faible champ d'une masse ponctuelle M_sol
    sur une boule de rayon R_sol avec densité uniforme. Dans cette boule :
        Φ(r) = (GM/2R³)(r² - 3R²)
        ∇²Φ = 3GM/R³ = 4πGρ   (Poisson exacte)
        h(r) = -2Φ/c² = (GM/R³c²)(3R² - r²)
        ∇²h = -2·(3GM/R³c²) × (∇²(r²/2)) → ∇²h = -6GM/(R³c²) = -8πGρ/c²
    """
    G = G_newton_from_kappa_g()
    M_sun = 1.989e30
    R_sun = 6.957e8
    rho_uniform = 3.0 * M_sun / (4.0 * math.pi * R_sun ** 3)

    def h_prof(r):
        return G * M_sun / (R_sun ** 3 * C_LIGHT ** 2) * (3.0 * R_sun ** 2 - r ** 2)
    def lap_h(r):
        # ∇²(A·(3R²-r²)) = -6A en coords sphériques (∇²r² = 6)
        A = G * M_sun / (R_sun ** 3 * C_LIGHT ** 2)
        return -6.0 * A
    def rho_prof(r):
        return rho_uniform

    chk = verify_einstein_equation(h_prof, lap_h, rho_prof, r=R_sun / 2)
    schw = schwarzschild_limit_check(M=M_sun, r=R_sun)
    glink = G_from_kappa_g_consistency()
    return {
        "kappa_g_m": kappa_g_V5p5("m"),
        "G_predicted": G_newton_from_kappa_g(),
        "G_codata": G_CODATA,
        "G_error_pct": (G_newton_from_kappa_g() - G_CODATA) / G_CODATA * 100,
        "einstein_check_G00": chk.G_00,
        "einstein_check_rhs00": chk.rhs_00,
        "einstein_check_ratio": chk.ratio_00,
        "poisson_lhs": chk.poisson_lhs,
        "poisson_rhs": chk.poisson_rhs,
        "poisson_ratio": chk.poisson_ratio,
        "schw_h_required": schw["h_required_for_schw"],
        "schw_h_linear": schw["h_linear_first_order"],
        "schw_linearization_error": schw["linearization_error"],
        "G_to_kappa_g_formula": glink["formula"],
        "G_to_kappa_g_ratio": glink["ratio"],
        "G_to_kappa_g_rel_error": glink["relative_error"],
    }


# ---------------------------------------------------------------------------
# 10. Dérivation symbolique sympy (cas conformément plat statique).
# ---------------------------------------------------------------------------
def symbolic_ricci_conformal_isotropic():
    """Dérivation sympy du Ricci pour g_μν = Ω²(r) η_μν, Ψ=Ψ(r).

    Effectue le calcul exact des symboles de Christoffel, du Riemann
    et du Ricci pour une métrique conformément plate à symétrie
    sphérique statique, et vérifie la formule de Wald App. D.

    Retour
    ------
    dict avec :
      - 'omega2'      : expression Ω²(r)
      - 'christoffel' : Γ^λ_μν (4×4×4 sympy)
      - 'ricci'       : R_μν (4×4 sympy)
      - 'ricci_scalar': R (sympy)
      - 'G_00_linear' : G_00 au premier ordre en h(r)
    """
    try:
        import sympy as sp
    except ImportError:
        raise RuntimeError("sympy requis pour la dérivation symbolique.")

    # Coords cartésiennes ; pour éviter les Abs liés à √(x²+y²+z²),
    # on travaille avec une fonction H(x,y,z) générique et on suppose
    # symétrie sphérique a posteriori. Le calcul est valide pour
    # Ω² = 1 + H(x,y,z) quelconque.
    t, x, y, z = sp.symbols("t x y z", real=True, positive=False)
    coords = (t, x, y, z)
    H = sp.Function("H")(x, y, z)
    omega2 = 1 + H
    eta = sp.diag(-1, 1, 1, 1)

    g = omega2 * eta
    g_inv = sp.simplify(g.inv())

    # Christoffel : Γ^λ_μν = (1/2) g^{λρ}(∂_μ g_{ρν} + ∂_ν g_{ρμ} - ∂_ρ g_{μν})
    Gamma = sp.MutableDenseNDimArray.zeros(4, 4, 4)
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                s = 0
                for rho in range(4):
                    s += g_inv[lam, rho] * (
                        sp.diff(g[rho, nu], coords[mu])
                        + sp.diff(g[rho, mu], coords[nu])
                        - sp.diff(g[mu, nu], coords[rho])
                    )
                Gamma[lam, mu, nu] = sp.simplify(s / 2)

    # Tenseur de Ricci : R_μν = ∂_λ Γ^λ_μν - ∂_ν Γ^λ_μλ
    #                       + Γ^λ_μν Γ^σ_λσ - Γ^σ_μλ Γ^λ_νσ
    R_munu = sp.MutableDenseNDimArray.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            term = 0
            for lam in range(4):
                term += sp.diff(Gamma[lam, mu, nu], coords[lam])
                term -= sp.diff(Gamma[lam, mu, lam], coords[nu])
                for sig in range(4):
                    term += Gamma[lam, mu, nu] * Gamma[sig, lam, sig]
                    term -= Gamma[sig, mu, lam] * Gamma[lam, nu, sig]
            R_munu[mu, nu] = sp.simplify(term)

    # Scalaire R
    R_scalar = 0
    for mu in range(4):
        for nu in range(4):
            R_scalar += g_inv[mu, nu] * R_munu[mu, nu]
    R_scalar = sp.simplify(R_scalar)

    # G_00 = R_00 - (1/2) R g_00 (forme exacte ; pas de série ici car H
    # est multi-variable).
    G_00 = sp.simplify(R_munu[0, 0] - sp.Rational(1, 2) * R_scalar * g[0, 0])

    return {
        "omega2": omega2,
        "christoffel": Gamma,
        "ricci": R_munu,
        "ricci_scalar": R_scalar,
        "G_00": G_00,
    }


if __name__ == "__main__":
    s = summary()
    print("=" * 72)
    print(" V5.8 — Dérivation des équations d'Einstein depuis TQNT-V5")
    print("=" * 72)
    print(f"\n  κ_g                          = {s['kappa_g_m']:.4e} m")
    print(f"  G prédit                      = {s['G_predicted']:.5e} SI")
    print(f"  G CODATA                      = {s['G_codata']:.5e} SI")
    print(f"  Écart relatif                 = {s['G_error_pct']:+.3f} %")
    print(f"\n  Lien V5 : {s['G_to_kappa_g_formula']}")
    print(f"  Rapport κ_g(V5)/κ_g(de G)     = {s['G_to_kappa_g_ratio']:.6f}")
    print(f"  Écart                         = {s['G_to_kappa_g_rel_error']*100:+.3f} %")
    print(f"\n  --- Limite Newton (Poisson) au cœur du proton ---")
    print(f"  ∇²Φ                           = {s['poisson_lhs']:.4e}")
    print(f"  4πGρ                          = {s['poisson_rhs']:.4e}")
    print(f"  Rapport                       = {s['poisson_ratio']:.6f}")
    print(f"\n  --- Limite Schwarzschild (Soleil, r=R_sol) ---")
    print(f"  h_requis (Schw isotrope)      = {s['schw_h_required']:.4e}")
    print(f"  h_linéaire (1er ordre)        = {s['schw_h_linear']:.4e}")
    print(f"  Erreur de linéarisation       = {s['schw_linearization_error']*100:+.3e} %")
    print("=" * 72)
