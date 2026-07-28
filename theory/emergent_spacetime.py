"""
Théorie Nodale — Espace-Temps Émergent

P5: Spacetime emerges from wave aberrations of nothingness.
The local intensity |Psi|² defines spacetime density.
Bi-conformal metric:
  ds² = -c²e^{-κ|Ψ|²}dt² + Ω²(dx² + dy² + dz²)   with Ω² = 1 + κ|Ψ|²
Temporal component e^{-κ|Ψ|²}: slows time near knots (gravity). β_PPN = γ_PPN = 1.
Spatial component Ω²: dilates space near knots (unchanged from v3).
"""
import numpy as np
from theory.postulates import KAPPA


def spacetime_density(psi: np.ndarray) -> np.ndarray:
    """Compute spacetime density rho = |Psi|²."""
    return np.abs(psi) ** 2


def emergent_metric_factor(psi: np.ndarray) -> np.ndarray:
    """Compute the conformal factor (1 + kappa * |Psi|²)."""
    return 1.0 + KAPPA * spacetime_density(psi)


def conformal_factor(psi: np.ndarray) -> np.ndarray:
    """Compute conformal factor Ω = sqrt(1 + κ|Ψ|²).

    The conformal metric is g_ij = Ω² δ_ij where Ω² = 1 + κ|Ψ|².
    """
    return np.sqrt(1.0 + KAPPA * spacetime_density(psi))


def temporal_dilation_factor(psi: np.ndarray) -> np.ndarray:
    """Compute temporal dilation factor e^{-κ|Ψ|²/2}.

    In the bi-conformal metric ds² = -c²e^{-κ|Ψ|²}dt² + Ω²(dx²+dy²+dz²),
    proper time is dτ = dt × e^{-κ|Ψ|²/2} < dt near knots (gravity).
    The exponential form gives PPN parameters β = γ = 1 (matches GR).
    """
    h = KAPPA * spacetime_density(psi)
    return np.exp(-h / 2.0)


def effective_light_speed(psi: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Compute effective coordinate light speed c_eff = c × e^{-h/2}/Ω.

    From ds² = 0: c²e^{-h}dt² = Ω²dℓ², so c_eff = c × e^{-h/2}/Ω
    where h = κ|Ψ|². Light slows near knots (Shapiro delay analogue).
    """
    h = KAPPA * spacetime_density(psi)
    omega = np.sqrt(1.0 + h)
    return c * np.exp(-h / 2.0) / omega


def gravitational_potential(r: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Compute effective gravitational potential from wave density."""
    rho = spacetime_density(psi)
    dr = r[1] - r[0] if len(r) > 1 else 1.0
    phi = np.zeros_like(r, dtype=float)
    for i in range(len(r)):
        dist = np.abs(r - r[i])
        dist[i] = dr
        phi[i] = -KAPPA * np.sum(rho * r**2 / dist) * dr
    return phi


# --- Boucle auto-cohérente énergie-temps (E=mc² + P4 + P5) ---


def gravitational_self_coupling(m, G=1.0, hbar=1.0, c=1.0):
    """Couplage gravitationnel adimensionné α_G = Gm²/(ℏc).

    Mesure l'intensité de l'auto-interaction gravitationnelle d'une particule.
    Pour le proton : α_G ≈ 5.91×10⁻³⁹.
    """
    return G * m**2 / (hbar * c)


def temporal_distortion_energy(energy, rho_J, alpha_G_cold, E_0=1.0):
    """Distorsion temporelle en fonction de l'énergie : Ω_t(E).

    Ω_t(E) = exp(-α_G · [ρ_J / (ρ_J + E/E₀)]²)

    Dérivée en combinant P5 (Ω_t = exp(-Gm²/(ℏc))) avec P4
    (m ∝ ρ(K,E) = ρ_J²/(ρ_J + E/E₀)) et E = mc².

    Propriétés :
    - E=0     → Ω_t = exp(-α_G)  (distorsion maximale, nucléon froid)
    - E=E₀ρ_J → Ω_t = exp(-α_G/4) (demi-rétractation)
    - E→∞     → Ω_t → 1  (pas de distorsion, dissolution)
    """
    energy = np.asarray(energy, dtype=float)
    ratio = rho_J / (rho_J + energy / E_0)
    return np.exp(-alpha_G_cold * ratio**2)


def proper_compton_time(energy, rho_J, m_cold, alpha_G_cold,
                        E_0=1.0, hbar=1.0, c=1.0):
    """Temps propre de Compton modifié par la distorsion temporelle.

    τ(E) = [ℏ(ρ_J + E/E₀) / (m₀·ρ_J·c²)] · Ω_t(E)

    Dérivé de τ_C = ℏ/(mc²) avec m = m₀·ρ_J/(ρ_J + E/E₀) et
    τ_propre = τ_C · Ω_t.

    Propriétés :
    - E=0  → τ = ℏΩ_t(0)/(m₀c²)
    - E→∞  → τ → ∞ (dissolution : m→0, Ω_t→1)
    """
    energy = np.asarray(energy, dtype=float)
    omega_t = temporal_distortion_energy(energy, rho_J, alpha_G_cold, E_0)
    return hbar * (rho_J + energy / E_0) / (m_cold * rho_J * c**2) * omega_t


def mass_from_energy(energy, rho_J, eta_F, E_0=1.0):
    """Masse topologique en fonction de l'énergie via P4.

    m(E) = η_F · ρ_J² / (ρ_J + E/E₀)

    C'est la masse émergente quand l'énergie E modifie la rétractation.
    - E=0     → m = η_F·ρ_J = m_cold
    - E=E₀ρ_J → m = m_cold/2
    - E→∞     → m → 0
    """
    energy = np.asarray(energy, dtype=float)
    return eta_F * rho_J**2 / (rho_J + energy / E_0)


def rest_energy(m, c=1.0):
    """Énergie au repos E = mc².

    Pivot reliant la masse topologique à l'énergie. Ferme la boucle
    auto-cohérente : E ↔ ρ(K,E) ↔ m ↔ Ω_t ↔ τ.
    """
    return m * c**2


def master_equation_roots(rho_J, eta_F, E_0=1.0, c=1.0):
    """Racine positive de l'équation maîtresse m²c²/E₀ + m·ρ_J = η_F·ρ_J².

    Dérivée en substituant E = mc² dans ρ(K,E) = ρ_J²/(ρ_J + E/E₀)
    puis dans m = η_F · ρ(K,E).

    Retourne la masse d'équilibre m_eq (racine physique positive).
    """
    # Coefficients de am² + bm + c = 0
    a = c**2 / E_0
    b = rho_J
    d = -eta_F * rho_J**2  # 'c' du trinôme (renommé d pour éviter shadowing)
    discriminant = b**2 - 4 * a * d
    return (-b + np.sqrt(discriminant)) / (2 * a)
