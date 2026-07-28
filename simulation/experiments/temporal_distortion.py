"""
Théorie Nodale — Distorsion temporelle macroscopique

Calcule la dilatation temporelle à la surface du Soleil et de la Terre
en intégrant les contributions de chaque nucléon sur le volume du corps.

Métrique bi-conforme (P5) :
  ds² = -c²e^{-κ|Ψ|²}dt² + Ω²(dx²+dy²+dz²),  Ω² = 1 + κ|Ψ|²

La perturbation métrique h(r) = κ|Ψ|² satisfait l'équation de Poisson
  ∇²h = -κ ρ_Ψ
dont la solution pour N nucléons uniformément distribués dans une sphère
de rayon R se réduit (théorème de Newton) à h(R) = κN/(4πR).

La dilatation temporelle à la surface est :
  δτ/τ = e^{-h/2} - 1 ≈ -h/2 = -GM/(Rc²)

La forme exponentielle donne β_PPN = γ_PPN = 1 (identique à la RG),
contrairement à la forme 1/Ω² qui donnerait β = 2 (falsifié).
"""
import numpy as np

# --- Physical constants (SI) ---
G = 6.67430e-11      # m³ kg⁻¹ s⁻²
c = 2.99792458e8      # m/s
hbar = 1.054571817e-34  # J·s
m_p = 1.67262192e-27  # kg (proton mass)
m_e = 9.1093837e-31   # kg (electron mass)

# Fundamental coupling length (P5/P19)
kappa_phys = 8 * np.pi * G * m_p / c**2  # m

# --- Celestial bodies ---
bodies = {
    "Soleil": {
        "M": 1.989e30,       # kg
        "R": 6.957e8,        # m
    },
    "Terre": {
        "M": 5.972e24,       # kg
        "R": 6.371e6,        # m
    },
}


def poisson_h_surface(N, R):
    """Metric perturbation h at surface of a uniform sphere of N nucleons.

    Uses Newton's shell theorem: the integral of the Green's function
    κ/(4π|r-r'|) over a uniform sphere gives κN/(4πR) at the surface.
    """
    return kappa_phys * N / (4 * np.pi * R)


def poisson_h_numerical(N, R, n_shells=1000):
    """Numerical radial integration of h(R) for a uniform density sphere.

    h(R) = κ ∫₀ᴿ ρ_n(r') G_Poisson(R, r') 4πr'² dr'

    where G_Poisson(R, r') is the spherically averaged Green's function
    of the Laplacian at r=R for a source at r':
      G(R, r') = 1/(4πR)   for r' < R   (shell theorem)
    """
    r_prime = np.linspace(0, R, n_shells + 1)
    dr = r_prime[1] - r_prime[0]
    r_mid = 0.5 * (r_prime[:-1] + r_prime[1:])

    # Uniform number density
    V = (4 / 3) * np.pi * R**3
    rho_n = N / V  # nucleons per m³

    # Integrand: κ × ρ_n × (1/(4πR)) × 4πr'² dr'
    # = κ × ρ_n × r'² / R × dr'
    integrand = kappa_phys * rho_n * r_mid**2 / R * dr
    h = np.sum(integrand)
    return h


def temporal_dilation(h):
    """Compute δτ/τ = e^{-h/2} - 1 from metric perturbation h = κ|Ψ|²."""
    return np.exp(-h / 2.0) - 1.0


def gr_prediction(M, R):
    """GR weak-field prediction: δτ/τ = -GM/(Rc²)."""
    return -G * M / (R * c**2)


def main():
    print("=" * 78)
    print("Distorsion temporelle macroscopique — Métrique bi-conforme (P5)")
    print("=" * 78)
    print()
    print(f"Constantes fondamentales :")
    print(f"  G     = {G:.5e} m³ kg⁻¹ s⁻²")
    print(f"  c     = {c:.8e} m/s")
    print(f"  ℏ     = {hbar:.6e} J·s")
    print(f"  m_p   = {m_p:.6e} kg")
    print(f"  κ     = {kappa_phys:.4e} m  (= 8πGm_p/c²)")
    print()

    # Single-particle gravitational couplings
    print("-" * 78)
    print("Couplages gravitationnels individuels α_G = Gm²/(ℏc)")
    print("-" * 78)
    for name, m in [("Électron", m_e), ("Proton", m_p)]:
        alpha_G = G * m**2 / (hbar * c)
        print(f"  {name:10s}: α_G = {alpha_G:.3e}")
    print()

    # Macroscopic bodies
    print("-" * 78)
    print(f"{'Corps':>10s} | {'N nucléons':>14s} | {'h(R) analytique':>16s} | "
          f"{'h(R) numérique':>16s} | {'δτ/τ (nodal)':>14s} | {'δτ/τ (RG)':>14s} | {'erreur rel.':>12s}")
    print("-" * 78)

    for name, params in bodies.items():
        M = params["M"]
        R = params["R"]
        N = M / m_p  # number of nucleons

        h_analytic = poisson_h_surface(N, R)
        h_numeric = poisson_h_numerical(N, R, n_shells=10000)

        dt_nodal = temporal_dilation(h_analytic)
        dt_gr = gr_prediction(M, R)

        err_rel = abs((dt_nodal - dt_gr) / dt_gr)

        print(f"{name:>10s} | {N:14.4e} | {h_analytic:16.6e} | "
              f"{h_numeric:16.6e} | {dt_nodal:14.6e} | {dt_gr:14.6e} | {err_rel:12.2e}")

    print()

    # Detailed derivation for the Sun
    print("=" * 78)
    print("Dérivation détaillée — Soleil")
    print("=" * 78)
    M_sun = bodies["Soleil"]["M"]
    R_sun = bodies["Soleil"]["R"]
    N_sun = M_sun / m_p

    print(f"  M☉ = {M_sun:.3e} kg")
    print(f"  R☉ = {R_sun:.3e} m")
    print(f"  N  = M☉/m_p = {N_sun:.4e} nucléons")
    print()

    h_sun = poisson_h_surface(N_sun, R_sun)
    print(f"  h(R☉) = κN/(4πR☉)")
    print(f"        = {kappa_phys:.4e} × {N_sun:.4e} / (4π × {R_sun:.3e})")
    print(f"        = {h_sun:.6e}")
    print()
    print(f"  Vérification : 2GM/(R☉c²) = {2 * G * M_sun / (R_sun * c**2):.6e}")
    print(f"  → h = 2GM/(Rc²) ✓  (identité P19)")
    print()

    dt_sun = temporal_dilation(h_sun)
    dt_gr_sun = gr_prediction(M_sun, R_sun)
    print(f"  δτ/τ (nodal) = e^(-h/2) - 1  = {dt_sun:.6e}")
    print(f"  δτ/τ (RG)    = -GM/(Rc²)     = {dt_gr_sun:.6e}")
    print(f"  Écart relatif : {abs((dt_sun - dt_gr_sun) / dt_gr_sun):.2e}")
    print()

    # Express in seconds per year
    seconds_per_year = 365.25 * 24 * 3600
    dt_sec_year = dt_sun * seconds_per_year
    print(f"  En pratique : une horloge à la surface du Soleil retarde de")
    print(f"  {abs(dt_sec_year):.2f} secondes par an par rapport à l'infini.")
    print()

    # Detailed derivation for the Earth
    print("=" * 78)
    print("Dérivation détaillée — Terre")
    print("=" * 78)
    M_earth = bodies["Terre"]["M"]
    R_earth = bodies["Terre"]["R"]
    N_earth = M_earth / m_p

    print(f"  M⊕ = {M_earth:.3e} kg")
    print(f"  R⊕ = {R_earth:.3e} m")
    print(f"  N  = M⊕/m_p = {N_earth:.4e} nucléons")
    print()

    h_earth = poisson_h_surface(N_earth, R_earth)
    dt_earth = temporal_dilation(h_earth)
    dt_gr_earth = gr_prediction(M_earth, R_earth)

    print(f"  h(R⊕) = κN/(4πR⊕) = {h_earth:.6e}")
    print(f"  δτ/τ (nodal) = {dt_earth:.6e}")
    print(f"  δτ/τ (RG)    = {dt_gr_earth:.6e}")
    print()

    dt_microsec_day = dt_earth * 24 * 3600 * 1e6
    print(f"  En pratique : une horloge à la surface de la Terre retarde de")
    print(f"  {abs(dt_microsec_day):.2f} μs par jour par rapport à l'infini.")
    print(f"  (C'est la correction gravitationnelle du GPS : ~45 μs/jour.)")
    print()

    # Ratio Sun/Earth
    print("=" * 78)
    print("Rapport Soleil/Terre")
    print("=" * 78)
    ratio = h_sun / h_earth
    ratio_gr = (M_sun / R_sun) / (M_earth / R_earth)
    print(f"  h(R☉)/h(R⊕) = {ratio:.1f}")
    print(f"  (M☉/R☉)/(M⊕/R⊕) = {ratio_gr:.1f}")
    print(f"  → Le Soleil distord le temps {ratio:.0f}× plus que la Terre.")


if __name__ == "__main__":
    main()
