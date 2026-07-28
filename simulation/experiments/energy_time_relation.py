# simulation/experiments/energy_time_relation.py
"""
Expérience : Boucle auto-cohérente énergie-temps du nucléon

Relation nodale entre énergie, masse, distorsion temporelle et temps propre.
L'énergie E = mc² ferme la boucle : E → ρ(K,E) → m → Ω_t → τ.

Équation maîtresse : m²c²/E₀ + m·ρ_J = η_F·ρ_J²
Distorsion temporelle : Ω_t(E) = exp(-α_G · [ρ_J/(ρ_J + E/E₀)]²)
Relation E-τ nodale : E·τ = ℏ·Ω_t
"""
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt

from theory.emergent_spacetime import (
    gravitational_self_coupling,
    temporal_distortion_energy,
    proper_compton_time,
    mass_from_energy,
    rest_energy,
    master_equation_roots,
)

OUTPUT_DIR = "output/energy_time_relation"

# --- Physical constants (SI) ---
HBAR_SI = 1.0546e-34       # J·s
C_SI = 2.9979e8            # m/s
G_SI = 6.6743e-11          # m³/(kg·s²)
M_PROTON_KG = 1.6726e-27   # kg

# --- Nucleon parameters (natural units) ---
RHO_NUC = 25.7             # ρ_J du nucléon (9₁+7₄+6₃)
E_0 = 1.0                  # échelle d'énergie (unités naturelles)
ETA_F = 1.0                # facteur de calibration (unités naturelles)
M_COLD = ETA_F * RHO_NUC   # m₀ = η_F · ρ_J


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 76)
    print("  Boucle auto-cohérente énergie-temps du nucléon")
    print("=" * 76)

    # --- Couplage gravitationnel physique ---
    alpha_G_phys = gravitational_self_coupling(
        M_PROTON_KG, G=G_SI, hbar=HBAR_SI, c=C_SI)
    print(f"\n  α_G(proton) = Gm_p²/(ℏc) = {alpha_G_phys:.4e}")

    # --- Paramètres en unités naturelles ---
    alpha_G_cold = 0.1  # valeur illustrative pour les graphiques
    E_K = E_0 * RHO_NUC  # énergie caractéristique

    print(f"  ρ_J (nucléon)  = {RHO_NUC}")
    print(f"  E_K = E₀·ρ_J  = {E_K}")
    print(f"  m_cold = η_F·ρ_J = {M_COLD}")

    # --- Équation maîtresse ---
    m_eq = master_equation_roots(RHO_NUC, ETA_F, E_0)
    E_eq = rest_energy(m_eq)
    print(f"\n  Équation maîtresse : m²/E₀ + m·ρ_J = η_F·ρ_J²")
    print(f"  Racine : m_eq = {m_eq:.6f}")
    print(f"  E_eq = m_eq·c² = {E_eq:.6f}")

    # Vérification
    lhs = m_eq**2 / E_0 + m_eq * RHO_NUC
    rhs = ETA_F * RHO_NUC**2
    print(f"  Vérification : LHS = {lhs:.6f}, RHS = {rhs:.6f}, "
          f"Δ = {abs(lhs - rhs):.2e}")

    # --- Conditions aux limites ---
    print(f"\n  {'Limite':<20} {'E':<12} {'m/m₀':<12} {'Ω_t':<16} {'τ·m₀c²/ℏ':<12}")
    print("  " + "-" * 68)

    boundaries = [
        ("Nucléon froid", 0.0),
        ("Demi-rétractation", E_K),
        ("Haute énergie", 100 * E_K),
    ]
    for label, E in boundaries:
        m = mass_from_energy(E, RHO_NUC, ETA_F, E_0)
        omega = temporal_distortion_energy(E, RHO_NUC, alpha_G_cold, E_0)
        tau = proper_compton_time(E, RHO_NUC, M_COLD, alpha_G_cold, E_0)
        tau_norm = tau * M_COLD  # τ·m₀c²/ℏ (avec ℏ=c=1)
        print(f"  {label:<20} {E:<12.2f} {m/M_COLD:<12.6f} "
              f"{omega:<16.10f} {tau_norm:<12.6f}")

    # --- Graphiques ---
    _make_plots(RHO_NUC, ETA_F, M_COLD, alpha_G_cold, E_0)

    print(f"\n  Graphiques sauvegardés dans {OUTPUT_DIR}/")


def _make_plots(rho_J, eta_F, m_cold, alpha_G_cold, E_0):
    """Génère le panel 2×2 de la boucle énergie-temps."""

    E_K = E_0 * rho_J
    energies = np.linspace(0, 5 * E_K, 1000)
    x = energies / E_K  # E/E_K (adimensionné)

    # Calculs
    omega = temporal_distortion_energy(energies, rho_J, alpha_G_cold, E_0)
    masses = mass_from_energy(energies, rho_J, eta_F, E_0)
    tau = proper_compton_time(energies, rho_J, m_cold, alpha_G_cold, E_0)
    E_rest = rest_energy(masses)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Boucle auto-cohérente énergie-temps du nucléon",
                 fontsize=14, fontweight='bold')

    # Lignes verticales pour conditions aux limites
    bc_lines = [
        (0.0, "E=0", "green"),
        (1.0, r"E=E$_K$", "orange"),
    ]

    # --- Panel (0,0) : Ω_t(E) ---
    ax = axes[0, 0]
    ax.plot(x, omega, 'b-', linewidth=2)
    ax.set_xlabel(r"$E / E_K$")
    ax.set_ylabel(r"$\Omega_t(E)$")
    ax.set_title(r"Distorsion temporelle $\Omega_t(E)$")
    ax.axhline(y=np.exp(-alpha_G_cold), color='green', linestyle='--',
               alpha=0.5, label=r"$e^{-\alpha_G}$")
    ax.axhline(y=np.exp(-alpha_G_cold / 4), color='orange', linestyle='--',
               alpha=0.5, label=r"$e^{-\alpha_G/4}$")
    ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.3, label=r"$1$")
    for xv, label, color in bc_lines:
        ax.axvline(x=xv, color=color, linestyle=':', alpha=0.4)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel (0,1) : m(E)/m_cold ---
    ax = axes[0, 1]
    ax.plot(x, masses / m_cold, 'r-', linewidth=2)
    ax.set_xlabel(r"$E / E_K$")
    ax.set_ylabel(r"$m(E) / m_0$")
    ax.set_title(r"Masse relative $m(E)/m_0$")
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5,
               label=r"$m_0/2$")
    for xv, label, color in bc_lines:
        ax.axvline(x=xv, color=color, linestyle=':', alpha=0.4)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel (1,0) : τ(E)·m₀c²/ℏ ---
    ax = axes[1, 0]
    tau_norm = tau * m_cold  # τ·m₀c²/ℏ avec ℏ=c=1
    ax.plot(x, tau_norm, 'm-', linewidth=2)
    ax.set_xlabel(r"$E / E_K$")
    ax.set_ylabel(r"$\tau(E) \cdot m_0 c^2 / \hbar$")
    ax.set_title(r"Temps propre normalisé $\tau \cdot m_0 c^2 / \hbar$")
    for xv, label, color in bc_lines:
        ax.axvline(x=xv, color=color, linestyle=':', alpha=0.4)
    ax.grid(True, alpha=0.3)

    # --- Panel (1,1) : E·τ/ℏ ---
    ax = axes[1, 1]
    E_tau = E_rest * tau  # E·τ/ℏ avec ℏ=1
    ax.plot(x, E_tau, 'g-', linewidth=2, label=r"$E \cdot \tau / \hbar$")
    ax.plot(x, omega, 'b--', linewidth=1.5, alpha=0.7,
            label=r"$\Omega_t(E)$")
    ax.set_xlabel(r"$E / E_K$")
    ax.set_ylabel(r"$E \cdot \tau / \hbar$")
    ax.set_title(r"Relation nodale $E \cdot \tau = \hbar \, \Omega_t$")
    ax.legend(fontsize=8)
    for xv, label, color in bc_lines:
        ax.axvline(x=xv, color=color, linestyle=':', alpha=0.4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "energy_time_panel.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
