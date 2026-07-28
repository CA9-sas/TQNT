"""
Théorie Nodale — Entrelacs neutrino-photon : Hopf link minimal

Calcule et trace la statistique de formation d'un lien de Hopf entre
un neutrino (reliquat ondulatoire, R ≈ 173 nm) et un photon (λ ~ R).

Résultat principal : taux nul aujourd'hui (CMB trop froid), significatif
pour z > 1720 (T > 4700 K) et dans les photosphères stellaires.

Génère 3 figures dans output/ :
  1. Spectre de Planck avec ligne E_match
  2. Taux d'interaction vs température
  3. Section efficace vs rayon du neutrino
"""
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.neutrino import HBAR_C_MEV_FM, boomerang_radius
from theory.linking import (
    linking_cross_section,
    spectral_matching_energy,
    planck_spectral_density,
    interaction_rate_density,
    mean_free_path,
    total_events_hubble,
    critical_temperature,
    critical_redshift,
    linking_rate_per_neutrino,
    linking_onset_mass,
    linking_mean_free_path_neutrino,
    hopf_link_properties,
    T_CMB, T_HUBBLE_S, N_GAMMA_TOTAL, N_NU_PER_FLAVOR,
    K_B_EV_PER_K, C_CM_PER_S,
)


def main():
    os.makedirs("output", exist_ok=True)

    # ── Paramètres du neutrino β ──────────────────────────────────────
    m_nu_eV = 0.57  # eV (désintégration β du neutron)
    m_nu_MeV = m_nu_eV * 1e-6
    R_nu_fm = boomerang_radius(m_nu_MeV)
    R_nu_nm = R_nu_fm * 1e-6  # fm → nm

    print("=" * 72)
    print("  ENTRELACS NEUTRINO-PHOTON : HOPF LINK MINIMAL")
    print("=" * 72)

    # ── 1. Section efficace ───────────────────────────────────────────
    sigma_fm2 = linking_cross_section(R_nu_fm)
    sigma_cm2 = sigma_fm2 * 1e-26

    print(f"\n  R_ν = {R_nu_fm:.3e} fm = {R_nu_nm:.1f} nm")
    print(f"  σ_link = π R² = {sigma_fm2:.3e} fm² = {sigma_cm2:.3e} cm²")
    print(f"  (σ_faible ~ 10⁻³⁸ cm² → ratio = {sigma_cm2 / 1e-38:.1e})")

    # ── 2. Condition spectrale ────────────────────────────────────────
    E_match_MeV = spectral_matching_energy(R_nu_fm)
    E_match_eV = E_match_MeV * 1e6

    print(f"\n  E_match = ℏc/R = {E_match_eV:.3f} eV (infrarouge proche)")
    print(f"  λ_match = {R_nu_nm:.1f} nm")

    # ── 3. Densité de photons au CMB ──────────────────────────────────
    n_gamma_cmb = planck_spectral_density(E_match_eV, T_CMB)
    print(f"\n  n_γ(CMB, {E_match_eV:.2f} eV) = {n_gamma_cmb:.3e} cm⁻³ eV⁻¹")
    print(f"  (x = E/kT = {E_match_eV / (K_B_EV_PER_K * T_CMB):.0f} → suppression)")

    # ── 4. Taux dans différents environnements ────────────────────────
    print(f"\n  {'─' * 60}")
    print(f"  {'Environnement':<25s} {'T (K)':<12s} {'n_γ (cm⁻³)':<14s} "
          f"{'Γ/V (cm⁻³s⁻¹)':<16s}")
    print(f"  {'─' * 60}")

    environments = [
        ("CMB actuel", T_CMB),
        ("Surface stellaire", 5778.0),
        ("Étoile chaude", 30000.0),
        ("Intérieur stellaire", 1.5e7),
        ("Univers jeune (z~1720)", 4700.0),
        ("Big Bang (T~10⁹ K)", 1e9),
    ]

    for name, T in environments:
        n_gamma = planck_spectral_density(E_match_eV, T) * 1.0  # bande 1 eV
        rate = interaction_rate_density(n_gamma, N_NU_PER_FLAVOR, sigma_fm2)
        print(f"  {name:<25s} {T:<12.1f} {n_gamma:<14.3e} {rate:<16.3e}")

    # ── 5. Température critique ───────────────────────────────────────
    T_crit = critical_temperature(R_nu_fm)
    z_crit = critical_redshift(T_crit)

    print(f"\n  T_critique (Wien) = {T_crit:.0f} K")
    print(f"  z_critique = {z_crit:.0f} (avant recombinaison z=1100)")

    # ── 6. Propriétés du Hopf link ────────────────────────────────────
    props = hopf_link_properties(R_nu_fm)

    print(f"\n  Hopf link (2 × unknot 0₁) :")
    print(f"    Linking number = {props['linking_number']}")
    print(f"    Composantes = {props['components']}")
    print(f"    ρ_J(0₁) = {props['rho_J']}")
    print(f"    Masse = {props['mass_MeV']:.5f} MeV = {props['mass_MeV']*1e3:.2f} keV")
    print(f"    Stabilité : {props['reason']}")

    # ── 7. Libre parcours moyen ───────────────────────────────────────
    n_star = planck_spectral_density(E_match_eV, 5778.0) * 1.0
    if n_star > 0:
        l_mfp = mean_free_path(n_star, sigma_fm2)
        print(f"\n  Libre parcours moyen (photosphère solaire) :")
        print(f"    l = {l_mfp:.3e} cm = {l_mfp * 1e-5:.3e} km")

    # ── 8. Seuil de masse pour entrelac dans le CMB ──────────────────
    m_onset = linking_onset_mass(T_CMB)
    dm2_sol = 7.53e-5  # eV²
    m2_min = dm2_sol ** 0.5  # ≈ 8.7 meV

    print(f"\n{'=' * 72}")
    print("  SEUIL DE MASSE POUR ENTRELAC DANS LE CMB ACTUEL")
    print(f"{'=' * 72}")
    print(f"\n  m_onset = {m_onset*1e3:.2f} meV  (Γ_ν = 1/t_H)")
    print(f"  √(Δm²_sol) = {m2_min*1e3:.2f} meV  (masse min de ν₂)")
    print(f"  Coïncidence : m_onset / √(Δm²_sol) = {m_onset/m2_min:.3f}")

    print(f"\n  {'─' * 64}")
    print(f"  {'m_ν (meV)':<12s} {'R_ν':<14s} {'l_mfp':<18s} "
          f"{'Γ_ν (s⁻¹)':<14s} {'τ_link':<16s}")
    print(f"  {'─' * 64}")

    import math
    for m_eV in [0.57, 0.05, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4]:
        R_fm = boomerang_radius(m_eV * 1e-6)
        gamma = linking_rate_per_neutrino(m_eV, T_CMB)
        l = linking_mean_free_path_neutrino(m_eV, T_CMB)

        # Format R
        if R_fm > 1e12:
            R_str = f"{R_fm*1e-12:.2f} mm"
        elif R_fm > 1e9:
            R_str = f"{R_fm*1e-9:.1f} μm"
        elif R_fm > 1e6:
            R_str = f"{R_fm*1e-6:.1f} nm"
        else:
            R_str = f"{R_fm:.0f} fm"

        # Format l
        if l == float('inf'):
            l_str = "∞"
        elif l > 3.086e24:
            l_str = f"{l/3.086e24:.1e} Mpc"
        elif l > 3.086e18:
            l_str = f"{l/3.086e18:.1e} pc"
        elif l > 1e5:
            l_str = f"{l*1e-5:.1e} km"
        else:
            l_str = f"{l:.2e} cm"

        # Format tau
        if gamma > 0:
            tau = 1.0 / gamma
            if tau > 3.156e16:
                tau_str = f"{tau/3.156e7:.1e} yr"
            elif tau > 3.156e7:
                tau_str = f"{tau/3.156e7:.1f} yr"
            elif tau > 1:
                tau_str = f"{tau:.1f} s"
            else:
                tau_str = f"{tau:.2e} s"
        else:
            tau_str = "∞"

        print(f"  {m_eV*1e3:<12.2f} {R_str:<14s} {l_str:<18s} "
              f"{gamma:<14.3e} {tau_str:<16s}")

    print(f"\n  Hiérarchie normale  : m₁ peut être ~ 0 → entrelac possible")
    print(f"  Hiérarchie inversée : m₃ ≥ 49 meV >> m_onset → pas d'entrelac")

    # ══════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════

    # ── Figure 1 : Spectre de Planck + ligne E_match ──────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for T, label, color in [(T_CMB, "CMB (2.73 K)", "blue"),
                            (5778, "Soleil (5778 K)", "orange"),
                            (4700, "T_crit (4700 K)", "red")]:
        E_range = np.linspace(1e-4, 5.0 if T > 100 else 0.01, 2000)
        dn_dE = np.array([planck_spectral_density(E, T) for E in E_range])
        ax1.plot(E_range, dn_dE, label=label, color=color, linewidth=2)

    ax1.axvline(E_match_eV, color="green", linestyle="--", linewidth=2,
                label=f"E_match = {E_match_eV:.2f} eV")
    ax1.set_xlabel("Énergie (eV)", fontsize=13)
    ax1.set_ylabel("dn/dE (cm⁻³ eV⁻¹)", fontsize=13)
    ax1.set_title("Spectre de Planck et condition spectrale\n"
                   "pour l'entrelacs neutrino-photon", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=1e-5)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig("output/linking_planck_spectrum.png", dpi=150)
    print("\n  → output/linking_planck_spectrum.png")
    plt.close(fig1)

    # ── Figure 2 : Taux vs température ────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    T_range = np.logspace(0, 9, 500)  # 1 K → 10⁹ K
    rates = []
    for T in T_range:
        n_gamma = planck_spectral_density(E_match_eV, T) * 1.0
        rate = interaction_rate_density(n_gamma, N_NU_PER_FLAVOR, sigma_fm2)
        rates.append(rate)
    rates = np.array(rates)

    # Masquer les zéros pour le log plot
    mask = rates > 0
    if np.any(mask):
        ax2.plot(T_range[mask], rates[mask], "b-", linewidth=2)

    ax2.axvline(T_crit, color="red", linestyle="--", linewidth=2,
                label=f"T_crit = {T_crit:.0f} K")
    ax2.axvline(T_CMB, color="blue", linestyle=":", linewidth=2,
                label=f"T_CMB = {T_CMB:.2f} K")
    ax2.axvline(5778, color="orange", linestyle="-.", linewidth=2,
                label="T_Soleil = 5778 K")

    ax2.set_xlabel("Température (K)", fontsize=13)
    ax2.set_ylabel("Γ/V (cm⁻³ s⁻¹)", fontsize=13)
    ax2.set_title("Taux d'interaction neutrino-photon\n"
                   "(formation de Hopf link)", fontsize=14)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("output/linking_rate_vs_temperature.png", dpi=150)
    print("  → output/linking_rate_vs_temperature.png")
    plt.close(fig2)

    # ── Figure 3 : σ_link vs R_ν pour différentes masses ─────────────
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

    m_range_eV = np.logspace(-2, 2, 200)  # 0.01 eV → 100 eV
    R_range_fm = np.array([boomerang_radius(m * 1e-6) for m in m_range_eV])
    sigma_range = np.array([linking_cross_section(R) * 1e-26 for R in R_range_fm])

    ax3a.loglog(m_range_eV, sigma_range, "b-", linewidth=2)
    ax3a.axhline(1e-38, color="gray", linestyle="--", linewidth=1,
                 label="σ_faible ~ 10⁻³⁸ cm²")
    ax3a.axvline(m_nu_eV, color="red", linestyle="--", linewidth=2,
                 label=f"m_ν(β) = {m_nu_eV} eV")
    ax3a.set_xlabel("Masse du neutrino (eV)", fontsize=13)
    ax3a.set_ylabel("σ_link (cm²)", fontsize=13)
    ax3a.set_title("Section efficace géométrique", fontsize=14)
    ax3a.legend(fontsize=10)
    ax3a.grid(True, alpha=0.3)

    # T_crit vs masse
    T_crit_range = np.array([critical_temperature(R) for R in R_range_fm])
    ax3b.loglog(m_range_eV, T_crit_range, "r-", linewidth=2)
    ax3b.axhline(T_CMB, color="blue", linestyle=":", linewidth=2,
                 label=f"T_CMB = {T_CMB:.2f} K")
    ax3b.axhline(5778, color="orange", linestyle="-.", linewidth=2,
                 label="T_Soleil")
    ax3b.axvline(m_nu_eV, color="red", linestyle="--", linewidth=2,
                 label=f"m_ν(β) = {m_nu_eV} eV")
    ax3b.set_xlabel("Masse du neutrino (eV)", fontsize=13)
    ax3b.set_ylabel("T_critique (K)", fontsize=13)
    ax3b.set_title("Température critique de Wien", fontsize=14)
    ax3b.legend(fontsize=10)
    ax3b.grid(True, alpha=0.3)

    fig3.suptitle("Propriétés de l'entrelacs en fonction de m_ν",
                  fontsize=15, y=1.02)
    fig3.tight_layout()
    fig3.savefig("output/linking_sigma_vs_mass.png", dpi=150,
                 bbox_inches="tight")
    print("  → output/linking_sigma_vs_mass.png")
    plt.close(fig3)

    # ── Figure 4 : Taux par neutrino vs masse (CMB) ────────────────────
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

    m_scan = np.logspace(-5, -0.5, 300)  # 10 μeV → 0.3 eV
    rates_cmb = np.array([linking_rate_per_neutrino(m, T_CMB) for m in m_scan])
    mfp_cmb = np.array([linking_mean_free_path_neutrino(m, T_CMB) for m in m_scan])

    # Taux par neutrino
    mask = rates_cmb > 0
    if np.any(mask):
        ax4a.loglog(m_scan[mask] * 1e3, rates_cmb[mask], "b-", linewidth=2)
    ax4a.axhline(1.0 / T_HUBBLE_S, color="red", linestyle="--", linewidth=2,
                 label=f"1/t_H = {1/T_HUBBLE_S:.1e} s⁻¹")
    ax4a.axvline(m_onset * 1e3, color="green", linestyle="--", linewidth=2,
                 label=f"m_onset = {m_onset*1e3:.1f} meV")
    ax4a.axvline(m2_min * 1e3, color="purple", linestyle=":", linewidth=2,
                 label=f"√Δm²_sol = {m2_min*1e3:.1f} meV")
    ax4a.set_xlabel("Masse du neutrino (meV)", fontsize=13)
    ax4a.set_ylabel("Γ_ν (s⁻¹)", fontsize=13)
    ax4a.set_title("Taux de liaison par neutrino\ndans le CMB actuel",
                    fontsize=14)
    ax4a.legend(fontsize=10)
    ax4a.grid(True, alpha=0.3)

    # Libre parcours moyen
    mask_mfp = np.isfinite(mfp_cmb) & (mfp_cmb > 0)
    if np.any(mask_mfp):
        ax4b.loglog(m_scan[mask_mfp] * 1e3, mfp_cmb[mask_mfp] / 3.086e24,
                    "b-", linewidth=2)
    R_H_Mpc = 4.4e28 / 3.086e24
    ax4b.axhline(R_H_Mpc, color="red", linestyle="--", linewidth=2,
                 label="R_Hubble")
    ax4b.axvline(m_onset * 1e3, color="green", linestyle="--", linewidth=2,
                 label=f"m_onset = {m_onset*1e3:.1f} meV")
    ax4b.axvline(m2_min * 1e3, color="purple", linestyle=":", linewidth=2,
                 label=f"√Δm²_sol = {m2_min*1e3:.1f} meV")
    ax4b.set_xlabel("Masse du neutrino (meV)", fontsize=13)
    ax4b.set_ylabel("Libre parcours moyen (Mpc)", fontsize=13)
    ax4b.set_title("Libre parcours moyen\navant formation de Hopf link",
                    fontsize=14)
    ax4b.set_ylim(1e-15, 1e20)
    ax4b.legend(fontsize=10)
    ax4b.grid(True, alpha=0.3)

    fig4.suptitle("Seuil de masse pour entrelac neutrino-photon dans le CMB",
                  fontsize=15, y=1.02)
    fig4.tight_layout()
    fig4.savefig("output/linking_onset_mass.png", dpi=150,
                 bbox_inches="tight")
    print("\n  → output/linking_onset_mass.png")
    plt.close(fig4)

    # ── Résumé final ──────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  CONCLUSION")
    print(f"{'=' * 72}")
    print(f"  • Neutrino β (m = 0.57 eV, R = 173 nm) :")
    print(f"    → AUCUN entrelac (CMB trop froid, exp(−4850) suppression)")
    print(f"  • Seuil de masse : m_onset = {m_onset*1e3:.1f} meV")
    print(f"    → Coïncide avec √(Δm²_sol) = {m2_min*1e3:.1f} meV")
    print(f"  • Hiérarchie normale (m₁ ~ 0) :")
    print(f"    → ν₁ peut former des Hopf links dans le CMB actuel !")
    print(f"  • Hiérarchie inversée (m₃ ≥ 49 meV) :")
    print(f"    → Aucun entrelac possible dans le CMB")
    print(f"  • Univers jeune (z > {z_crit:.0f}, T > {T_crit:.0f} K) :")
    print(f"    → Formation significative pour TOUTES les masses")
    print()


if __name__ == "__main__":
    main()
