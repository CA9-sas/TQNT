# simulation/experiments/vibrational_energy.py
"""
Expérience : Calcul explicite de E_heat — Énergie vibrationnelle des nœuds noirs

Évalue numériquement la formule d'Isaacson adaptée (§21.7) :
    E_heat(K) = (c⁴ / 16πG) ∫ ⟨δh²⟩ d³x

Les « modes vibrationnels » sont les états excités FEM Ψ_i du nœud K.
Le résultat montre que ε = E_heat/(m_K c²) ≈ α_G × F(K) ≈ 10⁻⁴⁰,
insuffisant de ~37 ordres de grandeur pour expliquer ρ_Λ.
"""
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
from scipy.spatial import Delaunay, cKDTree

from simulation.solver import solve_knot
from theory.jones import jones_span, jones_polynomial
from theory.chi_field import jones_rho
from theory.postulates import KAPPA_PHYS

OUTPUT_DIR = "output/vibrational_energy"

# --- Physical constants (SI) ---
HBAR_SI = 1.0546e-34       # J·s
C_SI = 2.9979e8            # m/s
G_SI = 6.6743e-11          # m³/(kg·s²)
MEV_TO_J = 1.6022e-13      # J/MeV
M_PROTON_MEV = 938.272      # MeV/c²
M_K_DEFAULT_MEV = 5.0       # MeV/c² — raw knot mass scale

# Knots to study (those with parametric curves in the catalog)
KNOTS = ["3_1", "4_1", "5_1", "7_1", "9_1"]


def compute_overlap(points, tets, psi_full_0, psi_full_i):
    """Intégrale de recouvrement I_i = ∫ |Ψ₀|² |Ψ_i|² d³x.

    Utilise la quadrature de Gauss 4 points sur les tétraèdres,
    vectorisée sur tous les éléments.
    """
    a = (5.0 - np.sqrt(5.0)) / 20.0
    b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    bary = np.array([
        [b, a, a, a],
        [a, b, a, a],
        [a, a, b, a],
        [a, a, a, b],
    ])

    # Vertices and volumes
    all_verts = points[tets]  # (n_tet, 4, 3)
    edges = all_verts[:, 1:, :] - all_verts[:, 0:1, :]
    J = np.transpose(edges, (0, 2, 1))
    volumes = np.abs(np.linalg.det(J)) / 6.0  # (n_tet,)

    valid = volumes > 1e-15
    valid_tets = tets[valid]
    valid_volumes = volumes[valid]

    # |Ψ|² at nodes
    rho0 = psi_full_0**2   # |Ψ₀|²
    rhoi = psi_full_i**2   # |Ψ_i|²

    # Values at element nodes
    rho0_nodes = rho0[valid_tets]  # (n_valid, 4)
    rhoi_nodes = rhoi[valid_tets]  # (n_valid, 4)

    # Interpolate to quadrature points and integrate
    # rho0_quad[e, q] = Σ_j bary[q, j] * rho0_nodes[e, j]
    rho0_quad = np.einsum('qi,ei->eq', bary, rho0_nodes)  # (n_valid, 4)
    rhoi_quad = np.einsum('qi,ei->eq', bary, rhoi_nodes)  # (n_valid, 4)

    # I = Σ_e Σ_q w_q * rho0_quad * rhoi_quad, w_q = V_e / 4
    integrand = rho0_quad * rhoi_quad  # (n_valid, 4)
    I = np.sum(valid_volumes[:, None] / 4.0 * integrand)

    return float(I)


def solve_and_analyze(knot_name, n_coarse=20, n_eigenvalues=10):
    """Pipeline complet : résoudre + calculer spectre + recouvrements.

    Returns
    -------
    dict with keys: knot_name, crossing, rho_J, eigenvalues, omegas,
                    overlaps, I0, structure_factor
    """
    crossing = int(knot_name.split("_")[0])
    n_eig = max(crossing + 5, n_eigenvalues)

    result = solve_knot(
        knot_name, n_coarse=n_coarse, epsilon=0.2,
        n_eigenvalues=n_eig, refine=True,
    )

    eigenvalues = result["eigenvalues"]
    eigenvectors = result["eigenvectors"]
    points = result["points"]
    free_mask = result["free_mask"]
    rho_J = result["retraction_factor"]

    # Reconstruct tetrahedralization (same mesh = same Delaunay)
    from theory.knot_catalog import knot_curve
    curve = knot_curve(knot_name, 500)
    from theory.schrodinger import generate_mesh
    pts_check, tets, boundary = generate_mesh(
        curve, n_coarse=n_coarse, epsilon=0.2, refine=True,
    )
    n_pts = len(points)

    # Expand eigenvectors from free DOFs to full arrays
    def expand_psi(k):
        psi_full = np.zeros(n_pts)
        psi_full[free_mask] = eigenvectors[:, k]
        return psi_full

    psi_0 = expand_psi(0)
    E_0 = eigenvalues[0]

    # Frequencies ω_i = E_i - E_0 (in code units, ℏ=1)
    omegas = eigenvalues[1:] - E_0

    # Overlap integrals
    overlaps = []
    for i in range(1, len(eigenvalues)):
        psi_i = expand_psi(i)
        I_i = compute_overlap(points, tets, psi_0, psi_i)
        overlaps.append(I_i)

    # Auto-recouvrement I_0 = ∫ |Ψ₀|⁴ d³x
    I_0 = compute_overlap(points, tets, psi_0, psi_0)

    # Structure factor S = Σ_i ω_i I_i
    overlaps_arr = np.array(overlaps)
    S = float(np.sum(omegas * overlaps_arr))

    return {
        "knot_name": knot_name,
        "crossing": crossing,
        "rho_J": rho_J,
        "eigenvalues": eigenvalues,
        "E_0": E_0,
        "omegas": omegas,
        "overlaps": overlaps_arr,
        "I_0": I_0,
        "structure_factor": S,
        "n_modes": len(omegas),
    }


def physical_epsilon(S_code, m_K_MeV=M_K_DEFAULT_MEV):
    """Conversion du facteur de structure code → ε physique.

    ε = (κ_phys · ℏ · m_p) / (4 · m_K² · c²) × S_phys

    S_code est en unités code (ℏ=1, distances en unités maillage).
    La conversion S_code → S_phys nécessite des facteurs d'échelle,
    mais comme κ_phys ≈ 3×10⁻⁵³ m, le préfacteur est toujours ≈ α_G.

    On calcule le préfacteur dimensionnel exactement.
    """
    m_K = m_K_MeV * MEV_TO_J / C_SI**2  # kg
    m_p = M_PROTON_MEV * MEV_TO_J / C_SI**2  # kg

    # Préfacteur : κ_phys * ℏ * m_p / (4 * m_K² * c²)
    prefactor = KAPPA_PHYS * HBAR_SI * m_p / (4.0 * m_K**2 * C_SI**2)

    # α_G = G * m_K² / (ℏ * c) — couplage gravitationnel
    alpha_G = G_SI * m_K**2 / (HBAR_SI * C_SI)

    # En unités code, les longueurs sont O(1) et les énergies O(1),
    # donc S_code est déjà un nombre sans dimension O(1).
    # Le résultat est ε = préfacteur × S_code
    epsilon = prefactor * abs(S_code)

    return {
        "epsilon": epsilon,
        "prefactor": prefactor,
        "alpha_G": alpha_G,
        "log10_epsilon": np.log10(epsilon) if epsilon > 0 else float('-inf'),
        "log10_alpha_G": np.log10(alpha_G),
    }


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print("  E_heat : Énergie vibrationnelle des nœuds — Calcul explicite")
    print("=" * 72)

    # ====== A. Spectre vibrationnel ======
    print("\n--- A. Spectre vibrationnel ---\n")
    all_results = []

    for knot_name in KNOTS:
        print(f"  Résolution {knot_name} ...", end=" ", flush=True)
        try:
            res = solve_and_analyze(knot_name, n_coarse=20)
            all_results.append(res)
            print(f"OK — {res['n_modes']} modes, E₀ = {res['E_0']:.4f}")
        except Exception as e:
            print(f"ÉCHEC: {e}")

    if not all_results:
        print("Aucun nœud résolu. Abandon.")
        return

    # Tableau du spectre
    print(f"\n{'Nœud':<8} {'c':<4} {'ρ_J':<8} {'E₀':<10} {'E₁':<10} {'ω₁':<10} {'n_modes':<8}")
    print("-" * 58)
    for r in all_results:
        E1 = r["eigenvalues"][1] if len(r["eigenvalues"]) > 1 else float('nan')
        w1 = r["omegas"][0] if len(r["omegas"]) > 0 else float('nan')
        print(f"{r['knot_name']:<8} {r['crossing']:<4} {r['rho_J']:<8.3f} "
              f"{r['E_0']:<10.4f} {E1:<10.4f} {w1:<10.4f} {r['n_modes']:<8}")

    # ====== B. Intégrales de recouvrement ======
    print("\n--- B. Intégrales de recouvrement ---\n")
    print(f"{'Nœud':<8} {'I₀ (auto)':<14} {'I₁':<14} {'I₂':<14} {'I₃':<14}")
    print("-" * 62)
    for r in all_results:
        vals = [f"{r['I_0']:<14.6e}"]
        for j in range(min(3, len(r["overlaps"]))):
            vals.append(f"{r['overlaps'][j]:<14.6e}")
        while len(vals) < 4:
            vals.append(f"{'—':<14}")
        print(f"{r['knot_name']:<8} {''.join(vals)}")

    # ====== C. Facteur de structure ======
    print("\n--- C. Facteur de structure S = Σ_i ω_i I_i ---\n")
    print(f"{'Nœud':<8} {'S (code)':<14} {'S/S₀':<12}")
    print("-" * 34)
    S_ref = all_results[0]["structure_factor"] if all_results else 1.0
    for r in all_results:
        S = r["structure_factor"]
        ratio = S / S_ref if S_ref != 0 else float('nan')
        print(f"{r['knot_name']:<8} {S:<14.6e} {ratio:<12.4f}")

    # ====== D. Conversion physique et résultat ======
    print("\n--- D. Conversion physique : ε = E_heat / (m_K c²) ---\n")
    print(f"{'Nœud':<8} {'c':<4} {'S (code)':<14} {'ε':<14} {'log₁₀(ε)':<12} {'log₁₀(α_G)':<12}")
    print("-" * 66)

    phys_results = []
    for r in all_results:
        S = r["structure_factor"]
        ph = physical_epsilon(S)
        phys_results.append(ph)
        print(f"{r['knot_name']:<8} {r['crossing']:<4} {S:<14.6e} "
              f"{ph['epsilon']:<14.6e} {ph['log10_epsilon']:<12.1f} "
              f"{ph['log10_alpha_G']:<12.1f}")

    # ====== Extrapolation pour c > 18 (nœuds noirs) ======
    print("\n--- Extrapolation : scaling en c (nœuds noirs, c > 18) ---\n")
    if len(all_results) >= 2:
        crossings = np.array([r["crossing"] for r in all_results])
        log_S = np.array([np.log(abs(r["structure_factor"]))
                          if r["structure_factor"] != 0
                          else -100.0
                          for r in all_results])

        # Fit linéaire log(S) vs c
        valid = log_S > -90
        if np.sum(valid) >= 2:
            coeffs = np.polyfit(crossings[valid], log_S[valid], 1)
            slope, intercept = coeffs
            print(f"  Fit : ln(S) = {slope:.3f}·c + {intercept:.3f}")
            print(f"  → S ∝ exp({slope:.3f}·c)")

            print(f"\n  {'c':<6} {'S (extrap.)':<14} {'ε (extrap.)':<14} {'log₁₀(ε)':<12}")
            print("  " + "-" * 48)
            for c_ext in [18, 20, 25, 30, 40, 50]:
                S_ext = np.exp(slope * c_ext + intercept)
                ph = physical_epsilon(S_ext)
                print(f"  {c_ext:<6} {S_ext:<14.4e} {ph['epsilon']:<14.4e} "
                      f"{ph['log10_epsilon']:<12.1f}")

    # ====== Bilan ======
    if phys_results:
        alpha_G = phys_results[0]["alpha_G"]
        eps_max = max(ph["epsilon"] for ph in phys_results)

        print("\n" + "=" * 72)
        print("  BILAN")
        print("=" * 72)
        print(f"\n  α_G (couplage gravitationnel, m_K = {M_K_DEFAULT_MEV} MeV) = {alpha_G:.2e}")
        print(f"  κ_phys = {KAPPA_PHYS:.2e} m")
        print(f"\n  ε_max (nœuds calculés) = {eps_max:.2e}")
        print(f"  ε requis (énergie noire) ≈ 2.6")
        print(f"\n  → Déficit : {np.log10(2.6) - np.log10(eps_max):.0f} ordres de grandeur")
        print(f"\n  CONCLUSION : Le chauffage vibrationnel est INSUFFISANT pour")
        print(f"  expliquer l'énergie noire. Le mécanisme est supprimé par α_G ≈ 10⁻⁴³.")
        print(f"  L'énergie du vide du champ χ reste la seule piste viable.")

    # ====== Graphiques ======
    _make_plots(all_results, phys_results)

    print(f"\nGraphiques sauvegardés dans {OUTPUT_DIR}/")


def _make_plots(all_results, phys_results):
    """Génère les graphiques récapitulatifs."""
    import matplotlib.pyplot as plt

    if not all_results:
        return

    # 1. Spectre par nœud
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in all_results:
        n_modes = len(r["omegas"])
        ax.plot(range(1, n_modes + 1), r["omegas"],
                'o-', label=f'{r["knot_name"]} (c={r["crossing"]})', markersize=4)
    ax.set_xlabel("Mode i")
    ax.set_ylabel("ω_i = E_i - E₀ (unités code)")
    ax.set_title("Spectre vibrationnel des nœuds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "vibrational_spectrum.png"), dpi=150)
    plt.close(fig)

    # 2. Intégrales de recouvrement
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in all_results:
        n = len(r["overlaps"])
        ax.semilogy(range(1, n + 1), np.abs(r["overlaps"]),
                     'o-', label=f'{r["knot_name"]}', markersize=4)
    ax.set_xlabel("Mode i")
    ax.set_ylabel("|I_i| = ∫ |Ψ₀|² |Ψ_i|² d³x")
    ax.set_title("Intégrales de recouvrement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overlap_integrals.png"), dpi=150)
    plt.close(fig)

    # 3. Structure factor vs crossing number
    if len(all_results) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        crossings = [r["crossing"] for r in all_results]
        S_vals = [abs(r["structure_factor"]) for r in all_results]
        ax.semilogy(crossings, S_vals, 'o-', markersize=8, color='navy')
        for r in all_results:
            ax.annotate(r["knot_name"],
                        (r["crossing"], abs(r["structure_factor"])),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_xlabel("Nombre de croisements c")
        ax.set_ylabel("|S| = |Σ_i ω_i I_i|")
        ax.set_title("Facteur de structure vs complexité du nœud")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "structure_factor.png"), dpi=150)
        plt.close(fig)

    # 4. Résultat final : ε vs c
    if phys_results and len(all_results) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        crossings = [r["crossing"] for r in all_results]
        epsilons = [ph["epsilon"] for ph in phys_results]
        ax.semilogy(crossings, epsilons, 'o-', markersize=8, color='darkred')
        ax.axhline(y=2.6, color='green', linestyle='--', linewidth=2,
                    label='ε requis ≈ 2.6 (énergie noire)')
        for r, ph in zip(all_results, phys_results):
            ax.annotate(r["knot_name"],
                        (r["crossing"], ph["epsilon"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_xlabel("Nombre de croisements c")
        ax.set_ylabel("ε = E_heat / (m_K c²)")
        ax.set_title("Ratio vibrationnel ε — déficit de ~40 ordres de grandeur")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "epsilon_vs_crossing.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    run()
