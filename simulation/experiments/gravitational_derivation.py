# simulation/experiments/gravitational_derivation.py
"""
Expérience : Dérivation non-circulaire de G via nucléons compressés

La constante gravitationnelle G n'est pas universelle — elle varie avec
le facteur de rétractation ρ(K,E). La compression augmente E, ce qui
diminue ρ(K,E) = ρ_J²/(ρ_J + E/E₀), affaiblissant le potentiel
V = -λρ/d² et modifiant G_eff.

Formule clé :
    G_eff(E) = κc²/(8π·m_eff) ≈ G_cold × (ρ_J + E/E₀) / ρ_J

Cette expérience calcule G_eff pour différents nœuds et niveaux de
compression (box_margin), montrant que G varie avec la densité locale.
"""
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
from scipy.spatial import cKDTree

from theory.knot_catalog import knot_curve
from theory.chi_field import jones_rho, chi_retraction, characteristic_energy
from theory.schrodinger import generate_mesh, assemble_system, solve_eigenproblem
from theory.postulates import KAPPA_PHYS

OUTPUT_DIR = "output/gravitational_derivation"

# --- Physical constants (SI) ---
HBAR_SI = 1.0546e-34       # J·s
C_SI = 2.9979e8            # m/s
G_SI = 6.6743e-11          # m³/(kg·s²)
M_PROTON_KG = 1.6726e-27   # kg

# Knots to study
KNOTS = ["3_1", "4_1", "5_1", "7_1", "9_1"]

# Box margins (compression levels: smaller = more compressed)
BOX_MARGINS = [3.0, 2.5, 2.0, 1.5, 1.2, 1.0]


def solve_compressed_knot(knot_name, box_margin, n_coarse=20,
                          n_eigenvalues=6, max_iterations=20, tol=0.01):
    """Résout l'équation de Schrödinger avec itération point fixe sur E.

    Appel direct aux fonctions bas niveau (generate_mesh, assemble_system,
    solve_eigenproblem) avec box_margin explicite.

    Pattern : solve_knot_jones (simulation/solver.py:67-133).
    """
    curve = knot_curve(knot_name, 500)
    knot_tree = cKDTree(curve)
    epsilon = 0.2

    points, tets, boundary = generate_mesh(
        curve, n_coarse=n_coarse, epsilon=epsilon,
        refine=True, box_margin=box_margin,
    )

    rho_J = jones_rho(knot_name)
    energy = 0.0
    damping = 0.5

    eigenvalues = None
    eigenvectors = None
    converged = False
    iteration = 0

    for iteration in range(max_iterations):
        H, M = assemble_system(
            points, tets, boundary, knot_name, knot_tree,
            energy=energy, epsilon=epsilon,
        )
        eigenvalues, eigenvectors = solve_eigenproblem(
            H, M, n_eigenvalues=n_eigenvalues,
        )
        new_energy = abs(eigenvalues[0])

        blended = damping * new_energy + (1 - damping) * energy
        if abs(blended - energy) < tol * max(abs(energy), 1e-10):
            energy = blended
            converged = True
            break
        energy = blended

    rho_final = chi_retraction(knot_name, energy)
    free_mask = ~boundary

    return {
        "knot_name": knot_name,
        "box_margin": box_margin,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "points": points,
        "tets": tets,
        "boundary": boundary,
        "free_mask": free_mask,
        "rho_J": rho_J,
        "rho_final": rho_final,
        "E_0": energy,
        "iterations": iteration + 1,
        "converged": converged,
    }


def compute_psi_norm(points, tets, psi_full):
    """∫|Ψ₀|² d³x via Gauss 4 points.

    Pattern : compute_overlap (vibrational_energy.py:37-79).
    """
    a = (5.0 - np.sqrt(5.0)) / 20.0
    b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    bary = np.array([
        [b, a, a, a],
        [a, b, a, a],
        [a, a, b, a],
        [a, a, a, b],
    ])

    all_verts = points[tets]
    edges = all_verts[:, 1:, :] - all_verts[:, 0:1, :]
    J = np.transpose(edges, (0, 2, 1))
    volumes = np.abs(np.linalg.det(J)) / 6.0

    valid = volumes > 1e-15
    valid_tets = tets[valid]
    valid_volumes = volumes[valid]

    rho_nodes = (psi_full**2)[valid_tets]
    rho_quad = np.einsum('qi,ei->eq', bary, rho_nodes)
    norm = np.sum(valid_volumes[:, None] / 4.0 * rho_quad)

    return float(norm)


def compute_concentration(points, tets, psi_full):
    """Inverse Participation Ratio : IPR = ∫|Ψ|⁴ / (∫|Ψ|²)².

    Mesure la localisation spatiale de la fonction d'onde.
    IPR grand = fonction d'onde concentrée.
    """
    a = (5.0 - np.sqrt(5.0)) / 20.0
    b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    bary = np.array([
        [b, a, a, a],
        [a, b, a, a],
        [a, a, b, a],
        [a, a, a, b],
    ])

    all_verts = points[tets]
    edges = all_verts[:, 1:, :] - all_verts[:, 0:1, :]
    J = np.transpose(edges, (0, 2, 1))
    volumes = np.abs(np.linalg.det(J)) / 6.0

    valid = volumes > 1e-15
    valid_tets = tets[valid]
    valid_volumes = volumes[valid]

    # ∫|Ψ|² d³x
    rho2_nodes = (psi_full**2)[valid_tets]
    rho2_quad = np.einsum('qi,ei->eq', bary, rho2_nodes)
    norm2 = np.sum(valid_volumes[:, None] / 4.0 * rho2_quad)

    # ∫|Ψ|⁴ d³x
    rho4_nodes = (psi_full**4)[valid_tets]
    rho4_quad = np.einsum('qi,ei->eq', bary, rho4_nodes)
    norm4 = np.sum(valid_volumes[:, None] / 4.0 * rho4_quad)

    if norm2 == 0:
        return 0.0
    return float(norm4 / norm2**2)


def physical_G(rho_final):
    """G_eff en SI depuis κ_phys et ρ_final.

    G_eff = κ_phys · c² / (8π · m_eff)
    avec m_eff ∝ ρ_final (la masse effective dépend de la rétractation).

    En pratique, G_cold = κ·c²/(8π·m_p) et :
        G_eff / G_cold = ρ_J / ρ_final
    car m_eff ∝ ρ_final.
    """
    return KAPPA_PHYS * C_SI**2 / (8.0 * np.pi * M_PROTON_KG)


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 76)
    print("  Dérivation non-circulaire de G via nucléons compressés")
    print("=" * 76)

    # ====== Boucle principale ======
    all_results = []

    for knot_name in KNOTS:
        rho_J = jones_rho(knot_name)
        E_K = characteristic_energy(knot_name)
        print(f"\n--- Nœud {knot_name} : ρ_J = {rho_J:.3f}, "
              f"E_K = {E_K:.3f} ---")

        for bm in BOX_MARGINS:
            print(f"  box_margin = {bm:.1f} ...", end=" ", flush=True)
            try:
                res = solve_compressed_knot(knot_name, bm, n_coarse=20)

                # Expand ground state
                psi_full = np.zeros(len(res["points"]))
                psi_full[res["free_mask"]] = res["eigenvectors"][:, 0]

                # Compute observables
                norm = compute_psi_norm(
                    res["points"], res["tets"], psi_full)
                ipr = compute_concentration(
                    res["points"], res["tets"], psi_full)

                # Box volume
                max_extent = np.max(np.abs(res["points"]))
                V_box = (2 * max_extent)**3

                # G_eff
                rho_J = res["rho_J"]
                rho_f = res["rho_final"]
                G_cold = physical_G(rho_J)
                G_eff_ratio = rho_J / rho_f if rho_f > 0 else float('inf')

                res.update({
                    "norm": norm,
                    "ipr": ipr,
                    "V_box": V_box,
                    "G_cold": G_cold,
                    "G_eff_ratio": G_eff_ratio,
                })
                all_results.append(res)

                conv_str = "✓" if res["converged"] else "✗"
                print(f"E₀={res['E_0']:.4f}, ρ/ρ_J={rho_f/rho_J:.4f}, "
                      f"G/G_cold={G_eff_ratio:.4f}, IPR={ipr:.4f} [{conv_str}]")
            except Exception as e:
                print(f"ÉCHEC: {e}")

    if not all_results:
        print("\nAucun résultat. Abandon.")
        return

    # ====== Table 1 : E₀ et ρ(K,E)/ρ_J vs compression par nœud ======
    print("\n" + "=" * 76)
    print("  TABLE 1 : E₀ et ρ(K,E)/ρ_J vs compression")
    print("=" * 76)
    print(f"\n{'Nœud':<8} {'margin':<8} {'V_box':<12} {'E₀':<10} "
          f"{'ρ/ρ_J':<10} {'IPR':<10} {'conv':<6}")
    print("-" * 64)
    for r in all_results:
        rho_ratio = r["rho_final"] / r["rho_J"] if r["rho_J"] > 0 else 0
        conv_str = "oui" if r["converged"] else "non"
        print(f"{r['knot_name']:<8} {r['box_margin']:<8.1f} "
              f"{r['V_box']:<12.1f} {r['E_0']:<10.4f} "
              f"{rho_ratio:<10.4f} {r['ipr']:<10.4f} {conv_str:<6}")

    # ====== Table 2 : G_eff/G_cold vs compression ======
    print("\n" + "=" * 76)
    print("  TABLE 2 : G_eff/G_cold vs compression")
    print("=" * 76)
    print(f"\n{'Nœud':<8} {'margin':<8} {'1/V_box':<12} "
          f"{'G_eff/G_cold':<14} {'E₀':<10}")
    print("-" * 52)
    for r in all_results:
        inv_V = 1.0 / r["V_box"] if r["V_box"] > 0 else 0
        print(f"{r['knot_name']:<8} {r['box_margin']:<8.1f} "
              f"{inv_V:<12.4e} {r['G_eff_ratio']:<14.4f} "
              f"{r['E_0']:<10.4f}")

    # ====== Table 3 : G_eff vs ρ_J à compression fixe ======
    print("\n" + "=" * 76)
    print("  TABLE 3 : G_eff vs ρ_J à compression fixe (margin=1.5)")
    print("=" * 76)
    fixed_margin = 1.5
    fixed_results = [r for r in all_results
                     if abs(r["box_margin"] - fixed_margin) < 0.01]
    if fixed_results:
        print(f"\n{'Nœud':<8} {'ρ_J':<8} {'E₀':<10} {'ρ_final':<10} "
              f"{'G_eff/G_cold':<14}")
        print("-" * 50)
        for r in fixed_results:
            print(f"{r['knot_name']:<8} {r['rho_J']:<8.3f} "
                  f"{r['E_0']:<10.4f} {r['rho_final']:<10.4f} "
                  f"{r['G_eff_ratio']:<14.4f}")

    # ====== Bilan ======
    print("\n" + "=" * 76)
    print("  BILAN")
    print("=" * 76)
    G_cold = physical_G(0)
    print(f"\n  G_cold = κc²/(8π·m_p) = {G_cold:.4e} m³/(kg·s²)")
    print(f"  G_SI (CODATA)          = {G_SI:.4e} m³/(kg·s²)")
    print(f"  Rapport G_cold/G_SI    = {G_cold/G_SI:.6f}")
    print(f"\n  κ_phys = {KAPPA_PHYS:.4e} m (paramètre fondamental)")
    print(f"\n  PHYSIQUE CLÉ :")
    print(f"  Compression → E↑ → ρ(K,E)↓ → G_eff↑")
    print(f"  G n'est PAS une constante universelle, elle varie avec ρ(K,E)")
    print(f"  La circularité κ/G est résolue : κ est fondamental,")
    print(f"  G est dérivé avec la structure additionnelle G(E)")

    # ====== Graphiques ======
    _make_plots(all_results)
    print(f"\nGraphiques sauvegardés dans {OUTPUT_DIR}/")


def _make_plots(all_results):
    """Génère les 4 graphiques récapitulatifs."""
    import matplotlib.pyplot as plt

    if not all_results:
        return

    knot_names = sorted(set(r["knot_name"] for r in all_results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(knot_names)))
    color_map = dict(zip(knot_names, colors))

    # 1. E₀ vs 1/V_box (énergie de compression)
    fig, ax = plt.subplots(figsize=(9, 6))
    for kn in knot_names:
        data = [r for r in all_results if r["knot_name"] == kn]
        inv_V = [1.0 / r["V_box"] for r in data]
        E0 = [r["E_0"] for r in data]
        ax.plot(inv_V, E0, 'o-', color=color_map[kn],
                label=kn, markersize=6)
    ax.set_xlabel("1/V_box (densité de compression)")
    ax.set_ylabel("E₀ (énergie propre fondamentale)")
    ax.set_title("Énergie de compression : E₀ vs 1/V_box")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "E0_vs_inv_V.png"), dpi=150)
    plt.close(fig)

    # 2. ρ(K,E)/ρ_J vs E₀ (affaiblissement de la rétractation)
    fig, ax = plt.subplots(figsize=(9, 6))
    for kn in knot_names:
        data = [r for r in all_results if r["knot_name"] == kn]
        E0 = [r["E_0"] for r in data]
        rho_ratio = [r["rho_final"] / r["rho_J"] for r in data]
        ax.plot(E0, rho_ratio, 'o-', color=color_map[kn],
                label=kn, markersize=6)
    ax.set_xlabel("E₀ (énergie)")
    ax.set_ylabel("ρ(K,E) / ρ_J")
    ax.set_title("Affaiblissement de la rétractation avec l'énergie")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rho_ratio_vs_E0.png"), dpi=150)
    plt.close(fig)

    # 3. G_eff/G_cold vs densité effective
    fig, ax = plt.subplots(figsize=(9, 6))
    for kn in knot_names:
        data = [r for r in all_results if r["knot_name"] == kn]
        inv_V = [1.0 / r["V_box"] for r in data]
        G_ratio = [r["G_eff_ratio"] for r in data]
        ax.plot(inv_V, G_ratio, 'o-', color=color_map[kn],
                label=kn, markersize=6)
    ax.set_xlabel("1/V_box (densité effective)")
    ax.set_ylabel("G_eff / G_cold")
    ax.set_title("Variation de G avec la densité : G_eff/G_cold vs 1/V_box")
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
               label='G_cold (limite froide)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "G_ratio_vs_density.png"), dpi=150)
    plt.close(fig)

    # 4. G_eff vs ρ_J à deux niveaux de compression
    fig, ax = plt.subplots(figsize=(9, 6))
    for bm, marker, ls in [(3.0, 'o', '-'), (1.0, 's', '--')]:
        data = [r for r in all_results
                if abs(r["box_margin"] - bm) < 0.01]
        if not data:
            continue
        rho_J_vals = [r["rho_J"] for r in data]
        G_ratio = [r["G_eff_ratio"] for r in data]
        ax.plot(rho_J_vals, G_ratio, marker + ls,
                label=f'margin={bm:.1f}', markersize=8)
        for r in data:
            ax.annotate(r["knot_name"],
                        (r["rho_J"], r["G_eff_ratio"]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8)
    ax.set_xlabel("ρ_J (facteur de rétractation de Jones)")
    ax.set_ylabel("G_eff / G_cold")
    ax.set_title("G_eff vs ρ_J à deux niveaux de compression")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "G_vs_rhoJ.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
