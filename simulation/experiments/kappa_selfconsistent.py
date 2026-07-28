# simulation/experiments/kappa_selfconsistent.py
"""
Détermination auto-consistante de κ — SANS G.

Calcul numérique réel :
1. |Ψ|²_EM depuis les facteurs de forme du proton (diffusion ep)
2. |Ψ|²_QM(κ) depuis le solveur Schrödinger–Poisson pour chaque κ
3. κ* = unique zéro de R(κ) = ∫|Φ(κ) - |Ψ|²_EM|² d³r

Entrées : ℏ, c, m_p, μ₀, e, topologie (ρ_J = 25.7)
Sortie  : κ en mètres, puis G = κc²/(8πm_p) comme vérification.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import brentq

from theory.knot_catalog import knot_curve
from theory.chi_field import jones_rho, chi_retraction
from theory.schrodinger import generate_mesh, assemble_system, solve_eigenproblem

# ── Constantes SI (aucune gravitationnelle) ──────────────────────
HBAR_SI = 1.054_571_817e-34   # J·s
C_SI    = 2.997_924_58e8      # m/s
M_P_SI  = 1.672_621_924e-27   # kg  (masse du proton)
MU_0_SI = 1.256_637_062e-6    # H/m (perméabilité du vide)
E_SI    = 1.602_176_634e-19   # C   (charge élémentaire)

# Facteur de forme dipolaire : Λ² = 0.71 GeV²
LAMBDA_SQ_GEV2 = 0.71                            # GeV²
GEV_TO_JOULE   = 1.602_176_634e-10               # J/GeV
LAMBDA_SQ_SI   = LAMBDA_SQ_GEV2 * GEV_TO_JOULE   # en J (pour Q en J/c)

# Moment magnétique du proton
MU_P_SI = 1.410_606_797e-26   # J/T


# ═══════════════════════════════════════════════════════════════════
# 1. |Ψ|²_EM depuis le facteur de forme magnétique du proton
# ═══════════════════════════════════════════════════════════════════

def proton_form_factor_realspace(r_fm, n_points=2000):
    """Densité de courant normalisée du proton en espace direct.

    Le facteur de forme magnétique dipolaire est :
        G_M(Q²) = μ_p / (1 + Q²/Λ²)²

    Sa transformée de Fourier 3D donne (Kelly 2004) :
        ρ_M(r) ∝ (Λ³/8π) × exp(-Λr)

    avec Λ = √(0.71) GeV/c = 4.27 fm⁻¹.

    On normalise : ∫ρ_M(r) 4πr² dr = 1.

    Parameters
    ----------
    r_fm : array, rayons en femtomètres.

    Returns
    -------
    rho : array, |Ψ|²_EM(r) normalisée [fm⁻³].
    """
    # Λ en fm⁻¹ :  Λ = √(Λ²) × (GeV/c) / (ℏc)
    # ℏc = 0.197327 GeV·fm
    hbar_c_gev_fm = 0.197_327  # GeV·fm
    Lambda_fm_inv = np.sqrt(LAMBDA_SQ_GEV2) / hbar_c_gev_fm  # fm⁻¹
    # Lambda_fm_inv ≈ 4.27 fm⁻¹

    r = np.asarray(r_fm, dtype=float)
    rho = (Lambda_fm_inv**3 / (8.0 * np.pi)) * np.exp(-Lambda_fm_inv * r)

    # Vérification numérique de la normalisation
    # ∫ρ(r)4πr²dr = (Λ³/8π)×4π × ∫r² exp(-Λr) dr
    #              = (Λ³/2) × 2/Λ³ = 1  ✓
    return rho


def psi2_em_on_mesh(points_code, R_proton_code):
    """Évalue |Ψ|²_EM sur les nœuds du maillage FEM.

    Le maillage FEM est en unités de code (taille ~1).
    Le proton physique a un rayon ~0.88 fm.
    On mappe : r_fm = |r_code| × (R_proton_fm / R_proton_code).

    Parameters
    ----------
    points_code : (N, 3) array, coordonnées du maillage FEM.
    R_proton_code : float, rayon caractéristique du nœud en code.

    Returns
    -------
    psi2_em : (N,) array, |Ψ|²_EM aux nœuds, en unités de code
              (normalisée sur le volume du maillage).
    """
    R_proton_fm = 0.8414  # fm (rayon de charge du proton, CODATA 2018)
    scale = R_proton_fm / R_proton_code  # fm par unité de code

    r_code = np.linalg.norm(points_code, axis=1)
    r_fm = r_code * scale

    # ρ en fm⁻³
    rho_fm3 = proton_form_factor_realspace(r_fm)

    # Conversion en unités de code : [code⁻³] = [fm⁻³] × scale³
    rho_code = rho_fm3 * scale**3

    return rho_code


# ═══════════════════════════════════════════════════════════════════
# 2. Solveur Schrödinger–Poisson avec κ variable
# ═══════════════════════════════════════════════════════════════════

def solve_selfconsistent_kappa(
    knot_name, kappa, points, tets, boundary, knot_tree,
    n_eigenvalues=6, max_iterations=30, tol=1e-4,
    epsilon=0.2, damping=0.5,
):
    """Solveur auto-consistant pour un κ donné (boucle interne).

    Identique à solve_knot_selfconsistent mais avec κ en paramètre
    au lieu de l'importer depuis postulates.

    Returns
    -------
    dict avec 'psi_full', 'energy', 'omega_sq', 'converged', 'iterations'.
    """
    n_pts = len(points)
    free_mask = ~boundary

    energy = 0.0
    omega_sq = np.ones(n_pts)

    converged = False
    iteration = 0

    for iteration in range(max_iterations):
        H, M_mat = assemble_system(
            points, tets, boundary, knot_name, knot_tree,
            energy=energy, epsilon=epsilon, omega_sq=omega_sq,
        )
        eigenvalues, eigenvectors = solve_eigenproblem(
            H, M_mat, n_eigenvalues=n_eigenvalues,
        )

        new_energy = abs(eigenvalues[0])
        blended_energy = damping * new_energy + (1 - damping) * energy
        delta_energy = abs(blended_energy - energy)

        psi_free = eigenvectors[:, 0]
        psi_full = np.zeros(n_pts)
        psi_full[free_mask] = psi_free

        # Ω² = 1 + κ|Ψ|² — avec κ paramétrique
        new_omega_sq = 1.0 + kappa * np.abs(psi_full)**2

        blended_omega_sq = damping * new_omega_sq + (1 - damping) * omega_sq
        delta_metric = np.max(np.abs(blended_omega_sq - omega_sq))

        energy = blended_energy
        omega_sq = blended_omega_sq

        energy_scale = max(abs(energy), 1e-10)
        if delta_energy < tol * energy_scale and delta_metric < tol:
            converged = True
            break

    return {
        "psi_full": psi_full,
        "energy": energy,
        "omega_sq": omega_sq,
        "converged": converged,
        "iterations": iteration + 1,
    }


def radial_profile(points, values, n_bins=50):
    """Profil radial sphérique moyen d'un champ sur le maillage.

    Returns
    -------
    r_centers : (n_bins,) array, centres des bins radiaux.
    profile : (n_bins,) array, valeur moyenne dans chaque coquille.
    """
    r = np.linalg.norm(points, axis=1)
    r_max = np.max(r)
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(mask):
            profile[i] = np.mean(values[mask])

    return r_centers, profile


def residual_kappa(kappa, knot_name, points, tets, boundary, knot_tree,
                   psi2_em, epsilon=0.2, n_bins=40):
    """Résidu R(κ) = distance L² entre profils radiaux.

    Compare le profil radial de |Ψ_κ|² (solveur) avec |Ψ|²_EM
    (facteurs de forme). Les deux sont normalisés en forme
    (intégrale = 1 sur les bins) pour que la comparaison soit
    indépendante de l'échelle absolue.

    Returns
    -------
    residual : float, résidu normalisé.
    """
    if kappa <= 0:
        kappa = 1e-20  # éviter κ = 0 exactement

    result = solve_selfconsistent_kappa(
        knot_name, kappa, points, tets, boundary, knot_tree,
        epsilon=epsilon,
    )
    psi2_qm = result["psi_full"]**2

    # Profils radiaux
    r_c, prof_qm = radial_profile(points, psi2_qm, n_bins=n_bins)
    _, prof_em = radial_profile(points, psi2_em, n_bins=n_bins)

    # Normalisation en forme (∫ profile × r² dr ≈ 1)
    dr = r_c[1] - r_c[0] if len(r_c) > 1 else 1.0
    w = 4.0 * np.pi * r_c**2 * dr  # poids sphérique

    norm_qm = np.sum(prof_qm * w)
    norm_em = np.sum(prof_em * w)

    if norm_qm < 1e-30 or norm_em < 1e-30:
        return 1e10

    shape_qm = prof_qm / norm_qm
    shape_em = prof_em / norm_em

    # Distance L² pondérée
    residual = np.sum((shape_qm - shape_em)**2 * w)
    return residual


# ═══════════════════════════════════════════════════════════════════
# 3. Boucle externe : scan de κ et recherche du minimum
# ═══════════════════════════════════════════════════════════════════

def scan_kappa(knot_name="9_1", n_coarse=20, kappa_values=None):
    """Scanne κ et trace R(κ) pour trouver le minimum.

    Par défaut, scanne de κ = 0.001 à κ = 1.0 en échelle log.
    """
    if kappa_values is None:
        kappa_values = np.logspace(-3, 0, 20)

    # Génération du maillage (une seule fois)
    curve = knot_curve(knot_name, 500)
    knot_tree = cKDTree(curve)
    points, tets, boundary = generate_mesh(
        curve, n_coarse=n_coarse, epsilon=0.2, refine=True,
    )

    # Rayon caractéristique du nœud en code
    R_knot_code = np.std(np.linalg.norm(curve, axis=1))

    # |Ψ|²_EM sur le maillage
    psi2_em = psi2_em_on_mesh(points, R_knot_code)

    print(f"Maillage : {len(points)} nœuds, {len(tets)} tétraèdres")
    print(f"R_knot_code = {R_knot_code:.4f}")
    print(f"max(|Ψ|²_EM) = {np.max(psi2_em):.4e}")
    print()

    residuals = []
    for kappa in kappa_values:
        print(f"  κ = {kappa:.6f} ...", end=" ", flush=True)
        try:
            R = residual_kappa(
                kappa, knot_name, points, tets, boundary,
                knot_tree, psi2_em,
            )
            residuals.append(R)
            print(f"R(κ) = {R:.6e}")
        except Exception as e:
            residuals.append(np.nan)
            print(f"ÉCHEC: {e}")

    return kappa_values, np.array(residuals)


def find_kappa_star(knot_name="9_1", n_coarse=20):
    """Trouve κ* par scan grossier puis affinement.

    Returns
    -------
    kappa_star_code : float, κ* en unités de code.
    results : dict, résultats détaillés.
    """
    # ── Maillage ──
    curve = knot_curve(knot_name, 500)
    knot_tree = cKDTree(curve)
    points, tets, boundary = generate_mesh(
        curve, n_coarse=n_coarse, epsilon=0.2, refine=True,
    )
    R_knot_code = np.std(np.linalg.norm(curve, axis=1))
    psi2_em = psi2_em_on_mesh(points, R_knot_code)

    print("=" * 60)
    print("  Détermination auto-consistante de κ")
    print("=" * 60)
    print(f"  Nœud         : {knot_name}")
    print(f"  Maillage     : {len(points)} nœuds")
    print(f"  R_knot_code  : {R_knot_code:.4f}")
    print()

    # ── Scan grossier (échelle log) ──
    print("── Scan grossier ──")
    kappa_scan = np.logspace(-3, 1, 30)
    residuals = []
    for kappa in kappa_scan:
        try:
            R = residual_kappa(
                kappa, knot_name, points, tets, boundary,
                knot_tree, psi2_em,
            )
        except Exception:
            R = np.nan
        residuals.append(R)
        print(f"  κ = {kappa:.4e}  →  R = {R:.6e}")

    residuals = np.array(residuals)
    valid = ~np.isnan(residuals)

    if not np.any(valid):
        print("ÉCHEC: aucun κ valide.")
        return None, {}

    # Trouver le minimum
    idx_min = np.nanargmin(residuals)
    kappa_star = kappa_scan[idx_min]
    R_min = residuals[idx_min]

    print(f"\n  Minimum grossier : κ* ≈ {kappa_star:.4e}, R = {R_min:.6e}")

    # ── Affinement autour du minimum ──
    if idx_min > 0 and idx_min < len(kappa_scan) - 1:
        kappa_lo = kappa_scan[idx_min - 1]
        kappa_hi = kappa_scan[idx_min + 1]
    else:
        kappa_lo = kappa_star * 0.3
        kappa_hi = kappa_star * 3.0

    print(f"\n── Affinement [{kappa_lo:.4e}, {kappa_hi:.4e}] ──")
    kappa_fine = np.linspace(kappa_lo, kappa_hi, 15)
    residuals_fine = []
    for kappa in kappa_fine:
        try:
            R = residual_kappa(
                kappa, knot_name, points, tets, boundary,
                knot_tree, psi2_em,
            )
        except Exception:
            R = np.nan
        residuals_fine.append(R)
        print(f"  κ = {kappa:.6e}  →  R = {R:.6e}")

    residuals_fine = np.array(residuals_fine)
    idx_fine = np.nanargmin(residuals_fine)
    kappa_star = kappa_fine[idx_fine]
    R_min = residuals_fine[idx_fine]

    print(f"\n  Minimum affiné : κ* = {kappa_star:.6e}, R = {R_min:.6e}")

    return kappa_star, {
        "kappa_scan": kappa_scan,
        "residuals_scan": residuals,
        "kappa_fine": kappa_fine,
        "residuals_fine": residuals_fine,
        "R_knot_code": R_knot_code,
        "n_points": len(points),
    }


# ═══════════════════════════════════════════════════════════════════
# 4. Conversion code → SI et vérification
# ═══════════════════════════════════════════════════════════════════

def kappa_code_to_si(kappa_code, R_knot_code):
    """Convertit κ en unités de code vers κ en SI (mètres).

    Dans le code :
    - Longueurs en unités de R_knot_code
    - |Ψ|² en unités de R_knot_code⁻³
    - κ × |Ψ|² est adimensionné → [κ_code] = [R_knot_code³]

    Mais h(r) = κ∫|Ψ|²/|r-r'| d³r' a dimension [κ][|Ψ|²][d³r'][1/r]
    = [κ_code] × 1 × 1 × 1 = adimensionné en code.

    En SI, h est adimensionné et r est en m, |Ψ|² en m⁻³ :
    [κ_SI] × [m⁻³] × [m³] × [m⁻¹] = [κ_SI/m] = adimensionné
    → [κ_SI] = m.

    La conversion : κ_SI = κ_code × L₀
    avec L₀ = (longueur physique du nœud) / (taille en code).

    Le proton a un rayon ~ 0.88 fm = 0.88e-15 m.
    L₀ = 0.8414e-15 / R_knot_code.
    """
    R_proton_m = 0.8414e-15   # m
    L0 = R_proton_m / R_knot_code
    return kappa_code * L0


def verify_G(kappa_si):
    """Calcule G depuis κ et compare à CODATA.

    G = κ c² / (8π m_p)
    """
    G_predicted = kappa_si * C_SI**2 / (8.0 * np.pi * M_P_SI)
    G_CODATA = 6.674_30e-11  # m³/(kg·s²)
    ratio = G_predicted / G_CODATA
    return G_predicted, G_CODATA, ratio


# ═══════════════════════════════════════════════════════════════════
# 5. Point d'entrée
# ═══════════════════════════════════════════════════════════════════

def run():
    """Exécute la détermination complète de κ."""
    print("=" * 70)
    print("  DÉTERMINATION AUTO-CONSISTANTE DE κ — SANS G")
    print("=" * 70)
    print()

    # On utilise le nœud 9_1 (composante principale du nucléon)
    # puis l'entrelacs complet si besoin
    for knot_name in ["9_1"]:
        kappa_star, info = find_kappa_star(knot_name, n_coarse=20)

        if kappa_star is None:
            continue

        R_knot_code = info["R_knot_code"]

        # Conversion en SI
        kappa_si = kappa_code_to_si(kappa_star, R_knot_code)

        print(f"\n{'=' * 60}")
        print(f"  RÉSULTAT pour {knot_name}")
        print(f"{'=' * 60}")
        print(f"  κ_code  = {kappa_star:.6e}")
        print(f"  κ_SI    = {kappa_si:.4e} m")

        G_pred, G_codata, ratio = verify_G(kappa_si)
        print(f"\n  Vérification :")
        print(f"  G_prédit = κc²/(8πm_p) = {G_pred:.4e} m³/(kg·s²)")
        print(f"  G_CODATA               = {G_codata:.4e} m³/(kg·s²)")
        print(f"  Rapport G_prédit/G_CODATA = {ratio:.6f}")

    print(f"\n{'=' * 70}")
    print(f"  FIN")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
