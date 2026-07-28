"""Tests V5.37b — Potentiel Cornell TQNT pour quarkonium.

10 tests pytest couvrant :

 1. Potentiel Cornell : signe Coulomb, signe linéaire, minimum
 2. Solveur radial 1D : oscillateur harmonique convergence sanity-check
 3. Solveur radial 1D : hydrogène ground state convergence
 4. α_TQNT et σ_TQNT : formes universelles vs topo
 5. Calibration K_alpha : α_TQNT(J/ψ) ≈ 0.4
 6. Calibration K_sigma : Δm(ψ(2S)/J/ψ) ≈ 589 MeV à 1 MeV près
 7. Spectre J/ψ : 3 niveaux croissants
 8. Mode universal : 4/4 splittings dans ±0.3 dex
 9. Mode topo : 2/4 splittings → diagnostic naïf
10. Mode universal améliore strictement V5.37 (Σ|log| < 0.76)
"""
from __future__ import annotations

import numpy as np
import pytest

from theory.v5p37b_cornell_quarkonium import (
    ALPHA_EM,
    BETA_MEV,
    HBAR_C_MEV_FM,
    M_CHARM_MEV,
    MODE_TOPO,
    MODE_UNIVERSAL,
    N_COLOR,
    QUARKONIUM_ASSIGNMENTS,
    alpha_topological,
    calibrate_cornell_constants,
    clear_calibration_cache,
    cornell_potential_TQNT,
    schrodinger_cornell_spectrum,
    sigma_topological,
    solve_radial_cornell,
    v5p37b_cornell_table,
    verdict_v5p37b,
)


@pytest.fixture(autouse=True)
def _clear_cal_cache():
    clear_calibration_cache()
    yield
    clear_calibration_cache()


# ---------------------------------------------------------------------------
# 1. Potentiel Cornell
# ---------------------------------------------------------------------------
def test_cornell_potential_shape():
    """V(r) doit être négatif près de 0 (Coulomb) et positif au loin (lin)."""
    r = np.array([0.01, 0.5, 5.0, 50.0])
    V = cornell_potential_TQNT(r, alpha_topo=0.4, sigma_topo_MeV_per_fm=900.0)
    # Très près : Coulomb domine ⇒ V < 0
    assert V[0] < 0, f"V(0.01 fm) = {V[0]} doit être < 0"
    # Loin : linéaire domine ⇒ V > 0
    assert V[-1] > 0, f"V(50 fm) = {V[-1]} doit être > 0"
    # Croissance asymptotique linéaire
    assert V[-1] > V[-2] > 0
    # Régularisation : pas de divergence à r=0
    V0 = cornell_potential_TQNT(np.array([0.0]),
                                alpha_topo=0.4,
                                sigma_topo_MeV_per_fm=900.0)
    assert np.isfinite(V0[0])


# ---------------------------------------------------------------------------
# 2. Solveur radial — oscillateur harmonique
# ---------------------------------------------------------------------------
def test_solver_harmonic_oscillator_sanity():
    """Pour V(r) = 0,5 k r² (oscillateur radial 3D ℓ=0), E_0 = 1,5 ℏω.

    Le solveur 1D radial avec u(r) = r·R(r) sur ℓ=0 donne le spectre
    E_n = (2n + 1 + 3/2) ℏω/2 pour le harmonique 3D ?  En fait pour
    l'éq. radiale ℓ=0 avec V = ½ μω² r² :
        E_n = ℏω · (2n + 3/2),  n = 0, 1, 2, ...
    Difficile à reproduire avec exactement Cornell ; on teste plutôt
    le **niveau fondamental d'une particule libre dans une boîte** :
        E_1 = π² ℏ² / (2 μ R²)
    en mettant α = σ = 0.
    """
    mu = 1000.0  # MeV
    R = 5.0      # fm
    spec = solve_radial_cornell(
        mu, alpha_topo=0.0, sigma_topo_MeV_per_fm=0.0,
        r_max_fm=R, n_grid=2000, n_eigenvalues=2,
    )
    E1_theo = np.pi ** 2 * HBAR_C_MEV_FM ** 2 / (2.0 * mu * R ** 2)
    rel = abs(spec.eigenvalues_MeV[0] - E1_theo) / E1_theo
    assert rel < 0.01, f"Erreur boîte {rel*100:.2f} % > 1 %"
    # E_2 ≈ 4 × E_1 pour la boîte
    rel2 = abs(spec.eigenvalues_MeV[1] - 4 * E1_theo) / (4 * E1_theo)
    assert rel2 < 0.02, f"Erreur E_2 boîte {rel2*100:.2f} % > 2 %"


# ---------------------------------------------------------------------------
# 3. Solveur radial — hydrogène (Coulomb pur ℓ=0)
# ---------------------------------------------------------------------------
def test_solver_hydrogen_ground_state():
    """E_1 hydrogène = -μ c² α² / 2 = -13.6 eV pour μ = m_e, α = α_EM.

    Ici : on prend μ = 1000 MeV (toy proton), α = 0.4 (pas hydrogène
    réel) et on vérifie que E_1 < 0 avec ordre de grandeur correct.
    En boîte assez large, E_1 = -μ c² α² / 2  (Coulomb-like).
    """
    mu = 1000.0
    alpha = 0.4
    spec = solve_radial_cornell(
        mu, alpha_topo=alpha, sigma_topo_MeV_per_fm=0.0,
        r_max_fm=20.0, n_grid=4000, n_eigenvalues=1,
    )
    E_th = -0.5 * mu * alpha ** 2
    # Erreur tolérée modérée (boîte finie, FD ordre 2) ≈ 5 %
    rel = abs(spec.eigenvalues_MeV[0] - E_th) / abs(E_th)
    assert rel < 0.05, (
        f"E_1 Coulomb = {spec.eigenvalues_MeV[0]:.2f}, "
        f"théorique = {E_th:.2f}, erreur {rel*100:.2f} %"
    )


# ---------------------------------------------------------------------------
# 4. α_TQNT et σ_TQNT
# ---------------------------------------------------------------------------
def test_alpha_sigma_formulas():
    """Forme exacte des constantes."""
    knots = QUARKONIUM_ASSIGNMENTS["J/psi"]["knots"]
    # Mode universal : pas de c_J_eff
    a_u = alpha_topological(knots, K_alpha=2.0, mode=MODE_UNIVERSAL)
    expected_u = 2.0 * ALPHA_EM * N_COLOR
    assert abs(a_u - expected_u) < 1e-12
    # Mode topo : multiplié par c_J_eff
    a_t = alpha_topological(knots, K_alpha=2.0, mode=MODE_TOPO)
    assert a_t > a_u  # c_J_eff > 1 pour J/ψ
    # σ universal
    s_u = sigma_topological(knots, mass_link_MeV=3097.0,
                            K_sigma=1.0, mode=MODE_UNIVERSAL)
    expected_su = BETA_MEV * M_CHARM_MEV / HBAR_C_MEV_FM
    assert abs(s_u - expected_su) < 1e-9
    # Mode inconnu
    with pytest.raises(ValueError):
        alpha_topological(knots, K_alpha=1.0, mode="invalid")


# ---------------------------------------------------------------------------
# 5. Calibration K_alpha
# ---------------------------------------------------------------------------
def test_calibration_K_alpha_targets_0p4():
    """α_TQNT(J/ψ) après calibration ≈ 0.4 (Cornell phéno)."""
    cal = calibrate_cornell_constants(mode=MODE_UNIVERSAL)
    assert abs(cal["alpha_TQNT_J_psi"] - 0.4) < 1e-6


# ---------------------------------------------------------------------------
# 6. Calibration K_sigma : Δm(ψ(2S)/J/ψ) ≈ 589 MeV
# ---------------------------------------------------------------------------
def test_calibration_K_sigma_matches_psi2S_J_psi():
    """La dichotomie sur K_sigma doit cibler 589.2 MeV à mieux que 1 MeV."""
    cal = calibrate_cornell_constants(mode=MODE_UNIVERSAL)
    err = abs(cal["dm_calibration_achieved_MeV"]
              - cal["dm_calibration_target_MeV"])
    assert err < 1.0, (
        f"dichotomie K_sigma : Δm = "
        f"{cal['dm_calibration_achieved_MeV']:.2f} vs cible "
        f"{cal['dm_calibration_target_MeV']:.2f} (écart {err:.2f} MeV)"
    )


# ---------------------------------------------------------------------------
# 7. Spectre J/ψ : 3 niveaux croissants
# ---------------------------------------------------------------------------
def test_spectrum_J_psi_increasing():
    """E_1 < E_2 < E_3 dans le spectre Cornell de la famille charmonium."""
    spec = schrodinger_cornell_spectrum("J/psi", mode=MODE_UNIVERSAL,
                                        n_levels=3)
    evs = spec["eigenvalues_MeV"]
    assert evs[0] < evs[1] < evs[2]
    # Spacing E_2 − E_1 = 589 MeV (calibration)
    assert abs((evs[1] - evs[0]) - 589.2) < 2.0


# ---------------------------------------------------------------------------
# 8. Mode universal : 4/4 dans seuil
# ---------------------------------------------------------------------------
def test_mode_universal_closes_quarkonium():
    """Mode universal : tous les splittings dans ±0.3 dex."""
    rows = v5p37b_cornell_table(mode=MODE_UNIVERSAL)
    v = verdict_v5p37b(rows, threshold_dex=0.3)
    assert v["n_within_threshold"] == 4
    assert v["verdict"] == "REPRODUIT"
    assert v["sum_abs_log"] < 0.5  # marge confortable


# ---------------------------------------------------------------------------
# 9. Mode topo : diagnostic (ne ferme pas)
# ---------------------------------------------------------------------------
def test_mode_topo_diagnostic_does_not_close():
    """Mode topo : reproduit charmonium mais pas bottomonium."""
    rows = v5p37b_cornell_table(mode=MODE_TOPO)
    v = verdict_v5p37b(rows, threshold_dex=0.3)
    # Mode topo doit donner < 4 paires dans seuil
    assert v["n_within_threshold"] < 4, (
        "Mode topo ne devrait pas tout fermer ; "
        "si tout passe, vérifier formules"
    )
    # Mais charmonium (paire 0) doit passer par construction (calibration)
    log_charm_21 = rows[0]["log_ratio"]
    assert abs(log_charm_21) < 0.05


# ---------------------------------------------------------------------------
# 10. Amélioration stricte sur V5.37
# ---------------------------------------------------------------------------
def test_v5p37b_strictly_improves_v5p37():
    """V5.37b (universal) : Σ|log| < V5.37 (0.76 dex rapporté commit 765f34d)."""
    rows = v5p37b_cornell_table(mode=MODE_UNIVERSAL)
    v = verdict_v5p37b(rows)
    sum_log_v5p37 = 0.76
    assert v["sum_abs_log"] < sum_log_v5p37, (
        f"V5.37b Σ|log| = {v['sum_abs_log']:.3f} doit être < "
        f"V5.37 = {sum_log_v5p37}"
    )
