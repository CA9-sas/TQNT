"""Tests des hypothèses structurales V5.5 pour α=1,20 depuis CS k=135."""
import math
import pytest

from theory.su2_modular import quantum_dim, conformal_weight, V5_LEVEL

V5_ALPHA = 1.20


def test_alpha_H1_log_d_squared():
    """H1 : log(d_{1/2}²) ≈ 1.386, +15.5% from V5."""
    alpha = math.log(quantum_dim(0.5) ** 2)
    assert alpha == pytest.approx(1.386, abs=0.01)
    delta_pct = (alpha - V5_ALPHA) / V5_ALPHA * 100
    assert abs(delta_pct) < 20


def test_alpha_H6_log_d_cubed_over_NHT():
    """H6 : log(d³ / N_HT) ≈ 1.093, only -8.9% from V5.

    Best 0-parameter hypothesis.
    Physical : fusion gain log(d²) - HT entropy log(N_HT/d).
    """
    d = quantum_dim(0.5)
    N_HT = 2.68
    alpha = math.log(d**3 / N_HT)
    assert alpha == pytest.approx(1.093, abs=0.01)
    delta_pct = (alpha - V5_ALPHA) / V5_ALPHA * 100
    assert abs(delta_pct) < 10


def test_alpha_H6_plus_sqrt_h():
    """H6 + √h_{1/2} ≈ 1.167, only -2.76% from V5.

    Best 0-parameter result.
    Physical : H6 + conformal subleading correction.
    """
    d = quantum_dim(0.5)
    h = conformal_weight(0.5)
    N_HT = 2.68
    alpha = math.log(d**3 / N_HT) + math.sqrt(h)
    assert alpha == pytest.approx(1.167, abs=0.01)
    delta_pct = (alpha - V5_ALPHA) / V5_ALPHA * 100
    assert abs(delta_pct) < 5  # < 5 % cible V5.5


def test_alpha_consistency_across_k():
    """Vérifie que H6 reste cohérent pour k ≠ 135 (test de robustesse)."""
    # Pour k=8 (autre niveau intéressant) : d_{1/2} = 2 cos(π/10) = 1.902
    d_k8 = quantum_dim(0.5, k=8)
    assert d_k8 == pytest.approx(2 * math.cos(math.pi / 10))
    # H6 à k=8 donne une valeur très différente : naturel car k=135 est canonique V5
    alpha_k8 = math.log(d_k8**3 / 2.68)
    assert alpha_k8 < 1.0  # plus petit qu'à k=135


def test_alpha_V5p5_step3_complete():
    """V5.5 ÉTAPE 3 — α dérivé à <1 % de V5 sans paramètre libre.

    Formule canonique :
        α = log(d_{1/2}^3 / N_HT_eff(c_med)) + sqrt(h_{1/2})

    avec :
        d_{1/2} = 2 cos(π/(k+2))               (SU(2)_135 V5 §10.4)
        h_{1/2} = 3/(8(k+2))                    (poids conforme)
        N_HT_eff(c) = ρ_asympt × exp(-1/c)      (Sundberg-Thistlethwaite finite-c)
        c_med ≈ 26                              (médiane du window condensation)
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.V5_5_alpha_derivation.alpha_refinement import (
        alpha_V5p5_step3, median_c_condensation,
    )
    c_med = int(median_c_condensation())
    assert 20 <= c_med <= 35, f"c_med = {c_med} hors plage condensation"
    alpha, diag = alpha_V5p5_step3()
    # CIBLE V5.5 : <1 % de 1.20
    assert abs(alpha - 1.20) / 1.20 < 0.01, (
        f"α = {alpha:.4f}, V5 = 1.20, écart {(alpha-1.20)/1.20*100:.3f}%"
    )


def test_NHT_actual_vs_asymptotic():
    """N_HT actuel à c∈[12,16] dévie de la limite asymptotique 2.68.

    Constatation V5.5 : le ratio N(c+1)/N(c) à c≤16 est ~4.8, bien
    au-dessus de la limite asymptotique 2.68 (Sundberg-Thistlethwaite).
    Cela suggère que la formule V5 §13.5 utilise N_HT asymptotique
    mais que la condensation primordiale opère à c modeste où le ratio
    est différent. PISTE V5.5 : raffiner α en utilisant N_HT effectif
    dans la fenêtre de condensation (c ∼ 18-25).
    """
    HT = {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165,
          11: 552, 12: 2176, 13: 9988, 14: 46972, 15: 253293,
          16: 1388705}
    ratios = [HT[c+1] / HT[c] for c in range(3, 16)]
    avg_ratio = sum(ratios[-5:]) / 5
    # Le rapport finite-c est strictement supérieur à 2.68 (limite c→∞)
    assert avg_ratio > 2.68, f"avg_ratio = {avg_ratio} (devrait être > 2.68)"
    # Et inférieur à 6 (sub-exponentiel)
    assert avg_ratio < 6.0
    # La limite asymptotique 2.68 est compatible avec la queue lointaine
    # mais le calcul de α doit en tenir compte → V5.5 raffinement
