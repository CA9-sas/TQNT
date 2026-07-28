"""
V5.5 — Tests de validation croisée α = 1.1991 (dérivé) vs α = 1.20 (V5 calibré).

Ces tests garantissent que le passage du paramètre calibré α=1.20 vers
la valeur structurelle α=1.1991, dérivée sans paramètre libre depuis
SU(2)_135, ne casse PAS la phénoménologie V5.3 :

1. Ratio Ω_DM/Ω_b reste dans 2σ Planck (5.45 ± 0.28)
2. Bipartition LZ-safe / LZ-excluded (53 / 64) préservée (structurelle)

Source : docs/TQNT_detaillee_V5_FR.tex §13.5 + docs/TQNT_catalog_V5_diffusion.xlsx
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.V5_5_alpha_derivation.abundance_solver import omega_ratio
from scripts.V5_5_alpha_derivation.validate_alpha_against_scan import (
    V5_ALPHA_CAL,
    V5P5_ALPHA,
    PLANCK_RATIO,
    PLANCK_SIGMA,
    load_catalog,
    bipartition,
    sigma_dm_n,
)


def test_omega_ratio_alpha_V5p5_in_2sigma_planck():
    """Avec α = 1.1991 (V5.5 structurel), Ω_DM/Ω_b doit rester dans la
    bande Planck 2σ (5.45 ± 0.56)."""
    ratio, _ = omega_ratio(V5P5_ALPHA)
    tension = abs(ratio - PLANCK_RATIO) / PLANCK_SIGMA
    # Critère V5.5 : doit rester dans 2σ Planck
    assert tension < 2.0, (
        f"α=1.1991 → Ω_DM/Ω_b = {ratio:.3f}, tension {tension:.2f}σ "
        f"(Planck 5.45±0.28). Hors 2σ."
    )
    # Information supplémentaire : ratio doit être de l'ordre de 5-6
    assert 5.0 < ratio < 6.5, f"Ratio aberrant : {ratio}"


def test_omega_ratio_drift_V5_to_V5p5_small():
    """Le passage α=1.20 → α=1.1991 ne doit pas faire diverger le ratio
    de plus de ~5 % (sensibilité locale ∂ratio/∂α)."""
    r_v5, _ = omega_ratio(V5_ALPHA_CAL)
    r_v55, _ = omega_ratio(V5P5_ALPHA)
    drift_pct = abs(r_v55 - r_v5) / r_v5 * 100
    assert drift_pct < 5.0, (
        f"Dérive ratio entre α=1.20 ({r_v5:.3f}) et α=1.1991 ({r_v55:.3f}) "
        f"trop grande : {drift_pct:.2f} % (cible < 5 %)"
    )
    # Le ratio croît quand α décroît (moins de pénalité sur c > 18)
    assert r_v55 > r_v5, (
        f"Sens du déplacement inattendu : r(1.1991)={r_v55:.3f} "
        f"<= r(1.20)={r_v5:.3f}"
    )


def test_bipartition_53_64_preserved():
    """La bipartition LZ-safe (53) / LZ-excluded (64) est STRUCTURELLE :
    elle vient de la clause Q_phys = 0 + critère Vassiliev sur c, pas de α.
    Elle doit donc être strictement préservée par tout changement de α."""
    catalog = load_catalog()
    assert len(catalog) == 117, f"Catalogue attendu : 117, trouvé {len(catalog)}"
    n_safe, n_excl = bipartition(catalog)
    assert n_safe == 53, f"LZ-safe attendu 53, trouvé {n_safe}"
    assert n_excl == 64, f"LZ-excluded attendu 64, trouvé {n_excl}"
    assert n_safe + n_excl == 117


def test_sigma_dm_n_independent_of_alpha():
    """σ_DM-N(c) = σ_EM × α_EM^(c-18) ne dépend QUE de c, pas de α de
    condensation. Cette propriété est centrale pour la validation : le
    verdict LZ ne change pas avec α=1.1991."""
    # σ à c=18 = σ_EM (référence)
    s18 = sigma_dm_n(18)
    s22 = sigma_dm_n(22)
    s28 = sigma_dm_n(28)
    # Décroissance exponentielle par α_EM = 1/137 par crossing au-delà de 18
    assert s22 < s18, f"σ(22) doit être < σ(18) : {s22} vs {s18}"
    assert s28 < s22
    # Ratio attendu : σ(c+4)/σ(c) = α_EM^4 ≈ (1/137)^4 ≈ 2.8e-9
    expected_ratio = (1.0 / 137.0) ** 4
    actual_ratio = s22 / s18
    assert abs(actual_ratio / expected_ratio - 1) < 1e-6


def test_alpha_V5p5_value_consistent():
    """La constante V5P5_ALPHA = 1.1991 doit reproduire la dérivation
    canonique log(d^3/N_HT_eff) + sqrt(h) à <0.1 % près."""
    from scripts.V5_5_alpha_derivation.alpha_refinement import alpha_V5p5_step3
    alpha, _ = alpha_V5p5_step3()
    assert abs(alpha - V5P5_ALPHA) < 1e-3, (
        f"V5P5_ALPHA={V5P5_ALPHA} ne reproduit pas le calcul "
        f"V5.5 étape 3 (= {alpha:.4f})."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
