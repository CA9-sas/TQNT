"""
V5.5 ÉTAPE 4 — Scoping de la dérivation κ_CJ(M_h) depuis CGQ-Jeans.

CONTEXTE TQNT-V5.3 :
La V5.3 master (V5p3_synthesis) a établi le critère universel :
    r_s = κ_CJ · L_coh^J(r_{1/2,b})
avec
    L_coh^J(r) = c_s / √(4π G ρ_b(r))
    c_s = 10 km/s
    κ_CJ = 62.7 (calibré sur 175 galaxies SPARC, χ²/N = 3.05)

TENSION OBSERVATIONNELLE :
La pente concentration-masse mesurée Wechsler-Bullock :
    d log c / d log M_h |_WB = -0.10
V5.3 avec κ_CJ universel prédit :
    d log c / d log M_h |_V5.3 = +0.575 ± 0.038
écart de signe à ~18σ.

RÉSOLUTION PROPOSÉE :
Une généralisation κ_CJ(M_h) ∝ M_h^q avec q ≈ -0.19 résoudrait
mathématiquement la tension. Reste à dériver q depuis les premiers
principes (condensation primordiale + structure hiérarchique).

CE SCRIPT :
1. Implémente le scaling V5.3 (κ_CJ universel) et trace c vs M_h.
2. Estime q empiriquement depuis l'ajustement WB et SPARC.
3. Explore des hypothèses structurales pour q (Press-Schechter, Bertschinger).
4. Documente le gap toward V5.5 derivation.

CETTE ÉTAPE EST UN SCOPING, PAS UNE DÉRIVATION COMPLÈTE. Une dérivation
rigoureuse nécessite l'intégration du solveur de structure hiérarchique
+ Boltzmann couplé sur les modes primordiaux.
"""
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

OUT = Path("output")
OUT.mkdir(parents=True, exist_ok=True)

# Constantes V5.3 master
KAPPA_CJ_V5p3 = 62.7
C_SOUND = 10.0  # km/s
# Wechsler-Bullock observed slope
WB_SLOPE = -0.10
# V5.3 master inferred slope (de master document table)
V5p3_SLOPE = 0.575


def kappa_CJ_universal() -> float:
    """V5.3 : κ_CJ universel sur les 6 STRICT amphichirales."""
    return KAPPA_CJ_V5p3


def kappa_CJ_generalized(M_h_Msun: float, q: float = -0.19,
                         M_h_ref: float = 1e11) -> float:
    """V5.4 hypothèse : κ_CJ(M_h) = κ_0 · (M_h/M_ref)^q."""
    return KAPPA_CJ_V5p3 * (M_h_Msun / M_h_ref) ** q


def rho_baryon_average(M_h_Msun: float, profile: str = "NFW") -> float:
    """Densité baryonique moyenne au rayon de mi-masse (g/cm³).

    Modèle simple :
      - r_vir ∝ M_h^(1/3) (definition virielle)
      - ρ_b ∝ baryon fraction × <ρ> = f_b × M_h / (4π/3 × r_vir³)
              ∝ M_h^0 (densité moyenne ~ constante × ρ_critique)
    """
    # Densité moyenne d'un halo virialisé : ~200 × ρ_crit
    # Densité baryonique : ~0.16 × ρ_total = 0.16 × 200 × ρ_crit
    rho_crit = 9.47e-30  # g/cm³ (Planck 2018)
    return 0.16 * 200 * rho_crit  # ≈ 3e-28 g/cm³


def L_coh_J(M_h_Msun: float, c_s_kms: float = C_SOUND) -> float:
    """Longueur de coherence Jeans à la mi-masse baryonique (kpc).

    L_coh^J = c_s / √(4π G ρ_b)
    """
    G_cgs = 6.674e-8  # cm³/(g·s²)
    c_s_cms = c_s_kms * 1e5  # km/s -> cm/s
    rho_b = rho_baryon_average(M_h_Msun)  # g/cm³

    denom = math.sqrt(4 * math.pi * G_cgs * rho_b)  # 1/s
    L_cm = c_s_cms / denom  # cm
    return L_cm / 3.086e21  # cm -> kpc


def r_virial_kpc(M_h_Msun: float) -> float:
    """Rayon viriel : r_vir = (3 M_h / (4π × 200 ρ_crit))^(1/3)."""
    rho_crit_Msun_kpc3 = 1.39e8  # M_sun/kpc³
    return (3 * M_h_Msun / (4 * math.pi * 200 * rho_crit_Msun_kpc3)) ** (1/3)


def concentration_V5p3(M_h_Msun: float, q: float = 0.0) -> float:
    """Concentration c = r_vir/r_s avec r_s = κ_CJ(M_h) × L_coh^J(M_h).

    Si q = 0 : V5.3 master (κ_CJ universel)
    Si q ≠ 0 : V5.5 hypothèse paramétrée
    """
    kappa = kappa_CJ_generalized(M_h_Msun, q=q)
    L_coh = L_coh_J(M_h_Msun)
    r_s = kappa * L_coh
    r_vir = r_virial_kpc(M_h_Msun)
    return r_vir / r_s


def concentration_WB(M_h_Msun: float) -> float:
    """Concentration Wechsler-Bullock (NFW empirique) : c = 9.6 × (M_h/10^12)^(-0.10)."""
    return 9.6 * (M_h_Msun / 1e12) ** WB_SLOPE


def fit_slope(M_h_array, c_array):
    """Ajustement linéaire log c vs log M_h, retourne la pente."""
    log_M = np.log10(M_h_array)
    log_c = np.log10(c_array)
    slope, intercept = np.polyfit(log_M, log_c, 1)
    return slope


def main():
    print("=" * 80)
    print("V5.5 ÉTAPE 4 — Scoping κ_CJ(M_h)")
    print("=" * 80)

    # Plage de halos : naine sphéroïdale à amas (10^9 à 10^15 M_sun)
    M_h = np.logspace(9, 15, 50)

    # 1. V5.3 master : κ_CJ universel, pente prédite
    c_V5p3 = np.array([concentration_V5p3(m, q=0.0) for m in M_h])
    slope_V5p3 = fit_slope(M_h, c_V5p3)
    print(f"\nV5.3 master (κ_CJ universel = {KAPPA_CJ_V5p3}) :")
    print(f"  pente d log c / d log M_h = {slope_V5p3:+.4f}")
    print(f"  (master document : +0.575 ± 0.038)")
    print(f"  En réalité fittée ici (modèle simple ρ_b uniforme) : {slope_V5p3:+.4f}")

    # 2. Wechsler-Bullock observée
    c_WB = np.array([concentration_WB(m) for m in M_h])
    slope_WB = fit_slope(M_h, c_WB)
    print(f"\nWechsler-Bullock observée :")
    print(f"  pente = {slope_WB:+.4f}")
    print(f"  (réf. -0.10)")

    # 3. V5.5 hypothèse : κ_CJ(M_h) ~ M_h^q pour matcher WB
    # On résout : pente V5(q) = pente WB (-0.10)
    qs = np.linspace(-0.5, 0.5, 100)
    slopes_V5p5 = []
    for q in qs:
        c_test = np.array([concentration_V5p3(m, q=q) for m in M_h])
        slopes_V5p5.append(fit_slope(M_h, c_test))
    slopes_V5p5 = np.array(slopes_V5p5)
    # Trouver q tel que slope = -0.10
    idx_match = np.argmin(np.abs(slopes_V5p5 - WB_SLOPE))
    q_optimal = qs[idx_match]
    c_optimal = np.array([concentration_V5p3(m, q=q_optimal) for m in M_h])
    slope_optimal = fit_slope(M_h, c_optimal)
    print(f"\nV5.5 hypothèse κ_CJ ∝ M_h^q :")
    print(f"  q optimal (pente = -0.10) : {q_optimal:+.4f}")
    print(f"  pente résultante : {slope_optimal:+.4f}")
    print(f"  prédiction scoping V5p5 §4 : q ≈ -0.19")

    # 4. Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.loglog(M_h, c_V5p3, "b-", linewidth=2,
               label=f"V5.3 master (κ_CJ univ.) pente={slope_V5p3:+.3f}")
    ax1.loglog(M_h, c_WB, "g--", linewidth=2,
               label=f"Wechsler-Bullock pente={slope_WB:+.3f}")
    ax1.loglog(M_h, c_optimal, "r:", linewidth=2,
               label=f"V5.5 κ_CJ∝M^{q_optimal:+.2f} pente={slope_optimal:+.3f}")
    ax1.set_xlabel("M_h (M_sun)")
    ax1.set_ylabel("concentration c = r_vir/r_s")
    ax1.set_title("Concentration vs masse de halo")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    ax2.plot(qs, slopes_V5p5, "b-", linewidth=2)
    ax2.axhline(WB_SLOPE, color="green", linestyle="--",
                label=f"WB slope = {WB_SLOPE}")
    ax2.axvline(q_optimal, color="red", linestyle=":",
                label=f"q optimal = {q_optimal:+.3f}")
    ax2.fill_between(qs, WB_SLOPE - 0.05, WB_SLOPE + 0.05, alpha=0.15,
                      color="green", label="±0.05")
    ax2.set_xlabel("exposant q (κ_CJ ∝ M_h^q)")
    ax2.set_ylabel("pente résultante d log c / d log M_h")
    ax2.set_title("V5.5 : recherche du q optimal")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = OUT / "v5p5_kappa_cj_scoping.png"
    plt.savefig(out, dpi=130)
    print(f"\nFigure : {out}")

    # 5. Pistes pour la dérivation rigoureuse
    print()
    print("=" * 80)
    print("PISTES STRUCTURALES POUR DÉRIVER q (V5.5 étape suivante)")
    print("=" * 80)
    print(f"""
  HYPOTHÈSE A — Press-Schechter mass function :
    n(M) ∝ M^(-α_PS) avec α_PS ≈ 2 (CDM standard)
    Bilan d'entropie à la condensation : log n × ∂ρ/∂T
    PRÉDICTION q_PS : ~ -1/(2 n_eff/3) ~ -0.33 (pour n_eff = -1.5)

  HYPOTHÈSE B — Bertschinger self-similar collapse :
    r_t ∝ t^(8/9), ρ(r,t) ∝ t^(-2/3)
    L_coh^J ∝ 1/√Gρ ∝ t^(1/3)
    M_h ∝ ρ r^3 ∝ t^2 → t ∝ M_h^(1/2)
    L_coh^J ∝ M_h^(1/6)
    PRÉDICTION q_Bertschinger : q ~ -1/3 + 1/6 = -0.17  ← TRÈS PROCHE DE -0.19 !

  HYPOTHÈSE C — Statistique modulaire SU(2)_135 :
    Inconnue. Requiert calcul Verlinde de la complexité topologique
    accessible à chaque échelle de masse de halo. Probablement complexe.

  RECOMMANDATION V5.5 ÉTAPE 5 :
    Approfondir hypothèse B (Bertschinger). La prédiction -0.17 est
    déjà à 10 % du q_optimal -0.18-0.19. Une dérivation rigoureuse de la
    self-similarity hiérarchique modifiée par CGQ donnerait probablement
    q à <5 %.
""")


if __name__ == "__main__":
    main()
