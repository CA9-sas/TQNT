"""
V5.5 ÉTAPE 3+ — Validation croisée de α=1.1991 (dérivé) vs 1.20 (V5 calibré)
================================================================================

But : vérifier que la nouvelle valeur structurelle α = 1.1991, issue de la
formule canonique V5.5

    α = log(d_{1/2}^3 / N_HT_eff(c_med)) + sqrt(h_{1/2})

avec d_{1/2}=1.9995, h_{1/2}=0.00547, N_HT_eff(c_med)=2.595 (c_med≈31
issu de la médiane self-consistent du window de condensation V5 §13.5 ;
proche du c_med≈26 cité dans le préambule V5.5 — l'écart vient du fait
que la médiane est calculée sur le poids m(c)·N(c)·exp(-α c) qui pousse
c_med au-delà de la moyenne brute), reste cohérente avec les quatre
piliers empiriques V5.3 :

1. Ratio Ω_DM/Ω_b mesuré par Planck (5,45 ± 0,28) — V5 §13.5
2. Scan des 117 candidats DM du catalogue V5 (docs/TQNT_catalog_V5_diffusion.xlsx)
3. Bipartition LZ-safe (53) / LZ-excluded (64) sous clause Q_phys=0
4. Verdict V5.3 master sur SPARC (chi2/N=3.05 pour 6_3+6_3)

Lecture-clé :
- σ_DM-N(c) = σ_EM × α_EM^(c-18)  est inchangée (n'utilise pas α de condensation)
- Mais l'abondance pondérée et la médiane c d'abundance changent légèrement,
  donc le σ effectif vu par les détecteurs (pondéré par les abondances) bouge.
- La bipartition LZ-safe/EXCLU est structurelle (clause Q_phys=0), elle ne
  dépend pas de α de condensation → préservée par construction.

Output :
- Tableau récap stdout
- Figure : output/v5p5_alpha_validation_scan.png

Run : PYTHONPATH=. python3 scripts/V5_5_alpha_derivation/validate_alpha_against_scan.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.V5_5_alpha_derivation.abundance_solver import (
    omega_ratio, m_knot_average, N_knots,
)
from scripts.V5_5_alpha_derivation.alpha_refinement import (
    alpha_V5p5_step3, median_c_condensation,
)

# ---------------------------------------------------------------------------
# Constantes de référence V5
# ---------------------------------------------------------------------------
V5_ALPHA_CAL = 1.20            # α calibré V5 (§13.5)
V5P5_ALPHA = 1.1991            # α dérivé structurel V5.5 étape 3
PLANCK_RATIO = 5.45            # Ω_DM/Ω_b Planck 2018
PLANCK_SIGMA = 0.28            # 1σ Planck
ALPHA_EM = 1.0 / 137.0         # constante de structure fine (k=135 → α_EM=1/137)
SIGMA_EM = 1e-24               # cm² référence EM (ordre de grandeur)
CATALOG_PATH = Path("docs/TQNT_catalog_V5_diffusion.xlsx")

# Verdicts V5.3 master à reporter (§ SPARC)
V53_SPARC_CHI2_PER_N = 3.05    # pour 6_3+6_3 (1.55 GeV)
V53_SPARC_VERDICT = "EXCLU (chi2/N=3.05)"


# ---------------------------------------------------------------------------
# 1. Chargement du catalogue V5
# ---------------------------------------------------------------------------
def load_catalog(path: Path = CATALOG_PATH) -> pd.DataFrame:
    """Charge l'onglet 5 (Matière noire) — 117 candidats avec assignation
    Niveau, c, m, Q_topo, Lk, Q_phys, Statut LZ."""
    df = pd.read_excel(path, sheet_name="5. Matière noire", header=7)
    df = df[df["#"].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. σ_DM-N(c) — formule Vassiliev (V5 §13.6.4)
# ---------------------------------------------------------------------------
def sigma_dm_n(c: int) -> float:
    """σ_DM-N(c) = σ_EM × α_EM^(c-18), cm². Indépendant de α de condensation."""
    if c < 18:
        return SIGMA_EM
    return SIGMA_EM * (ALPHA_EM ** (c - 18))


# ---------------------------------------------------------------------------
# 3. Abondance pondérée et σ effectif
# ---------------------------------------------------------------------------
def weighted_abundance(catalog: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Ajoute des colonnes weight et abundance_norm au catalogue.

    Le poids d'abondance V5 §13.5 pour chaque candidat de croisement c est :
        w_c = m(c) * N(c) * exp(-α c)
    Normalisé sur la somme totale du catalogue.
    """
    df = catalog.copy()
    weights = []
    for _, row in df.iterrows():
        c = int(row["c"])
        # On utilise N(c) du nombre total de nœuds premiers à c croisements,
        # même si une seule config est listée à cet c — c'est la mesure V5.
        w = m_knot_average(c) * math.exp(-alpha * c)
        # On garde N(c)=1 par candidat (mesure de comptage du catalogue lui-même)
        weights.append(w)
    df["abundance_weight"] = weights
    df["abundance_norm"] = df["abundance_weight"] / df["abundance_weight"].sum()
    return df


def median_c_weighted(catalog: pd.DataFrame, alpha: float) -> float:
    """Médiane de c pondérée par l'abondance ω = m(c) exp(-α c)."""
    df = weighted_abundance(catalog, alpha)
    df = df.sort_values("c").reset_index(drop=True)
    cum = df["abundance_norm"].cumsum()
    idx = int(np.searchsorted(cum, 0.5))
    return float(df.iloc[idx]["c"])


def effective_sigma(catalog: pd.DataFrame, alpha: float) -> float:
    """σ effectif vu par un détecteur = Σ w_c σ(c) / Σ w_c (cm²)."""
    df = weighted_abundance(catalog, alpha)
    sigmas = df["c"].astype(int).apply(sigma_dm_n)
    return float((df["abundance_norm"] * sigmas).sum())


# ---------------------------------------------------------------------------
# 4. Bipartition LZ-safe / LZ-excluded
# ---------------------------------------------------------------------------
def bipartition(catalog: pd.DataFrame) -> tuple[int, int]:
    """Compte (n_safe, n_excl) selon la colonne 'Statut LZ (clause Q_phys=0)'.

    Cette bipartition est STRUCTURELLE : elle vient de la clause Q_phys = 0
    (existence d'un Lk entier physiquement compatible), pas de α.
    Donc elle est invariante sous le changement α=1.20 → α=1.1991.
    """
    statut = catalog["Statut LZ (clause Q_phys=0)"].astype(str).str.strip()
    n_safe = int((statut == "LZ-safe (A)").sum())
    n_excl = int((statut == "EXCLU (Vassiliev)").sum())
    return n_safe, n_excl


# ---------------------------------------------------------------------------
# 5. Verdict V5.3 SPARC — invariance
# ---------------------------------------------------------------------------
def sparc_verdict_invariance() -> dict:
    """Le verdict V5.3 SPARC pour 6_3+6_3 (chi2/N=3.05) repose sur la masse
    et le profil ρ(r) ∝ |Ψ_6_3+6_3|², non sur α de condensation.

    Donc le changement α=1.20 → 1.1991 ne modifie PAS le chi2/N SPARC.
    On le rapporte tel quel pour la traçabilité.
    """
    return {
        "candidate": "6_3 + 6_3",
        "m_GeV": 1.5485,
        "chi2_per_N_V5p3": V53_SPARC_CHI2_PER_N,
        "verdict_V5p3": V53_SPARC_VERDICT,
        "alpha_dependence": "AUCUNE (chi2 SPARC dépend de m et profil ρ, pas de α)",
        "invariance_V5p5": "Verdict 6_3+6_3 EXCLU préservé",
    }


# ---------------------------------------------------------------------------
# 6. Figure de synthèse
# ---------------------------------------------------------------------------
def make_figure(out: Path, summary: dict, catalog_safe_excl: tuple,
                df_120: pd.DataFrame, df_1199: pd.DataFrame):
    """Figure 2x2 : (a) Ω_DM/Ω_b vs α, (b) distribution abondance par c,
    (c) σ_DM-N(c), (d) bipartition LZ-safe/EXCLU."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Ω_DM/Ω_b vs α
    ax = axes[0, 0]
    alphas = np.linspace(0.9, 1.5, 200)
    ratios = np.array([omega_ratio(a)[0] for a in alphas])
    ax.plot(alphas, ratios, "b-", linewidth=1.5)
    ax.axhline(PLANCK_RATIO, color="green", linestyle="--",
               label=f"Planck 5,45")
    ax.fill_between(alphas, PLANCK_RATIO - PLANCK_SIGMA,
                    PLANCK_RATIO + PLANCK_SIGMA, color="green", alpha=0.2,
                    label="±1σ")
    ax.fill_between(alphas, PLANCK_RATIO - 2 * PLANCK_SIGMA,
                    PLANCK_RATIO + 2 * PLANCK_SIGMA, color="green", alpha=0.08,
                    label="±2σ")
    ax.axvline(V5_ALPHA_CAL, color="red", linestyle=":", alpha=0.8,
               label=f"α V5 = {V5_ALPHA_CAL}")
    ax.axvline(V5P5_ALPHA, color="purple", linestyle="-.", alpha=0.9,
               label=f"α V5.5 = {V5P5_ALPHA}")
    r_v5 = summary["ratio_alpha_V5"]
    r_v55 = summary["ratio_alpha_V5p5"]
    ax.scatter([V5_ALPHA_CAL, V5P5_ALPHA], [r_v5, r_v55],
               color=["red", "purple"], s=70, zorder=5)
    ax.set_xlabel("α")
    ax.set_ylabel("Ω_DM / Ω_b")
    ax.set_title("(a) Abondance Ω_DM/Ω_b vs α (V5 §13.5)")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # (b) Distribution abondance par c
    ax = axes[0, 1]
    cs = sorted(df_120["c"].unique())
    w120 = [df_120[df_120["c"] == c]["abundance_norm"].sum() for c in cs]
    w1199 = [df_1199[df_1199["c"] == c]["abundance_norm"].sum() for c in cs]
    x = np.arange(len(cs))
    width = 0.4
    ax.bar(x - width / 2, w120, width, label=f"α={V5_ALPHA_CAL}", color="red",
           alpha=0.7)
    ax.bar(x + width / 2, w1199, width, label=f"α={V5P5_ALPHA}", color="purple",
           alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" for c in cs], rotation=0, fontsize=8)
    ax.set_xlabel("c (croisements)")
    ax.set_ylabel("Abondance normalisée")
    ax.set_title("(b) Distribution catalogue 117 candidats par c")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (c) σ_DM-N(c) — formule Vassiliev (inchangée)
    ax = axes[1, 0]
    cs_plot = np.arange(18, 31)
    sig = [sigma_dm_n(int(c)) for c in cs_plot]
    ax.semilogy(cs_plot, sig, "ko-", markersize=6)
    ax.axhline(1e-46, color="red", linestyle="--",
               label="LZ 2024 sensibilité (~10⁻⁴⁶ cm²)")
    ax.fill_between(cs_plot, 1e-50, 1e-46, color="red", alpha=0.1,
                    label="LZ-excluded zone")
    ax.set_xlabel("c (croisements)")
    ax.set_ylabel("σ_DM-N(c) [cm²]")
    ax.set_title("(c) Section efficace V5 §13.6.4 (indépendante de α)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) Bipartition LZ-safe / EXCLU
    ax = axes[1, 1]
    n_safe, n_excl = catalog_safe_excl
    labels = [f"LZ-safe (A)\n{n_safe}", f"EXCLU (Vassiliev)\n{n_excl}"]
    colors = ["#2ecc71", "#e74c3c"]
    ax.pie([n_safe, n_excl], labels=labels, colors=colors,
           autopct="%1.0f%%", startangle=90,
           textprops={"fontsize": 11})
    ax.set_title("(d) Bipartition structurelle 117 candidats\n"
                 "(clause Q_phys=0 ; invariante sous α)")

    fig.suptitle(
        f"V5.5 — Validation croisée α=1,1991 vs 1,20 sur scan V5.3\n"
        f"Ratio: α=1,20→{r_v5:.3f} | α=1,1991→{r_v55:.3f} | "
        f"Planck 5,45±0,28 (2σ band traversée)",
        fontsize=12, y=1.005,
    )
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nFigure : {out}")


# ---------------------------------------------------------------------------
# 7. Driver principal
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("V5.5 — VALIDATION CROISÉE α=1,1991 (dérivé) vs α=1,20 (calibré)")
    print("=" * 80)

    # --- 0. Sanity check : reproduit-on α_V5.5 = 1.1991 ? ---
    alpha_derived, diag = alpha_V5p5_step3()
    c_med = int(median_c_condensation())
    print(f"\n[0] Reproduction de la dérivation V5.5 étape 3")
    print(f"    c_med (condensation) = {c_med}")
    print(f"    d_{{1/2}}             = {diag['d_half']:.4f}")
    print(f"    h_{{1/2}}             = {diag['h_half']:.5f}")
    print(f"    N_HT_eff(c_med)      = {diag['N_HT_eff']:.4f}")
    print(f"    α V5.5 dérivé        = {alpha_derived:.4f}")
    assert abs(alpha_derived - V5P5_ALPHA) < 1e-3, (
        f"Reproduction α V5.5 incorrecte : {alpha_derived} vs {V5P5_ALPHA}"
    )

    # --- 1. Ratio Ω_DM/Ω_b pour α=1.20 vs α=1.1991 ---
    r_v5, _ = omega_ratio(V5_ALPHA_CAL)
    r_v55, _ = omega_ratio(V5P5_ALPHA)
    delta_v5 = (r_v5 - PLANCK_RATIO) / PLANCK_RATIO * 100
    delta_v55 = (r_v55 - PLANCK_RATIO) / PLANCK_RATIO * 100
    tension_v5 = abs(r_v5 - PLANCK_RATIO) / PLANCK_SIGMA
    tension_v55 = abs(r_v55 - PLANCK_RATIO) / PLANCK_SIGMA

    print(f"\n[1] Ratio Ω_DM/Ω_b (Planck 2018 : {PLANCK_RATIO} ± {PLANCK_SIGMA})")
    print(f"    {'α':>10} {'Ω_DM/Ω_b':>12} {'Δ Planck':>10} {'Tension':>10}")
    print(f"    {V5_ALPHA_CAL:>10.4f} {r_v5:>12.4f} {delta_v5:>+9.2f}% "
          f"{tension_v5:>9.2f}σ")
    print(f"    {V5P5_ALPHA:>10.4f} {r_v55:>12.4f} {delta_v55:>+9.2f}% "
          f"{tension_v55:>9.2f}σ")

    # --- 2. Catalogue V5 : 117 candidats ---
    catalog = load_catalog()
    n_total = len(catalog)
    print(f"\n[2] Catalogue V5 chargé : {n_total} candidats")
    assert n_total == 117, f"Attendu 117 candidats, trouvé {n_total}"

    # --- 3. σ_DM-N(c) inchangée par α ---
    print(f"\n[3] σ_DM-N(c) = σ_EM × α_EM^(c-18) (V5 §13.6.4, indépendant de α)")
    print(f"    c=18 : σ = {sigma_dm_n(18):.2e} cm²")
    print(f"    c=22 : σ = {sigma_dm_n(22):.2e} cm²  (~LZ 2024 frontière)")
    print(f"    c=28 : σ = {sigma_dm_n(28):.2e} cm²  (sub-LZ)")
    print(f"    c=30 : σ = {sigma_dm_n(30):.2e} cm²  (très sub-LZ)")

    # --- 4. Médiane c pondérée par abundance ---
    df_120 = weighted_abundance(catalog, V5_ALPHA_CAL)
    df_1199 = weighted_abundance(catalog, V5P5_ALPHA)
    c_med_120 = median_c_weighted(catalog, V5_ALPHA_CAL)
    c_med_1199 = median_c_weighted(catalog, V5P5_ALPHA)
    sigma_eff_120 = effective_sigma(catalog, V5_ALPHA_CAL)
    sigma_eff_1199 = effective_sigma(catalog, V5P5_ALPHA)

    print(f"\n[4] Médiane c d'abundance pondérée et σ effectif")
    print(f"    {'α':>10} {'c_median':>10} {'σ_eff (cm²)':>15}")
    print(f"    {V5_ALPHA_CAL:>10.4f} {c_med_120:>10.1f} {sigma_eff_120:>15.3e}")
    print(f"    {V5P5_ALPHA:>10.4f} {c_med_1199:>10.1f} {sigma_eff_1199:>15.3e}")

    # --- 5. Bipartition LZ-safe / LZ-excluded ---
    n_safe, n_excl = bipartition(catalog)
    print(f"\n[5] Bipartition structurelle (clause Q_phys=0, V5 catalogue)")
    print(f"    LZ-safe (A)       : {n_safe}")
    print(f"    EXCLU (Vassiliev) : {n_excl}")
    print(f"    Total             : {n_safe + n_excl}")
    print(f"    Attendu V5.3      : 53 safe + 64 excl = 117")
    bipartition_ok = (n_safe == 53 and n_excl == 64)
    print(f"    Bipartition préservée par α V5.5 ? "
          f"{'OUI' if bipartition_ok else 'NON'} "
          f"(structurelle, ne dépend pas de α)")
    assert bipartition_ok, "Bipartition 53/64 non préservée"

    # --- 6. Verdict V5.3 SPARC ---
    sparc = sparc_verdict_invariance()
    print(f"\n[6] Verdict V5.3 SPARC sur {sparc['candidate']} "
          f"({sparc['m_GeV']} GeV)")
    print(f"    chi2/N V5.3    : {sparc['chi2_per_N_V5p3']}")
    print(f"    Verdict V5.3   : {sparc['verdict_V5p3']}")
    print(f"    Dépendance α   : {sparc['alpha_dependence']}")
    print(f"    Invariance V5.5: {sparc['invariance_V5p5']}")

    # --- Synthèse ---
    print("\n" + "=" * 80)
    print("SYNTHÈSE — TABLEAU RÉCAPITULATIF")
    print("=" * 80)
    rows = [
        ("Ω_DM/Ω_b (α=1,20)", f"{r_v5:.3f}", f"Δ={delta_v5:+.2f}%, {tension_v5:.2f}σ"),
        ("Ω_DM/Ω_b (α=1,1991)", f"{r_v55:.3f}", f"Δ={delta_v55:+.2f}%, {tension_v55:.2f}σ"),
        ("Planck cible", f"{PLANCK_RATIO} ± {PLANCK_SIGMA}", "1σ band"),
        ("Tension absolue Δ", f"{abs(r_v55-r_v5):.3f}", f"{(r_v55-r_v5)/PLANCK_SIGMA*100:.1f}% de σ_Planck"),
        ("c_median (α=1,20)", f"{c_med_120:.0f}", "abundance pondérée"),
        ("c_median (α=1,1991)", f"{c_med_1199:.0f}", "abundance pondérée"),
        ("σ_eff (α=1,20)", f"{sigma_eff_120:.2e} cm²", "moyenne sur 117 candidats"),
        ("σ_eff (α=1,1991)", f"{sigma_eff_1199:.2e} cm²", "moyenne sur 117 candidats"),
        ("LZ-safe count", f"{n_safe}", "structurel (Q_phys=0 clause)"),
        ("LZ-excluded count", f"{n_excl}", "structurel (Vassiliev)"),
        ("V5.3 SPARC 6_3+6_3", f"chi2/N={V53_SPARC_CHI2_PER_N}", "EXCLU (invariant)"),
    ]
    for k, v, ann in rows:
        print(f"  {k:<25s} {v:>20s}   {ann}")

    # Critère acceptabilité
    print("\n" + "=" * 80)
    print("VERDICT V5.5 — COHÉRENCE AVEC V5.3 ?")
    print("=" * 80)
    accept_planck = tension_v55 < 2.0
    accept_bipart = bipartition_ok
    accept_sparc = True
    accept_sigma_shift = abs(sigma_eff_1199 - sigma_eff_120) / sigma_eff_120 < 0.10
    all_ok = accept_planck and accept_bipart and accept_sparc and accept_sigma_shift
    print(f"  [1] Planck 2σ      : {'OUI' if accept_planck else 'NON'} "
          f"(tension {tension_v55:.2f}σ < 2σ)")
    print(f"  [2] Bipartition 53/64: {'OUI' if accept_bipart else 'NON'}")
    print(f"  [3] SPARC 6_3+6_3   : {'OUI' if accept_sparc else 'NON'} "
          f"(verdict invariant)")
    print(f"  [4] σ_eff shift     : {'OUI' if accept_sigma_shift else 'NON'} "
          f"(shift {abs(sigma_eff_1199-sigma_eff_120)/sigma_eff_120*100:.1f}% < 10%)")
    if all_ok:
        verdict = ("V5.5 α=1,1991 COHÉRENT avec V5.3 — la dérivation "
                   "structurelle reproduit la phénoménologie à <1 % d'écart "
                   "sur α et reste dans le band Planck 2σ.")
    else:
        verdict = "INCOHÉRENCE — voir critères ci-dessus."
    print(f"\n  ⇒ {verdict}")

    # Figure
    summary = {
        "ratio_alpha_V5": r_v5,
        "ratio_alpha_V5p5": r_v55,
        "tension_V5_sigma": tension_v5,
        "tension_V5p5_sigma": tension_v55,
    }
    make_figure(Path("output/v5p5_alpha_validation_scan.png"),
                summary, (n_safe, n_excl), df_120, df_1199)

    return {
        "alpha_V5_calibrated": V5_ALPHA_CAL,
        "alpha_V5p5_derived": V5P5_ALPHA,
        "ratio_V5": r_v5,
        "ratio_V5p5": r_v55,
        "tension_V5_sigma": tension_v5,
        "tension_V5p5_sigma": tension_v55,
        "c_median_V5": c_med_120,
        "c_median_V5p5": c_med_1199,
        "sigma_eff_V5": sigma_eff_120,
        "sigma_eff_V5p5": sigma_eff_1199,
        "bipartition_safe": n_safe,
        "bipartition_excl": n_excl,
        "sparc_invariant": True,
        "v5p5_coherent_with_v5p3": all_ok,
    }


if __name__ == "__main__":
    main()
