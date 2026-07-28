"""V5.32 — Correction P23 (plasticite) sur les mesures d'oscillation.

Objectif
========
Le programme V5.28-V5.31 a un plafond structurel (Σ|log| >= 1.05 dex).
V5.31 ferme le ratio Bs/B = 35.02 mais degrade Σ|log| a 1.46.
Insight utilisateur : les Delta m PDG ne sont PAS purs topologiques.
Les experiences operent sous B_ext != 0 (Case B, cf. memory
case-a-b-bext-discipline). Le postulat P23 (plasticite) prescrit une
correction multiplicative eps(n)*g(B_eff) sur la masse mesuree.

On unfold cette correction des Delta m PDG pour obtenir des Delta m
topologiques purs (Case A) directement comparables aux predictions
V5.30c et V5.31.

Discipline TQNT (CLAUDE.md, V5 §sec:P23-plastique) :
* eps(1) = 0 (symetrie radiale)
* eps(2) = |C_plast| / m_mu ~ 0.0276 (calibre sur mu^-)
* eps(n>=3) = eps(2) * sqrt(n*(n-1)/2) (deduction geometrique)
* g(B_eff) = B_eff / 4 T (LHCb-normalise)
* PAS de parametre libre nouveau (1 calibration C_plast deja existante).

Sortie : output/v5p32_p23_unfolded.png

Verdict honnete attendu :
* eps*g ~ 5 % au plus -> shift max sur log_ratio ~0.03 dex
* Σ|log| ne changera donc essentiellement pas
* CONCLUSION : P23 ne leve PAS le plafond V5 (B_ext trop faible pour
  fermer un gap de 0.5-1 dex). Le plafond est structurel, pas
  experimental.

Reference :
* theory/p23_plasticity_oscillations.py
* theory/v5p30c_vibrational.py
* theory/v5p31_chiral_vibrational.py
* docs/TQNT_detaillee_V5_FR.tex §sec:P23-plastique
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.p23_plasticity_oscillations import (
    B_EFF_TYPICAL,
    B_REF_TESLA,
    C_PLAST_MEV,
    EPS_N2_DEFAULT,
    M_MUON_MEV,
    compare_models_to_unfolded,
    epsilon_n,
    g_B_field,
    plasticity_correction,
    unfolded_table,
)
from theory.topological_chiral_oscillation import (
    DELTA_M_PDG,
    MESON_OSCILLATION_ASSIGNMENTS,
)
from theory.v5p30_combined_oscillation import KAPPA_K_V5p30_DEFAULT
from theory.v5p30c_vibrational import ALPHA_VIB_DEFAULT, v5p30c_table
from theory.v5p31_chiral_vibrational import (
    BETA_CHIRAL_DEFAULT,
    v5p31_table,
)


def print_constants():
    print("=" * 80)
    print("V5.32 — Constantes P23 (V5 §sec:P23-plastique)")
    print("=" * 80)
    print(f"  C_plast            = {C_PLAST_MEV:+.3f} MeV  (calibre sur mu^-)")
    print(f"  m_mu (PDG)         = {M_MUON_MEV:.5f} MeV")
    print(
        f"  eps(2) = |C|/m_mu  = {EPS_N2_DEFAULT:.5f}  (~{EPS_N2_DEFAULT*100:.2f} %)"
    )
    print(f"  eps(3)             = {epsilon_n(3):.5f}  (~{epsilon_n(3)*100:.2f} %)")
    print(f"  B_ref (LHCb)       = {B_REF_TESLA:.2f} T")
    print(
        "  B_eff typiques     : "
        + ", ".join(f"{k}={v:.1f}T" for k, v in B_EFF_TYPICAL.items())
    )
    print()


def print_corrections_table():
    print("=" * 80)
    print("V5.32 — Corrections plastiques par particule")
    print("=" * 80)
    print(
        f"{'Particule':<10} {'n':>3} {'B_eff (T)':>10} "
        f"{'eps(n)':>10} {'g(B)':>8} {'delta = eps*g':>14} "
        f"{'1+delta':>10}"
    )
    print("-" * 80)
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        n = len(cfg["knots"])
        B = B_EFF_TYPICAL[name]
        c = plasticity_correction(n, B)
        print(
            f"{name:<10} {n:>3d} {B:>10.2f} "
            f"{c['eps_n']:>10.5f} {c['g_B']:>8.3f} "
            f"{c['delta']:>14.5f} {1 + c['delta']:>10.5f}"
        )
    print()


def print_unfolded_table():
    print("=" * 100)
    print("V5.32 — Δm_PDG -> Δm_topo_unfolded (Case B -> Case A)")
    print("=" * 100)
    rows = unfolded_table()
    print(
        f"{'Particule':<10} {'n':>3} {'B (T)':>7} {'factor':>10} "
        f"{'Δm_PDG':>13} {'Δm_unfolded':>14} "
        f"{'shift (dex)':>13}"
    )
    print("-" * 100)
    for r in rows:
        shift = float(
            np.log10(r["delta_m_topo_unfolded"] / r["delta_m_PDG"])
        )
        print(
            f"{r['particle']:<10} {r['n_components']:>3d} "
            f"{r['B_eff_T']:>7.2f} {r['correction_factor']:>10.5f} "
            f"{r['delta_m_PDG']:>13.3e} "
            f"{r['delta_m_topo_unfolded']:>14.3e} "
            f"{shift:>+13.4f}"
        )
    print()
    return rows


def gather_model_predictions():
    """V5.30c et V5.31 -> dicts {particle: Delta m MeV}."""
    rows_30c = v5p30c_table(
        alpha=ALPHA_VIB_DEFAULT, kappa_K=KAPPA_K_V5p30_DEFAULT
    )
    pred_30c = {r["name"]: r["delta_m_v5p30c_MeV"] for r in rows_30c}
    rows_31 = v5p31_table(
        beta=BETA_CHIRAL_DEFAULT,
        alpha=ALPHA_VIB_DEFAULT,
        kappa_K=KAPPA_K_V5p30_DEFAULT,
    )
    pred_31 = {r["name"]: r["delta_m_v5p31_MeV"] for r in rows_31}
    return pred_30c, pred_31


def print_comparison(pred_30c, pred_31, cmp_res):
    print("=" * 132)
    print(
        "V5.32 — Comparaison V5.30c et V5.31 vs PDG vs PDG_unfolded"
    )
    print("=" * 132)
    print(
        f"{'Particule':<8} "
        f"{'Δm_PDG':>11} {'Δm_unfolded':>12} "
        f"{'Δm V5.30c':>11} {'Δm V5.31':>11} "
        f"{'30c|PDG':>9} {'30c|unf':>9} "
        f"{'31|PDG':>9} {'31|unf':>9}"
    )
    print("-" * 132)
    for r in cmp_res["rows"]:
        n = r["name"]
        print(
            f"{n:<8} "
            f"{r['delta_m_PDG']:>11.3e} "
            f"{r['delta_m_topo_unfolded']:>12.3e} "
            f"{r['delta_m_v5p30c']:>11.3e} "
            f"{r['delta_m_v5p31']:>11.3e} "
            f"{r['log_ratio_v5p30c_vs_PDG']:>+9.3f} "
            f"{r['log_ratio_v5p30c_vs_unfolded']:>+9.3f} "
            f"{r['log_ratio_v5p31_vs_PDG']:>+9.3f} "
            f"{r['log_ratio_v5p31_vs_unfolded']:>+9.3f}"
        )
    print()
    print("Sommes Σ|log_ratio|:")
    print(
        f"  V5.30c vs PDG       : {cmp_res['sum_abs_log_v5p30c_vs_PDG']:.3f} dex"
    )
    print(
        f"  V5.30c vs unfolded  : {cmp_res['sum_abs_log_v5p30c_vs_unfolded']:.3f} dex"
    )
    print(
        f"  V5.31  vs PDG       : {cmp_res['sum_abs_log_v5p31_vs_PDG']:.3f} dex"
    )
    print(
        f"  V5.31  vs unfolded  : {cmp_res['sum_abs_log_v5p31_vs_unfolded']:.3f} dex"
    )
    print()
    print("Ratios Bs/B :")
    print(
        f"  V5.31      : {cmp_res['ratio_v5p31_Bs_B']:.2f}\n"
        f"  PDG        : {cmp_res['ratio_pdg_Bs_B']:.2f}\n"
        f"  PDG_unfolded : {cmp_res['ratio_pdg_unfolded_Bs_B']:.2f}"
    )
    print()


def make_plot(cmp_res, out_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    rows = cmp_res["rows"]
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    w = 0.18

    dm_pdg = [r["delta_m_PDG"] for r in rows]
    dm_unf = [r["delta_m_topo_unfolded"] for r in rows]
    dm_30c = [r["delta_m_v5p30c"] for r in rows]
    dm_31 = [r["delta_m_v5p31"] for r in rows]

    ax1.bar(x - 1.5 * w, dm_pdg, w, label="PDG", color="#ff7f0e")
    ax1.bar(x - 0.5 * w, dm_unf, w, label="PDG unfolded (V5.32)", color="#2ca02c")
    ax1.bar(x + 0.5 * w, dm_30c, w, label="V5.30c", color="#1f77b4")
    ax1.bar(x + 1.5 * w, dm_31, w, label="V5.31", color="#d62728")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel(r"$\Delta m$ (MeV)")
    ax1.set_title(r"V5.32 — $\Delta m$ : PDG, unfolded, V5.30c, V5.31")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, axis="y", which="both", alpha=0.3)

    log_30c_pdg = [r["log_ratio_v5p30c_vs_PDG"] for r in rows]
    log_30c_unf = [r["log_ratio_v5p30c_vs_unfolded"] for r in rows]
    log_31_pdg = [r["log_ratio_v5p31_vs_PDG"] for r in rows]
    log_31_unf = [r["log_ratio_v5p31_vs_unfolded"] for r in rows]
    ax2.bar(x - 1.5 * w, log_30c_pdg, w, label="V5.30c | PDG", color="#1f77b4")
    ax2.bar(
        x - 0.5 * w,
        log_30c_unf,
        w,
        label="V5.30c | unfolded",
        color="#1f77b4",
        alpha=0.5,
    )
    ax2.bar(x + 0.5 * w, log_31_pdg, w, label="V5.31 | PDG", color="#d62728")
    ax2.bar(
        x + 1.5 * w,
        log_31_unf,
        w,
        label="V5.31 | unfolded",
        color="#d62728",
        alpha=0.5,
    )
    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(0.5, color="g", ls="--", lw=0.8, alpha=0.5)
    ax2.axhline(-0.5, color="g", ls="--", lw=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel(r"$\log_{10}(\Delta m / \Delta m_{\rm ref})$")
    ax2.set_title("V5.32 — log_ratio avant / apres unfolding P23")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Plot sauvegarde : {out_path}")


def verdict(cmp_res):
    print("=" * 80)
    print("Verdict V5.32 (honnete)")
    print("=" * 80)
    s30c_pdg = cmp_res["sum_abs_log_v5p30c_vs_PDG"]
    s30c_unf = cmp_res["sum_abs_log_v5p30c_vs_unfolded"]
    s31_pdg = cmp_res["sum_abs_log_v5p31_vs_PDG"]
    s31_unf = cmp_res["sum_abs_log_v5p31_vs_unfolded"]
    d30c = s30c_unf - s30c_pdg
    d31 = s31_unf - s31_pdg

    print(
        f"  V5.30c : Σ|log| {s30c_pdg:.3f} (PDG) -> {s30c_unf:.3f} (unfolded)  "
        f"Δ = {d30c:+.3f} dex"
    )
    print(
        f"  V5.31  : Σ|log| {s31_pdg:.3f} (PDG) -> {s31_unf:.3f} (unfolded)  "
        f"Δ = {d31:+.3f} dex"
    )
    print()
    # Critere "plafond leve" : V5.30c ou V5.31 vs unfolded < 0.5 dex
    floor_lifted = (s30c_unf < 0.5) or (s31_unf < 0.5)
    improves = (d30c < -0.05) or (d31 < -0.05)
    marginal = (-0.05 <= d30c <= 0.05) and (-0.05 <= d31 <= 0.05)
    if floor_lifted:
        label = "SUCCESS"
        print(
            "VERDICT : SUCCESS — P23 unfolding amene au moins un modele\n"
            "         sous Σ|log| < 0.5 dex. Le plafond V5 est leve."
        )
    elif improves:
        label = "PARTIAL"
        print(
            "VERDICT : PARTIAL — P23 unfolding ameliore Σ|log| (au moins\n"
            "         un modele), sans descendre sous 0.5 dex. Le plafond\n"
            "         persiste mais n'est plus structurel."
        )
    elif marginal:
        label = "NULL"
        print(
            "VERDICT : NULL — P23 unfolding deplace Σ|log| de moins de\n"
            "         0.05 dex. La correction P23 est trop petite (champs\n"
            "         B ~1-4 T, eps(n)*g ~5 % max) pour fermer un gap de\n"
            "         0.5-1 dex. **Le plafond V5 est structurel, pas\n"
            "         experimental**. P23 sur les Δm est negligeable.\n"
            "         Honnete : un autre mecanisme est necessaire pour\n"
            "         lever le plafond V5 sur les oscillations."
        )
    else:
        label = "WORSE"
        print(
            "VERDICT : WORSE — P23 unfolding degrade Σ|log|.\n"
            "         L'hypothese 'mesure altere Δm' va dans le mauvais\n"
            "         sens : les modeles V5.30c/V5.31 sont en fait plus\n"
            "         proches des PDG bruts que des Δm topo purs."
        )
    print()
    return label


def main():
    print()
    print(
        "V5.32 — Plasticite P23 et unfolding des Δm experimentaux\n"
        "Reference V5 : sec:P23-plastique + sec:plast-experimental-alteration\n"
        "Discipline   : Case B (B_ext != 0) -> correction eps(n)*g(B_eff)\n"
    )

    print_constants()
    print_corrections_table()
    print_unfolded_table()
    pred_30c, pred_31 = gather_model_predictions()
    cmp_res = compare_models_to_unfolded(pred_30c, pred_31)
    print_comparison(pred_30c, pred_31, cmp_res)
    make_plot(cmp_res, "output/v5p32_p23_unfolded.png")
    label = verdict(cmp_res)
    print(f"Label final : {label}")


if __name__ == "__main__":
    main()
