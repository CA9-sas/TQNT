"""V5.35a — Terme 3-corps dans inertie chirale — test runner.

Δm V5.35a(L) = Δm V5.30c(L) × R_χ_v5p35a(L),
    R_χ_v5p35a = ω_χ_v5p35a(L) / ω_χ_v5p35a(K⁰),
    ω_χ_v5p35a² = F_torque / I_χ_v5p35a,
    I_χ_v5p35a = I_χ_v5p34_F3 + γ_3 × I_3body,
    I_3body = Σ_{i<j<k} b_i b_j b_k × √(c_i c_j c_k)
              × |Lk_ij · Lk_jk · Lk_ik|^{1/3}.

Sortie : output/v5p35a_3body.png
Référence : theory/v5p35a_3body_inertia.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p34_chiral_inertia import v5p34_table
from theory.v5p35a_3body_inertia import (
    calibrate_gamma_3,
    three_body_term,
    v5p35a_table,
)


# Baseline V5.34 (F3_coupling, sans paramètre libre)
def baseline_v5p34_f3():
    rows = v5p34_table("F3_coupling", 1.0)
    out = {}
    s = 0.0
    for r in rows:
        out[r["name"]] = {
            "delta_m_MeV": r["delta_m_v5p34_MeV"],
            "log_ratio": r["log_ratio_v5p34"],
        }
        s += abs(r["log_ratio_v5p34"])
    return out, s


def print_3body_diagnostics():
    print("=" * 88)
    print("V5.35a — Diagnostic du terme 3-corps par méson")
    print("=" * 88)
    print(
        f"{'Méson':<6} {'knots':<26} {'n_trip':>7} "
        f"{'I_3body':>12} {'détail triplet':<40}"
    )
    print("-" * 88)
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        tb = three_body_term(cfg["knots"])
        if tb["triplets"]:
            t = tb["triplets"][0]
            detail = (
                f"K={t['K_ijk']:.1f} √c³={t['c_term']:.2f} "
                f"|Lk³|^(1/3)={t['lk_term']:.2f}"
            )
        else:
            detail = "(n<3, terme 3-corps inactif)"
        print(
            f"{name:<6} {str(cfg['knots']):<26} {tb['n_triplets']:>7d} "
            f"{tb['I_3body']:>12.3e} {detail}"
        )
    print()


def print_compare(res, baseline, baseline_sum):
    print("=" * 88)
    print("V5.35a — Comparaison Δm V5.35a vs V5.34 (F3) vs PDG")
    print("=" * 88)
    print(f"γ_3* (calibré) = {res['gamma_3']:+.4f}")
    print(
        f"Σ|log| V5.35a = {res['sum_abs_log']:.3f} dex  "
        f"(V5.34 F3 baseline = {baseline_sum:.3f} dex)"
    )
    print(f"Ratio B_s/B = {res['ratio_Bs_B']:.2f}  (PDG = 35.05)")
    print()
    print(
        f"{'Méson':<6} {'Δm V5.35a':>12} {'log V5.35a':>11} "
        f"{'Δm V5.34_F3':>13} {'log V5.34':>10} {'Δm PDG':>12} "
        f"{'I_3body':>11}"
    )
    print("-" * 88)
    per = res["per_meson"]
    for name in ("K0", "D0", "B0", "Bs0"):
        p = per[name]
        b = baseline[name]
        print(
            f"{name:<6} {p['delta_m_MeV']:>12.3e} {p['log_ratio']:>+11.3f} "
            f"{b['delta_m_MeV']:>13.3e} {b['log_ratio']:>+10.3f} "
            f"{p['delta_m_pdg_MeV']:>12.3e} {p['I_3body']:>11.3e}"
        )
    print()


def verdict(res, baseline_sum):
    print("=" * 88)
    print("Verdict V5.35a (honnête)")
    print("=" * 88)
    s = res["sum_abs_log"]
    delta = baseline_sum - s
    per = res["per_meson"]
    log_B = per["B0"]["log_ratio"]
    log_Bs = per["Bs0"]["log_ratio"]
    log_D = per["D0"]["log_ratio"]
    if s < 0.50:
        label = "BREAKTHROUGH"
        print(
            f"BREAKTHROUGH — Σ|log| = {s:.3f} dex < 0.50.\n"
            f"   Le terme 3-corps clôt l'écart résiduel à 1 paramètre."
        )
    elif s < baseline_sum - 0.10:
        label = "IMPROVEMENT"
        print(
            f"IMPROVEMENT — Σ|log| = {s:.3f} dex, gain Δ = {delta:+.3f} dex\n"
            f"   sur V5.34 F3 ({baseline_sum:.3f}).  Le 3-corps aide mais\n"
            f"   ne ferme pas l'écart B⁰ (log = {log_B:+.3f})."
        )
    elif s < baseline_sum:
        label = "MARGINAL"
        print(
            f"MARGINAL — Σ|log| = {s:.3f} dex, gain Δ = {delta:+.3f} dex\n"
            f"   sur V5.34 F3 ({baseline_sum:.3f}).  Le terme 3-corps\n"
            f"   améliore légèrement mais B⁰ reste à log = {log_B:+.3f}."
        )
    elif s <= baseline_sum + 0.02:
        label = "NEUTRAL"
        print(
            f"NEUTRAL — Σ|log| = {s:.3f} dex, ~identique à V5.34 F3\n"
            f"   ({baseline_sum:.3f}).  Le terme 3-corps n'apporte rien."
        )
    else:
        label = "FAILURE"
        print(
            f"FAILURE — Σ|log| = {s:.3f} dex, AU-DESSUS de V5.34 F3\n"
            f"   ({baseline_sum:.3f}).  Le terme 3-corps dégrade la fit.\n"
            f"   γ_3* aurait dû être 0 ; si γ_3* ≠ 0 c'est un artefact."
        )
    print()
    # Diagnostic honnête : que change γ_3 sur chaque méson ?
    print("Détail honnête :")
    print(
        f"   B⁰  log = {log_B:+.3f}  (V5.34 F3 = -0.655)  "
        f"→ {'amélioré' if abs(log_B) < 0.655 else 'inchangé/dégradé'}"
    )
    print(
        f"   D⁰  log = {log_D:+.3f}  (V5.34 F3 = -0.122)  "
        f"→ {'amélioré' if abs(log_D) < 0.122 else 'inchangé/dégradé'}"
    )
    print(
        f"   B_s⁰ log = {log_Bs:+.3f}  (V5.34 F3 = -0.043)  "
        f"→ {'amélioré' if abs(log_Bs) < 0.043 else 'inchangé/dégradé (n=2, I_3body=0)'}"
    )
    print()
    return label


def make_plot(res, baseline, baseline_sum, out_path="output/v5p35a_3body.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    mesons = ["K0", "D0", "B0", "Bs0"]
    # Panneau 1 : log_ratio par méson, V5.34 vs V5.35a
    log_v534 = [baseline[m]["log_ratio"] for m in mesons]
    log_v535a = [res["per_meson"][m]["log_ratio"] for m in mesons]
    x = np.arange(len(mesons))
    w = 0.35
    ax1.bar(x - w / 2, log_v534, w, label="V5.34 (F3)", color="#1f77b4")
    ax1.bar(x + w / 2, log_v535a, w, label=f"V5.35a (γ_3={res['gamma_3']:+.3f})", color="#d62728")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(mesons)
    ax1.set_ylabel(r"$\log_{10}(\Delta m / \Delta m_{\rm PDG})$ (dex)")
    ax1.set_title(
        f"V5.35a — log par méson  "
        f"(Σ|log|: V5.34={baseline_sum:.2f} → V5.35a={res['sum_abs_log']:.2f})"
    )
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # Panneau 2 : Δm vs PDG (log scale)
    dm_v535a = [res["per_meson"][m]["delta_m_MeV"] for m in mesons]
    dm_pdg = [res["per_meson"][m]["delta_m_pdg_MeV"] for m in mesons]
    ax2.bar(x - w / 2, dm_v535a, w, label="V5.35a", color="#d62728")
    ax2.bar(x + w / 2, dm_pdg, w, label="PDG", color="#ff7f0e")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mesons)
    ax2.set_ylabel(r"$\Delta m$ (MeV)")
    ax2.set_title(f"V5.35a — Δm vs PDG (γ_3* = {res['gamma_3']:+.3f})")
    ax2.legend()
    ax2.grid(True, axis="y", which="both", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Plot sauvegardé : {out_path}")


def main():
    print()
    print(
        "V5.35a — Terme 3-corps dans inertie chirale\n"
        "I_χ_V5.35a = I_χ_V5.34_F3 + γ_3 × Σ_{i<j<k} b_i b_j b_k "
        "× √(c_i c_j c_k) × |Lk_ij·Lk_jk·Lk_ik|^{1/3}\n"
    )
    print_3body_diagnostics()
    baseline, baseline_sum = baseline_v5p34_f3()
    res = calibrate_gamma_3(gamma_range=(0.0, 10.0), n_grid=10001)
    print_compare(res, baseline, baseline_sum)
    label = verdict(res, baseline_sum)
    make_plot(res, baseline, baseline_sum)
    print(f"Label final : {label}")
    return label


if __name__ == "__main__":
    main()
