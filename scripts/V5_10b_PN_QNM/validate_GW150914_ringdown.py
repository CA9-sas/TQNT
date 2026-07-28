#!/usr/bin/env python3
"""
V5.10b — Validation QNM Kerr sur ringdown GW150914.

Reproduit la fréquence et le temps d'amortissement du mode (l=2, m=2, n=0)
pour le trou noir final de GW150914 (M_f ≈ 62 M_sun, a* ≈ 0.67) à partir
des tables Berti-Cardoso-Starinets 2009.

Référence LIGO :
  Abbott+ 2016, "Tests of general relativity with GW150914",
  PRL 116, 221101.

V5 hérite Kerr et ses QNM via la dérivation V5.8 de l'équation
d'Einstein depuis le χ-field. La validation ne teste PAS une
spécificité V5, seulement la cohérence V5 ↔ GR dans le ringdown.

Output :
  - output/v5p10b_GW150914_ringdown.png

Lancement :
    PYTHONPATH=. python3 scripts/V5_10b_PN_QNM/validate_GW150914_ringdown.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.ringdown_QNM_tabulated import (
    available_modes,
    damping_time_ms,
    kerr_QNM_mode,
    quality_factor,
    ringdown_frequency_Hz,
)


# Paramètres GW150914 post-merger (Abbott+ 2016, PRL 116, 221101)
M_FINAL = 62.0       # M_sun (intervalle 62 ± 4)
A_STAR_FINAL = 0.67  # spin réduit (intervalle 0.67 ± 0.05)

# Valeurs LIGO observées pour le ringdown (mode dominant)
F_LIGO_HZ = 250.0     # ordre de grandeur LIGO (Abbott+ 2016)
TAU_LIGO_MS = 4.0     # ordre de grandeur LIGO


def main() -> int:
    print("=" * 72)
    print("V5.10b — VALIDATION QNM RINGDOWN GW150914")
    print("=" * 72)
    print(f"Paramètres post-merger (Abbott+ 2016 PRL 116, 221101) :")
    print(f"  M_final = {M_FINAL:.1f} M_sun")
    print(f"  a*_final = {A_STAR_FINAL:.2f}")
    print()
    print(f"{'Mode':<14} {'f (Hz)':>10} {'τ (ms)':>10} {'Q':>8}")
    print("-" * 50)
    for (l, m, n) in available_modes():
        f_Hz = ringdown_frequency_Hz(M_FINAL, l, m, n, A_STAR_FINAL)
        tau_ms = damping_time_ms(M_FINAL, l, m, n, A_STAR_FINAL)
        Q = quality_factor(l, m, n, A_STAR_FINAL)
        print(f"({l},{m},{n})         {f_Hz:>10.2f} {tau_ms:>10.3f} {Q:>8.2f}")
    print("-" * 50)
    print()

    # Mode dominant (2,2,0)
    f_220 = ringdown_frequency_Hz(M_FINAL, 2, 2, 0, A_STAR_FINAL)
    tau_220 = damping_time_ms(M_FINAL, 2, 2, 0, A_STAR_FINAL)

    print(f"Mode (2,2,0) dominant — prédiction GR / LIGO :")
    print(f"  f_prédit  = {f_220:.1f} Hz    vs  f_LIGO ≈ {F_LIGO_HZ:.0f} Hz")
    print(f"  τ_prédit  = {tau_220:.2f} ms   vs  τ_LIGO ≈ {TAU_LIGO_MS:.1f} ms")
    print(f"  écart f   : {100.0 * abs(f_220 - F_LIGO_HZ) / F_LIGO_HZ:.1f} %")
    print(f"  écart τ   : {100.0 * abs(tau_220 - TAU_LIGO_MS) / TAU_LIGO_MS:.1f} %")
    print()
    print("Résultats GR standard tabulés depuis Berti-Cardoso-Starinets 2009,")
    print("hérités par V5 via la dérivation V5.8 de l'équation d'Einstein.")
    print("Aucune spécificité TQNT revendiquée ici.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1 : forme d'onde damped sinusoid (mode 2,2,0)
    ax = axes[0]
    f_Hz = f_220
    tau_s = tau_220 * 1.0e-3
    omega = 2.0 * np.pi * f_Hz
    t = np.linspace(0.0, 30.0e-3, 4000)  # 30 ms window
    h = np.exp(-t / tau_s) * np.cos(omega * t)
    ax.plot(t * 1.0e3, h, color="#d62728", lw=1.4, label="Mode (2,2,0)")
    ax.plot(t * 1.0e3, np.exp(-t / tau_s), color="black",
            ls="--", lw=0.8, label=f"Enveloppe e^(-t/τ), τ={tau_220:.2f} ms")
    ax.plot(t * 1.0e3, -np.exp(-t / tau_s), color="black", ls="--", lw=0.8)
    ax.axvline(tau_220, color="#1f77b4", ls=":", lw=0.8,
               label=f"τ = {tau_220:.2f} ms")
    ax.set_xlabel("Temps après merger (ms)")
    ax.set_ylabel("Strain ringdown (unités arb.)")
    ax.set_title(f"Ringdown amorti GW150914\n"
                 f"f = {f_220:.1f} Hz, τ = {tau_220:.2f} ms (mode 2,2,0)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2 : QNM (l,m,n) à différents spins
    ax = axes[1]
    a_grid = np.linspace(0.0, 0.99, 60)
    for mode in available_modes():
        l, m, n = mode
        freqs = np.array([
            ringdown_frequency_Hz(M_FINAL, l, m, n, a) for a in a_grid
        ])
        ax.plot(a_grid, freqs, lw=1.6,
                label=f"({l},{m},{n}), M={M_FINAL:.0f} M_⊙")
    ax.axvline(A_STAR_FINAL, color="black", ls=":", lw=0.8,
               label=f"a* GW150914 = {A_STAR_FINAL}")
    ax.axhline(F_LIGO_HZ, color="grey", ls="--", lw=0.8,
               label=f"f_LIGO ≈ {F_LIGO_HZ:.0f} Hz")
    ax.set_xlabel("Spin réduit a* = a/M")
    ax.set_ylabel("Fréquence ringdown (Hz)")
    ax.set_title(f"QNM Kerr f(a*) — tables BCS09\n"
                 f"M_BH = {M_FINAL} M_⊙")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("V5.10b — Ringdown QNM Kerr GW150914 "
                 "(GR standard, hérité V5)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "v5p10b_GW150914_ringdown.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
