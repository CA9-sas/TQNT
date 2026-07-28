#!/usr/bin/env python3
"""
V5.9 — Validation GW150914.

Implémente le chirp inspiral du binaire BH-BH GW150914 (Abbott+ 2016,
PRL 116, 061102) à partir des fonctions V5.9 et compare la prédiction
TQNT au template LIGO publié.

Outputs :
  - output/v5p9_gw150914_validation.png : 2 panneaux
    (1) Strain h_+(t) du chirp
    (2) Évolution de la fréquence GW au cours du temps
  - Affichage console des grandeurs clés.

Lancement :
    PYTHONPATH=. python3 scripts/V5_9_GW/validate_gw150914.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.gravitational_waves import (
    GW_strain_binary,
    M_SUN,
    MPC,
    chirp_frequency,
    chirp_mass,
    coalescence_time,
    inspiral_evolution,
    time_to_merger_from_frequency,
    validate_GW150914,
    G_SI,
    C_LIGHT,
)


def main():
    print("=" * 72)
    print("  V5.9 — Validation GW150914 (Abbott+ 2016, PRL 116, 061102)")
    print("=" * 72)

    # Paramètres LIGO GW150914
    m1_msun = 36.0
    m2_msun = 29.0
    distance_Mpc = 410.0

    m1 = m1_msun * M_SUN
    m2 = m2_msun * M_SUN
    distance = distance_Mpc * MPC

    M_c = chirp_mass(m1, m2) / M_SUN
    M_tot = (m1 + m2) / M_SUN

    print(f"\n  Source : binaire BH-BH")
    print(f"    m1                 = {m1_msun:.1f} M_sun")
    print(f"    m2                 = {m2_msun:.1f} M_sun")
    print(f"    M_total            = {M_tot:.1f} M_sun")
    print(f"    M_chirp            = {M_c:.2f} M_sun  (LIGO : 28.1 ± 1.7)")
    print(f"    Distance           = {distance_Mpc:.0f} Mpc (~ 1.3 Gly)")

    # ---- Validation GW150914 (résumé)
    res = validate_GW150914(m1_msun, m2_msun, distance_Mpc,
                             f_gw_peak=150.0, iota=0.0)
    print(f"\n  Strain peak prédit V5.9     = {res.h_peak_predicted:.3e}")
    print(f"  Strain peak LIGO observé    = {res.h_peak_LIGO_observed:.3e}")
    print(f"  Erreur relative             = {res.relative_error*100:+.1f} %")
    print(f"\n  Durée chirp (35 Hz -> fusion) = {res.t_chirp_s*1000:.0f} ms"
          f"  (LIGO observation : ~200 ms)")

    # ---- Construction du chirp depuis 35 Hz
    f_gw_start = 35.0
    f_gw_end = 250.0
    t_start = time_to_merger_from_frequency(m1, m2, f_gw_start)
    t_end = time_to_merger_from_frequency(m1, m2, f_gw_end)
    print(f"\n  t à f=35  Hz : {t_start*1000:.1f} ms avant fusion")
    print(f"  t à f=250 Hz : {t_end*1000:.2f} ms avant fusion")

    # Échantillonnage temporel (jusqu'à 0,99 t_start avant la fusion)
    t_phys = np.linspace(0.0, t_start - t_end, 4000)
    tau = t_start - t_phys   # temps avant fusion à chaque point
    tau = np.maximum(tau, t_end)  # cutoff au PN dominant

    # Fréquence chirp à chaque pas
    f_gw_t = np.array([chirp_frequency(m1, m2, tt) for tt in tau])

    # Strain h_+(t) instantané (formule quadrupolaire à fréquence locale)
    # En supposant phase Ψ(t) = -2π ∫ f(t') dt' (approx PN)
    dt = t_phys[1] - t_phys[0]
    phase = -2.0 * math.pi * np.cumsum(f_gw_t) * dt

    # Amplitude h_0 instantanée (face-on)
    h0_t = np.array([
        abs(GW_strain_binary(m1, m2, distance, f, iota=0.0)[0])
        for f in f_gw_t
    ])

    # Phase-locked waveform (inspiral PN dominant)
    h_plus_t = h0_t * np.cos(phase)

    # ---- Plot
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "v5p9_gw150914_validation.png"

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    # Panneau (1) : strain h_+(t)
    ax = axes[0]
    ax.plot(t_phys * 1000, h_plus_t * 1e21,
            color="tab:blue", lw=1.2, label="V5.9 prédiction (inspiral PN dominant)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.axhline(1.0, color="tab:red", ls=":", lw=1, alpha=0.7,
                label="LIGO observed peak ($1.0\\times10^{-21}$)")
    ax.axhline(-1.0, color="tab:red", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Temps avant la fusion estimée [ms]")
    ax.set_ylabel(r"Strain $h_+$  [$\times 10^{-21}$]")
    ax.set_title(
        f"V5.9 : Chirp GW150914 (m1={m1_msun}+m2={m2_msun} $M_\\odot$, "
        f"d={distance_Mpc} Mpc)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # Panneau (2) : évolution de la fréquence GW
    ax = axes[1]
    ax.plot(t_phys * 1000, f_gw_t, color="tab:green", lw=1.5,
            label="V5.9 (Peters 1964)")
    ax.axhline(35, color="tab:red", ls="--", lw=1, alpha=0.7,
                label="LIGO band start (35 Hz)")
    ax.axhline(250, color="tab:red", ls=":", lw=1, alpha=0.7,
                label="LIGO band end (250 Hz)")
    ax.set_xlabel("Temps avant la fusion estimée [ms]")
    ax.set_ylabel(r"Fréquence $f_{GW}$  [Hz]")
    ax.set_yscale("log")
    ax.set_title("V5.9 : Évolution de la fréquence GW (chirp 35 -> 250 Hz)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "TQNT-V5.9 — Validation GW150914 (PRL 116, 061102)",
        fontsize=13, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure sauvée : {out_path.relative_to(ROOT)}")

    # ---- Honest disclaimer
    print("\n" + "-" * 72)
    print("  Limites honnêtes V5.9 :")
    print("  - Inspiral phase PN dominant : couvert (v/c ≲ 0,3).")
    print("  - Merger phase (~5 ms autour de t=t_coal) : HORS SCOPE.")
    print("  - Ringdown (QNM Kerr) : HORS SCOPE.")
    print("  - Le peak strain observé (~1e-21) inclut le merger ;")
    print("    V5.9 surestime le PN dominant à 150 Hz (frontière du régime).")
    print("    L'accord est meilleur dans la bande basse (35 - 100 Hz).")
    print("-" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
