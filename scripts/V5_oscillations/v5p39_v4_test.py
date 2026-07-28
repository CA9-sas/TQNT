"""V5.39 - Test de la derivation rigoureuse du facteur 4 via Vassiliev v_4.

Genere output/v5p39_v4.png avec :
  1. v_4 par meson (K^0, D^0, B^0, B_s^0)
  2. Cinq voies de derivation du facteur 4
  3. delta predit (universel) vs delta V5.38

Reference : theory/v5p39_vassiliev_v4.py
            theory/v5p38_delta_derivation.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.v5p38_delta_derivation import delta_first_principles
from theory.v5p39_vassiliev_v4 import (
    factor_4_derivation,
    summary_report,
    v4_meson_table,
)


def print_report() -> None:
    print(summary_report())


def print_derivation_table() -> None:
    print()
    print("=" * 88)
    print("V5.39 - Cinq voies de derivation du facteur 4")
    print("=" * 88)
    for d in factor_4_derivation():
        marker = "[RETENU]" if d.is_retained else "[rejete]"
        print(f"  {marker:<10} {d.label:<32} ({d.rigor})")
        # Wrap descriptions on ~110 chars
        desc = d.description
        for i in range(0, len(desc), 110):
            print(f"             {desc[i:i+110]}")
        print()


def make_plot(out_path: str = "output/v5p39_v4.png") -> None:
    table = v4_meson_table()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes

    # Panel 1 : v_4 par meson
    mesons_with_v4 = [r for r in table if r["v4_value"] is not None]
    names = [r["meson"] for r in mesons_with_v4]
    v4s = [r["v4_value"] for r in mesons_with_v4]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in v4s]
    ax1.bar(names, v4s, color=colors)
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_ylabel(r"$v_4(L)$")
    ax1.set_title(
        r"V5.39 - Vassiliev $v_4$ pour les 3-links meson"
        "\n"
        r"(formule Polyak-Viro $v_4 = \sum_{i<j} c_2(K_i) c_2(K_j) + \mu_{123}^2$)"
    )
    for i, (name, v) in enumerate(zip(names, v4s)):
        c2s = mesons_with_v4[i]["c2_components"]
        ax1.text(
            i, v,
            f" c$_2$={c2s}",
            ha="center", va="bottom" if v >= 0 else "top",
            fontsize=8,
        )
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel 2 : delta predit (constante universelle) vs k
    ks = np.logspace(1, 4, 50)
    deltas = [-4.0 * np.log(2.0 * np.cos(np.pi / (k + 2))) for k in ks]
    ax2.plot(ks, deltas, "-", color="#1f77b4", lw=2,
             label=r"$\delta(k) = -4\ln d_{1/2}(k)$ (V5.39 - rigoureux)")
    ax2.axhline(
        -4.0 * np.log(2.0), color="#d62728", ls="--", lw=1,
        label=r"limite $k \to \infty$ : $-\ln 16$ (V5.36 D1)",
    )
    ax2.axhline(
        delta_first_principles(), color="#2ca02c", ls=":", lw=1.5,
        label=f"$\\delta$(V5.38, k=135) = {delta_first_principles():+.4f}",
    )
    ax2.axvline(135, color="k", ls=":", alpha=0.6, label="k = 135 (TQNT)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Chern-Simons level k")
    ax2.set_ylabel(r"$\delta$")
    ax2.set_title(
        r"V5.39 - $\delta$ universel ne depend pas du link choisi"
        "\n"
        r"(le 4 emerge de l'ordre Vassiliev primitif sur 3-link generique)"
    )
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Plot sauvegarde : {out_path}")


def main() -> None:
    print()
    print(
        "V5.39 - Derivation rigoureuse facteur 4 dans delta = -4 ln(d_{1/2})\n"
        "    via invariant Vassiliev v_4 et fusion 4-points SU(2)_135\n"
    )
    print_report()
    print_derivation_table()
    make_plot()


if __name__ == "__main__":
    main()
