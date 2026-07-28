"""
Figure récapitulative pour le brouillon "durées de vie tensorielles".

Deux panneaux :
 (a) Résidus log10(τ_calc/τ_exp) par particule de la famille faible :
     P15 scalaire vs dégradation corrigée par le tenseur de stabilité 1/S_K.
 (b) RMS (décades) des formes testées : P15, nœud 1/S_K, enlacements.

Sortie : output/lifetime_tensor_residuals.png

Usage :  python3 -m simulation.experiments.lifetime_tensor_figure
"""
from __future__ import annotations

import numpy as np

from simulation.experiments.lifetime_tensor_weight import fit_and_score, fit_additive
from simulation.experiments.lifetime_linking import fit_additive as link_additive

WEAK = ["d", "s", "mu-", "c", "tau-", "b"]
LABEL = {"d": "n (β)", "mu-": "μ", "s": "s", "c": "c", "tau-": "τ", "b": "b"}


def main():
    p15 = fit_and_score("P15 (w=1)")
    node = fit_additive("U/ρ_J²  (1/S_K)")
    link = link_additive("Σ1/√(S_iS_j)")

    print(f"P15 RMS={p15['rms']:.3f}  node 1/S_K RMS={node['rms']:.3f}  "
          f"link 1/√(SS) RMS={link['rms']:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib indisponible : {e})")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # (a) résidus par particule
    x = np.arange(len(WEAK))
    r_p15 = [p15["resid"][n] for n in WEAK]
    r_node = [node["resid"][n] for n in WEAK]
    w = 0.38
    ax1.bar(x - w / 2, r_p15, w, label="P15 scalaire", color="steelblue")
    ax1.bar(x + w / 2, r_node, w, label="corrigé tenseur $1/S_K$", color="darkorange")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels([LABEL[n] for n in WEAK])
    ax1.set_ylabel(r"résidu $\log_{10}(\tau_{\rm calc}/\tau_{\rm exp})$")
    ax1.set_title("(a) Résidus famille faible — le neutron $n$($\\beta$)")
    ax1.legend(fontsize=9)

    # (b) RMS des formes
    forms = ["P15\nscalaire", "nœud\n$1/S_K$", "lien\n$1/\\sqrt{S_iS_j}$"]
    rms = [p15["rms"], node["rms"], link["rms"]]
    colors = ["steelblue", "darkorange", "seagreen"]
    ax2.bar(forms, rms, color=colors)
    for i, v in enumerate(rms):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax2.set_ylabel("RMS (décades)")
    ax2.set_title("(b) RMS sur la famille faible (1–2 param)")
    ax2.set_ylim(0, max(rms) * 1.25)

    fig.tight_layout()
    out = "output/lifetime_tensor_residuals.png"
    fig.savefig(out, dpi=130)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
