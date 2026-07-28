"""
V5.6 SCOPING --- Alternatives Wilson-loop multipath en SU(2)_135
=================================================================

CONTEXTE
--------
Le re-fit V5p3 master standardise (commit b107195) a confirme une pente
empirique de la concentration NFW

    d log c / d log M_h = +0.573 +/- 0.036  (SPARC, N ~ 175 galaxies)

en tension a ~18 sigma avec la pente Wechsler-Bullock observee (-0.10).
La fermeture purement conformelle (V5.5 etape 5/8) via une metrique
Omega^2 saturee a echoue (etape 8c, commit 2d65a91) : aucune valeur de
(h_max, Delta_log_M) ne reproduit -0.10.

Ce script EXPLORE trois (3) mecanismes alternatifs dans SU(2)_135 qui
pourraient produire une pente NEGATIVE en M_h, sans renoncer a la
universalite de kappa_CJ.  Chacun fournit un scaling explicite
kappa_CJ(M_h) ou L_coh(M_h) ; on calcule la pente induite en propageant
log c = log r_vir - log r_s, avec r_s = kappa_CJ * L_coh.

OBJECTIF
--------
Identifier laquelle des trois pistes a le potentiel d'amener la pente
totale de +0.573 (V5p3 pur) vers la cible -0.10.

LES TROIS ALTERNATIVES
======================

(A) Framing anomalies des Wilson loops (Witten 1989, §2)
--------------------------------------------------------
A niveau k, la framing de la Wilson loop introduit une phase
    W_K(f) = exp(2 pi i h_j * f) * W_K(0)
ou h_j = j(j+1)/(k+2) est le poids conforme.  Si le framing
effectif depend de l'echelle de halo via
    f_eff(M_h) = f_0 + delta * log10(M_h / M_*)
alors |W_K| recoit une enveloppe cosinus :
    kappa_CJ(M_h) = kappa_0 * |cos(pi h_j (f_0 + delta log10 M_h/M_*))|

(B) Multipath fusion (Verlinde formula, finite c)
-------------------------------------------------
La fusion entre primaires j1, j2, j3 ouvre N_{j1 j2}^{j3} canaux.
Pour le condensat baryonique, N depend du spin max accessible
    j_max(M_h) = j_* * (M_h / M_*)^xi
La longueur de coherence multipath :
    L_coh(M_h) = L_J / sqrt(N_paths(M_h))
avec N_paths(M_h) ~ (j_max(M_h))^p, p ~ 1-2.

(C) 1/c expansion (corrections a charge centrale finie)
-------------------------------------------------------
La charge centrale c_135 = 3*135/137 ~ 2.956 est finie.  Les
corrections 1/c modifient l'echelle :
    kappa_CJ(M_h) = kappa_0 * (1 + (alpha_c / c_135) * log(M_h / M_*))
ou alpha_c est le coefficient anomal de la 1/c expansion (Witten 1989,
appendice ; Frohlich-Kerler).

METRIQUE D'EVALUATION
=====================
Pour chaque mecanisme, on calcule la pente totale induite :
    slope_total = +0.573_V5p3 + slope_mech(M_h)
ou slope_mech = d log(1/kappa_CJ) / d log M_h ou d log(1/L_coh) / d log M_h.
La meilleure piste est celle qui rapproche le plus de WB (-0.10).

LIVRABLES
=========
- output/v5p6_wilson_multipath_scoping.png
- print() resume des pentes et de l'alternative la plus prometteuse
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.su2_modular import (
    V5_LEVEL,
    central_charge,
    conformal_weight,
    quantum_dim,
)

# ---------------------------------------------------------------------------
# Cibles empiriques
# ---------------------------------------------------------------------------
V5P3_BASE_SLOPE = +0.573
V5P3_BASE_ERR = 0.036
WB_SLOPE = -0.10

# Plage de masses de halo SPARC (typique : 5e9 ... 1e13 Msun)
M_GRID = np.logspace(9.5, 13.0, 60)
M_REF = 1.0e12  # masse pivot (Msun) -- correspond a Vflat = 200 km/s

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Helper
# ===========================================================================

def slope_of_log_log(x: np.ndarray, y: np.ndarray) -> float:
    """Pente moindres carres de log10 y vs log10 x."""
    lx = np.log10(x)
    ly = np.log10(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    s, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(s)


# ===========================================================================
# (A) Framing anomalies
# ===========================================================================

def framing_envelope(M_h: np.ndarray, j: float, f0: float, delta: float,
                     M_ref: float = M_REF, k: int = V5_LEVEL) -> np.ndarray:
    """|kappa_CJ(M_h) / kappa_0| sous framing-induit.

    W_K(f) = exp(2 pi i h_j f) W_K(0). Pour le condensat coherent, on prend
    l'amplitude effective comme |Re W_K(f_eff(M_h))| = |cos(pi h_j f_eff)|.

    On ajoute un plancher 1e-3 pour eviter les zeros pathologiques.
    """
    h_j = conformal_weight(j, k)
    f_eff = f0 + delta * np.log10(M_h / M_ref)
    envelope = np.abs(np.cos(np.pi * h_j * f_eff))
    return np.maximum(envelope, 1e-3)


def slope_framing(j: float, f0: float, delta: float) -> tuple[np.ndarray, float]:
    """Pente induite par framing-anomalies. r_s ~ kappa_CJ ~ envelope(M)."""
    env = framing_envelope(M_GRID, j, f0, delta)
    # r_s = kappa * L_coh, ici on isole l'effet kappa(M)
    # log c = log r_vir - log r_s = (1/3) log M - log kappa - log L_coh_base
    # d log c / d log M += -d log kappa / d log M
    # Donc la contribution incrementale de A vaut -slope(envelope)
    slope_kappa = slope_of_log_log(M_GRID, env)
    return env, -slope_kappa


# ===========================================================================
# (B) Multipath Verlinde
# ===========================================================================

def n_fusion_paths(j_max: float, k: int = V5_LEVEL) -> float:
    """Estimation du nombre de canaux de fusion accessibles a j <= j_max.

    Verlinde : N_{j1 j2}^{j3} = sum_l S_{j1 l} S_{j2 l} S_{j3 l} / S_{0 l}.
    Pour un scaling exploratoire, on utilise la borne
        N_paths(j_max) ~ Sum_{j=0}^{j_max} d_j^2
    qui est l'extension partielle de D^2 jusqu'a j_max.
    """
    if j_max <= 0:
        return 1.0
    j_vals = np.arange(0.0, j_max + 0.5, 0.5)
    return float(sum(quantum_dim(j, k) ** 2 for j in j_vals))


def slope_multipath(j_star: float, xi: float, p: float = 1.0
                    ) -> tuple[np.ndarray, float]:
    """L_coh(M_h) = L_J / sqrt(N_paths(M_h)).

    j_max(M_h) = j_star * (M_h / M_ref)^xi
    N_paths(M_h) ~ N_fusion(j_max)^p
    """
    j_max_arr = j_star * (M_GRID / M_REF) ** xi
    N_arr = np.array([n_fusion_paths(jm) ** p for jm in j_max_arr])
    L_factor = 1.0 / np.sqrt(N_arr)
    slope_L = slope_of_log_log(M_GRID, L_factor)
    # r_s ~ L_coh ; log c -= log L_coh ; d log c / d log M += -d log L / d log M
    return L_factor, -slope_L


# ===========================================================================
# (C) 1/c expansion
# ===========================================================================

def envelope_1overc(M_h: np.ndarray, alpha_c: float,
                    M_ref: float = M_REF, k: int = V5_LEVEL) -> np.ndarray:
    """kappa_CJ(M_h) = kappa_0 (1 + (alpha_c / c_135) * log(M_h / M_ref))."""
    c_135 = central_charge(k)
    return 1.0 + (alpha_c / c_135) * np.log(M_h / M_ref)


def slope_1overc(alpha_c: float) -> tuple[np.ndarray, float]:
    env = envelope_1overc(M_GRID, alpha_c)
    env = np.maximum(env, 1e-3)
    slope_kappa = slope_of_log_log(M_GRID, env)
    return env, -slope_kappa


# ===========================================================================
# Scoping principal
# ===========================================================================

def scope_alternative_A() -> dict:
    """(A) Framing anomalies : grille (j, f0, delta), trouve la combinaison
    qui donne la contribution -slope la plus negative."""
    j_list = [0.5, 1.0, 2.0, 5.0, 10.0]  # representations basses (les + couplees)
    f0_list = [0.0, 0.5, 1.0, 2.0]
    delta_list = np.linspace(-2.0, 2.0, 21)

    best = {"contrib": +np.inf, "total": +np.inf}
    grid = []
    for j in j_list:
        for f0 in f0_list:
            for delta in delta_list:
                _, contrib = slope_framing(j, f0, delta)
                total = V5P3_BASE_SLOPE + contrib
                grid.append((j, f0, delta, contrib, total))
                if abs(total - WB_SLOPE) < abs(best["total"] - WB_SLOPE):
                    best = {
                        "j": j, "f0": f0, "delta": delta,
                        "contrib": contrib, "total": total,
                    }
    return {"best": best, "grid": grid}


def scope_alternative_B() -> dict:
    """(B) Multipath Verlinde : grille (j_star, xi, p)."""
    j_star_list = [0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
    xi_list = np.linspace(0.05, 2.0, 30)
    p_list = [0.5, 1.0, 2.0, 4.0]

    best = {"contrib": +np.inf, "total": +np.inf}
    grid = []
    for j_star in j_star_list:
        for xi in xi_list:
            for p in p_list:
                _, contrib = slope_multipath(j_star, xi, p)
                total = V5P3_BASE_SLOPE + contrib
                grid.append((j_star, xi, p, contrib, total))
                if abs(total - WB_SLOPE) < abs(best["total"] - WB_SLOPE):
                    best = {
                        "j_star": j_star, "xi": xi, "p": p,
                        "contrib": contrib, "total": total,
                    }
    return {"best": best, "grid": grid}


def scope_alternative_C() -> dict:
    """(C) 1/c expansion : grille alpha_c."""
    alpha_c_list = np.linspace(-30.0, 30.0, 121)
    best = {"contrib": +np.inf, "total": +np.inf}
    grid = []
    for alpha_c in alpha_c_list:
        _, contrib = slope_1overc(alpha_c)
        total = V5P3_BASE_SLOPE + contrib
        grid.append((alpha_c, contrib, total))
        if abs(total - WB_SLOPE) < abs(best["total"] - WB_SLOPE):
            best = {
                "alpha_c": alpha_c, "contrib": contrib, "total": total,
            }
    return {"best": best, "grid": grid}


# ===========================================================================
# Plot
# ===========================================================================

def plot_scoping(res_A: dict, res_B: dict, res_C: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ---- (A) framing : log10 envelope vs log10 M_h pour quelques (j, delta)
    ax = axes[0, 0]
    j_demo = res_A["best"]["j"]
    f0_demo = res_A["best"]["f0"]
    for delta in [-1.5, -0.5, res_A["best"]["delta"], +0.5, +1.5]:
        env = framing_envelope(M_GRID, j_demo, f0_demo, delta)
        label = f"delta={delta:+.2f}"
        if abs(delta - res_A["best"]["delta"]) < 1e-3:
            label += " (best)"
        ax.plot(M_GRID, env, lw=1.6, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("M_h [Msun]"); ax.set_ylabel("kappa_CJ / kappa_0")
    ax.set_title(f"(A) Framing anomalies, j={j_demo}, f0={f0_demo}")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    # ---- (B) multipath : L_coh factor
    ax = axes[0, 1]
    j_star_demo = res_B["best"]["j_star"]
    p_demo = res_B["best"]["p"]
    for xi in [0.05, 0.2, res_B["best"]["xi"], 0.5, 1.0]:
        j_max_arr = j_star_demo * (M_GRID / M_REF) ** xi
        N_arr = np.array([n_fusion_paths(jm) ** p_demo for jm in j_max_arr])
        L_factor = 1.0 / np.sqrt(N_arr)
        label = f"xi={xi:.2f}"
        if abs(xi - res_B["best"]["xi"]) < 1e-3:
            label += " (best)"
        ax.plot(M_GRID, L_factor, lw=1.6, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("M_h [Msun]"); ax.set_ylabel("L_coh / L_J")
    ax.set_title(f"(B) Multipath Verlinde, j*={j_star_demo}, p={p_demo}")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    # ---- (C) 1/c : kappa(M_h) pour quelques alpha_c
    ax = axes[1, 0]
    alpha_c_best = res_C["best"]["alpha_c"]
    for alpha_c in [-20.0, -10.0, alpha_c_best, +5.0, +20.0]:
        env = envelope_1overc(M_GRID, alpha_c)
        env = np.maximum(env, 1e-3)
        label = f"alpha_c={alpha_c:+.1f}"
        if abs(alpha_c - alpha_c_best) < 1e-2:
            label += " (best)"
        ax.plot(M_GRID, env, lw=1.6, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("M_h [Msun]"); ax.set_ylabel("kappa_CJ / kappa_0")
    ax.set_title("(C) 1/c expansion")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- Recap des pentes
    ax = axes[1, 1]
    labels = ["V5p3 base", "WB target",
              f"(A) j={res_A['best']['j']}",
              f"(B) j*={res_B['best']['j_star']}",
              f"(C) alpha_c={res_C['best']['alpha_c']:+.1f}"]
    slopes = [V5P3_BASE_SLOPE, WB_SLOPE,
              res_A["best"]["total"], res_B["best"]["total"],
              res_C["best"]["total"]]
    colors = ["tab:purple", "k", "tab:blue", "tab:green", "tab:orange"]
    bars = ax.barh(labels, slopes, color=colors, alpha=0.8)
    ax.axvline(WB_SLOPE, color="k", ls=":", lw=1.5, label="WB -0.10")
    ax.axvline(V5P3_BASE_SLOPE, color="tab:purple", ls="--", lw=1.0,
               label="V5p3 +0.573")
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("slope_total = d log c / d log M_h")
    ax.set_title("Comparaison des alternatives vs WB target")
    for bar, val in zip(bars, slopes):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("V5.6 scoping : alternatives Wilson-loop multipath SU(2)_135",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Entry point
# ===========================================================================

def main(verbose: bool = True) -> dict:
    if verbose:
        print("=" * 70)
        print("V5.6 SCOPING --- alternatives Wilson-loop multipath SU(2)_135")
        print("=" * 70)
        print(f"  cible empirique WB              : slope = {WB_SLOPE:+.3f}")
        print(f"  base V5p3 (re-fit b107195)      : slope = {V5P3_BASE_SLOPE:+.3f}")
        print(f"  ecart a compenser               : {WB_SLOPE - V5P3_BASE_SLOPE:+.3f}")
        print(f"  niveau SU(2)_k                  : k = {V5_LEVEL}")
        print(f"  charge centrale c_135           : {central_charge():.4f}")
        print()

    res_A = scope_alternative_A()
    res_B = scope_alternative_B()
    res_C = scope_alternative_C()

    if verbose:
        print("ALTERNATIVE (A) -- Framing anomalies (Witten 1989)")
        b = res_A["best"]
        print(f"  best : j={b['j']}, f0={b['f0']}, delta={b['delta']:+.3f}")
        print(f"  contribution slope_A   = {b['contrib']:+.4f}")
        print(f"  total = V5p3 + A       = {b['total']:+.4f}")
        print(f"  distance a WB          = {abs(b['total'] - WB_SLOPE):.4f}")
        print()
        print("ALTERNATIVE (B) -- Multipath Verlinde (finite c)")
        b = res_B["best"]
        print(f"  best : j_star={b['j_star']}, xi={b['xi']:+.3f}, p={b['p']}")
        print(f"  contribution slope_B   = {b['contrib']:+.4f}")
        print(f"  total = V5p3 + B       = {b['total']:+.4f}")
        print(f"  distance a WB          = {abs(b['total'] - WB_SLOPE):.4f}")
        print()
        print("ALTERNATIVE (C) -- 1/c expansion (corrections finite c_135)")
        b = res_C["best"]
        print(f"  best : alpha_c={b['alpha_c']:+.2f}")
        print(f"  contribution slope_C   = {b['contrib']:+.4f}")
        print(f"  total = V5p3 + C       = {b['total']:+.4f}")
        print(f"  distance a WB          = {abs(b['total'] - WB_SLOPE):.4f}")
        print()

    dist = {
        "A": abs(res_A["best"]["total"] - WB_SLOPE),
        "B": abs(res_B["best"]["total"] - WB_SLOPE),
        "C": abs(res_C["best"]["total"] - WB_SLOPE),
    }
    winner = min(dist, key=dist.get)
    names = {
        "A": "Framing anomalies (Witten 1989)",
        "B": "Multipath Verlinde (fusion path counting)",
        "C": "1/c expansion (finite c_135)",
    }

    if verbose:
        print("=" * 70)
        print(f"GAGNANT DU SCOPING : Alternative ({winner}) -- {names[winner]}")
        print(f"  distance a WB target : {dist[winner]:.4f}")
        print(f"  effort estime closure complete :")
        if winner == "A":
            print("    - relier f_eff(M_h) au profil baryonique SPARC")
            print("    - regression Bayesienne sur (j, f0, delta)")
            print("    - 1-2 semaines (analytique + fit + tests)")
        elif winner == "B":
            print("    - calcul exact N_{j1 j2}^{j3} via Verlinde + S-matrix")
            print("    - relier j_max au halo via le profil de coherence chi")
            print("    - 2-3 semaines (CS-state-sum + regression)")
        else:
            print("    - derivation analytique alpha_c via OPE 1/c")
            print("    - confronter au profile de bord cosmologique (P22)")
            print("    - 3-4 semaines (calcul OPE + simulation)")
        print("=" * 70)

    out_png = OUTPUT_DIR / "v5p6_wilson_multipath_scoping.png"
    plot_scoping(res_A, res_B, res_C, out_png)
    if verbose:
        print(f"\nFigure ecrite : {out_png}")

    return {
        "res_A": res_A, "res_B": res_B, "res_C": res_C,
        "distances": dist, "winner": winner,
        "winner_name": names[winner],
    }


if __name__ == "__main__":
    main(verbose=True)
