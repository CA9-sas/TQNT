#!/usr/bin/env python3
"""V5.18 — Test SMF haute-z avec boost P22 RIGOUREUX (V5 §16.4).

Remplace l'ansatz LINÉAIRE boost(z) = 1 + α·max(0, z-5) du script
jwst_smf_test.py (V5 nominal §11.5) par la FORME FERMÉE rigoureuse :

    boost(M_h, z) = exp(η · ξ(z, M_h))
    ξ(z, M_h) = κ_g · (|Ψ_halo|² - |Ψ_cosmo|²) · λbar_C(e) / r_vir(M_h, z)

et confronte à 17 points JWST (JADES + COSMOS-Web + Boylan-Kolchin 2023).

Output : output/v5p18_rigorous_boost_vs_jwst.png

Run :
    PYTHONPATH=. python3 scripts/V5_JWST/v5p18_rigorous_boost_test.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.smf_V5 import (
    SMF_LambdaCDM, sheth_tormen_HMF,
    SHMR_Behroozi, dM_halo_dM_star, M_halo_from_M_star,
    JWST_SMF_DATA, JWST_Z_MID,
)
from theory.p22_boost_rigorous import (
    boost_P22_rigorous, xi_at_halo_boundary,
    xi_required_for_target_boost, gap_to_jwst,
    diagnose, KAPPA_G_M, LAMBDABAR_C_E_M, ETA_P22_DEFAULT,
)


OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Wrap : SMF avec boost rigoureux V5.18
# ---------------------------------------------------------------------------
def SMF_V5p18(M_star_Msun: float, z: float, eta: float = 1.0) -> float:
    """SMF V5.18 = SMF_LambdaCDM(M*, z) · boost_P22_rigorous(M_h(M*), z).

    Le boost est appliqué à l'échelle du halo M_h (que l'on déduit
    numériquement de M*) puis transporté à la SMF via Jacobien SHMR.
    """
    smf_l = SMF_LambdaCDM(M_star_Msun, z)
    M_h = M_halo_from_M_star(M_star_Msun, z)
    return smf_l * boost_P22_rigorous(M_h, z, eta=eta)


def chi2_v5p18(eta: float = 1.0) -> tuple[float, int, dict]:
    """χ² total + détails par bin."""
    chi2_tot = 0.0
    N = 0
    per_bin = {}
    for bin_label, points in JWST_SMF_DATA.items():
        z = JWST_Z_MID[bin_label]
        chi2_bin = 0.0
        for log_M, log_phi_obs, sig in points:
            M = 10 ** log_M
            phi_pred = SMF_V5p18(M, z, eta=eta)
            log_phi_pred = math.log10(max(phi_pred, 1e-30))
            chi2_bin += ((log_phi_pred - log_phi_obs) / sig) ** 2
        per_bin[bin_label] = (chi2_bin, len(points))
        chi2_tot += chi2_bin
        N += len(points)
    return chi2_tot, N, per_bin


def chi2_lcdm() -> tuple[float, int, dict]:
    """χ² total ΛCDM (boost = 1)."""
    chi2_tot = 0.0
    N = 0
    per_bin = {}
    for bin_label, points in JWST_SMF_DATA.items():
        z = JWST_Z_MID[bin_label]
        chi2_bin = 0.0
        for log_M, log_phi_obs, sig in points:
            M = 10 ** log_M
            phi_pred = SMF_LambdaCDM(M, z)
            log_phi_pred = math.log10(max(phi_pred, 1e-30))
            chi2_bin += ((log_phi_pred - log_phi_obs) / sig) ** 2
        per_bin[bin_label] = (chi2_bin, len(points))
        chi2_tot += chi2_bin
        N += len(points)
    return chi2_tot, N, per_bin


def main():
    print("=" * 78)
    print("V5.18 — JWST SMF avec boost P22 RIGOUREUX (forme fermée V5 §16.4)")
    print("=" * 78)
    print(f"\nConstantes :")
    print(f"  κ_g             = {KAPPA_G_M:.3e} m")
    print(f"  λbar_C(e)       = {LAMBDABAR_C_E_M:.3e} m")
    print(f"  η_P22 (défaut)  = {ETA_P22_DEFAULT}")

    # -----------------------------------------------------------------
    # 1. Diagnostic à z=10, M_h=10^10
    # -----------------------------------------------------------------
    print(f"\n--- Diagnostic V5.18 à (z=10, M_h=10¹⁰ Msun) ---")
    d = diagnose(1e10, 10.0)
    print(f"  r_vir            = {d.r_vir_m:.3e} m = {d.r_vir_kpc:.3f} kpc")
    print(f"  n_baryon         = {d.n_baryon_m3:.3e} m⁻³")
    print(f"  h_halo = Ω²-1    = {d.h_halo:.3e}")
    print(f"  |∇Ω²|            = {d.grad_omega_squared:.3e} m⁻¹")
    print(f"  ξ                = {d.xi:.3e}")
    print(f"  boost V5.18      = {d.boost:.6f}")

    g = gap_to_jwst(1e10, 10.0)
    print(f"\n--- Comparaison cible JWST ---")
    print(f"  ξ requis pour boost ~10³ : {g['xi_required_JWST']:.3f}")
    print(f"  ξ rigoureux V5.18        : {g['xi_rigorous']:.3e}")
    print(f"  Écart log10              : {g['log10_gap']:.1f} décades")
    print(f"  Boost V5.5 nominal §11.5 : 1.500 (z=10)")
    print(f"  Boost V5.18 rigoureux    : {d.boost:.6f}")
    print(f"  Boost JWST requis        : ~10³")

    # -----------------------------------------------------------------
    # 2. χ²
    # -----------------------------------------------------------------
    print(f"\n--- χ² sur 17 points JWST ---")
    chi2_l, N_l, perbin_l = chi2_lcdm()
    chi2_v, N_v, perbin_v = chi2_v5p18(eta=1.0)
    print(f"  N data points                 = {N_l}")
    print(f"  χ²(ΛCDM)                      = {chi2_l:.2f}")
    print(f"  χ²(V5.18 rigoureux, η=1)      = {chi2_v:.2f}")
    print(f"  Δχ² = χ²(LCDM) - χ²(V5.18)    = {chi2_l - chi2_v:.2e}")
    print(f"  (V5.18 numériquement indistinct de ΛCDM)")
    for bin_label in JWST_SMF_DATA:
        cl, nl = perbin_l[bin_label]
        cv, nv = perbin_v[bin_label]
        print(f"    {bin_label:8s}: χ²(LCDM)={cl:7.2f}  χ²(V5.18)={cv:7.2f}  N={nl}")

    # -----------------------------------------------------------------
    # 3. Plot
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) SMF à z=8.5 et z=10.5 : LCDM, V5.5 nominal (boost 1.5), V5.18
    ax = axes[0, 0]
    log_M = np.linspace(8.0, 11.0, 60)
    for z, color in [(8.5, "tab:blue"), (10.5, "tab:red")]:
        smf_l = [SMF_LambdaCDM(10 ** lm, z) for lm in log_M]
        smf_v18 = [SMF_V5p18(10 ** lm, z, eta=1.0) for lm in log_M]
        # V5.5 nominal : ×1.5 à z=10
        boost_nom = 1.0 + 0.10 * max(0.0, z - 5.0)
        smf_nom = [s * boost_nom for s in smf_l]
        ax.plot(log_M, np.log10(smf_l), color=color, ls="--",
                label=fr"$\Lambda$CDM, $z={z}$")
        ax.plot(log_M, np.log10(smf_nom), color=color, ls=":",
                label=fr"V5.5 nominal ×{boost_nom:.2f}, $z={z}$")
        ax.plot(log_M, np.log10(smf_v18), color=color, ls="-", lw=2,
                label=fr"V5.18 rigoureux, $z={z}$")
    for z_bin, marker in [("z=8-9", "o"), ("z=10-12", "s")]:
        data = JWST_SMF_DATA[z_bin]
        Ms = [d[0] for d in data]
        phis = [d[1] for d in data]
        errs = [d[2] for d in data]
        col = "tab:blue" if z_bin == "z=8-9" else "tab:red"
        ax.errorbar(Ms, phis, yerr=errs, fmt=marker, color=col, capsize=3,
                    label=f"JWST {z_bin}", markersize=8,
                    markerfacecolor="white", markeredgecolor=col, mew=1.5)
    ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(\Phi /$Mpc$^{-3}$dex$^{-1})$")
    ax.set_title("(a) SMF : V5.18 vs ΛCDM vs V5.5 nominal vs JWST")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.set_xlim(8.0, 11.0)
    ax.set_ylim(-9.5, -2.5)
    ax.grid(alpha=0.3)

    # (b) ξ(z, M_h) cartographie
    ax = axes[0, 1]
    z_grid = np.linspace(5.0, 15.0, 60)
    for Mh, color, ls in [(1e8, "tab:blue", "--"),
                          (1e10, "tab:orange", "-"),
                          (1e12, "tab:red", ":")]:
        xis = [xi_at_halo_boundary(Mh, z) for z in z_grid]
        ax.semilogy(z_grid, xis, color=color, ls=ls, lw=2,
                    label=fr"$M_h = 10^{{{int(np.log10(Mh))}}}\,M_\odot$")
    xi_req = xi_required_for_target_boost(1e3)
    ax.axhline(xi_req, color="purple", ls="--", lw=1.5,
               label=fr"$\xi$ requis pour boost $10^3$ JWST")
    ax.axhline(1.0, color="k", ls=":", alpha=0.5, label=r"$\xi = 1$ (régime baryonique)")
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"$\xi = |\nabla\Omega^2|\,\bar\lambda_C/\Omega^2$")
    ax.set_title("(b) $\\xi$ V5.18 rigoureux vs requis JWST")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    # (c) boost(z, M_h) — toujours ≈ 1
    ax = axes[1, 0]
    for Mh, color, ls in [(1e8, "tab:blue", "--"),
                          (1e10, "tab:orange", "-"),
                          (1e12, "tab:red", ":")]:
        boosts = [boost_P22_rigorous(Mh, z) for z in z_grid]
        ax.semilogy(z_grid, boosts, color=color, ls=ls, lw=2,
                    label=fr"V5.18, $M_h = 10^{{{int(np.log10(Mh))}}}$")
    # V5.5 nominal
    boost_nom = [1.0 + 0.10 * max(0.0, z - 5.0) for z in z_grid]
    ax.semilogy(z_grid, boost_nom, "tab:green", lw=1.5, ls="-.",
                label="V5.5 nominal (ansatz lin.)")
    ax.axhline(1e3, color="purple", ls="--", lw=1.5, label="cible JWST")
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"boost P22")
    ax.set_title("(c) Boost V5.18 vs V5.5 nominal vs cible JWST")
    ax.set_ylim(1e-2, 1e5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # (d) χ²
    ax = axes[1, 1]
    bins = list(JWST_SMF_DATA.keys())
    chi2_l_list = [perbin_l[b][0] for b in bins]
    chi2_v_list = [perbin_v[b][0] for b in bins]
    x = np.arange(len(bins))
    w = 0.35
    ax.bar(x - w / 2, chi2_l_list, w, label=r"$\Lambda$CDM", color="tab:gray")
    ax.bar(x + w / 2, chi2_v_list, w, label="V5.18 rigoureux", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel(r"$\chi^2$ par bin")
    ax.set_title(f"(d) χ² par bin (total LCDM={chi2_l:.1f},"
                 f" V5.18={chi2_v:.1f})")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        "TQNT-V5.18 — Boost P22 rigoureux (V5 §16.4) confronté à JWST",
        fontsize=14, y=1.00,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "v5p18_rigorous_boost_vs_jwst.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved figure: {out}")

    # -----------------------------------------------------------------
    # 4. Verdict
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("VERDICT V5.18")
    print("=" * 78)
    if d.boost < 1.001 and abs(chi2_v - chi2_l) < 0.01:
        print("  Boost P22 rigoureux ≈ 1 partout (ξ ≪ 1 par 70+ décades).")
        print("  ⇒ V5.18 est NUMÉRIQUEMENT INDISTINCT de ΛCDM.")
        print("  ⇒ V5 §11.5 (boost ×1,5) était DÉJÀ surévalué : la forme")
        print("    rigoureuse de P22 (§16.4) NE produit PAS d'amplification HMF.")
        print("  ⇒ La 'crise des galaxies impossibles' JWST n'est PAS expliquée")
        print("    par P22 dans la formulation actuelle.")
        print()
        print("  Pistes pour V5.19+ :")
        print("   • réviser P22 (mécanisme cascade hiérarchique ?)")
        print("   • SHMR à haut z (les estimations stellaires JWST sont fragiles)")
        print("   • IMF top-heavy (boost effectif via lumière/masse)")
        print("   • réviser P5 (κ_g effectif augmenté dans régime primordial ?)")
    elif d.boost > 100:
        print(f"  Boost rigoureux V5.18 = {d.boost:.1f}, VICTOIRE majeure pour V5.")
    else:
        print(f"  Boost intermédiaire = {d.boost:.3f} ; V5 partiellement.")
    return {
        "xi_rigorous_z10_Mh10e10": d.xi,
        "boost_rigorous_z10_Mh10e10": d.boost,
        "boost_v5_nominal_z10": 1.5,
        "boost_target_jwst": 1e3,
        "chi2_LCDM": chi2_l,
        "chi2_V5p18": chi2_v,
        "delta_chi2": chi2_l - chi2_v,
        "N_data": N_l,
        "log10_gap": g["log10_gap"],
    }


if __name__ == "__main__":
    main()
