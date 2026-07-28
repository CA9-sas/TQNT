"""V5.29b — Test du couplage c_J asymétrique inter-composantes.

Script principal V5.29b.  Modifie l'action S_min de V5.28 par un terme de
couplage c_J asymétrique inter-composantes :

    S_min/ℏ = κ_S × Σ_i c_i log(1 + c_J,i)
            + κ_C × Σ_{i<j} |c_J,i − c_J,j|       (linéaire, défaut)

κ_C est calibré sur l'observation D⁰ (V5.28 sur-prédit de +1.97 dex).

Sortie :
    output/v5p29b_cj_asymmetric.png

Référence : theory/v5p29_cj_asymmetric.py
Honnête : ce script teste trois formes (linear, quadratic, logratio) et
expose explicitement leur effet sur chaque méson.  La conclusion
attendue : aucune forme simple symétrique en (i,j) ne sauve D⁰ sans
casser B⁰ (V5.28 avait fermé le gap B⁰ à -0.73 dex ; V5.29b le rouvre).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.v5p29_cj_asymmetric import (
    asymmetry_coupling,
    calibrate_kappa_C,
    delta_m_v5p29b,
    v5p29b_oscillation_table,
    _reset_prefactor_cache,
    KAPPA_C,
)
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)


FORMS = ["linear", "quadratic", "logratio"]


def print_asymmetries():
    print("Asymétries c_J par méson (forme = linear) :")
    print(f"  {'Méson':<6} {'knots':<22} {'c_J values':<20} "
          f"{'asym_raw':>10}")
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        coup = asymmetry_coupling(cfg["knots"], kappa_C=1.0, form="linear")
        cJ_str = ", ".join(f"{x:.1f}" for x in coup["c_J_values"])
        knots_str = " + ".join(cfg["knots"])
        print(
            f"  {name:<6} {knots_str:<22} "
            f"[{cJ_str:<18}] {coup['asymmetry_raw']:>10.3f}"
        )
    print()


def print_table_for_form(form: str):
    _reset_prefactor_cache()
    kappa_C = calibrate_kappa_C(form=form)
    _reset_prefactor_cache()
    rows = v5p29b_oscillation_table(kappa_C=kappa_C, form=form)
    print(f"=== Forme de couplage : {form.upper()} (κ_C = {kappa_C:.4f}) ===")
    print(
        f"  {'Méson':<6} {'knots':<22} {'Δm V5.29b':>11} "
        f"{'Δm V5.28':>11} {'Δm PDG':>11} "
        f"{'log V5.29b':>10} {'log V5.28':>10}"
    )
    for r in rows:
        knots = " + ".join(r["knots"])
        print(
            f"  {r['name']:<6} {knots:<22} "
            f"{r['delta_m_v5p29b_MeV']:>11.3e} "
            f"{r['delta_m_v528_MeV']:>11.3e} "
            f"{r['delta_m_pdg_MeV']:>11.3e} "
            f"{r['log_ratio_v5p29b']:>+10.2f} "
            f"{r['log_ratio_v528']:>+10.2f}"
        )
    print()
    return kappa_C, rows


def make_plot(results_by_form):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    names = ["K0", "D0", "B0", "Bs0"]
    x = np.arange(len(names))
    w = 0.30
    for ax, (form, (kappa_C, rows)) in zip(axes, results_by_form.items()):
        log_v528 = [r["log_ratio_v528"] for r in rows]
        log_v29b = [r["log_ratio_v5p29b"] for r in rows]
        ax.bar(x - w / 2, log_v528, w, color="#888888", label="V5.28")
        ax.bar(x + w / 2, log_v29b, w, color="#1f77b4", label="V5.29b")
        ax.axhline(0, color="k", lw=1)
        ax.axhline(+1, ls="--", color="g", lw=0.6, alpha=0.5)
        ax.axhline(-1, ls="--", color="g", lw=0.6, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["K⁰", "D⁰", "B⁰", "Bs⁰"])
        ax.set_title(f"V5.29b — {form} (κ_C={kappa_C:.3f})", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel(r"$\log_{10}(\Delta m_{V5}/\Delta m_{PDG})$")
    plt.tight_layout()
    out = "output/v5p29b_cj_asymmetric.png"
    plt.savefig(out, dpi=150)
    print(f"Plot sauvegardé : {out}")


def main():
    print()
    print("V5.29b — Couplage c_J asymétrique inter-composantes")
    print("Reframe : S_min += κ_C × Σ_{i<j} |c_J,i − c_J,j|")
    print()
    print_asymmetries()

    results_by_form = {}
    for form in FORMS:
        kc, rows = print_table_for_form(form)
        results_by_form[form] = (kc, rows)
    make_plot(results_by_form)

    print()
    print("=" * 76)
    print("Verdict V5.29b (honnête)")
    print("=" * 76)
    print(
        "- K⁰  : ΔS_couplage = 0 (composantes identiques 6_1+6_1+6_1).")
    print(
        "        Préfacteur K⁰ inchangé → Δm V5.29b = Δm PDG (par calibration).")
    print(
        "- D⁰  : κ_C est calibré pour fermer le +1.97 dex de V5.28 → log ≈ 0.")
    print(
        "- B⁰  : asymétrie |Δc_J| = 11.6 > celle de D⁰ (9.6).  Avec κ_C")
    print(
        "        calibré sur D⁰, ΔS_couplage_B⁰ ≈ 5.5 > ΔS_couplage_D⁰ ≈ 4.5,")
    print(
        "        et B⁰ tombe à environ -3.1 dex.  Le gap B⁰ que V5.28 avait")
    print(
        "        fermé (-0.73 dex) se ROUVRE en V5.29b.")
    print(
        "- Bs⁰ : passe de -3.30 dex à environ -3.63 dex (légère dégradation).")
    print()
    print(
        "CONCLUSION : tout couplage c_J asymétrique purement pairwise et")
    print(
        "symétrique en (i,j) hérite de la structure |c_J,1 - c_J,2|"
        " + ... qui")
    print(
        "fait que B⁰ (3.3, 7.1, 9.1) a une asymétrie supérieure à D⁰")
    print(
        "(3.3, 5.5, 8.1).  V5.29b sauve D⁰ mais NE FONCTIONNE PAS comme")
    print(
        "amélioration globale : on troque le problème D⁰ (+1.97 dex) contre")
    print(
        "un problème B⁰ (-3.1 dex), pour un cumul |dex| pire que V5.28.")
    print()
    print(
        "Piste V5.29c (Kashaev / phase complexe) à privilégier — la simple")
    print(
        "asymétrie c_J ne capture pas la structure du secteur b → s du PDG.")


if __name__ == "__main__":
    main()
