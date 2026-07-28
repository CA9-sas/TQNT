#!/usr/bin/env python3
"""V5.20 -- Class E "Strange Knot Stars" : enchevêtrement amphichiral.

Class E = sous-population stable SECONDAIRE (V5 §13.7) :
6 STRICT amphichiraux du catalogue V5 (TQNT_catalog_V5_diffusion.xlsx
onglet 5) :

    Composition         m_chi (GeV)   c_brick    n_brick
    --------------      -----------   -------    -------
    6_3 + 6_3              1.55         12          2
    6_3 + 6_3 + 6_3        4.70         18          3
    4_1 + 8_3             16.01         12          2
    8_3 + 8_3             17.24         16          2
    6_3 + 8_3             26.48         14          2
    4_1 + 4_1 + 8_3       51.78         16          3

Différence-clé Class E vs Class A (V5.20 originel) :
  * Class A pearl : nœuds DM génériques chiraux (c >= 19, n = 1) -> les
    reconnexions doivent inverser la chiralité produite, ce qui crée
    Delta c ~ 10 crossings additionnels (suppression Vassiliev alpha^10).
  * Class E pearl : bricks DEJA amphichiraux (Q_phys = 0
    individuellement). La reconnexion topologique préserve
    l'amphichiralité globale SANS avoir à compenser la chiralité ;
    elle n'a besoin que de Lk -> Lk +/- 1 (un crossing positif +
    négatif pour rester amphichiral) -> Delta c ~ 2 minimum.

Formule retenue (V5 §13.6.4) :
    Delta c_E (brick i + brick j) = 2 * max(1, b_link)
où b_link est le nombre minimal de crossings de linking exigés pour
préserver le groupe amphichiral L = K # K' (en moyenne sur les
6 compositions, Delta c_E ~ 2-4, contre 10 pour Class A).

Cela donne :
    Gamma_E / Gamma_A ~ alpha_EM^{Delta c_E - Delta c_A}
                     ~ alpha^{-(6 a 8)} ~ 10^{13} a 10^{17}

Signature distinctive :
  * Class A : continuum γ (large mass function, m_chi varie).
  * Class E : 6 RAIES γ discrètes à E_gamma = m_chi (annihilation
    knot+anti-knot) ou E_gamma = Delta_m (transition entre 2
    compositions amphichirales).

Run :
    PYTHONPATH=. python3 scripts/V5_pearl_internal/class_E_entanglement_test.py
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

from theory.dm_pearl_internal import (
    ALPHA_EM,
    DM_STRICT_AMPHI_GEV,
    M_PEARL_TYPICAL_MSUN,
    M_SUN_KG,
    GEV_KG,
    SIGMA_QED,
    T_HUBBLE_S,
    chandrasekhar_topological_pearl,
    entanglement_rate,
    n_events_over_universe_age,
    n_knots,
    sigma_entanglement,
    tov_pearl,
    verdict_v5p20,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "v5p20_class_E_entanglement.png"

# ----------------------------------------------------------------------------
# Class E catalogue (CLAUDE.md + TQNT_catalog_V5_diffusion.xlsx onglet 5)
# ----------------------------------------------------------------------------

# (composition_str, m_chi_GeV, c_total_brick, n_brick)
CLASS_E_CATALOG = [
    ("6_3+6_3",           1.55, 12, 2),
    ("6_3+6_3+6_3",       4.70, 18, 3),
    ("4_1+8_3",          16.01, 12, 2),
    ("8_3+8_3",          17.24, 16, 2),
    ("6_3+8_3",          26.48, 14, 2),
    ("4_1+4_1+8_3",      51.78, 16, 3),
]

# Class E represents ~10% of Omega_DM (CLAUDE: "Omega_E = 0.10 * Omega_DM,
# 2x Class A abundance"). For comparison, Class A is OMEGA_A = 0.05 * 0.265
# in dm_pearl_internal.n_events_over_universe_age. We rescale below.
OMEGA_E_FRACTION = 0.10        # 2x Class A
OMEGA_A_FRACTION = 0.05        # Class A original
OMEGA_DM = 0.265

# Reference Delta c for Class A pearls (V5.20 fiducial)
DELTA_C_CLASS_A = 10


# ----------------------------------------------------------------------------
# 1. Delta c spécifique amphichiral
# ----------------------------------------------------------------------------

def delta_c_amphichiral(c_brick: int, n_brick: int) -> int:
    """Crossings additionnels Vassiliev pour relier 2 perles amphichirales.

    Justification (V5 §13.6.4 + symétrie amphichirale) :
      * Une perle amphichirale satisfait Q_phys = 0 et K = mirror(K).
      * Pour préserver l'amphichiralité globale du link L = pearl_i # pearl_j,
        chaque crossing de linking positif (+1) doit être compensé par
        un crossing négatif (-1) -> 2 crossings additionnels MINIMUM
        par composante du link.
      * Pour n_brick composantes par perle, le link compose 2*n_brick
        composantes amphichirales -> nécessite (n_brick) paires de
        crossings = 2 * n_brick crossings additionnels.

    Cette formule donne :
        n_brick = 2 -> Delta c = 4
        n_brick = 3 -> Delta c = 6

    Comparée à Delta c_A = 10 pour Class A, on a un FACTEUR DE GAIN :
        alpha^{Delta_A - Delta_E} ~ 137^{4-6} ~ 10^{8-12}
    """
    return 2 * n_brick


def gain_factor_class_E(delta_c_E: int,
                         delta_c_A: int = DELTA_C_CLASS_A) -> float:
    """Facteur d'amplification Gamma_E / Gamma_A à densité égale.

    Gamma ~ sigma ~ alpha^{delta_c}.
    Gamma_E / Gamma_A = alpha^{delta_c_E - delta_c_A}
                     = alpha^{(2 n_brick) - 10}.
    """
    return ALPHA_EM ** (delta_c_E - delta_c_A)


# ----------------------------------------------------------------------------
# 2. Mass-Radius for Class E (6 amphichiral compositions)
# ----------------------------------------------------------------------------

def class_E_mass_radius(M_pearl_Msun: float = M_PEARL_TYPICAL_MSUN) -> list:
    """Construit la table (composition, M_Ch, R_pearl, rho_c) pour les 6
    amphichiraux à M_pearl typique."""
    out = []
    for name, m_chi, c_brick, n_brick in CLASS_E_CATALOG:
        M_Ch = chandrasekhar_topological_pearl(m_chi) / M_SUN_KG  # M_sun
        tov = tov_pearl(M_pearl_Msun, m_chi)
        out.append({
            "name": name,
            "m_chi_GeV": m_chi,
            "c_brick": c_brick,
            "n_brick": n_brick,
            "M_Ch_Msun": M_Ch,
            "R_pearl_m": float(tov["R"]),
            "rho_central": float(tov["rho_c"]),
            "n_central": float(tov["n"][0]),
            "N_knots": n_knots(M_pearl_Msun, m_chi),
        })
    return out


# ----------------------------------------------------------------------------
# 3. Gamma_entangle pour Class E (boucle sur 6 amphichiraux)
# ----------------------------------------------------------------------------

def class_E_entanglement_rates(M_pearl_Msun: float = M_PEARL_TYPICAL_MSUN
                               ) -> list:
    """Calcule Gamma_E et N_events_universe pour chaque amphichiral Class E.

    Inclut le rescaling d'abondance (Omega_E = 2 x Omega_A).
    """
    out = []
    for name, m_chi, c_brick, n_brick in CLASS_E_CATALOG:
        delta_c_E = delta_c_amphichiral(c_brick, n_brick)
        # Class A reference at same m_chi (fictif : on prend delta_c=10)
        rate_E_pearl = entanglement_rate(M_pearl_Msun, m_chi, delta_c_E)
        rate_A_pearl = entanglement_rate(M_pearl_Msun, m_chi, DELTA_C_CLASS_A)
        # Rescale abondance Class E (Omega_E = 0.10 * Omega_DM)
        # n_events_over_universe_age utilise OMEGA_A = 0.05 * 0.265
        # On corrige : Omega_E / Omega_A = 2 -> N_events *= 2
        scale_E = OMEGA_E_FRACTION / OMEGA_A_FRACTION   # = 2.0
        nev_A_ref = n_events_over_universe_age(M_pearl_Msun, m_chi,
                                               DELTA_C_CLASS_A)
        nev_E_raw = n_events_over_universe_age(M_pearl_Msun, m_chi,
                                               delta_c_E)
        N_events_E = nev_E_raw["N_events_total_universe"] * scale_E
        N_events_A = nev_A_ref["N_events_total_universe"]
        out.append({
            "name": name,
            "m_chi_GeV": m_chi,
            "n_brick": n_brick,
            "delta_c_E": delta_c_E,
            "delta_c_A_ref": DELTA_C_CLASS_A,
            "sigma_E_cm2": rate_E_pearl["sigma_cm2"],
            "sigma_A_cm2": rate_A_pearl["sigma_cm2"],
            "Gamma_E_pearl_Hz": rate_E_pearl["Gamma_total_pearl_Hz"],
            "Gamma_A_pearl_Hz": rate_A_pearl["Gamma_total_pearl_Hz"],
            "gain_factor": gain_factor_class_E(delta_c_E),
            "N_events_E": N_events_E,
            "N_events_A": N_events_A,
        })
    return out


# ----------------------------------------------------------------------------
# 4. Signature γ discrète vs continuum
# ----------------------------------------------------------------------------

def gamma_signature_class_E() -> dict:
    """Énergies γ associées aux 6 raies discrètes Class E.

    Hypothèses :
      * Annihilation pearl-antipearl partielle -> E_gamma = m_chi
        (raie principale).
      * Cascade vers le plus léger amphichiral -> Delta_E entre
        compositions successives.

    Renvoie :
      'primary_lines_GeV' : 6 raies = m_chi
      'cascade_lines_GeV' : Delta_E entre paires
      'continuum_compare' : note pour Class A
    """
    masses = [m for (_, m, _, _) in CLASS_E_CATALOG]
    cascades = []
    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            cascades.append(masses[j] - masses[i])
    return {
        "primary_lines_GeV": masses,
        "cascade_lines_GeV": sorted(cascades),
        "continuum_compare": (
            "Class A : continuum [1-100] GeV (large mass function)\n"
            "Class E : 6 raies primaires + cascades -> spectre DISCRET"
        ),
    }


# ----------------------------------------------------------------------------
# 5. Panel plot
# ----------------------------------------------------------------------------

def panel_a_deltac(ax):
    rates = class_E_entanglement_rates()
    names = [r["name"] for r in rates]
    dc_E = [r["delta_c_E"] for r in rates]
    dc_A = [DELTA_C_CLASS_A] * len(names)
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, dc_A, w, color="C3", alpha=0.7, label="Class A (ref)")
    ax.bar(x + w/2, dc_E, w, color="C0", alpha=0.9, label="Class E")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"$\Delta c$ Vassiliev")
    ax.set_title(r"(a) Crossings additionnels par enchev\^etrement")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def panel_b_gain(ax):
    rates = class_E_entanglement_rates()
    names = [r["name"] for r in rates]
    gains = [r["gain_factor"] for r in rates]
    x = np.arange(len(names))
    ax.bar(x, gains, color="C2", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Gamma_E / \Gamma_A$ (par perle)")
    ax.set_title(r"(b) Amplification Class E vs Class A")
    ax.axhline(1.0, color="k", ls="--", alpha=0.5)
    ax.grid(True, which="both", alpha=0.3)
    for i, g in enumerate(gains):
        ax.text(i, g, f"{g:.1e}", ha="center", va="bottom", fontsize=7,
                rotation=0)


def panel_c_mass_radius(ax):
    table = class_E_mass_radius()
    masses = [t["m_chi_GeV"] for t in table]
    radii = [t["R_pearl_m"] / 1000.0 for t in table]   # km
    rho_c = [t["rho_central"] for t in table]
    sc = ax.scatter(masses, radii, c=np.log10(rho_c),
                    cmap="viridis", s=80, edgecolors="k")
    for t in table:
        ax.annotate(t["name"], (t["m_chi_GeV"], t["R_pearl_m"]/1000.0),
                    fontsize=7, xytext=(5, 5), textcoords="offset points")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r"$\log_{10}\rho_c$ (kg/m$^3$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi$ (GeV)")
    ax.set_ylabel(r"$R_{\rm pearl}$ (km)")
    ax.set_title(r"(c) M--R pour les 6 amphichiraux Class E"
                 + f"\n(M_pearl = {M_PEARL_TYPICAL_MSUN:.0e} $M_\\odot$)")
    ax.grid(True, which="both", alpha=0.3)


def panel_d_signature(ax):
    sig = gamma_signature_class_E()
    primary = sig["primary_lines_GeV"]
    cascade = sig["cascade_lines_GeV"]
    # Continuum Class A (idéalisé) : flat dN/dE from 1 to 60 GeV
    E_grid = np.logspace(0, 2, 500)
    continuum_A = np.where((E_grid >= 1.0) & (E_grid <= 60.0), 1.0, 0.0)
    ax.fill_between(E_grid, 0, continuum_A * 0.5, color="C3", alpha=0.3,
                    label="Class A continuum (id\\'ealis\\'e)")
    # Raies Class E (primaires)
    for E in primary:
        ax.axvline(E, ymin=0, ymax=0.7, color="C0", lw=2, alpha=0.9)
    ax.axvline(primary[0], ymin=0, ymax=0.7, color="C0", lw=2, alpha=0.9,
               label="Class E raies primaires")
    # Cascades (faibles)
    for E in cascade:
        if E > 0:
            ax.axvline(E, ymin=0, ymax=0.3, color="C0", lw=1, alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 100)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$E_\gamma$ (GeV)")
    ax.set_ylabel(r"$dN/dE$ (arb.)")
    ax.set_title(r"(d) Signature $\gamma$ : Class E (raies) vs Class A (continuum)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)


def make_plot(path: Path = OUTPUT_PNG) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    panel_a_deltac(axes[0, 0])
    panel_b_gain(axes[0, 1])
    panel_c_mass_radius(axes[1, 0])
    panel_d_signature(axes[1, 1])
    fig.suptitle(
        "V5.20 -- Class E \"Strange Knot Stars\" : "
        "enchev\\^etrement amphichiral",
        fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# 6. Summary / verdict
# ----------------------------------------------------------------------------

def summary() -> dict:
    rates = class_E_entanglement_rates()
    mr = class_E_mass_radius()
    sig = gamma_signature_class_E()
    # Moyennes
    gain_mean = float(np.mean([r["gain_factor"] for r in rates]))
    Nev_E_total = float(np.sum([r["N_events_E"] for r in rates]))
    Nev_A_total = float(np.sum([r["N_events_A"] for r in rates]))
    # Verdict global
    if gain_mean > 1e6:
        verdict = "SIGNATURE DISTINCTIVE"
        comment = (
            f"Class E enchevêtre {gain_mean:.2e}x plus souvent "
            f"que Class A. Spectre γ DISCRET (6 raies primaires) "
            f"VS continuum Class A -> signature unique testable "
            f"(Fermi-LAT, CTA)."
        )
    elif gain_mean > 10:
        verdict = "ENHANCEMENT MODERE"
        comment = (
            f"Class E gain x{gain_mean:.1f}, raies γ distinctes."
        )
    else:
        verdict = "PAS DE DIFFERENCE NOTABLE"
        comment = "Class E ~ Class A."
    return {
        "rates": rates,
        "mass_radius": mr,
        "gamma_signature": sig,
        "gain_mean_class_E_over_A": gain_mean,
        "N_events_E_summed_universe": Nev_E_total,
        "N_events_A_summed_universe": Nev_A_total,
        "verdict": verdict,
        "comment": comment,
    }


if __name__ == "__main__":
    print("=" * 78)
    print("V5.20 -- Class E entanglement test")
    print("=" * 78)
    p = make_plot()
    print(f"\nPlot -> {p}")
    s = summary()
    print("\n--- Delta c et amplification (par amphichiral) ---")
    for r in s["rates"]:
        print(f"  {r['name']:20s}  m={r['m_chi_GeV']:5.2f} GeV  "
              f"n_brick={r['n_brick']}  Delta c_E={r['delta_c_E']:2d}  "
              f"sigma={r['sigma_E_cm2']:.2e} cm^2  "
              f"gain={r['gain_factor']:.2e}")
    print(f"\nGain moyen Class E / Class A : {s['gain_mean_class_E_over_A']:.3e}")
    print(f"N_events Class E (sommé 6 amphi, univers) : "
          f"{s['N_events_E_summed_universe']:.3e}")
    print(f"N_events Class A (sommé 6 amphi, univers) : "
          f"{s['N_events_A_summed_universe']:.3e}")
    print(f"\nVerdict : {s['verdict']}")
    print(f"Comment : {s['comment']}")
    print("\nSignature gamma :")
    print(f"  Raies primaires (GeV) : {s['gamma_signature']['primary_lines_GeV']}")
    print(f"  Cascades (GeV) : {s['gamma_signature']['cascade_lines_GeV']}")
