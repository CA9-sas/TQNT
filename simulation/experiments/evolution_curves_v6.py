"""Courbes d'évolution masse/spin/charge/durée-de-vie entre repos et haute énergie.

Modèle : la rétraction du champ χ donne la fraction de confinement
    f(E) = ρ_J_tot / (ρ_J_tot + E/E₀),   ρ_J_tot = Σρ_J,   E₀ = 33 MeV
qui passe de 1 (E=0, confiné) à 0 (E→∞, déconfiné), en valant 1/2 au SEUIL
    E_seuil = E₀·ρ_J_tot ~ ħc/r_link ~ 200 MeV.

Les quatre observables interpolent entre repos et haute énergie par f(E) :
    m(E)  = m_HE + (m_repos − m_HE)·f                 (déconfinement)
    Q(E)  = Σχ + Lk·f                                  (charge : total → constituants)
    S(E)  = S + (Σsᵢ − S)·(1−f)                        (spin : total → constituants)
    τ(E)  = τ_repos^f · τ_HE^(1−f)                     (vie : log-interpolation)

Le SEUIL E_seuil est le même pour les quatre (même f) : c'est l'énergie où la
particule bascule du régime confiné au régime déconfiné. Il diffère par particule.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.chi_field import jones_rho
from theory.jones import braid_index
from simulation.experiments.reassign_pdg import sigma_chi

HBARC, E0, HBAR = 197.327, 33.0, 6.582e-25
AMPHI = {"4_1", "6_3", "8_3"}
ISO = {"3_1": 0.51, "5_2": 1.96, "4_1": 4.87, "6_1": 19.04, "5_1": 23.04, "7_1": 51.34,
       "7_2": 50.46, "9_1": 81.64, "6_2": 88.65}


def chi(k): return 0 if k in AMPHI else sigma_chi(k)[1]
def s_knot(k): return (braid_index(k) - 1) / 2


def model(cl, Q_tot, S_tot, m_repos, tau_repos):
    rhoJ = sum(jones_rho(k) for k in cl)
    E_seuil = E0 * rhoJ
    m_HE = sum(ISO[k] for k in cl)
    Schi = sum(chi(k) for k in cl); Lk = Q_tot - Schi
    Ssum = sum(s_knot(k) for k in cl)
    tau_HE = HBAR / (E_seuil / 1000.0)
    E = np.logspace(0, 4, 400)                      # 1 MeV → 10 GeV
    f = rhoJ / (rhoJ + E / E0)
    return E, E_seuil, {
        "m": m_HE + (m_repos - m_HE) * f,
        "Q": Schi + Lk * f,
        "S": S_tot + (Ssum - S_tot) * (1 - f),
        "tau": tau_repos ** f * tau_HE ** (1 - f),
    }, {"m": (m_repos, m_HE), "Q": (Q_tot, Schi), "S": (S_tot, Ssum), "tau": (tau_repos, tau_HE)}


PARTS = [
    ("proton", ["3_1", "7_2", "7_2"], 1, 0.5, 938.5, 1e30),
    ("neutron", ["3_1", "7_2", "7_2"], 0, 0.5, 939.8, 879.0),
    ("muon", ["3_1", "6_1"], -1, 0.5, 105.7, 2.2e-6),
    ("pion", ["3_1", "3_1", "5_2"], 1, 0.0, 139.6, 2.6e-8),
]
LBL = {"m": "masse (MeV)", "Q": "charge", "S": "spin", "tau": "τ (s)"}

fig, axes = plt.subplots(len(PARTS), 4, figsize=(15, 3 * len(PARTS)))
print(f"{'particule':>9} | {'ρ_J_tot':>7} | {'E_seuil (MeV)':>13}")
for row, (nm, cl, Q, S, mr, tr) in enumerate(PARTS):
    E, Es, curves, ends = model(cl, Q, S, mr, tr)
    print(f"{nm:>9} | {sum(jones_rho(k) for k in cl):>7.1f} | {Es:>13.0f}")
    for col, key in enumerate(["m", "Q", "S", "tau"]):
        ax = axes[row, col]
        ax.plot(E, curves[key], "b-")
        ax.axvline(Es, color="r", ls="--", lw=1, label=f"E_seuil={Es:.0f}")
        ax.set_xscale("log")
        if key in ("m", "tau"):
            ax.set_yscale("log")
        ax.set_xlabel("E (MeV)"); ax.set_ylabel(LBL[key])
        ax.set_title(f"{nm} — {LBL[key]}", fontsize=9)
        ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("docs/evolution_curves_V6.png", dpi=110)
print("\nFigure : docs/evolution_curves_V6.png")
print("Les 4 observables basculent au MÊME E_seuil par particule (même f) :")
print("  proton 644, neutron 644, muon 337, pion 406 MeV.")
