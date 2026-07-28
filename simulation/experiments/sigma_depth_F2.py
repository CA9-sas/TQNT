"""
Profondeur du puits nucléaire — couplage g_σ du mode scalaire (TQNT).

Le mode scalaire 0+ du condensat χ (sigma_mode_F2.py) a m_σ ≈ 500 MeV.
Il médie une attraction de Yukawa scalaire :

    V_σ(r) = -(g_σ²/4π) (ħc) e^{-m_σ r/ħc} / r .

La profondeur dépend du couplage g_σ. On l'estime par la relation du
modèle σ linéaire g_σ = m_N / f_σ (la masse du nucléon vient du
condensat), avec f_σ l'amplitude du condensat χ (échelle forte). On
construit ensuite le potentiel NN complet (σ attractif + cœur
d'exclusion F3) et on lit la profondeur du puits.

Usage :  python3 -m simulation.experiments.sigma_depth_F2
"""
from __future__ import annotations

import numpy as np

HBARC = 197.327      # MeV·fm
M_N = 938.0          # MeV (nucléon)


def V_sigma(r_fm, g2_4pi, m_sigma_MeV):
    """Potentiel d'échange scalaire (Yukawa attractif), en MeV."""
    mu = m_sigma_MeV / HBARC          # fm^-1
    return -g2_4pi * HBARC * np.exp(-mu * r_fm) / r_fm


def main():
    print("=" * 66)
    print("  PROFONDEUR DU PUITS — couplage g_σ du mode scalaire (TQNT)")
    print("=" * 66)

    m_sigma = 500.0      # MeV — mode 0+ extrait (sigma_mode_F2.py)
    print(f"\n  m_σ = {m_sigma:.0f} MeV (mode de respiration 0+ du condensat χ)")
    print(f"  portée 1/μ = ħc/m_σ = {HBARC/m_sigma:.2f} fm")

    # --- (1) profondeur vs couplage ---
    print("\n(1) Profondeur de l'échange σ selon g_σ²/4π (à r = 1 fm) :")
    print(f"    {'g_σ²/4π':>8} {'V_σ(1 fm)':>12}")
    for g2 in (2.0, 4.0, 8.0, 12.0):
        print(f"    {g2:>8.1f} {V_sigma(1.0, g2, m_sigma):>10.1f} MeV")

    # --- (2) estimation TQNT-interne de g_σ ---
    print("\n(2) Estimation de g_σ (modèle σ linéaire : g_σ = m_N / f_σ)")
    print("    f_σ = amplitude du condensat χ (échelle forte).")
    for f_sigma, lab in ((93.0, "f_π (échelle chirale)"),
                         (130.0, "f_K-like"),
                         (180.0, "borne haute")):
        g = M_N / f_sigma
        g2 = g**2 / (4 * np.pi)
        print(f"      f_σ = {f_sigma:>5.0f} MeV [{lab:<20}] → "
              f"g_σ = {g:5.2f}, g_σ²/4π = {g2:5.2f}")
    # valeur centrale
    f_sigma = 93.0
    g2_4pi = (M_N / f_sigma)**2 / (4 * np.pi)
    print(f"\n    Valeur centrale (f_σ ~ f_π) : g_σ²/4π ≈ {g2_4pi:.1f}")
    print("    (réf. empirique σNN : g_σ²/4π ≈ 8) — accord à l'échelle ✓")

    # --- (3) potentiel NN complet : σ attractif + cœur répulsif (ω-like) ---
    print("\n(3) Potentiel nucléon-nucléon complet (σ + cœur répulsif F3)")
    # cœur répulsif court (analogue ω) : doit dominer à courte distance
    # ⇒ g_ω²/4π > g_σ²/4π. Valeur illustrative (F3 non encore dérivé).
    m_core = 782.0       # MeV (échelle ω)
    g2_core = 20.0       # g_ω²/4π (illustratif)

    r = np.linspace(0.5, 3.0, 500)
    Vatt = V_sigma(r, g2_4pi, m_sigma)
    Vrep = +g2_core * HBARC * np.exp(-(m_core / HBARC) * r) / r
    Vtot = Vatt + Vrep
    kmin = int(np.argmin(Vtot))
    Rmin, Vmin = r[kmin], Vtot[kmin]
    print(f"    cœur répulsif (F3 illustratif) : m_core = {m_core:.0f} MeV, "
          f"g_ω²/4π = {g2_core:.0f}")
    print(f"    σ seul à 1 fm        : {V_sigma(1.0, g2_4pi, m_sigma):.0f} MeV")
    print(f"    minimum du puits NET : R = {Rmin:.2f} fm, "
          f"profondeur V = {Vmin:.0f} MeV")
    print("\n    Échelle nucléaire empirique : puits NN ~ -50 à -100 MeV")
    print("    (à ~1 fm), liaison nette ~ -8 MeV/nucléon (après cinétique).")
    if -120 < Vmin < -20 and 0.7 < Rmin < 1.3:
        print("    → profondeur ET position du puits dans la fourchette "
              "nucléaire ✓")

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        ax.plot(r, Vatt, "--", color="green", label="$V_\\sigma$ (échange scalaire)")
        ax.plot(r, Vrep, ":", color="red", label="cœur d'exclusion (F3)")
        ax.plot(r, Vtot, "-", color="black", lw=2, label="$V_{NN}$ total")
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(Rmin, color="blue", lw=0.7, ls="--")
        ax.plot(Rmin, Vmin, "o", color="blue")
        ax.annotate(f"  puits {Vmin:.0f} MeV\n  à {Rmin:.2f} fm",
                    (Rmin, Vmin), fontsize=9, color="blue")
        ax.set_ylim(-160, 120)
        ax.set_xlabel("$r$ (fm)")
        ax.set_ylabel("$V_{NN}$ (MeV)")
        ax.set_title("Potentiel nucléon–nucléon TQNT "
                     "($m_\\sigma=500$ MeV, $g_\\sigma^2/4\\pi\\approx8$)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = "output/sigma_depth_F2.png"
        fig.savefig(out, dpi=130)
        print(f"\n  figure → {out}")
    except Exception as e:
        print(f"  (figure non générée : {e})")

    print("\n  CONCLUSION :")
    print("  • m_σ ≈ 500 MeV est CALCULÉ (mode 0+, sigma_mode_F2).")
    print("  • g_σ²/4π ≈ 8 (via g_σ=m_N/f_σ, f_σ~échelle forte) reproduit")
    print("    la profondeur nucléaire (~-100 MeV à 1 fm, puits net ~-50 MeV).")
    print("  • RESTE : dériver f_σ (l'amplitude du condensat χ) du lagrangien,")
    print("    pour PRÉDIRE g_σ au lieu de le caler sur l'échelle forte.")
    print("=" * 66)


if __name__ == "__main__":
    main()
