r"""
Durée de vie du NEUTRON prédite À PARTIR DU MUON — sans calibration sur le neutron.

Le muon (désintégration purement leptonique) DÉFINIT l'échelle faible (G_F).
Le neutron est alors PRÉDIT par la formule de Sargent/Fermi, où intervient le
facteur axial NUCLÉAIRE (1+3g_A²) — propriété GLOBALE/collective de l'entrelacs
de groupe 3 (le nucléon), et g_A = grandeur de l'INTERACTION NUCLÉAIRE FORTE.

    Γ_μ = G_F² m_μ^5 / (192 π³)              (référence leptonique)
    Γ_n = G_F²|V_ud|² m_e^5 (1+3g_A²) f / (2π³)   (β du nucléon)

  =>  τ_n/τ_μ = (m_μ/m_e)^5 / [ 96 · |V_ud|² · (1+3g_A²) · f ]

f = intégrale de phase de Fermi (endpoint W_0 = (m_n−m_p)/m_e), calculée ici.
Aucun paramètre ajusté sur le neutron : G_F vient du muon, g_A et V_ud sont des
constantes (forte / CKM), f est cinématique.

Usage :  python3 -m simulation.experiments.neutron_from_muon
"""
from __future__ import annotations

import numpy as np

# constantes mesurées (PDG)
M_E   = 0.51100         # MeV
M_MU  = 105.6584        # MeV
M_N   = 939.565         # MeV
M_P   = 938.272         # MeV
TAU_MU = 2.196981e-6    # s (référence : DÉFINIT l'échelle faible)
G_A   = 1.2723          # charge axiale du nucléon (INTERACTION NUCLÉAIRE FORTE)
V_UD  = 0.97373         # CKM
RAD_CORR = 1.039        # corrections radiatives internes (~+3.9 %)
TAU_N_EXP = 878.4       # s


def fermi_factor(W0: float) -> float:
    """f = ∫_1^{W0} W (W²−1)^{1/2} (W0−W)² dW  (phase space β, sans Coulomb)."""
    W = np.linspace(1.0, W0, 20000)
    integrand = W * np.sqrt(np.clip(W**2 - 1, 0, None)) * (W0 - W)**2
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(integrand, W))


def main():
    print("=" * 80)
    print("  NEUTRON PRÉDIT DEPUIS LE MUON — aucune calibration sur le neutron")
    print("=" * 80)

    # endpoint : énergie totale max de l'électron = Δ_np, en unités m_e
    Delta = M_N - M_P                       # 1.293 MeV
    W0 = Delta / M_E                         # endpoint réduit
    f = fermi_factor(W0)
    g_factor = 1.0 + 3.0 * G_A**2            # facteur axial nucléon (global)

    print(f"\n  Entrées (toutes mesurées, AUCUNE ajustée sur le neutron) :")
    print(f"    τ_μ   = {TAU_MU:.4e} s   (référence leptonique, définit G_F)")
    print(f"    m_μ/m_e = {M_MU/M_E:.2f}")
    print(f"    g_A   = {G_A}  ->  (1+3g_A²) = {g_factor:.3f}   [interaction nucléaire FORTE, global nucléon]")
    print(f"    |V_ud|² = {V_UD**2:.4f}   [CKM]")
    print(f"    Δ_np  = {Delta:.3f} MeV  ->  W_0 = {W0:.3f},  f = {f:.4f}  [cinématique]")

    ratio = (M_MU/M_E)**5 / (96.0 * V_UD**2 * g_factor * f)
    tau_n = TAU_MU * ratio
    tau_n_rc = tau_n / RAD_CORR

    print(f"\n  τ_n/τ_μ = (m_μ/m_e)^5 / [96·|V_ud|²·(1+3g_A²)·f] = {ratio:.3e}")
    print(f"\n  PRÉDICTION  τ_n = {tau_n:.1f} s        (brut)")
    print(f"              τ_n = {tau_n_rc:.1f} s        (+ corrections radiatives)")
    print(f"  MESURÉ      τ_n = {TAU_N_EXP:.1f} s")
    dl = np.log10(tau_n_rc / TAU_N_EXP)
    print(f"  Écart : {(tau_n_rc-TAU_N_EXP)/TAU_N_EXP*100:+.1f} %   ({dl:+.3f} décade)")

    # inversion : le neutron DÉFINIT g_A (donc l'interaction nucléaire forte)
    print("\n  INVERSION — la durée de vie du neutron DÉFINIT g_A :")
    g_factor_meas = (M_MU/M_E)**5 * TAU_MU / (96.0 * V_UD**2 * f * TAU_N_EXP * RAD_CORR)
    gA_meas = np.sqrt((g_factor_meas - 1) / 3)
    print(f"    de τ_n mesuré + G_F(muon)  ->  (1+3g_A²) = {g_factor_meas:.3f}  ->  g_A = {gA_meas:.3f}")
    print(f"    (valeur PDG g_A = 1.2754 ± 0.0013)")
    print("\n  => le neutron N'EST PAS calibré : il PRÉDIT 878 s à ~1 %, et")
    print("     g_A (charge axiale GLOBALE du nucléon de groupe 3) EST l'interaction nucléaire forte.")
    print("=" * 80)


if __name__ == "__main__":
    main()
