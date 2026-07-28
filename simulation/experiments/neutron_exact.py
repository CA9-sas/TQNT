r"""
Neutron : la STRUCTURE (Sargent Q^5 + facteur global nucléon 1+3g_A²) reproduit
878 s à +0,2 % AVEC les constantes précises. Démontre que mes 918 s venaient
d'approximations (rapport au muon, f grossier), pas de la structure.

    1/τ_n = (G_F² |V_ud|² m_e^5 / 2π³ħ) · (1+3λ²) · f · (1+Δ_R)

Toutes les constantes sont MESURÉES (G_F, λ, |V_ud|, f, Δ_R) : c'est le point
clé — aucune théorie (TQNT ou MS) ne produit 878,4 s sans ces inputs.

Usage :  python3 -m simulation.experiments.neutron_exact
"""
import numpy as np

# constantes (toutes MESURÉES)
G_F  = 1.1663787e-11    # MeV^-2  (1.166e-5 GeV^-2)
M_E  = 0.51099895       # MeV
HBAR = 6.582119e-22     # MeV·s
LAM  = 1.2756           # g_A/g_V (axial nucléon, force nucléaire forte)
VUD2 = 0.94815          # |V_ud|²
F_N  = 1.6887           # facteur de phase de Fermi (Coulomb + recul)
DR   = 1.039            # 1 + corrections radiatives
TAU_EXP = 878.4


def main():
    print("=" * 74)
    print("  NEUTRON — formule ABSOLUE, constantes précises")
    print("=" * 74)
    rate0 = G_F**2 * M_E**5 / (2 * np.pi**3) / HBAR     # s^-1 (échelle nue)
    rate = rate0 * VUD2 * (1 + 3*LAM**2) * F_N * DR
    tau = 1.0 / rate
    print(f"\n  G_F² m_e^5/(2π³ħ)        = {rate0:.4e} s⁻¹")
    print(f"  × |V_ud|²={VUD2}")
    print(f"  × (1+3λ²)={1+3*LAM**2:.3f}   [facteur axial GLOBAL du nucléon, g_A={LAM}]")
    print(f"  × f={F_N}              [phase space de Fermi]")
    print(f"  × (1+Δ_R)={DR}        [corrections radiatives]")
    print(f"\n  τ_n (prédit) = {tau:.1f} s")
    print(f"  τ_n (mesuré) = {TAU_EXP} s")
    print(f"  écart = {(tau-TAU_EXP)/TAU_EXP*100:+.2f} %   ({np.log10(tau/TAU_EXP):+.3f} décade)")
    print("\n  -> la STRUCTURE (Sargent Q^5 + 1+3g_A²) est EXACTE à +0,2 %.")
    print("     Mes 918 s = artefact du rapport au muon (corrections non annulées).")
    print("\n  MAIS : G_F, λ, |V_ud|, f, Δ_R sont TOUS mesurés. Obtenir 878,4 s")
    print("  SANS aucun input mesuré est impossible en principe — aucune théorie")
    print("  (ni TQNT ni MS) ne dérive G_F, g_A, V_ud de nombres purs.")
    print("=" * 74)


if __name__ == "__main__":
    main()
