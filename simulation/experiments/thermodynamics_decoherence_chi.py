"""
Formule maîtresse de la dissipation (TQNT) : chaleur et entropie depuis
le taux de décohérence Γ de la danse du puits χ.

  q̇   = (m*/n e²) Γ J²  = ρ J²            (chaleur, 1er principe)
  Ṡ   = q̇ / T  = (m*/n e² T) Γ J²  ≥ 0    (entropie, 2nd principe)
  2nd principe : Ṡ ≥ 0  ⟺  Γ ≥ 0
  Γ → 0 (danse cohérente, supra) : q̇ = Ṡ = 0 (réversible)

Usage : PYTHONPATH=. python3 -m simulation.experiments.thermodynamics_decoherence_chi
"""
from __future__ import annotations
import numpy as np

e = 1.602176634e-19
m_e = 9.109e-31
kB = 1.380649e-23


def decoherence_rate(rho: float, n: float, mstar: float = m_e) -> float:
    """Taux de décohérence Γ = 1/τ = n e² ρ / m* (déduit de la résistivité)."""
    return n * e ** 2 * rho / mstar


def heat_rate(Gamma: float, J: float, n: float, mstar: float = m_e) -> float:
    """q̇ = (m*/n e²) Γ J²  [W/m³]  (= ρ J²)."""
    return (mstar / (n * e ** 2)) * Gamma * J ** 2


def entropy_rate(Gamma: float, J: float, T: float, n: float, mstar: float = m_e) -> float:
    """Ṡ = q̇ / T  [W/(m³·K)]  ≥ 0 ⟺ Γ ≥ 0 (second principe)."""
    return heat_rate(Gamma, J, n, mstar) / T


if __name__ == "__main__":
    # Cuivre
    rho, n = 1.68e-8, 8.5e28
    Gamma = decoherence_rate(rho, n)
    J, T = 1e7, 300.0
    print(f"Cuivre : Γ = {Gamma:.3e} /s  (τ = {1/Gamma:.2e} s)")
    print(f"J = {J:.0e} A/m², T = {T:.0f} K\n")

    q = heat_rate(Gamma, J, n)
    print("Formule maîtresse (vérification q̇ = (m*/ne²)ΓJ² = ρJ²) :")
    print(f"  q̇ = {q:.4e} W/m³   ;   ρJ² = {rho*J**2:.4e} W/m³")
    print(f"  Ṡ = q̇/T = {entropy_rate(Gamma, J, T, n):.4e} W/(m³·K)\n")

    print("Second principe : Ṡ ≥ 0 ⟺ Γ ≥ 0")
    for label, G in [("danse cohérente (supra)", 0.0), ("danse incohérente", Gamma)]:
        q = heat_rate(G, J, n)
        print(f"  Γ = {G:.2e} /s [{label}] : q̇ = {q:.2e} W/m³, "
              f"Ṡ = {entropy_rate(G, J, T, n) if T else 0:.2e}")
    print("  -> Γ=0 : réversible (ni chaleur ni entropie) ; Γ>0 : irréversible.")
