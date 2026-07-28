r"""Intégrale de recouvrement du champ χ au croisement (vortex GL/Nielsen-Olesen).

Voir docs/brouillon_amplitude_reconnexion_chi.tex §(v). Résout le profil de vortex
f(ρ) (f''+f'/ρ−f/ρ²+f(1−f²)=0, f(0)=0, f(∞)=1, f'(0)=0.583) et calcule les
recouvrements adimensionnés. RÉSULTAT HONNÊTE : tous O(1) dans [1.3, 3.3] → confirme
c=O(1) (physique de cœur), MAIS aucun ne donne c=1.31 proprement (le plus proche
I_dep/I_cross=1.43 à 9%). Un balayage trouve √(1/f'(0))=1.3095 à 0.04% de 5π/12,
mais parmi ~10 combinaisons sans justification physique → CHERRY-PICKING, NON revendiqué.
Clore c = élément de matrice 3D complet (CS+matière) + meilleure mesure de σ_χ.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def _rhs(rho, y):
    f, fp = y
    if rho < 1e-9:
        return [fp, 0.0]
    return [fp, -fp / rho + f / rho ** 2 - f * (1 - f ** 2)]


def solve_vortex(rmax=14.0):
    shoot = lambda c1: solve_ivp(_rhs, [1e-6, rmax], [c1 * 1e-6, c1],
                                 rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.008)
    c1 = brentq(lambda c: shoot(c).y[0, -1] - 1.0, 0.1, 2.0, xtol=1e-9)
    sol = shoot(c1)
    return c1, sol


def overlap_integrals():
    c1, sol = solve_vortex()
    rho = np.linspace(1e-6, 14, 6000)
    f = np.clip(sol.sol(rho)[0], 0, 1.05)
    I = lambda g: np.trapz(g * rho, rho)
    return {
        'f_prime_0': c1,
        'I_dep': I(1 - f ** 2),
        'I_dep2': I((1 - f ** 2) ** 2),
        'I_cross': I(f ** 2 * (1 - f ** 2)),
        'I_dep/I_cross': I(1 - f ** 2) / I(f ** 2 * (1 - f ** 2)),
    }


if __name__ == '__main__':
    r = overlap_integrals()
    for kk, vv in r.items():
        print(f"  {kk:16s} = {vv:.4f}")
    print(f"  cible c ≈ 1.31 (5π/12={5*np.pi/12:.4f}, 4/3={4/3:.4f})")
    print("  → O(1) confirmé, mais c non pin (pas d'élément de matrice 3D dérivé). "
          "√(1/f'(0)) match 5π/12 mais = cherry-picking, non revendiqué.")
