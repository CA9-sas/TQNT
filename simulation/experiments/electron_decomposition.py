r"""
DÉCOMPOSITION DE LA MASSE DE L'ÉLECTRON EN 4 PARTS

Le trèfle 3_1 a :
   - 3 lobes (boucles)         → torsion τ (twist le long de l'arc)
   - 1 région de croisement    → courbure κ (bending là où les brins
                                  passent l'un sur l'autre)

On répartit m_e selon la décomposition NATURELLE de l'énergie
d'Abraham U_EM = U_κ + U_τ :

   U_κ = ∮ κ² ds   (énergie de courbure ↔ croisement de brins)
   U_τ = ∮ τ² ds   (énergie de torsion  ↔ lobes/boucles)

   m_crossing = m_e · U_κ / U_total
   m_loop     = m_e · U_τ / (3 · U_total)   (3 lobes par symétrie)

Total : m_crossing + 3 · m_loop = m_e   ✓ (pas forcément égales)
"""
from __future__ import annotations
import numpy as np

from theory.abraham import frenet_geometry

M_E = 0.51099895  # MeV


def compute_3_1_decomposition(n_points: int = 50000):
    """Calcule U_κ et U_τ pour le trèfle 3_1 à haute résolution."""
    geo = frenet_geometry("3_1", n_points)
    kappa = geo["kappa"]
    tau = geo["tau"]
    speed = geo["speed"]
    N = len(kappa)
    dt = 2 * np.pi / N
    ds = speed * dt

    L = float(np.sum(ds))
    int_kappa = float(np.sum(np.abs(kappa) * ds))
    int_tau = float(np.sum(np.abs(tau) * ds))
    U_kappa = float(np.sum(kappa ** 2 * ds))
    U_tau = float(np.sum(tau ** 2 * ds))
    U_total = U_kappa + U_tau

    return {
        "L": L,
        "int_kappa": int_kappa,
        "int_tau": int_tau,
        "U_kappa": U_kappa,
        "U_tau": U_tau,
        "U_total": U_total,
    }


def main():
    print("=" * 80)
    print("  DÉCOMPOSITION DE LA MASSE DE L'ÉLECTRON EN 4 PARTS")
    print("=" * 80)
    print(f"\n  Électron = trèfle 3_1 (cr = 3, 3 lobes, 3-fold symétrie)")
    print(f"  m(e⁻) = {M_E} MeV (référence)")

    print(f"\n  Calcul des intégrales de Frenet sur 3_1 à 50 000 points…")
    g = compute_3_1_decomposition()

    print(f"\n  ─── INTÉGRALES GÉOMÉTRIQUES DU TRÈFLE ───")
    print(f"  L          = {g['L']:>10.4f}  (longueur d'arc totale)")
    print(f"  ∮|κ|ds     = {g['int_kappa']:>10.4f}")
    print(f"  ∮|τ|ds     = {g['int_tau']:>10.4f}")
    print(f"  U_κ = ∮κ²ds = {g['U_kappa']:>10.4f}  ← courbure (croisement)")
    print(f"  U_τ = ∮τ²ds = {g['U_tau']:>10.4f}  ← torsion (boucles)")
    print(f"  U_total    = {g['U_total']:>10.4f}")
    print(f"  fraction κ = {g['U_kappa']/g['U_total']:>10.4f}  ({g['U_kappa']/g['U_total']*100:.2f} %)")
    print(f"  fraction τ = {g['U_tau']/g['U_total']*100:.2f} %")

    # Décomposition de m_e
    f_kappa = g["U_kappa"] / g["U_total"]
    f_tau = g["U_tau"] / g["U_total"]
    m_crossing = M_E * f_kappa
    m_loop_total = M_E * f_tau
    m_loop = m_loop_total / 3

    print(f"\n  ─── DÉCOMPOSITION m_e = 4 PARTS ───")
    print(f"  {'composante':<25} {'masse (MeV)':>12} {'fraction':>10}")
    print("  " + "-" * 50)
    print(f"  m_crossing (1 région)     {m_crossing:>12.6f} {f_kappa:>10.4f}")
    print(f"  m_loop_1                  {m_loop:>12.6f} {f_tau/3:>10.4f}")
    print(f"  m_loop_2                  {m_loop:>12.6f} {f_tau/3:>10.4f}")
    print(f"  m_loop_3                  {m_loop:>12.6f} {f_tau/3:>10.4f}")
    print("  " + "-" * 50)
    print(f"  TOTAL                     {m_crossing + 3*m_loop:>12.6f} "
          f"{1.0:>10.4f}")
    print(f"  m_e attendu               {M_E:>12.6f}")
    err = abs(m_crossing + 3*m_loop - M_E) / M_E * 100
    print(f"  écart                     {err:>11.2e} %")

    print(f"\n  ─── INTERPRÉTATION ───")
    print(f"  La région de croisement   : m = {m_crossing:.5f} MeV "
          f"(= {f_kappa*100:.1f} % de m_e)")
    print(f"  Chacune des 3 boucles     : m = {m_loop:.5f} MeV "
          f"(= {f_tau/3*100:.1f} % de m_e)")
    print(f"  Ratio croisement/lobe     : {m_crossing/m_loop:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
