r"""
DÉCOMPOSITION CROISEMENTS + LOBES pour 4_1, 5_1, 7_1, 7_2

Même méthode que pour l'électron (3_1) :
   m_crossing(K) = η · U_κ(K)            (énergie de courbure)
   m_loop_total(K) = η · U_τ(K)          (énergie de torsion)
   m_loop_par_lobe(K) = m_loop_total / n_lobes(K)
   m(K) = m_crossing + m_loop_total = η · U_total(K)  (Abraham)

avec η = m_e / U_total(3_1) (calibré sur l'électron).

Nombre de lobes :
   3_1 (2,3) torique : 3 lobes
   4_1 figure-huit   : 2 lobes
   5_1 (2,5) torique : 5 lobes
   7_1 (2,7) torique : 7 lobes
   7_2 twist knot    : 2 lobes (analogue figure-huit avec 5 twists)

Le nombre de croisements est cr(K) (toujours = numéro du nœud).
On compare m(7_2) calculé avec la valeur de Hölder précédente
(66.08 MeV, déduite du proton).
"""
from __future__ import annotations
import numpy as np

from theory.abraham import frenet_geometry

M_E = 0.51099895
M_72_HOLDER = 66.083   # valeur déduite du proton dans le fit Hölder

KNOTS_TARGET = ["3_1", "4_1", "5_1", "7_1", "7_2"]

# Nombre de lobes par nœud (par inspection du diagramme standard)
N_LOBES = {
    "3_1": 3,    # (2,3) torus
    "4_1": 2,    # figure-eight
    "5_1": 5,    # (2,5) torus
    "7_1": 7,    # (2,7) torus
    "7_2": 2,    # twist knot
}


def cr(K):
    return int(K.split("_")[0])


def compute_integrals(K, n_points=50000):
    """Compute U_κ, U_τ et longueur d'arc."""
    geo = frenet_geometry(K, n_points)
    kappa = geo["kappa"]
    tau = geo["tau"]
    speed = geo["speed"]
    N = len(kappa)
    dt = 2 * np.pi / N
    ds = speed * dt
    return {
        "L":       float(np.sum(ds)),
        "U_kappa": float(np.sum(kappa ** 2 * ds)),
        "U_tau":   float(np.sum(tau ** 2 * ds)),
    }


def main():
    print("=" * 95)
    print("  DÉCOMPOSITION CROISEMENTS + LOBES pour 4_1, 5_1, 7_1, 7_2")
    print("  (même méthode que électron 3_1 — calibration η sur électron)")
    print("=" * 95)

    # Calibration sur l'électron
    g31 = compute_integrals("3_1")
    U31_total = g31["U_kappa"] + g31["U_tau"]
    eta = M_E / U31_total
    print(f"\n  ─── CALIBRATION ───")
    print(f"  U_κ(3_1)   = {g31['U_kappa']:.4f}")
    print(f"  U_τ(3_1)   = {g31['U_tau']:.4f}")
    print(f"  U_total(3_1) = {U31_total:.4f}")
    print(f"  η = m_e / U_total(3_1) = {eta:.6e} MeV/unité")

    # Calcul pour chaque nœud
    print(f"\n  ─── INTÉGRALES ET DÉCOMPOSITION ───")
    print(f"  {'K':>5} {'cr':>3} {'lob.':>4} {'L':>7} "
          f"{'U_κ':>9} {'U_τ':>8} {'fκ %':>6} "
          f"{'m_cross':>10} {'m_lobe':>10} {'m_total':>10}")
    print("  " + "-" * 85)
    results = {}
    for K in KNOTS_TARGET:
        g = compute_integrals(K)
        U_tot = g["U_kappa"] + g["U_tau"]
        f_kappa = g["U_kappa"] / U_tot * 100
        m_crossing = eta * g["U_kappa"]
        m_loop_total = eta * g["U_tau"]
        nl = N_LOBES[K]
        m_loop_par = m_loop_total / nl
        m_total = eta * U_tot
        results[K] = {
            "U_kappa":   g["U_kappa"],
            "U_tau":     g["U_tau"],
            "U_total":   U_tot,
            "m_crossing": m_crossing,
            "m_loop_par": m_loop_par,
            "m_total":   m_total,
            "n_lobes":   nl,
        }
        anchor = " ★" if K == "3_1" else ""
        print(f"  {K:>5} {cr(K):>3} {nl:>4} {g['L']:>7.2f} "
              f"{g['U_kappa']:>9.4f} {g['U_tau']:>8.4f} {f_kappa:>5.1f}% "
              f"{m_crossing:>10.5f} {m_loop_par:>10.5f} "
              f"{m_total:>10.5f}{anchor}")

    # Tableau des masses déduites
    print(f"\n  ─── MASSES TOTALES m(K) PAR ABRAHAM ───")
    print(f"  {'K':>5} {'m(K) (MeV)':>13}")
    print("  " + "-" * 22)
    for K in KNOTS_TARGET:
        m = results[K]["m_total"]
        anchor = " ★ ancré (e⁻)" if K == "3_1" else ""
        print(f"  {K:>5} {m:>13.5f}{anchor}")

    # Vérification 7_2
    print(f"\n  ─── VÉRIFICATION : m(7_2) ABRAHAM vs HÖLDER ───")
    m72_abr = results["7_2"]["m_total"]
    print(f"  m(7_2) Abraham      = {m72_abr:>11.5f} MeV")
    print(f"  m(7_2) Hölder       = {M_72_HOLDER:>11.5f} MeV  (déduit du proton)")
    print(f"  ratio Hölder/Abraham = {M_72_HOLDER/m72_abr:>11.2f}×")
    print(f"  écart absolu        = {M_72_HOLDER - m72_abr:>11.4f} MeV")
    print(f"  écart relatif       = {(M_72_HOLDER/m72_abr - 1)*100:>11.1f} %")

    # Diagnostic
    print(f"\n  ─── DIAGNOSTIC ───")
    if abs(m72_abr - M_72_HOLDER) / M_72_HOLDER < 0.05:
        print(f"  ★ Les deux valeurs coïncident à <5 %.")
    elif m72_abr < M_72_HOLDER:
        print(f"  ⚠ m(7_2) Abraham ({m72_abr:.3f}) est très inférieur à la")
        print(f"    valeur Hölder ({M_72_HOLDER:.1f}). Facteur manquant : "
              f"{M_72_HOLDER/m72_abr:.1f}×")
        print(f"  La formule d'Abraham pure (linéaire en U_EM) ne reproduit")
        print(f"  pas l'amplification observée dans le spectre. Il manque un")
        print(f"  mécanisme additionnel (puissance, c_J, etc.)")
    print("=" * 95)


if __name__ == "__main__":
    main()
