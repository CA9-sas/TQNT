r"""
DÉCOMPOSITION par compte explicite (croisements + lobes)
                avec masses UNITAIRES déduites de l'électron.

L'électron (3_1) donne :
   m_per_crossing = m_crossing_total(3_1) / cr(3_1) = 0.443635 / 3 = 0.147878 MeV
   m_per_loop     = m_loop_total(3_1) / n_lobes(3_1) = 0.067364 / 3 = 0.022454 MeV

Formule générique :
   m(K) = cr(K) · m_per_crossing + n_lobes(K) · m_per_loop

Pour 7_2 (boucle serrée, isolée dans la plupart des configs) :
   m(7_2)_serré = k_s · m(7_2)_libre,  k_s = 2

Vérification : m(7_2)_serré vs valeur Hölder (66.08 MeV du proton).
"""
from __future__ import annotations

M_E = 0.51099895
F_KAPPA = 0.8682     # fraction de m_e dans la courbure (3_1 : 86.82%)
F_TAU = 1 - F_KAPPA  # fraction dans la torsion
K_S = 2.0            # coefficient boucle serrée

# Pour le trèfle 3_1 (calibration)
CR_31 = 3
N_LOBES_31 = 3

# Cibles
KNOTS = {
    # K : (cr, n_lobes, état du 7_2 si applicable)
    "3_1": (3, 3, "libre"),
    "4_1": (4, 2, "libre"),
    "5_1": (5, 5, "libre"),
    "7_1": (7, 7, "libre"),
    "7_2": (7, 2, "serré"),    # boucle isolée → k_s = 2
}

M_72_HOLDER = 66.083  # MeV — déduit du proton dans le fit Hölder


def main():
    print("=" * 90)
    print("  DÉCOMPOSITION PAR COMPTE EXPLICITE (croisements + lobes)")
    print("  + coefficient k_s pour boucle serrée")
    print("=" * 90)

    # ─── Masses unitaires depuis l'électron ─────────────────────────
    m_cross_tot_31 = M_E * F_KAPPA
    m_loop_tot_31 = M_E * F_TAU
    m_per_crossing = m_cross_tot_31 / CR_31
    m_per_loop = m_loop_tot_31 / N_LOBES_31

    print(f"\n  ─── MASSES UNITAIRES (calibrées sur l'électron 3_1) ───")
    print(f"  m_crossing_total(3_1) = m_e · f_κ  = {M_E} × {F_KAPPA} "
          f"= {m_cross_tot_31:.6f} MeV")
    print(f"  m_loop_total(3_1)     = m_e · f_τ  = {M_E} × {F_TAU:.4f} "
          f"= {m_loop_tot_31:.6f} MeV")
    print(f"  m_per_crossing = {m_cross_tot_31:.6f} / {CR_31} = "
          f"{m_per_crossing:.6f} MeV par croisement")
    print(f"  m_per_loop     = {m_loop_tot_31:.6f} / {N_LOBES_31} = "
          f"{m_per_loop:.6f} MeV par lobe")

    # ─── Calcul pour chaque nœud ────────────────────────────────────
    print(f"\n  ─── DÉCOMPOSITION POUR CHAQUE NŒUD ───")
    print(f"  {'K':>5} {'cr':>3} {'lob':>3} {'état':>6} "
          f"{'cr·m_c':>10} {'lob·m_l':>10} {'m_libre':>10} "
          f"{'k_s':>5} {'m_final':>11}")
    print("  " + "-" * 78)

    for K, (cr, nl, etat) in KNOTS.items():
        m_cross_part = cr * m_per_crossing
        m_loop_part = nl * m_per_loop
        m_libre = m_cross_part + m_loop_part
        ks_used = K_S if etat == "serré" else 1.0
        m_final = ks_used * m_libre
        anchor = " ★" if K == "3_1" else ""
        ks_str = f"{ks_used:.0f}" if etat == "serré" else "—"
        print(f"  {K:>5} {cr:>3} {nl:>3} {etat:>6} "
              f"{m_cross_part:>10.5f} {m_loop_part:>10.5f} "
              f"{m_libre:>10.5f} {ks_str:>5} {m_final:>11.5f}{anchor}")

    # ─── Vérification 7_2 ───────────────────────────────────────────
    cr72, nl72, _ = KNOTS["7_2"]
    m72_libre = cr72 * m_per_crossing + nl72 * m_per_loop
    m72_serre = K_S * m72_libre

    print(f"\n  ─── VÉRIFICATION : m(7_2)_serré ───")
    print(f"  Compte : cr = 7, lobes = 2")
    print(f"  m(7_2)_libre = 7 × {m_per_crossing:.6f} + 2 × {m_per_loop:.6f}")
    print(f"                = {cr72*m_per_crossing:.5f} + {nl72*m_per_loop:.5f}")
    print(f"                = {m72_libre:.5f} MeV")
    print(f"  m(7_2)_serré = k_s × libre = {K_S} × {m72_libre:.5f} "
          f"= {m72_serre:.5f} MeV")
    print(f"\n  Cible Hölder (déduit du proton) = {M_72_HOLDER:.5f} MeV")
    print(f"  ratio Hölder / décompo  = {M_72_HOLDER/m72_serre:>8.2f}×")
    print(f"  écart absolu            = {M_72_HOLDER - m72_serre:>8.4f} MeV")
    print(f"  écart relatif           = {(M_72_HOLDER/m72_serre - 1)*100:>8.1f} %")

    # ─── Diagnostic ─────────────────────────────────────────────────
    print(f"\n  ─── DIAGNOSTIC ───")
    if abs(m72_serre - M_72_HOLDER) / M_72_HOLDER < 0.05:
        print(f"  ★★ ACCORD PARFAIT : 7_2 décomposé coïncide avec la valeur Hölder")
    elif abs(m72_serre - M_72_HOLDER) / M_72_HOLDER < 0.20:
        print(f"  ✓ Accord raisonnable (<20%)")
    else:
        ratio = M_72_HOLDER / m72_serre
        print(f"  ⚠ Écart important : facteur {ratio:.1f}× manquant.")
        print(f"  Pour atteindre m(7_2) = {M_72_HOLDER}, il faudrait :")
        print(f"    - soit m_per_crossing ≈ {(M_72_HOLDER/K_S - nl72*m_per_loop)/cr72:.4f} MeV")
        print(f"      (au lieu de {m_per_crossing:.4f}, facteur "
              f"{((M_72_HOLDER/K_S - nl72*m_per_loop)/cr72)/m_per_crossing:.1f}×)")
        print(f"    - soit k_s ≈ {M_72_HOLDER/m72_libre:.2f}")
        print(f"      (au lieu de {K_S}, facteur {M_72_HOLDER/m72_libre/K_S:.1f}×)")
    print("=" * 90)


if __name__ == "__main__":
    main()
