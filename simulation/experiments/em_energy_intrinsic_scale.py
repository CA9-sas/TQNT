r"""
VOIE A — énergie EM avec ÉCHELLE INTRINSÈQUE (rope-length physique)
au lieu de normalisation L=100.

Motivation (cf. commit 44b516c) : normaliser toutes les courbes à L=100
tue toute discrimination de masse car U_self ∝ L · log(L/a).
On essaye maintenant : chaque nœud K est scalé à sa rope-length
publiée R_pub(K) (Cantarella-Kusner-Sullivan), interprétée comme la
TAILLE PHYSIQUE INTRINSÈQUE de la boucle.

Pour les composites : chaque composante garde sa propre taille R_pub(K_i),
les composantes sont placées en n-gone régulier avec séparation = mean
des tailles → embedding 'naturel'.

Hypothèse implicite : la TQNT prédit que chaque type de nœud a une
TAILLE PRÉFÉRÉE = sa rope-length minimale, qui détermine son énergie
EM = sa masse.

Trois variantes testées :
  V1. Courbes Pieranski scalées à R_pub : composantes touchant
  V2. Idem mais avec rayon de tube a = 0.5 (= demi-épaisseur)
  V3. Avec embedding plus dense (séparation = R_pub/2 : composantes
      qui s'interpénètrent)

Si une variante reproduit la hiérarchie de masse à <50% sur les 7
canoniques, on a un cadre prédictif.
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K
from simulation.experiments.em_energy_knots import (
    arc_length, segment_lengths, midpoints,
    d_min_non_adjacent, coulomb_self_energy, mobius_energy,
)
from simulation.experiments.em_energy_full import (
    relax_curve, coulomb_interaction, place_curves_triangular,
)

KNOTS = list(REFERENCE_N50K.keys())

# Rope-lengths publiées (Cantarella, Kusner, Sullivan + Pieranski)
PUBLISHED_ROPE = {
    "3_1":  16.37,  "4_1":  20.99,  "5_1":  23.55,  "5_2":  26.13,
    "6_1":  28.31,  "6_2":  28.13,  "6_3":  27.21,
    "7_1":  30.69,  "7_2":  32.96,  "7_3":  33.61,  "7_4":  34.97,
    "7_5":  35.16,  "7_6":  32.66,  "7_7":  31.93,
    "8_1":  37.49,  "8_2":  37.32,  "8_3":  37.30,
    "9_1":  38.62,  "10_1": 42.33,
}


def scale_to_rope(pts, R_target):
    """Pieranski donne d_min ≈ 1. On scale pour que L = R_target.
    Comme rope-length = L/d_min, on retrouve R = R_target.
    """
    L = arc_length(pts)
    return pts * (R_target / L)


def main():
    print("=" * 100)
    print("  VOIE A — ÉNERGIE EM AVEC ÉCHELLE INTRINSÈQUE (rope-length physique)")
    print("=" * 100)

    # ─── Relaxation Pieranski (cache) ─────────────────────────────────
    print(f"\n  Relaxation Pieranski 1 (N=200, 600 itérations)…")
    relax_curves = {}
    for K in KNOTS:
        pts = relax_curve(K)
        # Vérification : Pieranski doit donner d_min ≈ 1
        dm = d_min_non_adjacent(pts)
        relax_curves[K] = pts
        print(f"    {K} (R_pub={PUBLISHED_ROPE[K]:>5.2f}): d_min Pieranski = {dm:.3f}")

    # ─── Scale chaque courbe à sa R_pub ──────────────────────────────
    scaled_curves = {K: scale_to_rope(relax_curves[K], PUBLISHED_ROPE[K])
                     for K in KNOTS}

    # ─── Énergies individuelles à l'échelle intrinsèque ─────────────
    print(f"\n  ÉNERGIES INDIVIDUELLES (chaque K à sa R_pub) :")
    print(f"  {'K':>5} {'R_pub':>6} {'L':>7} {'d_min':>7} "
          f"{'U_self':>10}")
    print("  " + "-" * 50)
    for K in KNOTS:
        c = scaled_curves[K]
        L = arc_length(c)
        dm = d_min_non_adjacent(c)
        a = max(dm / 3, L / 200)
        Uself = coulomb_self_energy(c, a=a)
        print(f"  {K:>5} {PUBLISHED_ROPE[K]:>6.2f} {L:>7.2f} "
              f"{dm:>7.3f} {Uself:>10.2f}")

    # ─── Calcul des spectres ─────────────────────────────────────────
    SPECTRUM = [
        ("e⁻",  ["3_1"],               0.511),
        ("μ⁻",  ["3_1", "6_1"],        105.66),
        ("τ⁻",  ["3_1", "5_1", "7_1"], 1776.86),
        ("π⁰",  ["8_1"],               134.98),
        ("π±",  ["8_1"],               139.57),
        ("K±",  ["4_1", "4_1", "5_2"], 493.68),
        ("η",   ["3_1", "4_1", "5_1"], 547.86),
        ("p",   ["3_1", "7_2", "7_2"], 938.272),
        ("n",   ["3_1", "7_2", "7_2"], 939.565),
        ("Λ",   ["4_1", "7_4"],        1115.7),
        ("Δ",   ["3_1", "5_2", "6_2"], 1232.0),
        ("Ω⁻",  ["5_2", "5_2", "7_2"], 1672.0),
        ("W±",  ["4_1", "7_3", "8_3"], 80377.0),
        ("Z⁰",  ["5_1", "7_2", "8_3"], 91188.0),
        ("H",   ["6_1", "7_4", "8_3"], 125250.0),
        ("t",   ["7_4", "7_6", "8_3"], 173100.0),
    ]

    def compute_U_tot(comps, sep_factor=2.5, cutoff=0.3):
        curves = [scaled_curves[K] for K in comps]
        placed = place_curves_triangular(curves, sep_factor=sep_factor)
        U_self_sum = 0.0
        for c in placed:
            L_c = arc_length(c)
            dm_c = d_min_non_adjacent(c)
            a_c = max(dm_c / 3, L_c / 200)
            U_self_sum += coulomb_self_energy(c, a=a_c)
        U_int_sum = 0.0
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                U_int_sum += coulomb_interaction(
                    placed[i], placed[j], cutoff=cutoff)
        return U_self_sum + U_int_sum, U_self_sum, U_int_sum

    # Test V1 : séparation standard, cutoff standard
    print(f"\n  V1. Séparation 2.5×taille, cutoff = 0.3")
    print(f"  {'particule':>5} {'m_exp':>11} {'U_self':>10} {'U_int':>10} "
          f"{'U_tot':>10}")
    rows_V1 = []
    for nm, comps, me in SPECTRUM:
        U_tot, Us, Ui = compute_U_tot(comps, sep_factor=2.5, cutoff=0.3)
        rows_V1.append((nm, me, Us, Ui, U_tot))
        print(f"  {nm:>5} {me:>11.2f} {Us:>10.0f} {Ui:>10.0f} {U_tot:>10.0f}")

    # Test V3 : composantes interpénétrantes (sep < taille)
    print(f"\n  V3. Séparation 0.5×taille (composantes serrées), cutoff = 0.3")
    rows_V3 = []
    for nm, comps, me in SPECTRUM:
        U_tot, Us, Ui = compute_U_tot(comps, sep_factor=0.5, cutoff=0.3)
        rows_V3.append((nm, me, Us, Ui, U_tot))

    # ─── Calibration et prédictions ─────────────────────────────────
    print(f"\n  PRÉDICTIONS (calibration α = m_e / U_tot(e⁻)) :")
    print(f"  {'particule':>5} {'m_exp':>11} {'V1 U_tot':>11} {'V1 m_pred':>11} "
          f"{'Δ %':>9} {'V3 U_tot':>11} {'V3 m_pred':>11} {'Δ %':>9}")
    print("  " + "-" * 92)

    alpha_V1 = SPECTRUM[0][2] / rows_V1[0][4]
    alpha_V3 = SPECTRUM[0][2] / rows_V3[0][4]

    res_V1, res_V3 = [], []
    for (nm, me, _, _, U1), (_, _, _, _, U3) in zip(rows_V1, rows_V3):
        m1 = alpha_V1 * U1
        m3 = alpha_V3 * U3
        e1 = (m1 - me) / me * 100
        e3 = (m3 - me) / me * 100
        res_V1.append(np.log(m1 / me) if me > 0 else 0)
        res_V3.append(np.log(m3 / me) if me > 0 else 0)
        f1 = "★" if abs(e1) < 10 else (" " if abs(e1) < 50 else "?")
        f3 = "★" if abs(e3) < 10 else (" " if abs(e3) < 50 else "?")
        print(f"  {nm:>5} {me:>11.2f} {U1:>11.0f} {m1:>11.2f} "
              f"{e1:>+8.1f}% {f1} {U3:>11.0f} {m3:>11.2f} {e3:>+8.1f}% {f3}")

    rms_V1 = float(np.sqrt(np.mean(np.array(res_V1) ** 2)))
    rms_V3 = float(np.sqrt(np.mean(np.array(res_V3) ** 2)))
    print(f"\n  RMS log :")
    print(f"    V1 (sep=2.5): {rms_V1:.3f}  (= +{(np.exp(rms_V1)-1)*100:.0f}% relatif)")
    print(f"    V3 (sep=0.5): {rms_V3:.3f}  (= +{(np.exp(rms_V3)-1)*100:.0f}% relatif)")

    # ─── Diagnostic ─────────────────────────────────────────────────
    print(f"\n  ─── DIAGNOSTIC ───")
    print(f"  U_self (composante seule) à rope-length intrinsèque :")
    for K in ["3_1", "8_1", "7_2", "8_3"]:
        c = scaled_curves[K]
        L = arc_length(c)
        dm = d_min_non_adjacent(c)
        a = max(dm / 3, L / 200)
        Uself = coulomb_self_energy(c, a=a)
        print(f"    {K} (R={PUBLISHED_ROPE[K]:>5.2f}, L={L:.1f}): "
              f"U_self = {Uself:.1f}")

    print(f"\n  Range U_self individuel : "
          f"{min(rows_V1, key=lambda r: r[2])[2]:.0f} à "
          f"{max(rows_V1, key=lambda r: r[2])[2]:.0f}")
    print(f"  Range m_exp : 0.5 à 173 100 MeV")
    print(f"\n  Pour reproduire la hiérarchie 340 000×, il faut U_tot range")
    print(f"  également 340 000×. Si V1 et V3 plafonnent à ~10×, le mécanisme")
    print(f"  EM linéaire est INSUFFISANT, quelle que soit la calibration.")
    print("=" * 100)


if __name__ == "__main__":
    main()
