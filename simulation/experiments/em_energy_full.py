r"""
ÉTAPES 1 + 2 du programme du brouillon_masse_EM.tex :

  ÉTAPE 1 — relaxer les 19 courbes catalogue par Pieranski 1 et
            recalculer E_Möbius, U_ball, U_self sur les courbes
            propres. Comparer aux valeurs catalogue brutes.

  ÉTAPE 2 — étendre la formule aux entrelacs composites :
              U_tot(L) = Σ_i U(K_i) + Σ_{i<j} U_int(K_i, K_j)
            avec U_int(K_i, K_j) = ∮∮ ds ds' / |γ_i(s) - γ_j(s')|

Cf. docs/brouillon_masse_EM.tex (commit 3c07db0) pour le cadre.

Le script produit :
  - Tableau Pieranski recalculé (19 nœuds)
  - Tableau composites canoniques (p, n, π, Higgs, W, Z, top, ...)
  - Calibration α_EM par l'électron (étape 3 partielle)
  - Comparaison aux masses expérimentales
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K
from simulation.experiments.mass_pieranski import (
    pieranski_iteration, reparametrize, d_min_far,
)
from simulation.experiments.em_energy_knots import (
    arc_length, segment_lengths, midpoints, curvature_arr,
    d_min_non_adjacent, mobius_energy, coulomb_self_energy,
    em_field_energy_in_ball,
)

KNOTS = list(REFERENCE_N50K.keys())


# ──────────────────────────────────────────────────────────────────────
# Relaxation Pieranski 1 → courbes "propres"
# ──────────────────────────────────────────────────────────────────────
def relax_curve(name, n_points=200, n_iter=600, D=1.0,
                alpha=0.15, beta=0.5, skip_frac=0.03):
    pts = knot_curve(name, n_points + 1)[:-1]
    skip = max(int(n_points * skip_frac), 5)
    d0 = d_min_far(pts, skip)
    if d0 < 1e-10:
        d0 = 1e-3
    pts = pts * (D / d0)
    for it in range(n_iter):
        pts = pieranski_iteration(pts, alpha, D, beta, skip)
        if (it + 1) % 50 == 0:
            pts = reparametrize(pts, n_points)
    return pts


def normalize_to_L(pts, L_target=100.0):
    L = arc_length(pts)
    return pts * (L_target / L)


# ──────────────────────────────────────────────────────────────────────
# ÉTAPE 1 : energies sur courbes Pieranski normalisées
# ──────────────────────────────────────────────────────────────────────
def step1_pieranski_energies():
    print("=" * 100)
    print("  ÉTAPE 1 — ÉNERGIES EM SUR COURBES PIERANSKI (relaxées + normalisées L=100)")
    print("=" * 100)

    rows = []
    print(f"\n  Relaxation Pieranski 1 + calcul des énergies (N=200, n_iter=600)…")
    relax_curves = {}
    for K in KNOTS:
        pts = relax_curve(K)
        pts = normalize_to_L(pts)
        L = arc_length(pts)
        kap, _ = curvature_arr(pts)
        dmin = d_min_non_adjacent(pts)
        a_cutoff = max(dmin / 3, L / 200)
        E_M = mobius_energy(pts)
        U_self = coulomb_self_energy(pts, a=a_cutoff)
        U_ball, R_ball, R_max = em_field_energy_in_ball(pts, n_grid=18)
        rows.append({
            "K": K, "L": L, "R_max": R_max, "d_min": dmin,
            "k_max": float(kap.max()), "k_mean": float(kap.mean()),
            "E_M": E_M, "U_ball": U_ball, "U_self": U_self,
        })
        relax_curves[K] = pts
        print(f"    {K}: R_max={R_max:.2f}, d_min={dmin:.3f}, "
              f"E_M={E_M:.2f}, U_ball={U_ball:.1f}, U_self={U_self:.1f}")

    print(f"\n  TABLEAU ÉTAPE 1 (courbes Pieranski, L=100) :")
    print(f"  {'K':>5} {'R_max':>7} {'d_min':>7} {'κ_max':>8} "
          f"{'E_Möbius':>10} {'U_ball':>10} {'U_self':>10}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['K']:>5} {r['R_max']:>7.2f} {r['d_min']:>7.3f} "
              f"{r['k_max']:>8.2f} {r['E_M']:>10.2f} "
              f"{r['U_ball']:>10.2f} {r['U_self']:>10.2f}")

    # Comparaison avec valeurs catalogue brutes
    print(f"\n  COMPARAISON catalogue brut vs Pieranski (E_Möbius) :")
    print(f"  {'K':>5} {'E_M cat':>10} {'E_M pier':>10} {'ratio':>8}")
    for K in KNOTS:
        pts_cat = normalize_to_L(knot_curve(K, 201)[:-1])
        E_M_cat = mobius_energy(pts_cat)
        E_M_pier = [r["E_M"] for r in rows if r["K"] == K][0]
        ratio = E_M_cat / E_M_pier if E_M_pier > 0 else float("inf")
        flag = "✓" if 0.5 < ratio < 2 else ("↑" if ratio > 2 else "↓")
        print(f"  {K:>5} {E_M_cat:>10.2f} {E_M_pier:>10.2f} {ratio:>7.2f}× {flag}")

    return rows, relax_curves


# ──────────────────────────────────────────────────────────────────────
# ÉTAPE 2 : entrelacs composites (avec terme d'interaction Coulomb)
# ──────────────────────────────────────────────────────────────────────
def coulomb_interaction(pts_a, pts_b, cutoff=0.1):
    """U_int(a, b) = ∮_a ∮_b ds_a ds_b / |γ_a - γ_b| avec cutoff."""
    ds_a = segment_lengths(pts_a)
    ds_b = segment_lengths(pts_b)
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    d_eff = np.where(d < cutoff, cutoff, d)
    integrand = 1.0 / d_eff
    weights = np.outer(ds_a, ds_b)
    return float(np.sum(integrand * weights))


def em_field_energy_composite_ball(curves, n_grid=18, charge=1.0):
    """U_ball pour un composite : champ E total = somme des contributions
    de toutes les composantes, intégré sur sphère englobante."""
    all_pts = np.vstack(curves)
    centroid = all_pts.mean(axis=0)
    R_max = float(np.linalg.norm(all_pts - centroid, axis=1).max())
    R_ball = 1.5 * R_max
    xs = np.linspace(-R_ball, R_ball, n_grid)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    grid = np.stack([X, Y, Z], axis=-1) + centroid
    r_from_c = np.linalg.norm(grid - centroid, axis=-1)
    inside = r_from_c < R_ball
    dV = (2 * R_ball / n_grid) ** 3

    # Champ total = somme des contributions de chaque composante
    field_total = np.zeros(grid.shape)
    cutoff = R_ball / n_grid * 1.5
    for pts in curves:
        ds = segment_lengths(pts)
        mids = midpoints(pts)
        diff = grid[..., None, :] - mids[None, None, None, :, :]
        d3 = np.linalg.norm(diff, axis=-1)
        too_close = d3 < cutoff
        safe_d3 = np.where(d3 > cutoff, d3 ** 3, 1.0)
        contrib = charge * ds[None, None, None, :, None] * diff / safe_d3[..., None]
        contrib = np.where(too_close[..., None], 0.0, contrib)
        field_total += contrib.sum(axis=-2)

    E2 = np.sum(field_total ** 2, axis=-1)
    return float((E2 / 2 * inside).sum() * dV), R_ball


def place_curves_triangular(curves, sep_factor=2.5):
    """Place n composantes en n-gone régulier dans le plan xy."""
    placed = []
    n = len(curves)
    for i, c in enumerate(curves):
        if n == 1:
            offset = np.array([0., 0., 0.])
        else:
            theta = 2 * np.pi * i / n
            size = float(np.sqrt(np.mean(np.var(c, axis=0))))
            offset = sep_factor * size * np.array(
                [np.cos(theta), np.sin(theta), 0.])
        placed.append(c + offset)
    return placed


def step2_composites(relax_curves):
    print("\n" + "=" * 100)
    print("  ÉTAPE 2 — ENTRELACS COMPOSITES (avec Coulomb croisé inter-boucles)")
    print("=" * 100)

    SPECTRUM = [
        # (nom, composantes, masse_exp_MeV)
        ("e⁻",       ["3_1"],                  0.511),
        ("μ⁻",       ["3_1", "6_1"],           105.66),
        ("τ⁻",       ["3_1", "5_1", "7_1"],    1776.86),
        ("π⁰",       ["8_1"],                  134.98),
        ("π±",       ["8_1"],                  139.57),
        ("K±",       ["4_1", "4_1", "5_2"],    493.68),
        ("η",        ["3_1", "4_1", "5_1"],    547.86),
        ("ρ/ω",      ["5_2", "7_7"],           775.0),
        ("φ",        ["3_1", "4_1", "6_2"],    1019.46),
        ("D⁰",       ["6_2", "7_7"],           1864.84),
        ("B",        ["4_1", "10_1"],          5279.4),
        ("p",        ["3_1", "7_2", "7_2"],    938.272),
        ("n",        ["3_1", "7_2", "7_2"],    939.565),
        ("Λ",        ["4_1", "7_4"],           1115.7),
        ("Σ⁺",       ["6_2", "7_2"],           1189.4),
        ("Δ",        ["3_1", "5_2", "6_2"],    1232.0),
        ("Ω⁻",       ["5_2", "5_2", "7_2"],    1672.0),
        ("W±",       ["4_1", "7_3", "8_3"],    80377.0),
        ("Z⁰",       ["5_1", "7_2", "8_3"],    91188.0),
        ("H",        ["6_1", "7_4", "8_3"],    125250.0),
        ("t",        ["7_4", "7_6", "8_3"],    173100.0),
    ]

    print(f"\n  Calcul U_tot = Σ U(K_i) + Σ U_int(K_i, K_j)  sur arrangement")
    print(f"  triangulaire (séparation = 2.5 × taille caractéristique)…")

    rows = []
    for nm, comps, m_exp in SPECTRUM:
        curves = [relax_curves[K] for K in comps]
        curves_placed = place_curves_triangular(curves, sep_factor=2.5)

        # Self-energy de chaque composante (U_self)
        U_self_sum = 0.0
        for c in curves_placed:
            L_c = arc_length(c)
            dmin_c = d_min_non_adjacent(c)
            a_c = max(dmin_c / 3, L_c / 200)
            U_self_sum += coulomb_self_energy(c, a=a_c)

        # Interactions croisées
        U_int_sum = 0.0
        for i in range(len(curves_placed)):
            for j in range(i + 1, len(curves_placed)):
                U_int_sum += coulomb_interaction(
                    curves_placed[i], curves_placed[j], cutoff=0.3)

        # Énergie totale champ EM (méthode directe sur grille)
        U_ball, R_ball = em_field_energy_composite_ball(curves_placed, n_grid=16)

        U_tot_self = U_self_sum + U_int_sum
        rows.append({
            "name": nm, "comps": comps, "n": len(comps), "m_exp": m_exp,
            "U_self_sum": U_self_sum, "U_int": U_int_sum,
            "U_tot": U_tot_self, "U_ball_composite": U_ball,
        })
        print(f"    {nm:>5} ({len(comps)} comp.): U_self={U_self_sum:>9.0f}, "
              f"U_int={U_int_sum:>8.0f}, U_tot={U_tot_self:>9.0f}, "
              f"U_ball={U_ball:>9.0f}")

    # ── Calibration α_EM par l'électron ──────────────────────────────
    e_row = next(r for r in rows if r["name"] == "e⁻")
    alpha_EM_self = e_row["m_exp"] / e_row["U_tot"]
    alpha_EM_ball = e_row["m_exp"] / e_row["U_ball_composite"]
    print(f"\n  CALIBRATION α_EM (étape 3 partielle) :")
    print(f"    via U_tot (self+int) : α = m_e / U_tot(e⁻) = {alpha_EM_self:.4e} MeV/(unit)")
    print(f"    via U_ball composite : α = m_e / U_ball     = {alpha_EM_ball:.4e} MeV/(unit)")

    # ── Tableau final prédictions ────────────────────────────────────
    print(f"\n  TABLEAU PRÉDICTIONS (α_EM calibré sur e⁻ via U_tot self+int) :")
    print(f"  {'particule':>5} {'n':>2} {'m_exp':>11} {'U_tot':>10} "
          f"{'m_pred':>12} {'Δ %':>9} {'U_ball comp.':>13} {'m_b':>12}")
    print("  " + "-" * 90)
    rms_self = []
    rms_ball = []
    for r in rows:
        m_self = alpha_EM_self * r["U_tot"]
        m_ball = alpha_EM_ball * r["U_ball_composite"]
        e_self = (m_self - r["m_exp"]) / r["m_exp"] * 100
        e_ball = (m_ball - r["m_exp"]) / r["m_exp"] * 100
        rms_self.append(np.log(m_self / r["m_exp"]))
        rms_ball.append(np.log(m_ball / r["m_exp"]))
        flag = "★" if abs(e_self) < 5 else (" " if abs(e_self) < 50 else "?")
        print(f"  {r['name']:>5} {r['n']:>2} {r['m_exp']:>11.2f} "
              f"{r['U_tot']:>10.0f} {m_self:>12.2f} {e_self:>+8.1f}% "
              f"{r['U_ball_composite']:>13.0f} {m_ball:>11.2f} {flag}")

    rms_s = float(np.sqrt(np.mean(np.array(rms_self) ** 2)))
    rms_b = float(np.sqrt(np.mean(np.array(rms_ball) ** 2)))
    print(f"\n  RMS log :")
    print(f"    via U_tot self+int   : {rms_s:.3f}  (= +{(np.exp(rms_s)-1)*100:.0f}% relatif)")
    print(f"    via U_ball composite : {rms_b:.3f}  (= +{(np.exp(rms_b)-1)*100:.0f}% relatif)")

    return rows, alpha_EM_self


def main():
    rows1, relax_curves = step1_pieranski_energies()
    rows2, alpha = step2_composites(relax_curves)

    # ─── Export LaTeX pour brouillon ─────────────────────────────────
    tex_path = "docs/em_energy_pieranski_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"% Étape 1 : énergies EM sur courbes Pieranski (L=100)" + "\n")
        f.write(r"\begin{center}" + "\n")
        f.write(r"\small\renewcommand{\arraystretch}{1.1}" + "\n")
        f.write(r"\begin{tabular}{lrrrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"$K$ & $R_{\max}$ & $d_{\min}$ & $\kappa_{\max}$ & "
                r"$E_{\mathrm{M\"ob}}$ & $U_{\mathrm{ball}}$ & "
                r"$U_{\mathrm{self}}$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in rows1:
            knot_tex = r["K"].replace("_", r"_")
            f.write(f"  ${knot_tex}$ & {r['R_max']:.2f} & "
                    f"{r['d_min']:.3f} & {r['k_max']:.2f} & "
                    f"{r['E_M']:.2f} & {r['U_ball']:.1f} & "
                    f"{r['U_self']:.1f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{center}" + "\n")
    print(f"\n  ✓ Tableau étape 1 exporté dans : {tex_path}")

    # ─── Tableau composites en LaTeX ─────────────────────────────────
    tex_path2 = "docs/em_energy_composites_table.tex"
    with open(tex_path2, "w", encoding="utf-8") as f:
        f.write(r"% Étape 2 : entrelacs composites avec interaction" + "\n")
        f.write(r"\begin{center}" + "\n")
        f.write(r"\small\renewcommand{\arraystretch}{1.1}" + "\n")
        f.write(r"\begin{tabular}{lrrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"particule & $m_{\text{exp}}$ (MeV) & $U_{\text{self}}^{\Sigma}$ & "
                r"$U_{\text{int}}^{\Sigma}$ & $U_{\text{tot}}$ & "
                r"$m_{\text{pred}}$ (MeV) \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in rows2:
            m_pred = alpha * r["U_tot"]
            err = (m_pred - r["m_exp"]) / r["m_exp"] * 100 if r["m_exp"] > 0 else 0
            f.write(f"  {r['name']} & {r['m_exp']:.3f} & "
                    f"{r['U_self_sum']:.0f} & {r['U_int']:.0f} & "
                    f"{r['U_tot']:.0f} & {m_pred:.2f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{center}" + "\n")
    print(f"  ✓ Tableau étape 2 exporté dans : {tex_path2}")
    print("=" * 100)


if __name__ == "__main__":
    main()
