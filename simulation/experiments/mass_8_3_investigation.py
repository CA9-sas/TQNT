r"""
Chantier 1 — investigation U_EM(8_3) : intrinsèque ou artefact ?

Le nœud 8_3 est l'OUTLIER MAJEUR de la corrélation U_EM ↔ c_J : son
U_EM = 1492 est 12,4× la valeur prédite par la corrélation moyenne
(~120). C'est ce qui place le secteur EW (W/Z/H/top, tous porteurs de
8_3) à l'échelle de 100 GeV.

Question : ce U_EM démesuré est-il INTRINSÈQUE au nœud 8_3 (propriété
topologique fondamentale), ou ARTEFACT de la paramétrisation Fourier
haute fréquence choisie dans le catalogue ?

Tests :
1. Convergence de U_EM en n_points (numérique stable ?).
2. Profile κ(t), τ(t) le long de la courbe — y a-t-il des pics localisés ?
3. Variantes paramétriques (sans terme haute fréquence, ou amplitudes
   réduites) et leur U_EM.
4. Si U_EM chute fortement avec amplitudes réduites → artefact.
   Si stable → intrinsèque.

ATTENTION : les variantes paramétriques peuvent ne PAS conserver la
topologie 8_3 (vérifier par Jones serait l'étape suivante). On
mesure ici la SENSIBILITÉ de U_EM aux choix paramétriques, pas la
valeur intrinsèque pour un 8_3 idéal.

Cf. docs/brouillon_hamiltonien_multipart.tex.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.knot_catalog import _fourier_curve
from theory.abraham import frenet_geometry, geometric_integrals


def compute_UEM_from_points(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Calcule (L, ∮κ²ds, ∮τ²ds, U_EM=somme) depuis une courbe (n+1, 3)
    avec doublon de bouclage. Réutilise la logique Frenet-Serret d'abraham."""
    pts = pts[:-1]
    N = len(pts)
    dt = 2 * np.pi / N
    p_next = np.roll(pts, -1, axis=0)
    p_prev = np.roll(pts, 1, axis=0)
    p_next2 = np.roll(pts, -2, axis=0)
    p_prev2 = np.roll(pts, 2, axis=0)
    dr = (p_next - p_prev) / (2 * dt)
    ddr = (p_next - 2 * pts + p_prev) / dt ** 2
    dddr = (p_next2 - 2 * p_next + 2 * p_prev - p_prev2) / (2 * dt ** 3)
    speed = np.linalg.norm(dr, axis=1)
    cross = np.cross(dr, ddr)
    cross_norm = np.linalg.norm(cross, axis=1)
    kappa = cross_norm / (speed ** 3 + 1e-30)
    triple = np.sum(cross * dddr, axis=1)
    tau = triple / (cross_norm ** 2 + 1e-30)
    ds = speed * dt
    L = float(np.sum(ds))
    int_k2 = float(np.sum(kappa ** 2 * ds))
    int_t2 = float(np.sum(tau ** 2 * ds))
    return L, int_k2, int_t2, int_k2 + int_t2


def make_variant(amp_high: float, n: int) -> np.ndarray:
    """Variante 8_3 avec amplitude des harmoniques 5/6 multipliée par amp_high.
    amp_high = 1 → paramétrisation catalogue ; amp_high = 0 → sans haute fréq."""
    return _fourier_curve(
        x_terms=[(2.0, 1, np.pi / 2), (1.0, 3, np.pi / 2), (0.4 * amp_high, 5, np.pi / 2)],
        y_terms=[(2.0, 1, 0),        (-1.0, 3, 0),         (0.4 * amp_high, 5, 0)],
        z_terms=[(1.0, 2, 0),        (-1.0, 4, 0),         (0.3 * amp_high, 6, 0)],
        n_points=n,
    )


def main():
    print("=" * 70)
    print("  INVESTIGATION 8_3 — U_EM intrinsèque ou artefact d'embedding ?")
    print("=" * 70)

    # (1) Convergence en n_points (catalogue) -----------------------------
    print("\n  (1) Convergence U_EM en n_points (paramétrisation catalogue) :")
    print(f"  {'n':>7} {'sqrt_U':>10} {'U_EM':>10}")
    for n_test in (500, 1000, 5000, 10000, 50000):
        gi = geometric_integrals("8_3", n_test)
        sU = gi["sqrt_U"]
        print(f"  {n_test:>7} {sU:>10.4f} {sU**2:>10.2f}")

    # (2) Profile κ²+τ² catalogue -----------------------------------------
    print("\n  (2) Profile (κ² + τ²)(t) le long de la courbe catalogue :")
    n_main = 20000
    geo = frenet_geometry("8_3", n_main)
    t = geo["t"]
    kappa = geo["kappa"]
    tau = geo["tau"]
    speed = geo["speed"]
    ds = speed * (2 * np.pi / n_main)
    L_cat = float(np.sum(ds))
    int_k2_cat = float(np.sum(kappa ** 2 * ds))
    int_t2_cat = float(np.sum(tau ** 2 * ds))
    U_cat = int_k2_cat + int_t2_cat
    print(f"    L = {L_cat:.2f}")
    print(f"    ∮κ²ds = {int_k2_cat:.2f}    ∮τ²ds = {int_t2_cat:.2f}")
    print(f"    U_EM (somme) = {U_cat:.2f}")

    # localiser les pics (top 10 % de κ²+τ²)
    integrand = (kappa ** 2 + tau ** 2) * ds
    cum_share = np.sort(integrand)[::-1].cumsum() / integrand.sum()
    n_top10 = np.searchsorted(cum_share, 0.5)   # combien de points pour 50 % ?
    n_top90 = np.searchsorted(cum_share, 0.9)
    print(f"    {n_top10}/{n_main} points portent 50 % de l'intégrale "
          f"({n_top10/n_main*100:.1f} %)")
    print(f"    {n_top90}/{n_main} points portent 90 % de l'intégrale "
          f"({n_top90/n_main*100:.1f} %)")

    # (3) Variantes paramétriques ---------------------------------------
    print("\n  (3) Variantes avec amplitude haute fréquence réduite :")
    print(f"  {'amp_high':>10} {'L':>8} {'∮κ²ds':>10} {'∮τ²ds':>10} {'U_EM':>10}")
    n_var = 10000
    results = []
    for amp in (1.0, 0.5, 0.2, 0.1, 0.0):
        pts = make_variant(amp, n_var)
        L, k2, t2, U = compute_UEM_from_points(pts)
        results.append((amp, L, k2, t2, U))
        print(f"  {amp:>10.2f} {L:>8.2f} {k2:>10.2f} {t2:>10.2f} {U:>10.2f}")
    U_no_high = results[-1][4]

    # (4) Synthèse ---------------------------------------------------------
    print("\n  (4) Synthèse :")
    print(f"    U_EM catalogue (amp=1)     : {U_cat:>10.2f}")
    print(f"    U_EM sans haute fréq (amp=0) : {U_no_high:>10.2f}")
    ratio = U_cat / U_no_high
    print(f"    Ratio = {ratio:.1f}×")
    print()
    if ratio > 5:
        print(f"    => ARTEFACT D'EMBEDDING : la haute fréquence inflate U_EM")
        print(f"       d'un facteur {ratio:.0f}×.")
        print(f"       Sans elle, U_EM tomberait à {U_no_high:.0f}, donc DANS")
        print(f"       la corrélation moyenne (~120).")
        print(f"       ⇒ la spécificité de 8_3 dans V4 vient du CHOIX paramétrique.")
        print(f"       ⇒ l'échelle EW dépend du choix de paramétrisation du catalogue.")
    elif ratio > 2:
        print(f"    => SENSIBILITÉ NOTABLE : facteur {ratio:.1f}× ; ambivalent.")
    else:
        print(f"    => INTRINSÈQUE : U_EM reste élevé sans haute fréq")
        print(f"       (facteur {ratio:.1f}× seulement).")

    print()
    print(f"    ATTENTION : sans vérification que les variantes restent")
    print(f"    topologiquement 8_3 (calcul de Jones depuis la courbe 3D),")
    print(f"    le test mesure la SENSIBILITÉ paramétrique, pas la valeur")
    print(f"    intrinsèque d'un 8_3 idéal.")

    # Sauvegarde figure
    try:
        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        ax[0].plot(t, kappa, lw=0.7, color="C0")
        ax[0].set_ylabel("κ(t)")
        ax[0].set_title(f"8_3 catalogue : profile κ et τ (haute fréquence visible)")
        ax[1].plot(t, tau, lw=0.7, color="C1")
        ax[1].set_xlabel("t (paramètre)")
        ax[1].set_ylabel("τ(t)")
        ax[1].axhline(0, color="k", lw=0.3)
        fig.tight_layout()
        out = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..", "..", "output", "8_3_kappa_tau.png"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"\n  Figure → {out}")
    except Exception as e:
        print(f"  (figure non générée : {e})")

    print("=" * 70)


if __name__ == "__main__":
    main()
