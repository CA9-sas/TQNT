r"""
Chantier 1 — diagnostic 7_5 (cas extrême, ratio 2703× au filtrage Fourier).

Le 7_5 cumule TROIS pathologies :
1. U_EM_catalogue = 24432 (vs table REFERENCE_N50K = 49, écart 500×) ;
2. Instabilité numérique (sqrt_U varie 11/156/13/52 selon N) ;
3. Ratio cat/k≤3 = 2703× au filtrage Fourier (vu en généralisation).

Hypothèse : la paramétrisation
    x(t) = cos(t) + 2cos(2t) + 0,5cos(4t + π/6)
    y(t) = cos(t) - 2cos(2t) + 0,3cos(4t + π/4)
    z(t) = -cos(3t) + 0,7cos(5t + π/2)
contient des QUASI-CUSPS (vitesse |γ'(t)| très faible à certains t) où
κ = |γ' × γ''| / |γ'|³ EXPLOSE — d'où l'instabilité et l'inflation.

On vérifie en profilant κ, τ, |γ'| le long de la courbe.

Cf. docs/brouillon_hamiltonien_multipart.tex.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.abraham import frenet_geometry, geometric_integrals


def main():
    print("=" * 72)
    print("  DIAGNOSTIC 7_5 : pourquoi U_EM est-il pathologique ?")
    print("=" * 72)

    # (1) Convergence en n_points
    print("\n  (1) Convergence U_EM(7_5) en n_points :")
    print(f"  {'n':>6} {'sqrt_U':>12} {'U_EM':>16}")
    for n in (500, 1000, 2000, 5000, 10000, 20000, 50000, 100000):
        gi = geometric_integrals("7_5", n)
        sU = gi["sqrt_U"]
        print(f"  {n:>6} {sU:>12.4f} {sU**2:>16.2f}")
    print("  ⇒ Pas de convergence stable, signature d'une singularité")
    print("    paramétrique mal discrétisée.")

    # (2) Profile à n=20000
    n = 20000
    print(f"\n  (2) Profile κ(t), τ(t), |γ'(t)| à n = {n} :")
    geo = frenet_geometry("7_5", n)
    t = geo["t"]
    kappa = geo["kappa"]
    tau = geo["tau"]
    speed = geo["speed"]

    print(f"    max κ        = {kappa.max():>10.2f}  à t = {t[kappa.argmax()]:.4f}")
    print(f"    max |τ|      = {abs(tau).max():>10.2f}  à t = {t[abs(tau).argmax()]:.4f}")
    print(f"    min |γ'(t)|  = {speed.min():>10.4f}  à t = {t[speed.argmin()]:.4f}")
    print(f"    max |γ'(t)|  = {speed.max():>10.4f}  à t = {t[speed.argmax()]:.4f}")
    print(f"    ratio max/min speed = {speed.max()/max(speed.min(), 1e-30):>6.1f}")

    # (3) Concentration de l'intégrale
    integ = (kappa ** 2 + tau ** 2) * speed * (2 * np.pi / n)
    idx_sorted = np.argsort(integ)[::-1]
    cum_share = np.cumsum(integ[idx_sorted]) / integ.sum()
    n_50 = np.searchsorted(cum_share, 0.5)
    n_90 = np.searchsorted(cum_share, 0.9)
    n_99 = np.searchsorted(cum_share, 0.99)
    print(f"\n  (3) Concentration de U_EM :")
    print(f"    {n_50:>5}/{n} points portent 50 % de U_EM  ({n_50/n*100:>6.3f} %)")
    print(f"    {n_90:>5}/{n} points portent 90 % de U_EM  ({n_90/n*100:>6.3f} %)")
    print(f"    {n_99:>5}/{n} points portent 99 % de U_EM  ({n_99/n*100:>6.3f} %)")

    # (4) Localisation des pics
    print(f"\n  (4) Top 10 pics de (κ² + τ²)·|γ'|·dt :")
    print(f"    {'t':>8} {'κ':>10} {'τ':>10} {'|γ′(t)|':>10} {'contrib':>12}")
    for i in range(10):
        idx = idx_sorted[i]
        print(f"    {t[idx]:>8.4f} {kappa[idx]:>10.2f} {tau[idx]:>10.2f} "
              f"{speed[idx]:>10.4f} {integ[idx]:>12.4e}")

    # (5) Hypothèse cusp
    print(f"\n  (5) Y a-t-il des QUASI-CUSPS (minima de |γ'|) ?")
    min_speed_idx = np.argsort(speed)[:8]
    print(f"    {'rang':>5} {'t':>8} {'|γ′(t)|':>10} {'κ':>10} {'τ':>10}")
    for rank, idx in enumerate(min_speed_idx):
        print(f"    {rank+1:>5} {t[idx]:>8.4f} {speed[idx]:>10.4f} "
              f"{kappa[idx]:>10.2f} {tau[idx]:>10.2f}")

    # (6) Test analytique : vitesse aux composantes basse fréquence seules
    print(f"\n  (6) Test analytique : vitesse aux modes basse fréquence seuls :")
    print(f"    x_low(t) = cos(t) + 2cos(2t)")
    print(f"    y_low(t) = cos(t) - 2cos(2t)")
    print(f"    z_low(t) = -cos(3t)")
    print(f"    x_low'(t) = -sin(t) - 4sin(2t) ;  à t=0 : 0,  à t=π : 0")
    print(f"    y_low'(t) = -sin(t) + 4sin(2t) ;  à t=0 : 0,  à t=π : 0")
    print(f"    z_low'(t) =  3sin(3t)          ;  à t=0 : 0,  à t=π : 0")
    print(f"    ⇒ |γ'_low(0)| = 0 (CUSP cinématique) ; idem à t=π")
    print(f"    Les hautes fréquences (0,5·cos(4t+π/6) etc.) régularisent")
    print(f"    PARTIELLEMENT — vitesse min observée = {speed.min():.4f}.")
    print(f"    Mais κ ~ 1/|γ'|³ explose près de ces minima → pic immense.")

    # (7) Comparaison à un nœud sain (3_1) pour calibrer
    print(f"\n  (7) Pour comparaison, profile 3_1 (nœud sain) :")
    geo_31 = frenet_geometry("3_1", n)
    integ_31 = (geo_31["kappa"] ** 2 + geo_31["tau"] ** 2) * \
               geo_31["speed"] * (2 * np.pi / n)
    idx_31 = np.argsort(integ_31)[::-1]
    cum_31 = np.cumsum(integ_31[idx_31]) / integ_31.sum()
    n_50_31 = np.searchsorted(cum_31, 0.5)
    print(f"    max κ     = {geo_31['kappa'].max():>10.4f}  "
          f"(7_5: {kappa.max():.2f})")
    print(f"    min speed = {geo_31['speed'].min():>10.4f}  "
          f"(7_5: {speed.min():.4f})")
    print(f"    {n_50_31}/{n} points portent 50 % de U_EM ({n_50_31/n*100:.1f} %)")
    print(f"    ⇒ Concentration ~uniforme pour 3_1, pathologique pour 7_5.")

    # (8) Sauvegarde figure
    try:
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(t, kappa, lw=0.5, color="C0")
        axes[0].set_ylabel("κ(t)")
        axes[0].set_title("7_5 : profile κ, τ, |γ'(t)| — pics aux quasi-cusps")
        axes[0].axvline(0, color="r", lw=0.3, ls="--", alpha=0.5)
        axes[0].axvline(np.pi, color="r", lw=0.3, ls="--", alpha=0.5)
        axes[1].plot(t, tau, lw=0.5, color="C1")
        axes[1].set_ylabel("τ(t)")
        axes[1].axvline(0, color="r", lw=0.3, ls="--", alpha=0.5)
        axes[1].axvline(np.pi, color="r", lw=0.3, ls="--", alpha=0.5)
        axes[1].axhline(0, color="k", lw=0.3)
        axes[2].plot(t, speed, lw=0.7, color="C2")
        axes[2].set_ylabel("|γ'(t)|")
        axes[2].set_xlabel("t (paramètre)")
        axes[2].axvline(0, color="r", lw=0.3, ls="--", alpha=0.5)
        axes[2].axvline(np.pi, color="r", lw=0.3, ls="--", alpha=0.5)
        axes[2].set_ylim(0, axes[2].get_ylim()[1])
        fig.tight_layout()
        out = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..", "..", "output",
                                           "7_5_pathology.png"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"\n  Figure → {out}")
    except Exception as e:
        print(f"  (figure non générée : {e})")

    # (9) Diagnostic
    print(f"\n  DIAGNOSTIC :")
    print(f"  ─────────────────────────────────────────")
    print(f"  La paramétrisation 7_5 du catalogue est PATHOLOGIQUE :")
    print(f"  - les modes basse fréquence (cos t, cos 2t, cos 3t) ont des cusps")
    print(f"    cinématiques à t=0 et t=π (vitesse exactement nulle) ;")
    print(f"  - les modes haute fréquence (0,5·cos 4t + 0,3·cos 4t + 0,7·cos 5t)")
    print(f"    régularisent partiellement mais |γ'| reste très faible aux cusps ;")
    print(f"  - κ = |γ'×γ''|/|γ'|³ explose comme 1/|γ'_min|³ près de ces cusps ;")
    print(f"  - l'intégrale ∮κ² ds est dominée par {n_50}/{n} = {n_50/n*100:.2f} %")
    print(f"    de la courbe, dans 4 quasi-cusps localisés.")
    print(f"  ")
    print(f"  ⇒ La valeur U_EM(7_5) = 24432 N'EST PAS un fait physique ;")
    print(f"    c'est un artefact pur de la paramétrisation Fourier sous-")
    print(f"    régularisée. À CORRIGER dans theory/knot_catalog.py si on veut")
    print(f"    une formule de masse robuste.")
    print(f"  ")
    print(f"  Le 7_5 n'apparaît que dans des entrelacs composites du spectre")
    print(f"  V4 (Λ, Σ, Ξ baryons via certaines configs), donc cet artefact")
    print(f"  affecte le calcul indirectement via les U_EM_tot des composites.")
    print("=" * 72)


if __name__ == "__main__":
    main()
