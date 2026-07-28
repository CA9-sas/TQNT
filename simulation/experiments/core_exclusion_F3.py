"""
Cœur répulsif nucléon-nucléon (F3) DÉRIVÉ de l'exclusion topologique (P16).

OBJECTIF. Dans sigma_depth_F2.py le cœur répulsif était ILLUSTRATIF (un
analogue oméga, m_core = 782 MeV, g^2/4pi = 20 posés à la main). L'oméga est
un méson du Modèle Standard ; la TQNT le remplace. Le cœur doit venir de la
TQNT elle-même : c'est l'EXCLUSION TOPOLOGIQUE (P16).

P16. « Le quantum d'action ħ (Chern-Simons) empêche une onde de se localiser »
dans la région d'action de ses voisins. Pour deux nucléons qui s'approchent,
le condensat chi partagé ne peut être co-localisé en dessous d'une échelle
sans payer l'énergie cinétique (de point zéro) de confinement. C'est un cœur
répulsif de type Heisenberg, PAS un échange de méson.

DÉRIVATION du cœur. Confiner le quantum partagé (le mode sigma, masse m_sigma)
à une taille r coûte l'énergie cinétique de Heisenberg

    E_cin(r) = (ħc)^2 / (2 m_sigma r^2)        (par quantum exclu),

amplifiée par le nombre de quanta exclus = complexité topologique du nucléon
c_J_eff (P16 : un nœud plus complexe résiste davantage). Le condensat a une
extension finie (longueur de cohérence ħc/m_sigma), d'où une coupure
exponentielle de même portée que le sigma :

    V_core(r) = c_J_eff * (ħc)^2/(2 m_sigma) * e^{-m_sigma r/ħc} / r^2 .

Tous les ingrédients sont internes à la TQNT (m_sigma calculé, c_J_eff
topologique) : AUCUN paramètre ajusté, contrairement à l'oméga illustratif.

POTENTIEL NN. On l'ajoute au sigma attractif DÉRIVÉ (g_sigma^2/4pi issu de
sigma_coupling_F2.py, ≈ 6 avec la taille réelle du nucléon) :

    V_NN(r) = V_sigma(r) + V_core(r),
    V_sigma(r) = -(g_sigma^2/4pi) ħc e^{-m_sigma r/ħc} / r .

Le 1/r^2 (cœur) bat le 1/r (sigma) à courte distance -> répulsion ; le sigma
domine à ~1 fm -> attraction. Un puits se forme entre les deux.

Usage :  python3 -m simulation.experiments.core_exclusion_F3
"""
from __future__ import annotations

import numpy as np

HBARC = 197.327      # MeV·fm
M_SIGMA = 500.0      # MeV — mode 0+ calculé (sigma_mode_F2)
G2_4PI = 6.0         # couplage sigma DÉRIVÉ (sigma_coupling_F2, R = r_p)
C_J_EFF = 15.0       # complexité topo. effective du nucléon : f(3)*c_J_tot = 0.77*19.5


def V_sigma(r, g2_4pi=G2_4PI, m_sigma=M_SIGMA):
    """Attraction scalaire dérivée (Yukawa), MeV."""
    mu = m_sigma / HBARC
    return -g2_4pi * HBARC * np.exp(-mu * r) / r


def V_core(r, c_J_eff=C_J_EFF, m_sigma=M_SIGMA):
    """Cœur d'exclusion P16 : Heisenberg 1/r^2, coupé à la cohérence."""
    mu = m_sigma / HBARC
    K = c_J_eff * HBARC**2 / (2.0 * m_sigma)     # MeV·fm^2
    return K * np.exp(-mu * r) / r**2


def main():
    print("=" * 70)
    print("  CŒUR RÉPULSIF NN (F3) — DÉRIVÉ de l'exclusion topologique (P16)")
    print("=" * 70)
    K = C_J_EFF * HBARC**2 / (2.0 * M_SIGMA)
    print(f"\n  Ingrédients (tous internes à la TQNT) :")
    print(f"    m_sigma  = {M_SIGMA:.0f} MeV          (mode 0+ calculé)")
    print(f"    g^2/4pi  = {G2_4PI:.1f}              (couplage sigma DÉRIVÉ)")
    print(f"    c_J_eff  = {C_J_EFF:.1f}             (complexité topo. du nucléon)")
    print(f"    portée cœur = ħc/m_sigma = {HBARC/M_SIGMA:.3f} fm")
    print(f"    force du cœur K = c_J_eff(ħc)^2/(2 m_sigma) = {K:.0f} MeV·fm^2")

    r = np.linspace(0.3, 3.0, 600)
    Vatt = V_sigma(r)
    Vrep = V_core(r)
    Vtot = Vatt + Vrep
    kmin = int(np.argmin(Vtot))
    Rmin, Vmin = r[kmin], Vtot[kmin]

    # rayon du mur répulsif : V_NN = 0 (zéro intérieur)
    sign = np.sign(Vtot)
    cross = np.where(np.diff(sign) < 0)[0]   # passage + -> - (mur intérieur)
    r_wall = r[cross[0]] if len(cross) else np.nan

    print("\n  POTENTIEL NN = V_sigma (attraction) + V_core (exclusion P16) :")
    print(f"    {'r (fm)':>7} {'V_sigma':>9} {'V_core':>9} {'V_NN':>9}")
    for rr in (0.4, 0.5, 0.6, 0.7, 0.86, 1.0, 1.3):
        print(f"    {rr:>7.2f} {V_sigma(rr):>8.0f} {V_core(rr):>8.0f} "
              f"{V_sigma(rr)+V_core(rr):>8.0f}")

    print(f"\n  Mur répulsif (V_NN = 0)     : r ≈ {r_wall:.2f} fm")
    print(f"  Minimum du puits            : R = {Rmin:.2f} fm, "
          f"profondeur V = {Vmin:.0f} MeV")
    print("\n  Référence nucléaire empirique : puits ~ -50 à -100 MeV à "
          "~0,7–1 fm,")
    print("  cœur répulsif dur à ~0,4–0,5 fm. Liaison nette ~ -8 MeV/nucléon")
    print("  (après compensation par l'énergie cinétique).")
    if -120 < Vmin < -30 and 0.5 < Rmin < 1.1:
        print("  → profondeur ET position du puits dans la fourchette nucléaire ✓")

    # bande sur les O(1) (c_J_eff entre c_J_tot/2 et c_J_tot ; masse effective)
    print("\n  Sensibilité O(1) (mécanisme dérivé, préfacteur à ±) :")
    print(f"    {'c_J_eff':>8} {'R_min(fm)':>10} {'V_min(MeV)':>11}")
    for cj in (10.0, 15.0, 19.5):
        Vt = V_sigma(r) + V_core(r, c_J_eff=cj)
        k = int(np.argmin(Vt))
        print(f"    {cj:>8.1f} {r[k]:>10.2f} {Vt[k]:>11.0f}")

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.8, 4.5))
        ax.plot(r, Vatt, "--", color="green",
                label="$V_\\sigma$ (échange scalaire, $g^2/4\\pi=6$ dérivé)")
        ax.plot(r, Vrep, ":", color="red",
                label="$V_{\\rm core}$ (exclusion P16, $\\propto 1/r^2$)")
        ax.plot(r, Vtot, "-", color="black", lw=2, label="$V_{NN}$ total")
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(Rmin, color="blue", lw=0.7, ls="--")
        ax.plot(Rmin, Vmin, "o", color="blue")
        ax.annotate(f"  puits {Vmin:.0f} MeV\n  à {Rmin:.2f} fm",
                    (Rmin, Vmin), fontsize=9, color="blue")
        ax.set_ylim(-160, 160)
        ax.set_xlim(0.3, 2.5)
        ax.set_xlabel("$r$ (fm)")
        ax.set_ylabel("$V_{NN}$ (MeV)")
        ax.set_title("Potentiel NN TQNT — cœur d'exclusion P16 (dérivé)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = "output/core_exclusion_F3.png"
        fig.savefig(out, dpi=130)
        print(f"\n  figure → {out}")
    except Exception as e:
        print(f"  (figure non générée : {e})")

    print("\n  CONCLUSION :")
    print("  • Le cœur répulsif n'est PLUS un oméga illustratif : c'est")
    print("    l'EXCLUSION TOPOLOGIQUE (P16), un coût cinétique de Heisenberg.")
    print("  • Forme et portée DÉRIVÉES : 1/r^2, portée ħc/m_sigma, force ∝ c_J_eff.")
    print("  • Combiné au sigma dérivé, il produit un puits NN de profondeur et")
    print("    de position nucléaires SANS aucun paramètre ajusté.")
    print("  • RÉSIDU : facteur O(1) (c_J_eff exact, masse effective du quantum).")
    print("=" * 70)


if __name__ == "__main__":
    main()
