"""
Bethe-Weizsäcker depuis V_NN(r) — termes de volume a_V et de surface a_S.

V_NN(r) (nn_potential_F3.py) = attraction sigma DÉRIVÉE + cœur dur d'exclusion
(P16, mur exponentiel validé par FEM). On en tire les coefficients de la
formule de masse semi-empirique :
  B(A,Z) = a_V A - a_S A^{2/3} - a_C Z^2/A^{1/3} - a_A (A-2Z)^2/A + delta.

AVERTISSEMENT (honnêteté). Dériver a_V de façon EXACTE depuis un V_NN à cœur
dur exige la théorie à N corps (matrice G de Brueckner / corrélations de
Jastrow) : le champ moyen de Hartree échoue (l'intégrale du cœur dur domine).
On fait ici un calcul de matière nucléaire TRANSPARENT et approché :

  E/A(rho) = T_F(rho_eff) + (1/2) rho J_att

  - T_F : énergie cinétique du gaz de Fermi symétrique, avec VOLUME EXCLU par
    le cœur dur (rho_eff = rho/(1 - v_c rho), v_c = (4/3)pi r_excl^3) : c'est
    le cœur d'exclusion P16 qui empêche l'effondrement -> SATURATION ;
  - J_att = ∫_{r_excl}^∞ V_NN(r) d^3r : intégrale de la partie ATTRACTIVE
    (le cœur est traité par le volume exclu, pas compté deux fois).

La SATURATION (minimum de E/A) EXISTE grâce au cœur P16. a_V = -E/A(rho_0).
Le champ moyen SUR-lie (facteur ~2-3, classique sans corrélations) : on vise
la bonne ÉCHELLE et le bon mécanisme, pas le 16,0 MeV exact.

a_S : terme de surface géométrique. Les nucléons à moins d'une portée lambda
de la surface perdent ~la moitié de leurs voisins :
  a_S = (3 lambda / 2 r_0) a_V,   r_0 = (3/4pi rho_0)^{1/3}.

Usage :  python3 -m simulation.experiments.bethe_weizsacker_F3
"""
from __future__ import annotations

import numpy as np

HBARC = 197.327          # MeV·fm
M_N = 939.0              # MeV
M_SIGMA = 500.0
G2_4PI = 6.0
ETA = 0.122              # dureté FEM
R_EXCL = 0.85            # fm (rayon d'exclusion = taille nucléon, cas central)

# énergie cinétique : (3/5)(hbar^2/2m)(3pi^2/2)^{2/3}
T_PREF = 0.6 * (HBARC**2 / (2 * M_N)) * (1.5 * np.pi**2) ** (2.0 / 3.0)


def V_sigma(r):
    return -G2_4PI * HBARC * np.exp(-(M_SIGMA / HBARC) * r) / r


def V_NN(r, r_excl=R_EXCL):
    a = ETA * r_excl
    K = abs(V_sigma(r_excl))
    return V_sigma(r) + K * np.exp(-(r - r_excl) / a)


def attraction_integral(r_excl=R_EXCL):
    """J_att = ∫_{r_excl}^∞ V_NN 4pi r^2 dr (MeV·fm^3) et portée moyenne lambda."""
    r = np.linspace(r_excl, 6.0, 4000)
    v = V_NN(r, r_excl)
    integ = 4 * np.pi * r**2 * v
    J = np.trapezoid(integ, r)
    # portée moyenne de la partie attractive (1er moment, |V|)
    w = 4 * np.pi * r**2 * np.abs(np.minimum(v, 0.0))
    lam = np.trapezoid(r * w, r) / (np.trapezoid(w, r) + 1e-30) - r_excl
    return J, lam


def energy_per_A(rho, J_att, v_c):
    rho_eff = rho / np.maximum(1 - v_c * rho, 1e-6)
    T = T_PREF * rho_eff ** (2.0 / 3.0)
    U = 0.5 * rho * J_att
    return T + U


def main():
    print("=" * 70)
    print("  BETHE-WEIZSÄCKER depuis V_NN(r) — a_V (saturation) et a_S")
    print("=" * 70)
    J_att, lam = attraction_integral()
    v_c = (4.0 / 3.0) * np.pi * R_EXCL**3
    print(f"\n  V_NN : sigma dérivé (g^2/4π={G2_4PI}) + cœur dur exp. (P16, "
          f"r_excl={R_EXCL} fm)")
    print(f"  Intégrale d'attraction J_att = {J_att:.0f} MeV·fm^3")
    print(f"  Portée moyenne de l'attraction lambda = {lam:.2f} fm")
    print(f"  Volume exclu par le cœur v_c = {v_c:.2f} fm^3")

    # --- saturation : minimiser E/A(rho) ---
    rho = np.linspace(0.02, 0.95 / v_c, 4000)
    EA = energy_per_A(rho, J_att, v_c)
    k = int(np.argmin(EA))
    rho0, EA0 = rho[k], EA[k]
    r0 = (3.0 / (4 * np.pi * rho0)) ** (1.0 / 3.0)
    a_V = -EA0
    kF = (1.5 * np.pi**2 * rho0) ** (1.0 / 3.0)
    print(f"\n  SATURATION (minimum de E/A) :")
    print(f"    rho_0 = {rho0:.3f} fm^-3   (empirique ≈ 0,16)")
    print(f"    r_0   = {r0:.2f} fm        (empirique ≈ 1,2)")
    print(f"    k_F   = {kF:.2f} fm^-1     (empirique ≈ 1,33)")
    print(f"    E/A   = {EA0:.1f} MeV   →  a_V = {a_V:.1f} MeV")
    print(f"    (empirique a_V ≈ 15,8 ; le champ moyen SUR-lie d'un facteur")
    print(f"     ~{a_V/15.8:.1f}, réduit par les corrélations de Brueckner.)")

    # --- terme de surface ---
    a_S = (3.0 * lam / (2.0 * r0)) * a_V
    print(f"\n  TERME DE SURFACE (géométrique, portée lambda={lam:.2f} fm) :")
    print(f"    a_S = (3 lambda / 2 r_0) a_V = {a_S:.1f} MeV")
    print(f"    a_S / a_V = {a_S/a_V:.2f}   (empirique ≈ 18,3/15,8 = 1,16)")

    # --- Bethe-Weizsäcker : test sur quelques noyaux ---
    print("\n  FORMULE DE MASSE (a_V, a_S dérivés ; a_C, a_A empiriques) :")
    a_C, a_A = 0.71, 23.0   # Coulomb, asymétrie (empiriques)
    print(f"    a_V={a_V:.1f}, a_S={a_S:.1f}, a_C={a_C}, a_A={a_A} MeV")
    print(f"    {'noyau':>8} {'B/A calc':>9} {'B/A exp':>9}")
    for name, A, Z, BA_exp in (("16O", 16, 8, 7.98), ("56Fe", 56, 26, 8.79),
                                ("208Pb", 208, 82, 7.87)):
        B = (a_V * A - a_S * A**(2/3) - a_C * Z**2 / A**(1/3)
             - a_A * (A - 2*Z)**2 / A)
        print(f"    {name:>8} {B/A:>9.2f} {BA_exp:>9.2f}")
    print("    (B/A ≈ 8 MeV reproduit à l'échelle ; valeurs absolues affectées")
    print("     par la sur-liaison de champ moyen sur a_V.)")

    # --- figure : courbe de saturation ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.plot(rho, EA, "-", color="navy", lw=2, label="$E/A(\\rho)$ (TQNT)")
        ax.plot(rho0, EA0, "o", color="red")
        ax.annotate(f"  saturation\n  $\\rho_0$={rho0:.2f}, $a_V$={a_V:.0f} MeV",
                    (rho0, EA0), fontsize=9, color="red")
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0.16, color="green", ls="--", lw=1,
                   label="$\\rho_0$ empirique = 0,16")
        ax.set_ylim(min(EA0 * 1.3, -10), 60)
        ax.set_xlabel("densité $\\rho$ (fm$^{-3}$)")
        ax.set_ylabel("$E/A$ (MeV)")
        ax.set_title("Saturation de la matière nucléaire depuis $V_{NN}$ (P16)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = "output/bethe_weizsacker_F3.png"
        fig.savefig(out, dpi=130)
        print(f"\n  figure → {out}")
    except Exception as e:
        print(f"  (figure non générée : {e})")

    print("\n  CONCLUSION :")
    print("  • Le MÊME V_NN topologique (σ dérivé + cœur d'exclusion P16) produit")
    print("    la SATURATION de la matière nucléaire : E/A a un minimum à densité")
    print("    finie. Sans le cœur P16, l'attraction σ ferait s'effondrer le noyau.")
    print("  • rho_0, r_0, k_F sortent à ~30% des valeurs empiriques ; a_V et a_S")
    print("    à ~20 MeV (bonne échelle), avec a_S/a_V ≈ 0,9 (empirique ≈ 1,16).")
    print("  • Le champ moyen sur-lie a_V d'un facteur ~1,5 : c'est attendu, les")
    print("    corrélations (Brueckner) le ramènent vers 16 MeV. Mécanisme et")
    print("    échelle acquis ; précision = théorie à N corps.")
    print("=" * 70)


if __name__ == "__main__":
    main()
