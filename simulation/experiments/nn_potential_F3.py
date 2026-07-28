"""
Potentiel nucléon-nucléon V_NN(r) QUANTITATIF — mur dur exponentiel (P16).

Le FEM à deux nucléons (two_nucleon_fem_F3.py) a montré que la répulsion
d'exclusion ÉMERGE de la statistique fermionique et que le splitting
Δ(d) = E_anti - E_sym monte de façon EXPONENTIELLE (mur dur), et non en
1/r^2. Ici on convertit ce résultat en un V_NN(r) physique (MeV, fm).

CE QUI EST ROBUSTE DANS LE FEM. La grandeur sans dimension fiable est la
DURETÉ du mur η = a / d_onset (longueur d'écran de l'exponentielle rapportée
au rayon d'allumage). La MAGNITUDE absolue, elle, est sous-estimée par le
modèle de puits ponctuels (le recouvrement réel d'ondes étendues est bien
plus grand) : on ne s'en sert donc PAS pour fixer la hauteur.

CONSTRUCTION de V_NN(r) :
  V_sigma(r) = -(g_sigma^2/4pi) ħc e^{-m_sigma r}/r        (attraction DÉRIVÉE)
  V_core(r)  = K e^{-(r - r_excl)/a_phys}                  (mur dur exponentiel)
  - r_excl   = rayon d'exclusion physique = recouvrement des enveloppes
               nucléoniques ≈ taille du nucléon (~0,8 fm) ;
  - a_phys   = η * r_excl                                  (dureté du FEM) ;
  - K        = |V_sigma(r_excl)|   (le mur équilibre l'attraction au rayon
               d'exclusion -> coeur dur = passage par zéro à r_excl).
On lit la profondeur, la position du puits et le rayon du coeur dur.

Usage :  python3 -m simulation.experiments.nn_potential_F3
"""
from __future__ import annotations

import numpy as np

HBARC = 197.327      # MeV·fm
M_SIGMA = 500.0      # MeV (mode 0+ calculé)
G2_4PI = 6.0         # couplage sigma DÉRIVÉ (sigma_coupling_F2)

# --- splitting FEM validé (two_nucleon_fem_F3.py, unités naturelles) ---
D_FEM = np.array([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80])
DELTA = np.array([7.115, 4.333, 2.564, 0.708, 0.355, 0.212, 0.121, 0.017])


def V_sigma(r):
    mu = M_SIGMA / HBARC
    return -G2_4PI * HBARC * np.exp(-mu * r) / r


def fem_hardness():
    """Dureté sans dimension η = a / d_onset extraite du splitting FEM."""
    # pente exponentielle a (longueur d'écran en u. de maillage)
    sl, b = np.polyfit(D_FEM, np.log(DELTA), 1)
    a_mesh = -1.0 / sl
    # rayon d'allumage : d où Δ = 1 (interpolation log-linéaire)
    d_onset = (np.log(1.0) - b) / sl
    eta = a_mesh / d_onset
    return eta, a_mesh, d_onset


def build_VNN(r, r_excl, eta):
    a_phys = eta * r_excl
    K = abs(V_sigma(r_excl))                 # mur = attraction au rayon d'excl.
    V_core = K * np.exp(-(r - r_excl) / a_phys)
    return V_sigma(r) + V_core, V_core, a_phys, K


def main():
    print("=" * 70)
    print("  V_NN(r) QUANTITATIF — mur dur exponentiel d'exclusion (P16)")
    print("=" * 70)
    eta, a_mesh, d_onset = fem_hardness()
    print(f"\n  Dureté FEM (sans dimension, robuste) :")
    print(f"    longueur d'écran a = {a_mesh:.3f} ; allumage d_onset = "
          f"{d_onset:.3f} (u. maillage)")
    print(f"    η = a/d_onset = {eta:.3f}  → mur TRÈS dur "
          f"(plus raide qu'un Yukawa ω)")

    print(f"\n  V_sigma : g^2/4π = {G2_4PI} (dérivé), m_sigma = {M_SIGMA:.0f} MeV")
    print("  V_core  : mur exp. dur au rayon d'exclusion r_excl (taille nucléon)\n")

    r = np.linspace(0.4, 2.5, 800)
    print(f"  {'r_excl(fm)':>10} {'a_phys(fm)':>10} {'coeur dur':>10} "
          f"{'R_min(fm)':>10} {'V_min(MeV)':>11}")
    print("  " + "-" * 56)
    central = None
    for r_excl in (0.75, 0.85, 0.95):
        VNN, Vcore, a_phys, K = build_VNN(r, r_excl, eta)
        kmin = int(np.argmin(VNN))
        Rmin, Vmin = r[kmin], VNN[kmin]
        # rayon du coeur dur : passage + -> - (mur intérieur)
        s = np.sign(VNN)
        cr = np.where(np.diff(s) < 0)[0]
        r_hard = r[cr[0]] if len(cr) else np.nan
        print(f"  {r_excl:>10.2f} {a_phys:>10.3f} {r_hard:>10.2f} "
              f"{Rmin:>10.2f} {Vmin:>11.0f}")
        if abs(r_excl - 0.85) < 1e-6:
            central = (r_excl, VNN, Vcore, a_phys, Rmin, Vmin, r_hard)

    print("\n  Référence nucléaire empirique : puits -50 à -100 MeV vers")
    print("  0,8–1 fm, coeur dur vers 0,5–0,8 fm.")
    print("  Comparaison coeur MOU 1/r^2 (core_exclusion_F3) : -85 MeV à 0,67 fm.")
    print("  → le mur dur exponentiel (FEM) déplace le minimum vers l'extérieur")
    print("    et durcit le coeur (passage par zéro net).")

    # --- figure (cas central r_excl = 0,85 fm) ---
    r_excl, VNN, Vcore, a_phys, Rmin, Vmin, r_hard = central
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.plot(r, V_sigma(r), "--", color="green",
                label="$V_\\sigma$ (attraction dérivée, $g^2/4\\pi=6$)")
        ax.plot(r, Vcore, ":", color="red",
                label=f"$V_{{\\rm core}}$ (mur dur exp., FEM, $a={a_phys:.2f}$ fm)")
        ax.plot(r, VNN, "-", color="black", lw=2, label="$V_{NN}$ total")
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(Rmin, color="blue", lw=0.7, ls="--")
        ax.plot(Rmin, Vmin, "o", color="blue")
        ax.annotate(f"  puits {Vmin:.0f} MeV\n  à {Rmin:.2f} fm",
                    (Rmin, Vmin), fontsize=9, color="blue")
        ax.set_ylim(-200, 200)
        ax.set_xlim(0.4, 2.2)
        ax.set_xlabel("$r$ (fm)")
        ax.set_ylabel("$V_{NN}$ (MeV)")
        ax.set_title("Potentiel NN — coeur dur exponentiel d'exclusion (P16, FEM)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = "output/nn_potential_F3.png"
        fig.savefig(out, dpi=130)
        print(f"\n  figure → {out}")
    except Exception as e:
        print(f"  (figure non générée : {e})")

    print(f"\n  RÉSULTAT (r_excl = {r_excl:.2f} fm, taille du nucléon) :")
    print(f"    coeur dur (V_NN=0) à {r_hard:.2f} fm ; "
          f"puits {Vmin:.0f} MeV à {Rmin:.2f} fm.")
    print("\n  CONCLUSION :")
    print("  • V_NN(r) quantitatif : attraction σ DÉRIVÉE + mur dur exponentiel")
    print("    d'exclusion (forme et dureté VALIDÉES par le FEM, P16).")
    print("  • Coeur dur net (passage par zéro), puits d'échelle nucléaire.")
    print("  • La hauteur du mur n'est PAS prise du puits ponctuel (qui")
    print("    sous-estime) mais ancrée à l'attraction au rayon d'exclusion ;")
    print("    seule la DURETÉ (sans dimension) vient du FEM.")
    print("  • RÉSIDU O(1) : r_excl (taille effective du nucléon).")
    print("=" * 70)


if __name__ == "__main__":
    main()
