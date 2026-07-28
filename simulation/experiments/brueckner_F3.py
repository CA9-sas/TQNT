"""
Chantier Brueckner — corrélations à deux corps et réduction de a_V.

Le champ moyen (bethe_weizsacker_F3) sur-lie : a_V ≈ 24 MeV (empirique 15,8).
La sur-liaison est universelle (pas propre à la TQNT) : le champ moyen ignore
les CORRÉLATIONS à deux corps qui soignent le cœur dur. Un G-matrix de
Brueckner complet est un code de recherche ; on implémente ici la méthode
variationnelle à plus bas ordre (LOCV / Jastrow, Pandharipande-Bethe), qui en
capture l'essentiel et est tractable.

PRINCIPE. On corrèle la paire par une fonction de Jastrow f(r) : f≈0 dans le
cœur dur (l'onde s'évite), f→1 au-delà de la distance de cicatrisation d.
L'énergie par nucléon :

  E/A = T_F + (rho/2) ∫ [ (ħ²/m) f'(r)² + f(r)² V_NN(r) ] 4pi r² dr

  - f² V_NN : potentiel CORRÉLÉ (le cœur dur, multiplié par f²≈0, ne diverge
    plus -> le champ moyen, lui, divergeait) ;
  - (ħ²/m) f'² : ÉNERGIE CINÉTIQUE DE BLESSURE (le coût de courber l'onde pour
    éviter le cœur). À haute densité, d ~ espacement inter-particules diminue,
    f' augmente -> la blessure monte -> SATURATION et réduction de la liaison.

On minimise E/A sur la forme de f (cœur r_c, cicatrisation d) à chaque densité,
puis sur la densité -> rho_0 et a_V. f=1 partout (sans corrélation) redonnerait
le champ moyen (divergent sur le cœur). On compare a_V corrélé au champ moyen.

Usage :  python3 -m simulation.experiments.brueckner_F3
"""
from __future__ import annotations

import numpy as np

HBARC = 197.327
M_N = 939.0
M_SIGMA = 500.0
G2_4PI = 6.0
ETA = 0.122
R_EXCL = 0.79                      # rayon d'exclusion DÉRIVÉ (2ħc/m_sigma)

HBAR2_M = HBARC**2 / M_N           # ħ²/m (mouvement relatif), MeV·fm²
HBAR2_2M = HBARC**2 / (2 * M_N)    # ħ²/2m, MeV·fm²


def V_sigma(r):
    return -G2_4PI * HBARC * np.exp(-(M_SIGMA / HBARC) * r) / r


def V_NN(r, r_wall=R_EXCL):
    a = ETA * r_wall
    return V_sigma(r) + abs(V_sigma(r_wall)) * np.exp(-(r - r_wall) / a)


def jastrow(r, r_c, d):
    """f(r) : 0 dans le cœur (r<r_c), montée sin² jusqu'à 1 à r=d, puis 1."""
    f = np.ones_like(r)
    core = r < r_c
    heal = (r >= r_c) & (r < d)
    f[core] = 0.0
    x = (r[heal] - r_c) / (d - r_c)
    f[heal] = np.sin(0.5 * np.pi * x) ** 2
    return f


def jastrow_deriv(r, r_c, d):
    fp = np.zeros_like(r)
    heal = (r >= r_c) & (r < d)
    x = (r[heal] - r_c) / (d - r_c)
    fp[heal] = (np.pi / (d - r_c)) * np.sin(0.5 * np.pi * x) * np.cos(0.5 * np.pi * x)
    return fp


_R = np.linspace(1e-3, 8.0, 6000)


def pair_integral(r_c, d, VNN_grid):
    """J = ∫ [ (ħ²/m) f'² + f² V_NN ] 4pi r² dr  (MeV·fm³)."""
    f = jastrow(_R, r_c, d)
    fp = jastrow_deriv(_R, r_c, d)
    integ = (HBAR2_M * fp**2 + f**2 * VNN_grid) * 4 * np.pi * _R**2
    return np.trapezoid(integ, _R)


def energy_per_A(rho, VNN_grid, r_wall):
    """E/A(rho) corrélé. CŒUR DUR STRICT : f=0 pour r<r_wall (l'intérieur
    singulier du sigma n'est JAMAIS sondé -> pas d'effondrement). Cicatrisation
    d = r_0 (contrainte LOCV) : à haute densité d->r_wall, la blessure monte
    -> SATURATION."""
    kF = (1.5 * np.pi**2 * rho) ** (1.0 / 3.0)
    T_F = 0.6 * HBAR2_2M * kF**2
    d = (3.0 / (4 * np.pi * rho)) ** (1.0 / 3.0)      # = r_0, contrainte LOCV
    if d <= r_wall + 0.05:
        return T_F + 1e3                              # exclusion totale (dense)
    return T_F + 0.5 * rho * pair_integral(r_wall, d, VNN_grid)


def saturation_for_core(r_wall):
    """Sature E/A(rho) (LOCV) pour un cœur dur de rayon r_wall ; -> rho0, a_V."""
    VNN_grid = V_NN(_R, r_wall)
    rhos = np.linspace(0.04, 0.40, 28)
    EAs = np.array([energy_per_A(rho, VNN_grid, r_wall) for rho in rhos])
    k = int(np.argmin(EAs))
    rho0 = rhos[k]
    r0 = (3.0 / (4 * np.pi * rho0)) ** (1.0 / 3.0)
    interior = 0 < k < len(rhos) - 1                  # vrai minimum (saturation) ?
    return rho0, r0, -EAs[k], rhos, EAs, interior


def main():
    print("=" * 70)
    print("  CHANTIER BRUECKNER — corrélations (LOCV) et réduction de a_V")
    print("=" * 70)
    print(f"\n  V_NN : sigma dérivé (g²/4π={G2_4PI}) + cœur dur exp. (P16).")
    print(f"  ħ²/m = {HBAR2_M:.1f} MeV·fm². LOCV : f soigne le cœur, blessure")
    print("  cinétique (ħ²/m)f'² -> réduit la sur-liaison du champ moyen.")

    a_tube = np.sqrt(2) * HBARC / M_SIGMA      # rayon de tube = vrai cœur dur
    print(f"\n  SENSIBILITÉ au rayon de cœur dur r_wall (le LOCV y est très")
    print(f"  sensible). Le cœur PHYSIQUE en coordonnée relative est le rayon")
    print(f"  de TUBE a = √2ħc/m_σ = {a_tube:.2f} fm (cœurs de condensat qui se")
    print(f"  touchent), PAS l'onset d'exclusion r_excl = {R_EXCL} fm (enveloppes).")
    print(f"\n  {'r_wall (fm)':>11} {'rho_0':>8} {'E/A_min':>9} {'saturation ?':>14}")
    print("  " + "-" * 50)
    results = {}
    for r_wall, note in ((0.45, "compact"), (a_tube, "tube √2ħc/m_σ"),
                         (0.65, "intermédiaire"), (R_EXCL, "onset")):
        rho0, r0, a_V, rhos, EAs, interior = saturation_for_core(r_wall)
        results[round(r_wall, 2)] = (rhos, EAs, rho0, a_V, interior)
        diag = "interne ✓" if interior else ("effondre." if rho0 > 0.3
                                             else "non lié")
        print(f"  {r_wall:>11.2f} {rho0:>8.3f} {-a_V:>9.1f}   {diag:>12} "
              f"[{note}]")
    print("  empirique :       0,16     -15,8        interne")

    print("\n  LECTURE (HONNÊTE) :")
    print("  • Aucun cœur ne donne de SATURATION INTERNE propre : cœur compact")
    print("    -> effondrement (haute densité) ; cœur physique (≥0,56) -> non lié")
    print("    (le minimum fuit vers la basse densité). Le LOCV cru ÉCHOUE ici.")
    print("  • Cause : (i) la queue sigma -1/r est trop attractive (g²/4π=6) ;")
    print("    (ii) il manque l'opérateur de Pauli (Q) qui, dans un VRAI G-matrix,")
    print("    affaiblit l'attraction à haute densité de façon contrôlée, et le")
    print("    spectre auto-cohérent à une particule. Le profil de Jastrow fixe")
    print("    et l'absence de Pauli rendent le résultat pathologique.")
    print("  • Le champ moyen (volume exclu, a_V≈24, ρ_0≈0,22) reste donc le")
    print("    MEILLEUR estimateur SIMPLE ; le 16 MeV exige le G-matrix complet.")

    # --- figure : E/A(rho) pour le cœur physique (tube) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        colors = {"compact": "purple", "tube": "navy", "grand": "darkred"}
        for r_wall, lab, c in ((0.45, "$r_{\\rm wall}=0{,}45$ (effondre)", "purple"),
                               (round(a_tube, 2),
                                f"tube $a={a_tube:.2f}$ (non lié)", "navy"),
                               (round(R_EXCL, 2),
                                f"onset $r_{{\\rm excl}}={R_EXCL}$", "darkred")):
            rhos, EAs, rho0, a_V, interior = results[round(r_wall, 2)]
            ax.plot(rhos, EAs, "-", color=c, lw=2, label=lab)
        ax.axhline(0, color="gray", lw=0.6)
        ax.axhline(-15.8, color="green", ls=":", lw=1.2, label="$-a_V$ empirique")
        ax.axvline(0.16, color="green", ls="--", lw=1)
        ax.set_ylim(-30, 40)
        ax.set_xlabel("densité $\\rho$ (fm$^{-3}$)")
        ax.set_ylabel("$E/A$ (MeV)")
        ax.set_title("LOCV cru : pas de saturation propre (cœur-dépendant)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = "output/brueckner_F3.png"
        fig.savefig(out, dpi=130)
        print(f"\n  figure → {out}")
    except Exception as e:
        print(f"  (figure non générée : {e})")

    print("\n  CONCLUSION (honnête) :")
    print("  • Le LOCV CRU (profil de Jastrow fixe, SANS opérateur de Pauli ni")
    print("    spectre auto-cohérent) N'ARRIVE PAS à reproduire la saturation :")
    print("    effondrement (cœur compact) ou non-liaison (cœur physique).")
    print("  • L'énergie cinétique de blessure EST grande (le mécanisme de")
    print("    réduction existe), mais sans le Pauli Q elle sur-corrige.")
    print("  • VERDICT : le 16 MeV exige le G-matrix de Brueckner COMPLET")
    print("    (ladders + Pauli Q + spectre auto-cohérent + ondes partielles) —")
    print("    un code de recherche. Le raccourci variationnel est insuffisant.")
    print("  • Estimateur simple le plus fiable : champ moyen (a_V≈24, ρ_0≈0,22).")
    print("=" * 70)


if __name__ == "__main__":
    main()
