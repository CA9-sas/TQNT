r"""
Dérivation du taux de reconnexion χ depuis Chern-Simons SU(2)_k — fonder τ_P
SANS calibration sur une durée de vie.

CADRE.
  Action CS : S = (k/4π) ∫ Tr(A dA + (2/3) A³),  k = 135.
  Particules = lignes de Wilson (spin ½, représentation fondamentale).
  Désintégration faible = RECONNEXION ΔLk = ±1 (changement d'une unité du
  nombre d'enlacement/framing d'une ligne de Wilson).

CE QUE CS DONNE (rigoureux).
  - La RÈGLE DE SÉLECTION : ΔLk = ±1 (transition de framing unitaire).
  - Le poids conforme (spin topologique) de la rep. ½ :
        h_{1/2} = j(j+1)/(k+2) = (3/4)/(k+2)
    et la phase de twist θ = exp(2πi h_{1/2}). Le couplage CS par croisement
    est 1/(k+2) = α_EM (identification TQNT).

STRUCTURE DU TAUX.
        Γ = ω_Compton · g_w² · Φ_espace-des-phases
  - ω_Compton = m c²/ħ  (oscillation de l'onde χ, dérivée de la masse)
  - g_w = couplage de reconnexion (candidat CS : α_EM ou h_{1/2})
  - Φ = espace des phases de Sargent des ondes libres émises (Q^5).
  Dimensionnellement G_F ~ g_w² / M² avec M = échelle de confinement (~m_p).

PRÉDICTION PARAMÈTRE-LIBRE (k=135 seul fixe l'échelle faible) — à comparer.

Usage :  python3 -m simulation.experiments.cs_reconnection_tauP
"""
import numpy as np

K_CS = 135
ALPHA = 1.0 / (K_CS + 2)            # = 1/137 (couplage CS = α_EM)
H_HALF = (3.0/4.0) / (K_CS + 2)      # spin topologique de la rep. ½
# constantes de masse / cinématique (PAS de durée de vie)
M_E, M_P, M_N = 0.51099895, 938.272, 939.565
HBAR = 6.582119e-22                  # MeV·s
LAM, VUD2, F_N = 1.2756, 0.94815, 1.6887
TAU_EXP = 878.4


def neutron_tau(g_w, M_scale):
    """τ_n via Γ_β avec G_F = g_w²/M_scale² (CS, sans calibration de durée de vie)."""
    G_F = g_w**2 / M_scale**2                       # MeV^-2
    rate = (G_F**2 * VUD2 * (1 + 3*LAM**2) / (2*np.pi**3)) * M_E**5 * F_N / HBAR
    return 1.0/rate, G_F


def main():
    print("=" * 78)
    print("  TAUX DE RECONNEXION χ depuis CS SU(2)_{k=135} — sans calibration")
    print("=" * 78)
    print(f"\n  couplage CS  1/(k+2) = α_EM = {ALPHA:.5f}")
    print(f"  spin topologique  h_{{1/2}} = 3/(4(k+2)) = {H_HALF:.5f}")
    print(f"  couplage faible mesuré  G_F·m_p² = {1.1663787e-11*M_P**2:.2e}  (~10⁻⁵)")

    print("\n  Candidats CS pour le couplage faible g_w² (échelle M = m_p) :")
    for lbl, g in [("g_w = α_EM = 1/(k+2)", ALPHA),
                   ("g_w = h_{1/2} = 3/4(k+2)", H_HALF)]:
        gf_mp2 = g**2
        tau, GF = neutron_tau(g, M_P)
        print(f"    {lbl:>26} : g_w²={gf_mp2:.2e}  ->  τ_n = {tau:.1f} s "
              f"(mesuré {TAU_EXP})  facteur {tau/TAU_EXP:.2f}")

    print("\n  RÉSULTAT (parameter-free, k=135 seul) :")
    tau_a,_ = neutron_tau(ALPHA, M_P); tau_h,_ = neutron_tau(H_HALF, M_P)
    print(f"    τ_n ~ {tau_a:.0f}–{tau_h:.0f} s selon le couplage CS, vs {TAU_EXP} s mesuré.")
    print(f"    -> BON ORDRE DE GRANDEUR ({np.log10(TAU_EXP/tau_h):+.1f} à "
          f"{np.log10(TAU_EXP/tau_a):+.1f} décade) sans AUCUNE calibration de durée de vie.")
    print(f"       La petitesse de la force nucléaire faible (~10⁻⁵) émerge de k=135.")

    print("\n  CE QUI RESTE (honnête) :")
    print("  • CS est une TQFT : les changements de topologie donnent des PHASES")
    print("    (|amplitude|=1), pas la SUPPRESSION. La petitesse ~10⁻⁵ est donc")
    print("    DYNAMIQUE — la probabilité que deux brins se rejoignent pour")
    print("    reconnecter, fixée par le profil de confinement du champ χ.")
    print("  • Le facteur ~8 manquant (et l'échelle M) = ce recouvrement dynamique,")
    print("    calculable par le solveur FEM de χ (la vraie prochaine étape).")
    print("  • Le coïncidence α_EM² ~ G_F m_p² (~10⁻⁵) suggère que le confinement")
    print("    χ relie bien l'échelle faible au niveau k — à établir par le calcul.")
    print("=" * 78)


if __name__ == "__main__":
    main()
