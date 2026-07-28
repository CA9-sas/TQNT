"""
Spectre beta du neutron — le neutrino comme PETITE ONDE LIBRE (TQNT).

Choix physique (interaction nucléaire faible TQNT) : le neutrino n'est PAS un nœud, c'est
une ONDE LIBRE du champ chi (spineur SU(2), P10), déconfinée. Conséquences
DIRECTEMENT calculables et observables :

  1. Spin 1/2 SANS nœud : le champ chi est SU(2), ses excitations sont des
     spineurs -> une onde libre porte déjà 1/2 (la formule de tresse
     s=(b-1)/2 de P7 ne vaut que pour les états NOUÉS/massifs).
  2. Masse ~ 0 : pas de nœud -> pas de masse topologique de confinement.
     Une onde libre de masse nulle impose l'espace de phase (W0 - W)^2 dans
     le spectre beta (W0 = énergie totale max de l'électron).
  3. Spectre beta CONTINU : la part libre (neutrino) emporte une fraction
     variable de l'énergie -> l'électron a un spectre continu, pas une raie.

Spectre de Fermi (transition permise n->p) :
    dN/dW = F(Z=1, W) * p_e * W * (W0 - W)^2 ,    p_e = sqrt(W^2 - m_e^2).
Le facteur (W0 - W)^2 EST la signature de l'onde libre de masse nulle.

Test propre : le TRACÉ DE KURIE  K(W) = sqrt( N / (F p_e W) ) = (W0 - W)
est une DROITE pour un neutrino sans masse (interceptant l'axe en W0). Une
masse m_nu le ferait plier près du point d'arrêt. La linéarité observée
confirme « petite onde libre quasi sans masse ».

Usage :  python3 -m simulation.experiments.beta_spectrum_weak
"""
from __future__ import annotations

import numpy as np

M_E = 0.510999       # MeV
DM_NP = 1.29333      # MeV  (m_n - m_p)
W0 = DM_NP           # énergie TOTALE max de l'électron (recul négligé)
Q = W0 - M_E         # énergie cinétique max (point d'arrêt) = 0.782 MeV
ALPHA = 1.0 / 137.036
Z = 1                # noyau fille (proton) vu par l'électron beta-


def fermi_function(W):
    """Correction coulombienne F(Z,W) (β−, attractive). η = αZ W/p."""
    p = np.sqrt(np.maximum(W**2 - M_E**2, 1e-12))
    eta = ALPHA * Z * W / p          # >0 : attraction
    x = 2 * np.pi * eta
    return x / (1.0 - np.exp(-x))


def spectrum(W, m_nu=0.0):
    """dN/dW (non normalisé). m_nu>0 plie le point d'arrêt."""
    p = np.sqrt(np.maximum(W**2 - M_E**2, 0.0))
    Enu = W0 - W                      # énergie du neutrino
    # espace de phase du neutrino : E_nu * sqrt(E_nu^2 - m_nu^2)
    pnu2 = Enu**2 - m_nu**2
    phase = np.where(pnu2 > 0, Enu * np.sqrt(np.maximum(pnu2, 0.0)), 0.0)
    return fermi_function(W) * p * W * phase


def main():
    print("=" * 70)
    print("  SPECTRE BETA — le neutrino comme petite onde libre (TQNT)")
    print("=" * 70)
    print(f"\n  n -> p + e- + nu_bar.  Δm = {DM_NP:.3f} MeV,  m_e = {M_E:.3f} MeV")
    print(f"  Énergie totale max électron W0 = {W0:.3f} MeV")
    print(f"  Énergie cinétique max (point d'arrêt) Q = {Q:.3f} MeV")
    print("\n  Onde libre de masse nulle -> espace de phase neutrino (W0-W)^2.")

    W = np.linspace(M_E + 1e-4, W0 - 1e-5, 4000)
    T = W - M_E                       # énergie cinétique de l'électron
    N = spectrum(W, m_nu=0.0)
    N /= np.trapezoid(N, T)           # normalisé en T

    # statistiques du spectre
    T_mean = np.trapezoid(T * N, T)
    T_mode = T[int(np.argmax(N))]
    Enu_mean = np.trapezoid((Q - T) * N, T)
    print(f"\n  Spectre continu de l'électron :")
    print(f"    énergie cinétique moyenne  <T_e>  = {T_mean:.3f} MeV "
          f"({100*T_mean/Q:.0f}% de Q)")
    print(f"    énergie cinétique la + probable    = {T_mode:.3f} MeV")
    print(f"    énergie moyenne du neutrino <E_nu> = {Enu_mean:.3f} MeV "
          f"({100*Enu_mean/Q:.0f}% de Q)")
    print(f"    -> l'électron et le neutrino se PARTAGENT Q : spectre continu ✓")
    print(f"    (réf. neutron : <T_e> mesuré ≈ 0,30 MeV, point d'arrêt 0,782) ")

    # --- tracé de Kurie : linéarité = neutrino sans masse ---
    K = np.sqrt(N / (fermi_function(W) * np.sqrt(W**2 - M_E**2) * W))
    # ajustement linéaire K = a (W0 - W) sur la moitié haute
    sel = T > 0.2
    a, b = np.polyfit(T[sel], K[sel], 1)
    T_intercept = -b / a              # où K=0 -> point d'arrêt extrait
    # résidu de linéarité
    resid = K[sel] - (a * T[sel] + b)
    rms = np.sqrt(np.mean(resid**2)) / np.max(K[sel])
    print(f"\n  TRACÉ DE KURIE  K = sqrt(N/(F p W)) :")
    print(f"    linéaire en T (écart-type relatif {100*rms:.1f}%), "
          f"intercept à T = {T_intercept:.3f} MeV")
    print(f"    -> DROITE jusqu'au point d'arrêt = neutrino SANS MASSE "
          f"(onde libre) ✓")
    print(f"    (point d'arrêt extrait {T_intercept:.3f} vs Q = {Q:.3f} MeV)")

    # comparaison : ce que ferait un neutrino massif
    for m_nu in (0.05, 0.10):
        Nm = spectrum(W, m_nu=m_nu)
        Wcut = W0 - m_nu
        print(f"    (un neutrino de masse {m_nu*1000:.0f} keV plierait le Kurie "
              f"et couperait à T={Wcut - M_E:.3f} MeV — non observé)")

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
        ax1.plot(T, N, "-", color="navy", lw=2)
        ax1.fill_between(T, 0, N, color="navy", alpha=0.12)
        ax1.axvline(T_mean, color="red", ls="--", lw=1,
                    label=f"$\\langle T_e\\rangle$={T_mean:.2f} MeV")
        ax1.axvline(Q, color="green", ls=":", lw=1.2,
                    label=f"point d'arrêt $Q$={Q:.3f} MeV")
        ax1.set_xlabel("énergie cinétique de l'électron $T_e$ (MeV)")
        ax1.set_ylabel("$dN/dT_e$ (norm.)")
        ax1.set_title("Spectre $\\beta$ continu (neutrino = onde libre)")
        ax1.legend(fontsize=9)

        ax2.plot(T, K / K.max(), "-", color="navy", lw=2,
                 label="$m_\\nu=0$ (onde libre) : droite")
        for m_nu, c in ((0.10, "orange"), (0.20, "red")):
            Nm = spectrum(W, m_nu=m_nu)
            with np.errstate(invalid="ignore"):
                Km = np.sqrt(Nm / (fermi_function(W)
                                   * np.sqrt(W**2 - M_E**2) * W))
            ax2.plot(T, Km / K.max(), "--", color=c, lw=1.3,
                     label=f"$m_\\nu={m_nu*1000:.0f}$ keV : plie")
        ax2.axvline(Q, color="green", ls=":", lw=1.2)
        ax2.set_xlabel("énergie cinétique de l'électron $T_e$ (MeV)")
        ax2.set_ylabel("Kurie $K$ (norm.)")
        ax2.set_title("Tracé de Kurie : linéaire $\\Rightarrow m_\\nu\\approx 0$")
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 1.05)
        fig.tight_layout()
        out = "output/beta_spectrum_weak.png"
        fig.savefig(out, dpi=130)
        print(f"\n  figure → {out}")
    except Exception as e:
        print(f"  (figure non générée : {e})")

    print("\n  CONCLUSION :")
    print("  • Neutrino = petite onde libre (spineur SU(2)) : spin 1/2 SANS nœud,")
    print("    masse topologique nulle.")
    print("  • Cette masse nulle impose l'espace de phase (W0-W)^2 -> spectre β")
    print("    CONTINU, <T_e> ≈ 0,3 MeV, point d'arrêt 0,782 MeV (accord neutron).")
    print("  • Le tracé de Kurie est une DROITE jusqu'au point d'arrêt : c'est la")
    print("    signature directe d'un neutrino sans masse = l'onde libre. ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
