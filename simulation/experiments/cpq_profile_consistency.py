r"""
Profil et cohérence dimensionnelle du champ gravitationnel quantique (CGQ).

Point ouvert : avec κ_g = (96π/7) α¹⁸ λ̄_C (une LONGUEUR) et |Ψ|², comment
Ω²−1 = κ_g|Ψ|² est-il sans dimension ET reproduit-il h₀ ≈ 3×10⁻³⁹, en accord
avec G (chap. Gravitation par exclusion) et la suppression Vassiliev
h₀ = (6/7) α¹⁸ ?

Résultats :
  1. |Ψ|² est une densité LINÉIQUE (1/longueur) ; κ_g (longueur) -> κ_g|Ψ|²
     sans dimension. Toute la petitesse (α¹⁸) est dans κ_g.
  2. Les deux déterminations de h₀ (profil P5 vs Vassiliev) coïncident SSI
     |Ψ(a)|² = 1/(16π λ̄_C) ≈ 0,095/fm. L'α¹⁸ s'ANNULE : |Ψ|² est géométrique
     O(0,1/fm), la petitesse est entièrement dans κ_g.
  3. Le champ CGQ à la surface g_q(r_p) = c²h₀/(2r_p) = G m_p/r_p² (cohérence
     interne : le CGQ d'un proton = son auto-champ newtonien).
  4. Profil radial Ω²(r), g_q(r) : bosse intérieure + queue 1/r (figure).

Usage :  python3 -m simulation.experiments.cpq_profile_consistency
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- constantes physiques ---
ALPHA = 1.0 / 137.036          # α_EM
HBARC = 197.3269631            # MeV·fm
M_P = 938.272                  # MeV (proton)
G_SI = 6.67430e-11             # m³/(kg s²) (CODATA, pour comparaison)
HBAR = 1.054571817e-34; C = 2.99792458e8; M_P_KG = 1.67262192e-27
LBAR = HBARC / M_P             # λ̄_C = ħ/(m_p c) en fm
A_EXCL = 4.0 * LBAR            # rayon d'exclusion a = r_p = 4 λ̄_C (V4)

# --- invariants du nucléon (V4) ---
CN_NUCLEON = 17                # c_n(3_1)+2 c_n(7_2) = 3+7+7
CN_EFF = CN_NUCLEON + 1        # = 18 (phase de Gauss, Lk impair)
C_PREF = 6.0 / 7.0            # préfacteur C (= 4·(3/7)/2)
# longueur dimensionnelle de l'entrelacs (abraham REFERENCE_N50K)
L_DIM = 28.83 + 2 * 30.96      # L(3_1)+2 L(7_2)


def main():
    print("=" * 74)
    print("  CGQ — cohérence dimensionnelle et profil de κ_g|Ψ|²")
    print("=" * 74)

    a18 = ALPHA ** CN_EFF
    kappa_g_fm = (96 * np.pi / 7) * a18 * LBAR        # fm
    h0 = C_PREF * a18                                  # = (6/7) α¹⁸

    print(f"\n  λ̄_C = ħc/m_p = {LBAR:.5f} fm ;  a = r_p = 4λ̄_C = {A_EXCL:.4f} fm")
    print(f"  α¹⁸ = {a18:.3e}   (la petitesse de la gravité)")
    print(f"  κ_g = (96π/7) α¹⁸ λ̄_C = {kappa_g_fm:.3e} fm = {kappa_g_fm*1e-15:.3e} m")
    print(f"        (V4 : 3,12×10⁻⁵³ m)")
    print(f"  h₀  = (6/7) α¹⁸ = {h0:.3e}   (Vassiliev/exclusion, V4)")

    # 1. round-trip G
    G_pred = (12/7) * a18 * (HBAR*C) / M_P_KG**2
    print(f"\n  [1] Round-trip : G = (12/7) α¹⁸ ħc/m_p² = {G_pred:.4e} "
          f"(CODATA {G_SI:.4e}, écart {(G_pred/G_SI-1)*100:+.1f}%)")

    # 2. densité linéique à la surface : annulation de α¹⁸
    psi2_a = h0 / kappa_g_fm                           # 1/fm
    psi2_a_exact = 1.0 / (16*np.pi*LBAR)               # forme fermée
    print(f"\n  [2] Cohérence des deux h₀ : |Ψ(a)|² = h₀/κ_g = {psi2_a:.4f} /fm")
    print(f"      forme fermée 1/(16π λ̄_C) = {psi2_a_exact:.4f} /fm  "
          f"(l'α¹⁸ S'ANNULE)")
    print(f"      adimensionné |Ψ(a)|² λ̄_C = 1/(16π) = {psi2_a*LBAR:.5f}")
    print(f"      -> |Ψ|² est une densité LINÉIQUE géométrique O(0,1/fm) ;")
    print(f"         TOUTE la petitesse (α¹⁸ ~ 10⁻³⁹) est portée par κ_g.")

    # longueur effective vs longueur géométrique du nœud
    L_eff = 1.0 / psi2_a                               # fm (si surface ~ moyenne)
    print(f"\n      Longueur effective L_eff = 1/|Ψ(a)|² = 16π λ̄_C = {L_eff:.2f} fm")
    print(f"      Nœud nucléon (abraham) : L_dim = {L_DIM:.1f} (unités catalogue)")
    print(f"      -> échelle {L_eff/L_DIM:.3f} fm/unité ; arc enroulé {L_eff:.1f} fm")
    print(f"         dans une sphère de Ø {2*A_EXCL:.2f} fm = {L_eff/(2*A_EXCL):.1f}× le")
    print(f"         diamètre -> plausible pour un entrelacs à 17 croisements.")

    # 3. champ CGQ à la surface = auto-champ newtonien
    a_m = A_EXCL * 1e-15
    gq_surf = C**2 * h0 / (2*a_m)                      # m/s²
    g_newton = G_SI * M_P_KG / a_m**2
    print(f"\n  [3] Champ CGQ à la surface : g_q(r_p) = c²h₀/(2r_p) = {gq_surf:.3e} m/s²")
    print(f"      auto-champ newtonien G m_p/r_p² = {g_newton:.3e} m/s²  "
          f"(écart {(gq_surf/g_newton-1)*100:+.1f}%)")
    print(f"      -> le CGQ d'un proton isolé EST son champ de gravité propre.")

    # 4. profil radial (modèle sphère uniforme : intérieur lisse + queue 1/r)
    r = np.linspace(0.01, 5*A_EXCL, 800)              # fm
    h = np.where(r >= A_EXCL,
                 h0 * A_EXCL / r,                       # queue Laplace 1/r
                 h0 * (3 - (r/A_EXCL)**2) / 2)          # intérieur uniforme
    r_m = r * 1e-15                                     # mètres (pour c² en m/s²)
    gq = np.where(r >= A_EXCL,
                  C**2 * h0 * a_m / (2*r_m**2),          # ~1/r² (m/s²)
                  C**2 * h0 * r_m / (2*a_m**2))          # ~r (->0 au centre)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(r/A_EXCL, h/h0, lw=2, color="#1f4e8c")
    ax1.axvline(1.0, ls="--", color="0.5"); ax1.set_yscale("linear")
    ax1.set_xlabel(r"$r / r_p$"); ax1.set_ylabel(r"$h(r)/h_0 = (\Omega^2-1)/h_0$")
    ax1.set_title(r"Profil métrique du CGQ ($h_0=\frac{6}{7}\alpha^{18}\approx3\times10^{-39}$)")
    ax1.annotate("bosse intérieure", (0.1, 1.35), color="#1f4e8c")
    ax1.annotate(r"queue $1/r \to$ gravité", (1.6, 0.45), color="#1f4e8c")
    ax2.plot(r/A_EXCL, gq, lw=2, color="#8c1f1f")
    ax2.axvline(1.0, ls="--", color="0.5")
    ax2.set_xlabel(r"$r / r_p$"); ax2.set_ylabel(r"$g_q(r)$  [m/s$^2$]")
    ax2.set_title(r"Champ CGQ $g_q=-\frac{c^2}{2}\nabla h$ (centripète)")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "..", "output", "cpq_profile.png")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n  [4] Figure -> {out}")

    print("\n  RÉSOLUTION DU POINT OUVERT :")
    print("  • Pas de structure « puits profond + queue » dans κ_g|Ψ|² : c'est un")
    print("    champ à UNE échelle, ~10⁻³⁹, dont la petitesse vient de α¹⁸ DANS κ_g.")
    print("  • |Ψ|² est une densité linéique géométrique (0,095/fm) ; α¹⁸ s'annule")
    print("    dans |Ψ(a)|²=h₀/κ_g -> les deux dérivations de h₀ sont cohérentes.")
    print("  • Le puits profond ~mc² (confinement) est V_mag (P4), champ DISTINCT.")
    print("=" * 74)


if __name__ == "__main__":
    main()
