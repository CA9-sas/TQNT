r"""
Profil de la métrique émergente Ω²(r) = 1 + κ_g|Ψ|² autour d'un neutron,
LIBRE vs LIÉ, et le gradient P22 ξ = |∇Ω²|λ_C/Ω².

Question : le gradient de la métrique GRAVITATIONNELLE gouverne-t-il la selle à
l'échelle du nucléon ? On compare ξ_libre (neutron seul dans le vide χ) à
ξ_lié (entouré de nucléons qui « remplissent » le voisinage).

Modèle : Ω²−1 ≈ 2Φ/c² (champ faible). Pour le neutron, Φ_surface = G m_n/r_n.
λ_C = ħ/(m_n c). ξ ~ |∇Ω²| λ_C ~ (2G m_n/r_n² c²)(ħ/m_n c) = 2Għ/(r_n² c³).

Usage :  python3 -m simulation.experiments.omega_profile_neutron
"""
import numpy as np

G=6.674e-11; HBAR=1.0546e-34; C=2.998e8
m_n=1.675e-27          # kg
r_n=0.84e-15           # m (rayon de charge)
lamC=HBAR/(m_n*C)      # longueur de Compton (m)
R_nuc=5e-15            # m (rayon d'un noyau moyen, ~A=40)

def main():
    print("="*72)
    print("  PROFIL Ω²(r) = 1 + κ_g|Ψ|²  —  neutron LIBRE vs LIÉ (gradient P22)")
    print("="*72)
    # amplitude gravitationnelle de la métrique à la surface du neutron
    dOmega2 = 2*G*m_n/(r_n*C**2)
    print(f"\n  Ω²−1 à la surface du neutron (gravitationnel) = 2Gm_n/(r_n c²)")
    print(f"     = {dOmega2:.2e}   -> métrique quasi plate (gravité ultra-faible)")
    print(f"  longueur de Compton λ_C = {lamC*1e15:.3f} fm")

    # gradient P22 : ξ = |∇Ω²| λ_C / Ω²
    grad = 2*G*m_n/(r_n**2*C**2)          # |∇Ω²| à la surface (1/m)
    xi_free = grad*lamC/1.0
    print(f"\n  NEUTRON LIBRE (vide χ autour) :")
    print(f"     |∇Ω²| ~ {grad:.2e} /m ;  ξ_libre = |∇Ω²|λ_C ~ {xi_free:.2e}")
    # neutron LIÉ : les voisins remplissent le voisinage -> gradient lissé
    # le contraste tombe de (r_n -> R_nuc) : suppression ~ (r_n/R_nuc)
    supp = r_n/R_nuc
    xi_bound = xi_free*supp
    print(f"\n  NEUTRON LIÉ (entouré de nucléons, voisinage rempli) :")
    print(f"     gradient lissé sur ~R_noyau={R_nuc*1e15:.0f} fm -> suppression ~{supp:.2f}")
    print(f"     ξ_lié ~ {xi_bound:.2e}")
    print(f"\n  -> directionnellement ξ_lié < ξ_libre (ton intuition : le dense lisse).")
    print(f"     MAIS les deux sont ~10^-39, soit {1/xi_free:.0e}× SOUS le seuil ξ~1.")

    # seuil de production : ξ~1 demande un gradient ~ 1/λ_C
    grad_crit = 1.0/lamC
    print(f"\n  Seuil de production (ξ~1) : |∇Ω²| ~ 1/λ_C = {grad_crit:.1e} /m")
    print(f"     soit {grad_crit/grad:.0e}× le gradient d'un neutron -> JAMAIS au")
    print(f"     niveau nucléon ; réservé aux horizons / champs critiques (cosmo).")

    print("\n  CONCLUSION HONNÊTE :")
    print("  • La métrique GRAVITATIONNELLE Ω² est négligeable au nucléon (ξ~10^-39).")
    print("    Elle confirme la DIRECTION (ξ_lié<ξ_libre) mais ne gouverne PAS la β.")
    print("  • Le vrai champ opérant à l'échelle nucléon est la densité du CHAMP χ")
    print("    (le milieu nucléaire = V_NN) : le neutron libre est dans le vide χ")
    print("    (gradient à sa peau), le lié dans un milieu χ dense (voisins) qui")
    print("    comble le gradient -> reconnexion freinée. C'est V_NN, pas Ω²_grav.")
    print("  • P22 (ξ~1) reste cosmologique (horizons, peau baryonique), pas nucléon.")
    print("="*72)

if __name__ == "__main__":
    main()
