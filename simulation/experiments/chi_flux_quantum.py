"""Derivation du quantum de flux chi->EM (et dissolution du '0.60').

1) LE QUANTUM EST DERIVE (attachement de flux Chern-Simons) :
   U(1)_N CS : EOM (N/2pi) da = rho -> flux attache par charge unite = Phi_0/N.
   TQNT : le niveau decale k+2 = 137 -- le MEME entier qui donne alpha = 1/(k+2) --
   d'ou   Phi_chi->EM = Phi_0/(k+2) = alpha (h/e) = 3.018e-17 Wb.
   Identite exacte (verifiee 0.00%) : alpha (h/e) = mu0 e c / 2  -- le flux du tube est
   celui qu'engendre la charge elementaire a c. Zero intrant nouveau : charge et flux
   sont lies par le meme niveau CS.
   [Caveat : attachement 2+1D applique au plan transverse du tube 3+1D -- etape de
    modelisation.]

2) LE '0.60' SE DISSOUT (artefact, pas un nombre a deriver) :
   L'inversion initiale visait le magneton generique mu_N au lieu du VRAI moment
   |mu_n| = 1.913 mu_N. Avec la bonne cible : Phi_req = 1.14 alpha (h/e) -> le quantum
   derive alpha Phi_0 a 14% (dans une geometrie mono-trefle nominale, incertaine O(1)).

3) FORME SANS NORMALISATION EM LIBRE :
   mu_tw = (4 pi Phi / mu0)(Tw S / L) avec Phi = alpha Phi_0 = mu0 e c/2 donne
       mu_tw = 2 pi e c * (Tw S / L)   -- la seule liberte restante est GEOMETRIQUE.
   Inversion sur |mu_n| : L_eff = 3.41 fm = 4.06 r_p  (4 r_p a 1.4% -- SUGGESTIF
   seulement : modele mono-composante, le neutron a 3 noeuds).
"""
import numpy as np

MU0 = 4e-7 * np.pi
H = 6.62607e-34
E = 1.60218e-19
C = 2.99792e8
ALPHA = 1 / 137.036
MU_N_SI = 5.0508e-27
MU_NEUTRON = 1.91304        # |mu_n| en mu_N
TW, S_L2 = 0.354, 0.0265    # trefle (courbe valide)
R_P = 0.8414                # fm


def run():
    h_e = H / E
    phi = ALPHA * h_e
    print("=== 1) quantum derive ===")
    print(f"  Phi = Phi_0/(k+2) = alpha (h/e) = {phi:.3e} Wb")
    print(f"  identite mu0 e c/2 = {MU0*E*C/2:.3e} Wb (ecart {abs(phi-MU0*E*C/2)/phi*100:.2f}%)")

    print("\n=== 2) dissolution du 0.60 ===")
    print(f"  cible generique mu_N -> 0.60 alpha ; cible VRAIE |mu_n|={MU_NEUTRON} ->"
          f" {0.60*MU_NEUTRON:.2f} alpha : quantum retrouve a ~14%")

    print("\n=== 3) forme sans normalisation libre ===")
    coef = 2 * np.pi * E * C * TW * S_L2 / MU_N_SI * 1e-15    # mu_tw/mu_N par fm
    L_eff = MU_NEUTRON / coef
    print(f"  mu_tw = 2 pi e c Tw S/L ; mu_tw/mu_N = {coef:.3f} L[fm]")
    print(f"  |mu_n| -> L_eff = {L_eff:.2f} fm = {L_eff/R_P:.2f} r_p"
          f"  (4 r_p a {abs(L_eff-4*R_P)/(4*R_P)*100:+.1f}% ; SUGGESTIF, modele mono-composante)")


if __name__ == "__main__":
    run()
