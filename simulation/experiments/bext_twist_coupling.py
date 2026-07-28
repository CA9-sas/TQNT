"""Derivation du couplage B_ext <-> Tw (twist du tube de flux).

CHAINE DERIVEE (electrodynamique classique du tube mince) :
  1. Tube uniformement tordu (taux q rad/m) : a l'interieur B_phi/B_z = q r.
  2. Ampere au bord (r=a) : I = 2 pi a B_phi/mu0 = 2 q Phi/mu0.
     => LE TWIST FAIT DU COURANT AXIAL : I = 2 q Phi/mu0, q = 2 pi Tw/L.
  3. Boucle fermee de courant : moment mu_tw = I * S_vec (S_vec = 1/2 INT r x dl).
  => COUPLAGE : E = -(4 pi Phi/mu0) (Tw/L) (S . B_ext)  -- LINEAIRE en Tw et en B.

RESULTATS :
  A. Inversion du flux : mu_tw ~ mu_N (moment structurel du neutron) exige
     Phi = 4.3e-3 (h/e) = 0.60 * alpha * (h/e) -- le quantum de flux ALPHA-SUPPRIME
     (EM = limite faible de chi au couplage alpha). Coherence d'ordre remarquable (x0.6).
  B. VERDICT D'ECHELLE : en labo, E = mu_N*B ~ 3e-8 eV (1 T). C_plast = -2.92 MeV
     exigerait B ~ 1e14 T. => P23 ne peut PAS etre un couplage magnetique direct au
     twist (echo memoire g-2 electron : champ de piege 29 decades trop petit).
     C_plast doit etre un residu de theorie ou un mecanisme non magnetique.
  C. RAFFINEMENT du lock : mu_tw ∝ Tw*S/L -- le twist FOURNIT le courant. Canal (a)
     twist->writhe : dmu/mu = -delta/2 + O(delta) [courant domine ; l'ancien
     (Tw/Wr)delta/2 = geometrie seule etait sous-estime x10]. Neutron (si bouteille/
     faisceau = reequilibrage) : |dmu_n/mu_n| ~ 7e-6 = 30x la precision (2.4e-7) ->
     nettement testable. Canal (b) : mu-silencieux SEULEMENT si S/L invariant.
"""
import numpy as np
from theory.knot_catalog import knot_curve
from theory.abraham import frenet_geometry

MU0 = 4e-7 * np.pi
H_E = 4.1357e-15        # h/e (Wb)
MU_N = 5.0508e-27       # magneton nucleaire (J/T)
ALPHA = 1 / 137.036
L_FM = 3e-15            # taille hadronique (m)


def trefoil_geometry():
    P = np.asarray(knot_curve('3_1', 4000), float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    dl = np.roll(P, -1, 0) - P
    mid = P + 0.5 * dl
    S = 0.5 * np.cross(mid, dl).sum(0)
    L = np.linalg.norm(dl, axis=1).sum()
    g = frenet_geometry('3_1', 20000)
    tau, sp = np.asarray(g['tau']), np.asarray(g['speed'])
    Tw = float((tau * sp * (g['t'][1] - g['t'][0])).sum() / (2 * np.pi))
    return np.linalg.norm(S) / L ** 2, Tw


def run():
    s_over_L2, Tw = trefoil_geometry()
    S = s_over_L2 * L_FM ** 2
    print("=== couplage derive : E = -(4 pi Phi/mu0)(Tw/L)(S.B) ===")
    print(f"  trefle : |S|/L^2={s_over_L2:.4f}, Tw={Tw:.3f} ; L=3 fm -> S={S:.2e} m^2")

    phi_need = MU_N * MU0 * L_FM / (4 * np.pi * Tw * S)
    print(f"\n=== A) inversion du flux (mu_tw = mu_N) ===")
    print(f"  Phi requis = {phi_need:.2e} Wb = {phi_need/H_E:.2e} (h/e)"
          f" = {phi_need/H_E/ALPHA:.2f} x alpha(h/e)  <- quantum alpha-supprime, O(1)")

    print(f"\n=== B) verdict d'echelle (P23) ===")
    for B in (1.0, 10.0):
        print(f"  B={B:>4} T : E = mu_N B = {MU_N*B/1.602e-19:.1e} eV")
    print(f"  C_plast=-2.92 MeV exigerait B ~ {2.92e6*1.602e-19/MU_N:.1e} T")
    print("  => P23 n'est PAS un couplage magnetique direct au twist (neV vs MeV).")

    print(f"\n=== C) lock raffine : mu ∝ Tw*S/L ===")
    print("  canal (a) : dmu/mu = -delta/2 (courant) + O(delta) (geometrie)")
    print(f"  neutron delta=1.4e-5 -> |dmu_n/mu_n| ~ {1.4e-5/2:.1e} (precision 2.4e-7 : testable x30)")


if __name__ == "__main__":
    run()
