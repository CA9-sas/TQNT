"""Cadre 3-cas de la circulation de charge (idee utilisateur) :
   T = torsion du tube de flux ; B = boucle bloquee ; T+B = les deux.

CAS T (torsion) : mu_T = 2 pi e c Tw* S/L (courant de twist, regle Tw*=round(Wr)-Wr).
  Porte par les composantes NON relaxees (trefle : Tw*=0.354). Lk-aveugle.

CAS B (boucle bloquee) : concept DEJA dans V6 (definition 'Boucle bloquee', le 7_{2,b} du
  neutron bloque par la contrainte Lk ; coefficient k_b de l'axiomatique). Mecanisme
  quantitatif natif : PIEGEAGE FLUXOIDE dans le condensat chi coherent (cf. supraconductivite
  bipolaron-chi) -- une boucle enfilee par Lk quanta alpha*Phi_0 (quantum DERIVE) porte un
  courant persistant I = Phi_thread/L_ind :
     mu_B = Lk * alpha Phi_0 * pi r^2 / [mu0 r (ln(8r/a)-2)] * cos(theta)
  ECHELLE AUTOMATIQUEMENT BONNE : 4.5-6.4 mu_N/Lk pour r=0.5-1 fm (dependance log faible),
  avec a = lambda_C/2 (rayon de tube du calcul 3-composantes).

DECOMPOSITION DU NUCLEON :
  3_1  : cas T (Tw*=+0.354, porte chi=-2)      -> mu_T = -1.913 mu_N (d=lambda_C)
  7_2  : cas B (Tw*=-0.06 quasi relaxe, ENFILE) -> porte la dependance Lk
  n->p : +1 boucle bloquee (theta~0) -> +4.71 mu_N = differentiel (r_boucle=0.54 fm exact ;
         toute valeur hadronique 0.5-1 fm donne 4.5-6.4 : robuste).

HYPERONS RELUS : la reponse mesuree/Lk fixe l'ORIENTATION de la boucle bloquee (vs axe de spin):
  nucleon theta ~ 2 deg (alignee) ; Sigma ~ 67 deg (inclinee) ; Xi ~ 97 deg (quasi
  perpendiculaire, legerement ANTI -> explique le signe faux STRUCTURELLEMENT).
  Honnete : 1 angle par doublet = autant de params que de donnees -> PAS predictif encore ;
  le gain = mecanisme a la bonne echelle + le bon degre de liberte identifie ; les angles
  doivent etre CONTRAINTS par la geometrie des configs du catalogue (quelle paire porte
  Delta Lk et son orientation), pas fittes.
"""
import numpy as np

MU0, E, H, C = 4e-7 * np.pi, 1.602e-19, 6.626e-34, 2.998e8
MU_N = 5.0508e-27
ALPHA = 1 / 137.036
A_TUBE = 0.105e-15          # rayon de tube = lambda_C/2 (calcul 3-composantes)


def mu_blocked(r_fm, lk=1, cos_theta=1.0):
    """Moment d'une boucle bloquee (fluxoide) : Lk quanta alpha*Phi_0 enfiles, rayon r."""
    r = r_fm * 1e-15
    L_ind = MU0 * r * (np.log(8 * r / A_TUBE) - 2)
    return lk * ALPHA * H / E / L_ind * np.pi * r ** 2 / MU_N * cos_theta


def run():
    print("=== cas B : echelle fluxoide (quantum derive alpha*Phi_0) ===")
    for r in (0.5, 0.54, 0.84, 1.0):
        print(f"  r={r:.2f} fm : mu_B = {mu_blocked(r):.2f} mu_N/Lk")
    print("  -> ~5 mu_N/Lk pour tout rayon hadronique (log faible) : la BONNE echelle, sans reglage")

    print("\n=== decomposition 3 cas du nucleon ===")
    print("  3_1 : cas T (Tw*=+0.354) -> mu_T=-1.913 ; 7_2 : cas B (Tw*=-0.06, enfiles)")
    print(f"  n->p : +1 boucle bloquee alignee -> {mu_blocked(0.542):.2f} mu_N = differentiel 4.71 OK")

    print("\n=== hyperons : l'orientation de la boucle bloquee ===")
    muB = 4.71
    for nm, resp in (('nucleon', 4.706), ('Sigma', 1.809), ('Xi', -0.599)):
        ct = resp / muB
        print(f"  {nm:8}: {resp:+.3f} mu_N/Lk -> cos(theta)={ct:+.3f}, theta={np.degrees(np.arccos(ct)):.0f} deg")
    print("  -> Xi : boucle quasi perpendiculaire legerement ANTI : le signe faux devient GEOMETRIQUE.")
    print("  Honnete : angles a contraindre par le catalogue (pas predictif encore).")


if __name__ == "__main__":
    run()
