"""Calcul 3-composantes du moment magnetique du neutron (3_1 + 7_2 + 7_2).

Chaine (tout herite des brouillons torsion/couplage/quantum) :
  mu_i = 2 pi e c * Tw_i * S_i / L_i   [Phi = alpha(h/e) = mu0ec/2 DERIVE, zero normalisation EM]
  L_i = l_i * d_tube (ropelength x diametre commun) ; S_i = s_i L_i^2 (facteur de forme).

REGLE DE FRAMING FONDAMENTAL (hypothese physique, validee sur le trefle) :
  le tube relaxe choisit l'entier de framing le plus proche de son writhe ->
     Tw* = round(Wr) - Wr   (twist residuel fractionnaire, |Tw*| <= 1/2).
  Validation : 3_1 (Wr=-3.354) -> Tw*=+0.354 = EXACTEMENT le Tw Frenet utilise partout.
  (Le Tw Frenet d'une courbe arbitraire n'est PAS physique : il depend de l'embedding ;
   le scan du 7_2 lisse l'a montre : Wr converge (6.06-6.12), Tw Frenet erratique.)

RESULTATS :
  1. 7_2 quasi RELAXE : Wr(7_2 repare)=+6.06 proche de l'entier 6 -> Tw*=-0.06.
     => LE MOMENT DU NEUTRON EST PORTE PAR LE COEUR TREFLE (X: 3_1=+0.154 ; 2x7_2=-0.045).
  2. Inversion sur |mu_n|=1.913 mu_N :
       trefle seul     : d_tube = 0.209 fm = lambda_C(proton)=0.2103 fm a 1%
       3 comp. (bornes): d_tube in [0.16, 0.30] fm (signe relatif 7_2 non fixe)
  3. COHERENCE (3 nombres = 1 enonce) : d_tube = lambda_C(p) <=> L(3_1)=l*d=3.44 fm = 4.09 r_p
     (le '4 r_p' de l'inversion mono-composante) <=> avec r_p=4*lambda_C (CGQ),
     l(3_1)=16.37 = 16 a 2.3%.
Honnete : signe de mu_n non derive (orientations/spin) ; s(7_2), Wr(7_2) sur courbe reparee
(pas l'embedding ideal) ; aires vectorielles supposees colineaires ; Tw* = hypothese
(etat fondamental de twist minimal), motivee et validee sur 3_1.
"""
import numpy as np

E, C, MU_N = 1.602e-19, 2.998e8, 5.0508e-27
MU_NEUTRON = 1.91304
LAMBDA_C_P = 0.2103      # hbar/(m_p c) fm
R_P = 0.8414             # fm

# (Wr converge, facteur de forme s=|S|/L^2, ropelength l)
DATA = {'3_1': dict(Wr=-3.354, s=0.0265, l=16.37),
        '7_2': dict(Wr=+6.06,  s=0.0114, l=32.96)}


def tw_star(Wr):
    """Framing fondamental : twist residuel fractionnaire."""
    return round(Wr) - Wr


def run():
    print("=== regle Tw* = round(Wr) - Wr ===")
    for K, d in DATA.items():
        d['Tw'] = tw_star(d['Wr'])
        print(f"  {K}: Wr={d['Wr']:+.3f} -> Tw*={d['Tw']:+.3f}")
    print("  (3_1 : reproduit le 0.354 Frenet ; 7_2 : quasi relaxe, Wr proche de 6)")

    X31 = DATA['3_1']['Tw'] * DATA['3_1']['s'] * DATA['3_1']['l']
    X72 = DATA['7_2']['Tw'] * DATA['7_2']['s'] * DATA['7_2']['l']
    print(f"\n=== contributions X=Tw*s*l : 3_1={X31:+.4f}, 7_2 x2={2*X72:+.4f} ===")
    print("  => moment porte par le coeur TREFLE")

    coef = 2 * np.pi * E * C * 1e-15 / MU_N   # mu/mu_N par (X * d_tube[fm])
    print("\n=== inversion |mu_n| -> d_tube ===")
    for nm, X in (('trefle seul', X31), ('3 comp. opposes', X31 + 2 * X72),
                  ('3 comp. alignes', X31 + 2 * abs(X72))):
        d = MU_NEUTRON / (coef * abs(X))
        print(f"  {nm:16}: d_tube = {d:.3f} fm")
    d0 = MU_NEUTRON / (coef * X31)
    print(f"\n  central = {d0:.3f} fm vs lambda_C(p)={LAMBDA_C_P} ({abs(d0-LAMBDA_C_P)/LAMBDA_C_P*100:.0f}%)")
    print(f"  coherence : L(3_1) = {16.37*LAMBDA_C_P:.2f} fm = {16.37*LAMBDA_C_P/R_P:.2f} r_p ;"
          f" r_p=4 lambda_C (CGQ) -> l(3_1)=16.37 = 16 a 2.3%")


if __name__ == "__main__":
    run()
