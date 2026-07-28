"""Raffinement H0 : la rotation autour du centre de la galaxie
(24/07 -- demande utilisateur, suite de l'elasticite d'echelle).

QUESTION : la rotation galactique (le Soleil a ~230 km/s autour du
centre ; les masses arrivent et se retirent en chaque point) peut-elle
pomper la trame elastique et raffiner la tension H0 ?

QUATRE CANAUX, TOUS CHIFFRES :
 [R1] LE THEOREME DE LA RIVIERE (stationnarite) : la trame elastique
      repond a la DENSITE rho(x,t), pas aux masses individuelles.
      Une rotation axisymetrique stationnaire = un motif d'ecrasement
      CONSTANT en chaque point (comme une riviere : l'eau passe, le
      niveau est fixe) -> d rho/dt = 0 -> la trame tient sa forme,
      AUCUN pompage. L'elasticite (la trame revient quand la masse se
      retire) n'est sollicitee que par la partie NON stationnaire.
 [R2] LA PARTIE NON STATIONNAIRE (bras spiraux, barre) : pompage
      quadrupolaire -- l'analogue de la luminosite gravitationnelle
      d'une barre en rotation, P ~ (32/5)(G/c^5) I^2 eps^2 Omega^6.
 [R3] LE FROTTEMENT D'ACCELERATION (etabli) : la rotation est un
      mouvement ACCELERE -> temperature d'Unruh kT = hbar a/(2 pi c).
 [R4] LE COTE MESURE : la rotation solaire est corrigee (repere CMB) ;
      SH0ES coupe a z > 0,023 precisement pour ecraser les vitesses
      particulieres.
"""
import math

G = 6.674e-11
C = 3e8
HBAR = 1.055e-34
KB = 1.381e-23
M_SUN = 1.989e30
KPC = 3.086e19
H0_SI = 2.2e-18          # s^-1 (~67,4 km/s/Mpc)

# Voie lactee : barre
M_BAR = 1e10 * M_SUN
R_BAR = 3 * KPC
EPS = 0.2                # ellipticite effective
OMEGA = 2e5 / R_BAR      # ~200 km/s a 3 kpc
M_GAL = 1e12 * M_SUN     # masse totale (halo compris)

# orbite solaire
V_SUN = 2.3e5
R_SUN = 8 * KPC


def main():
    print(__doc__)
    print("=" * 78)
    print("[R1] LE THEOREME DE LA RIVIERE")
    print("=" * 78)
    print("""  La trame repond a rho(x,t) : pour un disque en rotation
  stationnaire, rho est CONSTANT en chaque point (l'eau passe, le
  niveau est fixe). L'elasticite n'est pas sollicitee : le motif
  d'ecrasement d'une galaxie en rotation est aussi STATIQUE que
  celui d'un atome -- c'est le cas (A1) du triptyque (systeme lie,
  trame tenue, zero poussee seculaire). La rotation N'AJOUTE RIEN
  par elle-meme : seul d rho/dt compte. (Corollaire : la trame ne
  distingue pas « matiere qui tourne » de « densite posee » --
  aucune modification des courbes de rotation : les courbes plates
  restent l'affaire du reservoir sombre, PAS d'une trame non
  lineaire type MOND -- coherent avec « indiscernable de la RG ».)""")

    print("=" * 78)
    print("[R2] LE POMPAGE NON STATIONNAIRE (barre/bras), CHIFFRE")
    print("=" * 78)
    I = M_BAR * R_BAR**2
    P = (32 / 5) * (G / C**5) * (I * EPS) ** 2 * OMEGA**6
    rate = P / (M_GAL * C**2)
    print("""  barre : I = %.1e kg m^2, eps = %.1f, Omega = %.1e s^-1
  P_pompage ~ (32/5)(G/c^5)(I eps)^2 Omega^6 = %.1e W
  taux relatif : P/(M_gal c^2) = %.1e s^-1  vs  H0 = %.1e s^-1
  -> %.0f ORDRES DE GRANDEUR sous H0 : le pompage quadrupolaire de
  la rotation est totalement negligeable (c'est la luminosite
  gravitationnelle de la barre -- elle n'a jamais chauffe personne).""" % (
        I, EPS, OMEGA, P, rate, H0_SI, math.log10(H0_SI / rate)))

    print("=" * 78)
    print("[R3] LE FROTTEMENT D'ACCELERATION (Unruh, etabli)")
    print("=" * 78)
    a = V_SUN**2 / R_SUN
    T_u = HBAR * a / (2 * math.pi * C * KB)
    print("""  a_centripete (Soleil) = v^2/r = %.1e m/s^2
  T_Unruh = hbar a/(2 pi c k_B) = %.1e K
  -> ~30 ordres sous le CMB : le frottement de trame de la rotation
  galactique est strictement rien (conforme au bloc frottement/
  temperature etabli : jamais de frottement du mouvement uniforme,
  et l'accelere est thermique a T_U -- ici infinitesimale).""" % (a, T_u))

    print("=" * 78)
    print("[R4] LE COTE MESURE")
    print("=" * 78)
    z_min = 0.023
    v_err = 0.0           # rotation corrigee (repere CMB)
    print("""  La rotation solaire (~230 km/s) et le mouvement du Groupe local
  (~630 km/s) sont corriges par le dipole CMB ; SH0ES coupe son flot
  de Hubble a z > %.3f (cz > 6 900 km/s) pour ecraser les vitesses
  particulieres residuelles (< qq %%, modelisees). Une erreur de
  correction de rotation biaiserait le DIPOLE de H0, pas son
  monopole -- et le dipole mesure est conforme. Rien a extraire.""" % z_min)

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("""  LE CANAL ROTATION EST QUADRUPLEMENT NUL :
   [R1] stationnaire -> pas de sollicitation elastique (theoreme de
        la riviere -- et corollaire : pas de MOND de trame, les
        courbes plates restent au reservoir) ;
   [R2] pompage non stationnaire : ~22 ordres sous H0 ;
   [R3] Unruh rotationnel : ~30 ordres sous le CMB ;
   [R4] cote mesure : deja corrige (dipole conforme).
  La position H0 = 67,4 recoit sa QUATRIEME confirmation par une
  quatrieme voie. GAIN DURABLE du raffinement : le theoreme de la
  riviere complete le triptyque -- l'elasticite est sollicitee par
  d rho/dt SEUL, ce qui unifie (A1) systemes lies ET rotation
  stationnaire sous un meme enonce : « la trame repond a la densite,
  pas au mouvement » (coherent avec la protection du mouvement
  uniforme, et maintenant du mouvement orbital stationnaire).""")


if __name__ == "__main__":
    main()
