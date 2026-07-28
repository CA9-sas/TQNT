"""Frottement d'une particule sur le socle et origine de la temperature (23/07).

VISION UTILISATEUR : la dispersion de l'energie en chaleur lors du
frottement d'une particule sur le socle serait l'ORIGINE de la
temperature.

CADRE CANONIQUE (les pieces existantes) :
 - protection de Lorentz : le socle a ZERO dispersion structurelle --
   un mouvement UNIFORME ne frotte pas (sinon repere privilegie =
   ether, refute ; nuls GRB E_QG > 9,3 E_P) ; la « route friction »
   (verrouillage T_c, derives c/G) est MORTE au critere ;
 - MAIS le document principal contient deja DEUX temperatures qui sont des
   frottements de socle : UNRUH (P22, 3e regime : 4pi/xi = mc^2/kT_U,
   identite exacte) et HAWKING (S4''' : horizon = dents arretees) ;
 - la chaleur = puits chi qui vibrent (formule maitresse de
   dissipation, q_dot = rho J^2 via Gamma de la danse) ;
 - la T EMERGENTE de la chaine de Boltzmann (fond d'ondes chi +
   equiprobabilite -> danse + T) : la temperature est DEJA emergente
   du socle au niveau formel.

L'ETUDE : quatre volets quantitatifs pour situer la vision.
[A] Le theoreme de protection (uniforme : aucun frottement, borne GRB).
[B] Le frottement EXISTE en acceleration : Unruh = SA temperature.
[C] La dispersion en chaleur : le canal de Larmor (modes du socle).
[D] L'origine de la temperature : fluctuation-dissipation sur le
    CONTENU d'ondes du socle (la T unique d'Einstein).
"""
import numpy as np

HBAR = 1.0546e-34
C = 2.9979e8
KB = 1.3807e-23
ME = 9.109e-31
QE = 1.602e-19
EPS0 = 8.854e-12
SIGMA_T = 6.652e-29                    # Thomson, m^2
A_RAD = 7.566e-16                      # J m^-3 K^-4
LP = 1.616e-35


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] PROTECTION : le socle NU ne frotte pas en mouvement uniforme")
    print("=" * 78)
    print(f"""  Zero dispersion structurelle (Lorentz = symetrie de []chi = 0) :
  un frottement du mouvement uniforme definirait un repere absolu.
  Borne experimentale : nul GRB E_QG > 9,3 E_P -> toute echelle de
  « grain frottant » est < l_P/9,3 = {LP / 9.3:.1e} m : le socle nu est
  sans frottement JUSQUE SOUS son propre grain. Corollaire quotidien :
  a = g = 9,81 m/s^2 -> T_Unruh = hbar a/(2 pi c k) =
  {HBAR * 9.81 / (2 * np.pi * C * KB):.1e} K : la chaleur du quotidien N'EST PAS du
  frottement de socle nu. (Et la route « friction » du chantier T_c
  reste morte -- aucun retour.)""")

    print("=" * 78)
    print("[B] LE FROTTEMENT EXISTE EN ACCELERATION : UNRUH = SA TEMPERATURE")
    print("=" * 78)
    a = 1e20
    lamC = HBAR / (ME * C)
    xi = 2 * a * lamC / C**2
    lhs = 4 * np.pi / xi
    kTu = HBAR * a / (2 * np.pi * C)
    rhs = ME * C**2 / kTu
    print(f"""  L'acceleration PROPRE est absolue (invariante) : « frotter sur le
  socle » a un sens sans repere -- c'est le 3e regime de P22.
  Identite canonique verifiee (electron, a = 1e20 m/s^2) :
     4 pi / xi          = {lhs:.6e}
     m c^2 / k T_Unruh  = {rhs:.6e}   (rapport {lhs / rhs:.9f})
  kT_U = hbar a / (2 pi c) : LA temperature du frottement du socle.
  Cas extreme : l'horizon BH (dents arretees, S4''') -> Hawking = le
  MEME frottement a la limite. Le document principal possede donc deja deux
  temperatures 100 %% socle-frottement : Unruh et Hawking.""")

    print("=" * 78)
    print("[C] LA DISPERSION EN CHALEUR : LE CANAL DE LARMOR (modes du socle)")
    print("=" * 78)
    T = 300.0
    v_th = np.sqrt(3 * KB * T / ME)
    tau_coll = 1e-13                    # collision typique (gaz/solide)
    a_coll = v_th / tau_coll
    P_lar = QE**2 * a_coll**2 / (6 * np.pi * EPS0 * C**3)
    E_rad = P_lar * tau_coll
    print(f"""  Une charge acceleree DISPERSE son energie dans les modes du socle
  (rayonnement ; couplage = le QED a alpha = 1/137 derive du socle).
  Electron thermique a 300 K : v_th = {v_th:.1e} m/s, collision
  ~{tau_coll:.0e} s -> a ~ {a_coll:.1e} m/s^2 :
     P_Larmor = {P_lar:.1e} W -> E_rayonnee/collision = {E_rad:.1e} J
     vs kT = {KB * T:.1e} J : fraction {E_rad / (KB * T):.1e}
  -> la dispersion RADIATIVE par collision est infime : la matiere
  thermalise d'abord CINETIQUEMENT (puits chi qui vibrent = la
  chaleur, formule de dissipation du document principal), et le rayonnement
  s'equilibre sur ~1e10 collisions -- pourquoi la matiere chauffe
  vite et « rougeoie » lentement.""")

    print("=" * 78)
    print("[D] L'ORIGINE DE LA TEMPERATURE : FROTTEMENT SUR LE CONTENU DU SOCLE")
    print("=" * 78)
    for Tb in (2.725, 300.0, 6000.0, 1e7):
        U = A_RAD * Tb**4
        tau_drag = 3 * ME * C / (4 * SIGMA_T * U)
        print(f"  fond thermique de photons T = {Tb:10.1f} K : tau_frottement(e-) = "
              f"{tau_drag:.1e} s")
    print(f"""  Le frottement LICITE n'est pas sur le socle nu mais sur son
  CONTENU d'ondes (le fond thermique de photons/ondes chi PRESENTES) : le fond thermique
  definit un repere (comme le CMB) sans violer Lorentz. La relation
  d'Einstein (fluctuation-dissipation) verrouille : la trainee
  (dissipation) et les coups (fluctuations du fond thermique) partagent UNE
  temperature -- D/mu = kT. C'est LA formalisation de la vision :
     temperature = parametre du contenu d'ondes du socle ;
     equilibrage = frottement sur ce contenu (traine de Thomson,
     bremsstrahlung, Gamma de la danse) ;
     chaleur = energie dispersee vers puits (vibrations) et modes
     (rayonnement) du socle.
  La chaine de Boltzmann du document principal (fond d'ondes + equiprobabilite ->
  danse + T emergente) est la version formelle ; le frottement decrit
  ICI est son MECANISME microscopique d'equilibrage.""")

    print("=" * 78)
    print("BILAN (statut : etude exploratoire, brouillon)")
    print("=" * 78)
    print("""  LA VISION TIENT SOUS UNE SCISSION PRECISE :
  (i) socle NU : frottement INTERDIT en mouvement uniforme (theoreme
      de protection -- jamais reintroduire) ; il n'existe qu'en
      ACCELERATION propre et sa temperature est UNRUH (exacte,
      canonique via P22) ; cas extreme = Hawking. Deux temperatures
      du document principal SONT deja du frottement de socle.
  (ii) matiere ordinaire : la temperature = frottement sur le CONTENU
      d'ondes du socle (fluctuation-dissipation, T unique) + puits
      qui vibrent (dissipation Gamma) -- l'origine de la temperature
      EST le socle en ce sens (la T emergente canonique recoit son
      mecanisme d'equilibrage).
  (iii) garde-fous chiffres : T_U(g) ~ 4e-20 K (le quotidien n'est
      pas du socle nu) ; dispersion radiative/collision ~1e-10 kT
      (la chaleur est d'abord cinetique/puits) ; borne GRB (pas de
      grain frottant au-dessus de l_P).""")


if __name__ == "__main__":
    main()
