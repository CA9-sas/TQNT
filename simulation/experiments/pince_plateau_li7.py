"""Pincer la valeur du plateau de Spite : le calcul d'efficacite
propre -- et son verdict (24/07).

AVANT les 3 raffinements (multipole, survie douce, etalement), il faut
verrouiller le maillon le plus fragile de la chaine : l'EFFICACITE
photon -> 7Be, posee a ~3e-3 « ordre litterature » dans le premier
passage. ELLE SE CALCULE :

 [E1] sigma_gamma(7Be) par BALANCE DETAILLEE depuis la capture
      3He(alpha,gamma)7Be (S34(0) ~ 0,56 keV b, Sommerfeld exact) ;
 [E2] la perte dominante du photon : Compton sur les electrons libres
      (Klein-Nishina a 2 MeV ; ~1 diffusion suffit a passer sous le
      seuil) ;
 [E3] P(destruction) = sigma_Be n_Be / (sigma_KN n_e) -- les deux
      densites en a^-3 : le rapport est INDEPENDANT du temps.
PUIS le bilan : photons requis par baryon vs le budget du mecanisme.
"""
import math

# nucleaire
S34 = 0.56e-3          # MeV b (S-facteur 3He(a,g)7Be a E->0)
MU = 1710.0            # MeV, masse reduite 3He-4He
Z1Z2 = 4
ALPHA = 1 / 137.036
SEUIL_BE = 1.587       # MeV
E_G = 2.0              # MeV, photon type du mecanisme
# plasma
SIG_T = 6.652e-25      # cm^2
M_E = 0.511
BE_H = 5e-10           # 7Be/H (BBN)
NH_NB, NE_NB = 0.75, 0.88
# mecanisme
N_MX = 8.6e-2          # paires mixtes par baryon (cascade, mesure)
FRAC_DESTRUCTION = 2 / 3


def sigma_capture(E_cm):
    eta = Z1Z2 * ALPHA * math.sqrt(MU / (2 * E_cm))
    return S34 / E_cm * math.exp(-2 * math.pi * eta)   # b


def sigma_gamma_be7(E_g):
    E_cm = E_g - SEUIL_BE
    if E_cm <= 0:
        return 0.0
    p2c2 = 2 * MU * E_cm               # MeV^2
    boost = p2c2 / E_g**2              # balance detaillee (ordre, stat O(1))
    return sigma_capture(E_cm) * boost * 1e-24   # cm^2


def sigma_kn(E):
    x = E / M_E
    return math.pi * (2.818e-13) ** 2 * (1 / x) * (math.log(2 * x) + 0.5)


def main():
    print(__doc__)
    print("=" * 78)
    print("[E1-E3] L'EFFICACITE CALCULEE")
    print("=" * 78)
    sg = sigma_gamma_be7(E_G)
    sk = sigma_kn(E_G)
    p = (sg / sk) * (BE_H * NH_NB / NE_NB)
    print("""  sigma_capture(0,41 MeV) = %.1e b (Sommerfeld e^-2pi eta = %.1e)
  sigma_gamma(7Be, 2 MeV)  = %.1e cm^2  (~0,1 mb, balance detaillee)
  sigma_KN(2 MeV)          = %.1e cm^2
  n_Be7/n_e = %.1e
  -> P(destruction par photon) = %.1e   (INDEPENDANT du temps)
  L'« efficacite ~3e-3 » du premier passage etait FAUSSE de ~9 ordres
  (elle confondait le requis par photon et le requis par GeV injecte :
  les solutions de la litterature exigent bien ~GeV/baryon).""" % (
        sigma_capture(E_G - SEUIL_BE), math.exp(-2 * math.pi * Z1Z2 * ALPHA
        * math.sqrt(MU / (2 * (E_G - SEUIL_BE)))), sg, sk,
        BE_H * NH_NB / NE_NB, p))

    print("=" * 78)
    print("LE BILAN : requis vs budget")
    print("=" * 78)
    n_dest = FRAC_DESTRUCTION * BE_H * NH_NB
    n_req = n_dest / p
    deficit = n_req / N_MX
    print("""  a detruire : %.1e 7Be/baryon -> photons requis : %.0f /baryon
  (soit ~%.1f GeV/baryon a 2 MeV -- l'ordre des solutions publiees a
  particules lourdes en desintegration ; thermalise sans trace CMB a
  ces epoques profondes : la contrainte n'est PAS le CMB, c'est le
  BUDGET).
  budget du mecanisme : n_MX/n_b = %.1e (mesure), 1 photon chacun,
  x e^-t_eff/tau <= 1 ->
     DEFICIT >= x%.0f  (~4 ordres de grandeur).""" % (
        n_dest, n_req, n_req * E_G * 1e-3, N_MX, deficit))

    print("=" * 78)
    print("LES CANAUX DE SAUVETAGE, FERMES UN A UN")
    print("=" * 78)
    print("""  - resonance nucleaire dans la fenetre : le premier niveau du
    continuum 7Be utile est a ~4,6 MeV (hors fenetre, au-dessus du
    seuil D) -> pas de rehaussement resonant ;
  - marche aleatoire Compton (diffusions vers l'avant qui laissent
    E > seuil) : facteur <~3, il en faut 1e4 ;
  - multiplicite : les paires naissent a |Lk| = 1 (mesure) -> 1
    photon chacune ; |Lk| >= 2 : 0/187 mesure ;
  - abondance : 860/baryon requis vs 2,5 noeuds sombres/baryon AU
    TOTAL -> impossible par construction (x340 au-dessus de TOUTE la
    matiere noire) ;
  - canal neutronique (7Be(n,p)7Li puis destruction) : exigerait une
    injection HADRONIQUE ; le mecanisme est EM pur, et les photons
    > 2,22 MeV des familles DD meurent thermiquement avant le D ;
  - retarder tau (survie douce, etalement) : ne change RIEN -- le
    deficit est en AMONT de la survie (budget brut insuffisant).""")

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("""  LE MECANISME « CENDRES EXCITEES DU GEL » EST REFUTE AU MUR
  D'EFFICACITE : il livre au mieux ~1,7e-4 GeV/baryon quand la
  photodestruction du 7Be en exige ~1 (efficacite reelle ~3e-13 par
  photon, calculee -- pas les ~3e-3 supposes). Les 3 raffinements
  prevus (multipole, survie douce, etalement) sont SANS OBJET : ils
  raffinaient un facteur e^-t/tau en AVAL d'un budget deja court de
  4 ordres en amont. CE QUI SURVIT : la chaine structurelle
  (abondance mesuree des paires mixtes, fenetre geometrique beta
  sqrt(cJcJ'), protection par enlacement, norme radiative alpha
  G_F^2 M^2 Q/8pi) reste DERIVEE et correcte -- elle decrit une
  population reelle de metastables du gel et son rayonnement ; elle
  ne detruit simplement pas assez de 7Be. LE LITHIUM-7 REDEVIENT
  HERITE. A consigner : l'erreur du premier passage (efficacite
  posee ~3e-3 « ordre litterature » sans calcul -- 9 ordres) ; a
  amender : la consolidation V6 du jour (statut du paragraphe
  li7-cendres, B.6, tables).""")


if __name__ == "__main__":
    main()
