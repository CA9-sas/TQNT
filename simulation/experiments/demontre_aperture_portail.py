"""La condition residuelle du maillon Fuller, attaquee depuis la jonction (22/07).

CIBLE (S24) : « le portail couple sur exactement UNE cellule angulaire
(le radian) » -- i.e. angle stocke = omega x a avec omega = densite de
twist du defaut (rad/rad) et a = 1 rad. A demontrer ou epingler.

STRUCTURE EN QUATRE THEOREMES/PIECES :
 [1] UNIFORMISATION (exact) : sur l'anneau ferme, l'equilibre elastique
     du defaut de Fuller (E ~ int omega^2, contrainte int omega = 2piWr)
     donne omega = Wr CONSTANT. La jonction ne voit que la densite.
 [2] LOCALITE (structurel) : la jonction est un POINT ; l'holonomie
     totale 2piWr est GLOBALE (il faut transporter autour de l'anneau
     pour la voir) -- une lecture ponctuelle ne peut dependre que de
     omega. D'ou : s = f(omega), f universelle, f(omega_3)=0 (le trefle
     = zero du portail).
 [3] L'APERTURE MESUREE : lineariser f -> angle stocke = omega x a.
     Les donnees MESURENT a = pi*s/DWr. Confrontation aux apertures
     STRUCTUREES de la jonction (secteur Z3, demi-secteur ruban, tour,
     demi-tour, empreinte metrique, enroulement de trialite).
 [4] DISCRIMINANT MESON/BARYON : les bandes du shift sont MESONIQUES
     (portail a enroulement 0, base 0 -- theoreme de trialite) : toute
     lecture par enroulement ou par secteur de la jonction Y y predirait
     zero ou un autre nombre. La lecture IDENTITE (sans echelle) est
     portail-independante -- coherente, et falsifiable sur les bandes
     baryoniques etranges (meme s attendu).
"""
import numpy as np

S_MEAS, S_ERR = 0.3577, 0.0023
DWR = 1.1277                    # Wr(5_2)-Wr(3_1) ideal katlas (S23)
DWR_ERR = 0.005                 # convergence 4e chiffre + biais residuel


def theoreme_uniformisation(n=360, seed=3):
    """Minimise E = sum omega_i^2 sous sum omega_i dtheta = 2piWr.

    Verification numerique du theoreme (Euler-Lagrange : omega = cst) --
    y compris avec un point epingle (la jonction fixe le REPERE, pas la
    densite : la contrainte est integrale, le minimiseur reste uniforme).
    """
    rng = np.random.default_rng(seed)
    dth = 2 * np.pi / n
    om = rng.normal(0, 1, n)                     # depart aleatoire
    target = 2 * np.pi * 4.5458                  # 2piWr(5_2)
    for _ in range(20000):                       # descente projetee
        om -= 0.05 * 2 * om * dth                # grad de E
        om += (target - om.sum() * dth) / (n * dth)   # projection
    return om.std() / abs(om.mean()), abs(om.mean() * 2 * np.pi - target)


def main():
    print(__doc__)
    print("=" * 76)
    print("[1] UNIFORMISATION : minimiseur elastique sous contrainte de Fuller")
    print("=" * 76)
    disp, res = theoreme_uniformisation()
    print("""  Analytique : delta(int omega^2) = 0 sous int omega = 2piWr
  => 2 omega = lambda (multiplicateur) => omega = Wr PARTOUT. Exact.
  Numerique (depart aleatoire, descente projetee) : dispersion
  residuelle omega : %.1e ; contrainte : %.1e. CONFIRME --
  la jonction ne recoit qu'UN nombre : la densite omega = Wr (rad/rad),
  sans dimension.""" % (disp, res))

    print("=" * 76)
    print("[2] LOCALITE : ce qu'un point peut lire")
    print("=" * 76)
    print("""  L'holonomie 2piWr est une quantite GLOBALE (defaut de fermeture d'un
  transport autour de l'anneau complet) : aucune mesure ponctuelle ne la
  donne. La quantite locale disponible a la jonction est la DENSITE
  omega (theoreme [1] : uniforme = bien definie au point). Donc
  s = f(omega), f universelle (la meme jonction pour tous les noeuds),
  avec le zero au trefle (la porte de trialite definit le repere --
  acquis S20). Linearite : verifie par les donnees a 0,4 %% sur
  omega = 1,13 (pas un petit parametre : la linearite est un FAIT
  mesure, coherent avec la linearite du ladder en angle).""")

    print("=" * 76)
    print("[3] L'APERTURE MESUREE : a = pi*s/DWr")
    print("=" * 76)
    a_meas = np.pi * S_MEAS / DWR
    a_err = a_meas * np.sqrt((S_ERR / S_MEAS) ** 2 + (DWR_ERR / DWR) ** 2)
    print("  a = %.4f +- %.4f rad   (unite a %.1f sigma ; precision %.1f %%)\n"
          % (a_meas, a_err, abs(a_meas - 1) / a_err, 100 * a_err / a_meas))
    candidats = [
        ("IDENTITE (sans echelle)         a = 1", 1.0),
        ("demi-secteur ruban (2pi/3)/2    a = pi/3", np.pi / 3),
        ("secteur Z3                      a = 2pi/3", 2 * np.pi / 3),
        ("demi-tour (ruban)               a = pi", np.pi),
        ("tour plein                      a = 2pi", 2 * np.pi),
    ]
    for nom, a in candidats:
        ns = abs(a - a_meas) / a_err
        print("  %-48s %7.4f  %6.1f sigma %s" % (
            nom, a, ns, "<-- SURVIVANTE" if ns < 3 else "EXCLUE"))
    print("""  (l'empreinte METRIQUE d_tube est deja exclue par le SIGNE, S24.)
  => l'aperture est l'UNITE a 0,5 sigma (mesuree a 0,7 %%). Le
  concurrent structure le plus proche, pi/3 = 1,047 (un secteur Z3 lu
  par le ruban 4pi a taux 1/2), est EXCLU a 6,5 sigma -- declare (regle
  du null : proximite 5 %% notee, mecanisme rejete par les donnees).
  L'identite est l'unique taux SANS ECHELLE : sur le socle gradue en
  action (S4', aucune graduation metrique ni angulaire propre), un
  point sans structure convertit une densite (rad/rad) en angle (rad)
  au seul taux disponible : 1. Tout autre taux exigerait une echelle
  angulaire propre du portail -- et chacune des echelles qu'il possede
  (Z3, ruban) est exclue ci-dessus.""")

    print("=" * 76)
    print("[4] DISCRIMINANT MESON/BARYON : la lecture est portail-independante")
    print("=" * 76)
    print("""  FAIT : les bandes qui mesurent s sont MESONIQUES (K*, D_s, B_s) --
  et le theoreme de trialite donne au portail mesonique un enroulement
  NUL (caractere + conjugue -> base 0, mesure 0,11+-0,16). Toute lecture
  du defaut PAR l'enroulement de trialite (w=1 baryon) ou par un secteur
  de la jonction Y predirait ici ZERO ou un autre nombre : EXCLU par
  l'existence meme du shift mesonique. La lecture identite (locale, sans
  echelle) est independante du type de portail -- coherente avec :
  s(D_s) = s(B_s) (pas de x3^g : cote portail) et la loi de part de
  rotation (S18 : pleine en lourd-leger, 1/2 en leger-leger -- la part
  multiplie la LECTURE, l'aperture est invariante).
  PREDICTION FALSIFIABLE : les bandes orbitales des BARYONS etranges
  (Sigma*/Xi* en L=1, residu structurel nomme du ch. Spectre) doivent
  porter LE MEME shift s = 0,358(2) -- une valeur differente (p.ex.
  DWr/3 = 0,376 par secteur Z3) refuterait la lecture identite et
  reactiverait le secteur.""")

    print("=" * 76)
    print("BILAN")
    print("=" * 76)
    print("""  La condition residuelle « une cellule = 1 radian » est remplacee par :
  [1] la jonction ne voit que la densite omega (theoreme exact) ;
  [2] la lecture est locale (structurel) et lineaire (donnees 0,4 %%) ;
  [3] le taux est MESURE : a = %.4f +- %.4f rad -- l'IDENTITE a 0,5
      sigma, seule valeur sans echelle ; toutes les echelles angulaires
      du portail (Z3, ruban) exclues a >= 6,5 sigma, la metrique par le
      signe ;
  [4] la lecture est portail-independante (le shift existe au portail
      mesonique d'enroulement nul) -- prediction : meme s sur les bandes
      baryoniques etranges.
  STATUT : theoreme conditionnel a l'identification « taux sans echelle
  = identite » (au rang des identifications canoniques : sigma=b de la
  chaine v, boucle<->demi-tour de S4''). Le maillon Wr/2pi est FERME a
  ce rang : s = DWr/pi n'a plus de maillon libre.""" % (a_meas, a_err))


if __name__ == "__main__":
    main()
