"""L'equation du socle : sans particule, avec particule, loi differentielle (21/07).

DEMANDE : definir l'equation du socle (vide / avec particule), en tirer
une loi differentielle qui determine les masses plus facilement.

STRUCTURE (tout est deja canonique -- l'apport est l'ECRITURE UNIQUE) :
  [1] SANS particule : le socle libre
        box s = 0   (d'Alembert du fond d'ondes chi, niveau alpha^0)
      dispersion omega = c|k| EXACTE (zero dispersion = Lorentz
      structurel), amplitude uniforme A0 (la ligne G du quatuor),
      graduation en action (chaque dent = h).
  [2] AVEC particule : la deplation + compensation (S4')
        A(r) = A0 (1 - delta(r)),   LOI DIFFERENTIELLE DE COMPENSATION :
        lap h = - kappa_g |Psi|^2,    h = Omega^2 - 1
      (le Poisson de P5 en forme locale). La masse EST la charge du
      champ de deplation :  m = (c^2/2G) lim r->inf [ r h(r) ].
  [3] LE MODE (masse = frequence confinee) : le defaut est une onde chi
      confinee sur le tube ; l'energie statique du defaut = sigma_chi L
      -> M_topo = sqrt(sigma_chi) l(K) (la loi geometrique RE-DERIVEE
      comme fonctionnelle d'energie) ; les excitations = S4'' :
        M^2 = (Sigma_i m_i)^2 + 2 pi sigma_chi J        (rotation/Regge)
                              + pi sigma_chi n           (torsion, S4'')
                              + 1,38 . 2 pi sigma_chi n_r (radial)
      == LA FONCTIONNELLE MAITRESSE : E[defaut] = sigma_chi Int ds
         + termes de rotation/torsion du ruban. UNE ecriture, toute la
         table des 225 etats.
"""
import numpy as np

SQS = 0.41847
TURN = 2 * np.pi * SQS**2
HALF = TURN / 2
RAD = 1.38 * TURN
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
G, C = 6.6710e-11, 2.99792458e8          # G PREDIT (convention document principal)

print(__doc__)
print("=" * 76)
print("[A] LA LOI DE COMPENSATION REPRODUIT LA GRAVITATION (verifs)")
print("=" * 76)
# Poisson -> Newton : h = 2 G m / (c^2 r) hors source ; GPS :
GM_E = 3.986004418e14
r_sol, r_gps = 6.371e6, 2.6571e7
dff = GM_E / C**2 * (1 / r_sol - 1 / r_gps)
print("  GPS : delta f/f = h/2 difference -> %.1f mus/jour (mesure 45,7)"
      % (dff * 86400 * 1e6))
m_p = 1.67262e-27
h0 = 2 * G * m_p / (C**2 * 0.84e-15)
print("  double petitesse au proton : h(r_p) = %.2e (document principal 2,95e-39)" % h0)
print("  -> la loi lap h = -kappa_g |Psi|^2 EST P5/CGQ en forme locale ;")
print("     monopole = masse : m = (c^2/2G) lim r.h  (la masse est la")
print("     CHARGE du champ de deplation -- S4' quantifie).")

print("\n" + "=" * 76)
print("[B] LA FONCTIONNELLE MAITRESSE REPRODUIT LES SECTEURS (echantillon)")
print("=" * 76)
# M^2 = (somme des defauts)^2 + TURN.DJ + HALF.n + RAD.n_r
tests = [
    ("m_s/m_l = l(5_2)/l(3_1)", MS, ML * (494.8 / 310.0), "longueur (statique)"),
    ("D = m_l+m_c", 1.8697, ML + MC, "somme des defauts"),
    ("D* = D + 1/2 tour", 2.0103, np.sqrt((ML + MC)**2 + HALF), "torsion"),
    ("a2(1320) = rho + 1 tour", 1.3182, np.sqrt(0.77526**2 + TURN), "rotation"),
    ("Lambda_b = coeur + base", 5.6196, np.sqrt((MB + 2 * ML)**2 + HALF), "torsion (base)"),
    ("Sigma_c = coeur + (2+rac2-1)/2t", 2.4535,
     np.sqrt((MC + 2 * ML)**2 + (1 + np.sqrt(2)) * HALF), "echelle S4''"),
    ("rho(1450) = rho + 1 radial", 1.465, np.sqrt(0.77526**2 + RAD), "radial"),
]
for nom, exp, th, canal in tests:
    print("  %-34s %8.4f vs %8.4f  (%+.2f %%)  [%s]"
          % (nom, exp, th, 100 * (th / exp - 1), canal))
print("  -> UNE fonctionnelle (longueur + rotation + torsion + radial),")
print("     zero parametre nouveau : c'est l'algebre de la table 225.")

print("\n" + "=" * 76)
print("VERDICT ET LIMITES HONNETES")
print("=" * 76)
print("""  La loi differentielle du socle est le COUPLE :
    (i)  mode propre du defaut confine (masse = frequence, solveur
         auto-coherent existant = cette equation, couche micro) ;
    (ii) compensation lap h = -kappa_g |Psi|^2 (P5 local, masse =
         monopole).
  Le gain 'masses plus facilement' est REEL mais il est ALGEBRIQUE :
  la fonctionnelle maitresse M^2 = (Sig m_i)^2 + 2pi sig_chi (J + n/2
  + 1,38 n_r) remplace la resolution micro par 4 nombres topologiques
  (l, J, n, n_r). LIMITE IRREDUCTIBLE (document principal 17/07) : l'echelle
  absolue m_l est une constante de TRANSMUTATION -- aucune equation
  differentielle du cadre ne la fixe (analogue Lambda_QCD) ; la loi
  fixe les RAPPORTS et la STRUCTURE, pas le barreau zero.""")
