"""Dérivation de H1b — la discrétisation du flot (barreaux ln(1/alpha)).

H1b disait : « le RG de la trame est discret, barreau = croisement,
espacement L = ln(1/alpha) » — une identification. Dérivation en 5 pièces :

  [D1, canonique + données] TRAME LIBRE EXACTEMENT SANS ÉCHELLE :
       box chi = 0, dispersion nulle STRUCTURELLE (verrou topologique
       alpha = 1/(k+2), k entier ; nuls GRB E_QG > 9,3 E_P). Un champ
       libre sans échelle n'a AUCUNE anomalie -> beta continu du milieu
       nu = 0 EXACTEMENT. Toute course exige un VERTEX.
  [D2, théorème TL (fermeture d'hier)] L'unique vertex du cadre est le
       portail de croisement, et son unique insertion de potentiel est
       le contact quartique (TL_2 : {1, e}, phase pure sur 1).
       => SUPPORT DU FLOT = L'ENSEMBLE DES CROISEMENTS. Le flot est
       discret par ontologie, pas par hypothèse.
  [D3, données ~50 sigma] DÉCOUPLAGE DES BOUCLES DE MATIÈRE : si le
       flot du milieu recevait la contribution une-boucle du MS
       (top + jauge + lambda), beta_MS = -0,023/e-fold -> m_H = 119,6
       GeV (-51 sigma) ; même une contamination de 10 % -> -4,5 sigma.
       Le PARTAGE DES REGISTRES est FORCÉ par les données : les boucles
       de matière courent les MASSES de matière (la loi de course,
       gamma_m — bien réelle et utilisée), pas la raideur du milieu.
  [D4, auto-boucle CALCULÉE] La seule course continue propre du mode
       (sa boucle lambda^2) : au minimum lambda(v) = -B ->
       3 lambda^2/16pi^2 = 8e-5 (3e-4 relatif) ; courbure induite
       eps = -1,2e-3 -> kappa_lambda décalé de -0,05 %, m_H inchangé.
       LE LOG PUR EST EXACT AU NIVEAU DES OBSERVABLES. Portée déclarée :
       forme effective au voisinage du barreau EW (lambda(a) croît vers
       l'UV ; à ~10 barreaux le développement sort de son domaine —
       hors de toute observable utilisée).
  [D5, espacement DOUBLEMENT MESURÉ] barreau = ln(1/alpha) : le ratio
       d'échelle par croisement est lu sur DEUX ancres indépendantes —
       l'ancre EW (v = rac(2pi) alpha^8 M_Pl (1+alpha/4pi), 8 croisements,
       ~4-5 ppm PAR CROISEMENT) et la chaîne G (18 croisements,
       (7h0/6)^(1/18) = alpha à -0,03 %, canonique) ; et alpha lui-même
       est dérivé au verrou entier (k+2 = round((8pi/3) l(3_1)) = 137).

  => beta_lambda = Delta lambda_barreau / L : H1b DÉRIVÉ CONDITIONNEL.
  Reste H1a : l'absence de -mu^2 NU (le radiatif est déjà interdit par
  [D2]/TL) — c'est exactement la liberté d'échelle que kappa_lambda
  testera : la fourche expérimentale, pas un maillon manquant.
"""
import numpy as np

# ---------------------------------------------------------------- entrées
alpha_inv = 137.035999
alpha = 1 / alpha_inv
L = np.log(alpha_inv)
v = 246.21965
M_Pl = 1.220890e19
mH, s_mH = 125.20, 0.11
m_W, m_Z, m_t = 80.3692, 91.1880, 172.57
B = 1 / (np.pi * L)
beta_discret = 4 * B
lam = mH**2 / (2 * v**2)

print("=" * 74)
print("DÉRIVATION DE H1b — LA DISCRÉTISATION DU FLOT   (2026-08-27 soir)")
print("=" * 74)

# ------------------------------------------ D3 : découplage de la matière
y_t = np.sqrt(2) * m_t / v
g2 = 4 * m_W**2 / v**2
gp2 = 4 * (m_Z**2 - m_W**2) / v**2
beta_SM = (24 * lam**2 - 6 * y_t**4 + 12 * lam * y_t**2
           + (3/8) * (2 * g2**2 + (g2 + gp2)**2)) / (16 * np.pi**2)
print(f"""
[D3] — le test de contamination par les boucles de matière :
   beta discret (4B)                = {beta_discret:+.4f} / e-fold
   beta une-boucle MS (top domine)  = {beta_SM:+.4f} / e-fold
   si le flot du milieu recevait beta_MS :""")
for f in (1.0, 0.5, 0.1):
    beta_eff = beta_discret + f * beta_SM
    mh_eff = 2 * v * np.sqrt(beta_eff / 4)
    print(f"      contamination {100*f:3.0f} % : m_H = {mh_eff:6.1f} GeV "
          f"({(mh_eff-mH)/s_mH:+.1f} sigma)")
print("   -> le découplage n'est pas une hypothèse : il est FORCÉ par les")
print("      données (~51 sigma au plein, 4,5 sigma à 10 %). Cohérent avec")
print("      la loi de course : les boucles de matière courent les masses")
print("      de matière (gamma_m), pas la raideur du milieu.")

# ------------------------------------------ D4 : l'auto-boucle du mode
lam_v = -B                                   # lambda(v) = 4B(-1/4)
self_loop = 3 * lam_v**2 / (16 * np.pi**2)
d2lam = (6 * lam_v / (16 * np.pi**2)) * beta_discret
eps = (d2lam / 2) / beta_discret
kl_shift = (2/3) * eps / (5/3)
print(f"""
[D4] — l'auto-boucle du mode (seule course continue propre) :
   lambda(v) = -B = {lam_v:+.4f} -> 3 lambda^2/16pi^2 = {self_loop:.1e} / e-fold
      ({self_loop/beta_discret:.1e} relatif au beta discret)
   courbure induite : eps = {eps:+.2e} -> kappa_lambda {100*kl_shift:+.3f} %
   -> LE LOG PUR EST EXACT au niveau des observables (m_H inchangé).
   Portée déclarée : |ln(a/v)| ~ quelques barreaux (lambda(a) ~ 4B ln
   croît vers l'UV ; le développement local suffit à E'', E''', E'''').""")

# ------------------------------------------ D5 : l'espacement doublement lu
v_pred = np.sqrt(2 * np.pi) * alpha**8 * M_Pl * (1 + alpha / (4 * np.pi))
per_step = (v / (np.sqrt(2 * np.pi) * M_Pl * (1 + alpha/(4*np.pi))))**(1/8)
dev_step = per_step / alpha - 1
print(f"""
[D5] — l'espacement mesuré deux fois :
   ancre EW  : (v_GF / [rac(2pi) M_Pl (1+a/4pi)])^(1/8) = 1/{1/per_step:.3f}
               vs alpha = 1/{alpha_inv} : {1e6*dev_step:+.1f} ppm PAR CROISEMENT
   chaîne G  : (7h0/6)^(1/18) = alpha à -0,03 % (canonique, 18 croisements)
   verrou    : alpha = 1/(k+2), k+2 = round((8pi/3) l(3_1)) = 137 (entier)
   -> le ratio alpha par croisement est pincé au ppm sur une ancre, au
      3e-4 sur l'autre, et dérivé au verrou entier.

BILAN : H1b = [D1] beta libre nul (champ sans échelle, zéro dispersion
structurelle) + [D2] vertex = croisements seuls (TL) + [D3] découplage
matière forcé par les données (~51 sigma) + [D4] auto-boucle 3e-4
calculée + [D5] espacement ln(1/alpha) doublement mesuré.
   => beta_lambda = Delta lambda_barreau / L  : DÉRIVÉ CONDITIONNEL
      (identifications restantes : celles du cadre déjà en place).
   RESTE H1a : pas de -mu^2 NU — le -mu^2 RADIATIF est interdit par
   [D2] ; le nu est la liberté d'échelle elle-même = LA FOURCHE que
   kappa_lambda (5/3 vs 1) testera. Ce n'est pas un maillon manquant,
   c'est la prédiction.""")
