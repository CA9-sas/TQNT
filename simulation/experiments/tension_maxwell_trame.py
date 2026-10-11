"""Tension de Maxwell et trame — premiers nombres du brouillon.

Conjecture centrale : la tension de Faraday d'une ligne de champ du tube
(le stress de Maxwell T_par = B^2/2mu_0 intégré sur la section) est la
fraction alpha de la tension de corde :

      T_EM = alpha · sigma_chi

Elle est IMPLIQUÉE par la composition de deux canoniques :
  (1) E_EM = alpha · m_l  (la fraction EM du tube retrouvée par la
      géométrie du courant de framing, -0,3 %, programme m_e) ;
  (2) m = sigma_chi · l   (la loi de masse : le tube EST de la tension).
  => énergie EM par unité de longueur = alpha · sigma_chi.

Trois calculs d'ouverture :
  N1  les échelles : alpha·sigma_chi en MeV/fm et en newtons ;
  N2  l'IDENTITÉ du préfacteur : le stress de Biot-Savart du courant de
      framing (I = e·c/L, un tour par période) donne
        mu_0 I^2/4pi = alpha·hbar c/L^2 = (16/9)·alpha·sigma_chi EXACT
      — car L = m_l/sigma_chi et m_l(nu) = (3/4)·rac(sigma_chi) : le
      16/9 est le (3/4)^-2 de Koide. La forme alpha·sigma_chi est
      FORCÉE ; le O(1) restant = la géométrie du framing (cible T1).
  N3  le null du « quantum de flux libre » : le rayon qu'exigerait une
      ligne de vide portant Phi_0 = h/e à la tension alpha·sigma_chi —
      aucune échelle canonique ne le porte (cohérent : la dichotomie
      lumière/matière interdit les lignes quantifiées du vide).

Structure théorique à calculer (T1, cible) : sur le tube idéal, le
courant de framing a deux composantes -> deux stress de signes opposés :
  - composante solénoïdale (twist, B_z interne)   -> TENSION  +B_z^2/2mu0
  - composante longitudinale (circulation, B_phi) -> COMPRESSION -B_phi^2/2mu0
  T_net = somme sur la section ; prédiction : T_net = +alpha·sigma_chi.
"""
import numpy as np

# ---------------------------------------------------------------- entrées
hbarc = 0.19733            # GeV.fm
alpha_inv = 137.035999
alpha = 1 / alpha_inv
sqrt_sigma = 0.41847       # GeV
sigma = sqrt_sigma**2      # GeV^2
sigma_fm = sigma / hbarc   # GeV/fm
GeV_per_fm_to_N = 1.602177e-10 / 1e-15   # J/m par GeV/fm

m_l_nu = 0.75 * sqrt_sigma            # GeV — masse nue Koide (3/4)rac(sigma)
L = m_l_nu / sigma_fm                 # fm — longueur de tube canonique

print("=" * 72)
print("TENSION DE MAXWELL ET TRAME — premiers nombres   (2026-08-28)")
print("=" * 72)

# ------------------------------------------------- N1 : les échelles
T_strong = sigma_fm * GeV_per_fm_to_N
T_EM = alpha * sigma_fm * GeV_per_fm_to_N
print(f"""
N1 — échelles :
   sigma_chi        = {sigma_fm:.4f} GeV/fm  =  {T_strong:.3e} N  (~{T_strong/9.81/1000:.1f} tonnes-force)
   alpha·sigma_chi  = {1e3*alpha*sigma_fm:.3f} MeV/fm  =  {T_EM:.1f} N
   -> la « tension de Faraday » d'une ligne-tube élémentaire : ~10^3 N —
      la tension hadronique divisée par 137.""")

# ------------------------------------------------- N2 : l'identité 16/9
ah_over_L2 = alpha * hbarc / L**2      # GeV/fm
ratio = ah_over_L2 / (alpha * sigma_fm)
print(f"""
N2 — l'identité du préfacteur (Biot-Savart du framing) :
   L = m_l(nu)/sigma_chi = {L:.4f} fm  (canonique 0,354)
   mu_0 I^2/4pi  (I = e c/L)  =  alpha hbar c / L^2 = {1e3*ah_over_L2:.2f} MeV/fm
   rapport à alpha·sigma_chi = {ratio:.6f}  vs 16/9 = {16/9:.6f}
   ({1e2*abs(ratio*9/16-1):.2e} % — EXACT : L = m/sigma et m = (3/4)rac(sigma)
    => sigma L^2 = (9/16) => alpha hbar c/L^2 = (16/9) alpha sigma_chi)
   -> la forme alpha·sigma_chi est FORCÉE par charge + loi de masse ;
      le O(1) (16/9 -> 1) = la géométrie du framing : CIBLE T1.""")

# ------------------------------------------------- N3 : null du flux libre
# tube de vide portant Phi_0 = h/e, tension imposée alpha·sigma_chi :
# T = Phi^2/(2 mu_0 A)  ->  A = Phi^2/(2 mu_0 T)
h = 6.62607e-34; e = 1.602177e-19; mu0 = 4e-7 * np.pi
for name, Phi in (("h/e", h/e), ("h/2e", h/(2*e))):
    A = Phi**2 / (2 * mu0 * T_EM)
    r = np.sqrt(A / np.pi) * 1e15
    print(f"""
N3 — null « ligne de vide quantifiée » ({name}) :
   rayon requis à T = alpha·sigma_chi : r = {r:.1f} fm""")
print("""   -> ni r_link (~1 fm), ni lambda_C_barre (386), ni D (0,0214) :
   AUCUNE échelle canonique. Lecture : le vide n'a PAS de lignes
   quantifiées (dichotomie lumière/matière) — la quantification des
   lignes est une propriété des TUBES DE MATIÈRE, pas du champ libre.
   La limaille de fer trace le champ moyen de sources linéaires, pas
   des lignes-objets du vide. (Null consigné, périmètre déclaré.)""")

print("""
Tâches du brouillon :
  T1  stress de Maxwell COMPLET du courant de framing sur le tube
      trèfle idéal (machinerie du programme m_e : White, Lk=+1,
      Tw=-2,42) — décomposition twist/circulation, intégrale de section,
      cible T_net = +alpha·sigma_chi (fermer le 16/9 -> 1) ;
  T2  la structure de signe (tension solénoïdale vs compression
      azimutale) = prédiction de structure, à vérifier au passage ;
  T3  hors périmètre déclaré : la géométrie des lignes du VIDE
      (limaille) — champ moyen U(1), pas d'objet-ligne.""")
