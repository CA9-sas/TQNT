"""Chronologie du Big Bang topologique — contrôles d'époques (synthèse).

t(T) en ère de radiation : t = 2,42 g*^-1/2 (T/MeV)^-2 s ; vérifie que
les époques canoniques tombent aux bons temps (t_c = 1,6e-4 s au T_c
dérivé 62 MeV), et rappelle les nombres de la chaîne (aucun nouveau).
"""
import numpy as np

def t_of_T(T_MeV, g_star):
    return 2.42 / np.sqrt(g_star) * T_MeV**-2

print("=" * 72)
print("BIG BANG TOPOLOGIQUE — LA CHRONOLOGIE CONTRÔLÉE    (2026-09-02)")
print("=" * 72)
rows = [
    ("transition du condensat (HotQCD pincé)", 156.5, 40, ""),
    ("LE GEL : T_c = c_eff sigma_chi D0",      62.0, 15.2,
     "<- t_c : marche de G (référence : 1,6e-4 s)"),
    ("BBN (début)",                              1.0, 10.75, ""),
    ("BBN (fin, gel n/p)",                       0.07, 3.4, ""),
]
for nom, T, g, note in rows:
    print(f"  T = {T:7.1f} MeV  (g* = {g:5.1f})  ->  t = "
          f"{t_of_T(T, g):.2e} s   {nom} {note}")
print("""
Nombres de la chaîne (référence, rappel — aucun recalcul) :
  inflation Starobinsky : n_s = 1-2/N ~ 0,965 ; r = 12/N^2 ~ 0,003 ;
    A_s = 1,89e-9 SANS parametre libre (V5.15) ; F1 fermee (V5.14)
  rechauffage en cascade : w = 1/3 exact, T_reh >> T_BBN
    (marge 16,6 ordres) ; F2 FERMEE (2026-07-17)
  gel : T_c = 62 +/- 3 MeV (c_eff = 3,26, double reconnexion P15) ;
    P_occ = (8/9) x f_frac x 1/lambda ; f_frac = 0,34 ;
    n_knot/L = 0,304/xi ; chaleur latente ~ 39 GeV/domaine
  baryogenese AU GEL : eta_B = 7,1e-10 (Planck -13 %) — gel thermique
    refute, chainon lessivage
  Omega_DM : ~0,6 [0,2, 1,3] au T_c derive (ordre de Planck 5,45) ;
    eremnions c >= 23 (bord 1,7-1,9 GeV)
  G : UNE marche a t_c (G = hbar c / M_Pl^2 ; avant : pas de puits)
  Friedmann mecanise : H^2 = 8 pi G rho/3 ; Omega_tot = 1 THEOREME ;
    w = -1 THEOREME ; theta* SMICA 0,06 % ; S8 = 0,826""")

# ---- W-1/W-2 : murs, plasmas, combinatoire depuis les noeuds primaires ---
import numpy as np
HBARC = 0.19733      # GeV fm
SIG_CHI = 0.887      # GeV/fm
LAM_CHI = 0.2092     # GeV
XI = 2.98 / SIG_CHI  # fm (quantum sigma_chi.xi = 2,98 GeV)
KAPPA_DOM = 1.14
L_dom = KAPPA_DOM * XI
sig_w = LAM_CHI**3 / HBARC**2          # tension de paroi ~ Lambda_chi^3
A_dom = 6 * L_dom**2
E_wall = sig_w * A_dom
tau_collapse = L_dom * 1e-15 / 3e8     # s
t_c = 1.6e-4
print(f"""
W-1 — LES PAROIS Z3 : bornees par les cordes, effondrees avant d'exister
  xi = {XI:.2f} fm ; L_dom = kappa_dom.xi = {L_dom:.2f} fm
  tension de paroi sigma_w ~ Lambda_chi^3 = {sig_w:.3f} GeV/fm^2
  E_paroi/domaine ~ 6 sigma_w L^2 = {E_wall:.1f} GeV
    vs chaleur latente 39 GeV/domaine -> ratio {E_wall/39:.2f} :
    LES PAROIS SONT UNE PART DU RESERVOIR LATENT (deja comptees)
  structure Z3 : 3 phases -> parois BORNEES PAR LES CORDES (jonctions
  triples de la trialite) -> reseau hybride qui S'EFFONDRE :
  tau ~ L/c = {tau_collapse:.1e} s << t_c = {t_c:.1e} s (x{t_c/tau_collapse:.0e})
  -> AUCUNE domination de parois (probleme de Zel'dovich dissous par
  la topologie meme du gel : la trialite fabrique des parois mortelles).
W-2 + COMBINATOIRE : des noeuds primaires aux planchers
  1. POPULATION PRIMAIRE (gel) : noeuds seuls, P(K) ~ N(c) e^-alpha.c ;
     trefle 59 % du noue ; PAIRES MIROIR obligatoires (parite du poids
     de Neumann — theoreme rails) : localement K + Kbar partout.
  2. ASSEMBLAGE (plasma dense) : enlacement par P15/fenetre de capture ;
     entonnoir des rails (référence §45) : P(3_1)=0,07-0,19, P(7_2)~1,2e-3,
     paire 7_2+7_2bar naturelle (50/50), P(Lk=+3)=0,056
     -> assemblage proton ~ 4e-9 ; n/p brut 2,0 -> 1,6.
  3. ENTONNOIR DU DESERT : tout entrelacs non protege cascade en
     ~1e-18 s vers les PLANCHERS (e, p, n lie ; marche 1,29 MeV =
     Delta(n-p)) + eremnions c>=23 : la combinatoire, si riche
     soit-elle, converge sur 4 especes en une nanoseconde d'univers.
  4. ANNIHILATION : le gros symetrique s'annihile LOCALEMENT (paires
     miroir co-localisees -> pas de separation spatiale) ; survit la
     miette CP eta_B = 7,1e-10, UNIVERSELLE (constante de la dynamique
     du lessivage, pas condition initiale)
  -> AUCUN domaine matiere/antimatiere, AUCUN mur baryonique :
     la contrainte Cohen-De Rujula-Glashow est passee PAR STRUCTURE.
  FALSIFIABLE DATE : tout anti-helium primordial (candidats AMS-02
  non confirmes) refuterait l'universalite de la miette.""")

# ---- W-3 : la magnetogenese de Biermann (le Seebeck primordial) ----------
# (a) la batterie est INEVITABLE : Gamma_therm/H = 0,6 -> gradients O(1)
#     mal alignes garantis ; croissance vers eps_B x B_equipartition
#     quasi instantanee (la survie est le vrai probleme, pas la source).
# (b) comptabilite comobile (B_com = B_phys (a/a0)^2 ; a/a0 = T0/T) et
#     longueurs de coherence ; (c) cascade inverse HELICALE (B^2.lambda
#     conserve — Banerjee-Jedamzik) ; source d'helicite = LA MEME
#     asymetrie CP que eta_B (lessivage) [identification nommee] ;
# (d) confrontation : borne blazars B_IGM >= 3e-16 G (Neronov-Vovk).
B_C_G = 4.41e13          # G (champ critique, reference d'echelle)
ME_GEV = 5.11e-4
T0_GEV = 2.35e-13        # temperature CMB actuelle en GeV
def B_eq(T_GeV):         # equipartition ~ T^2 (echelle B_c a m_e^2)
    return B_C_G * (T_GeV / ME_GEV)**2
def a_ratio(T_GeV):      # a/a0 ~ T0/T (facteurs g* omis, ordre)
    return T0_GEV / T_GeV
print("""
W-3 — LA MAGNETOGENESE DE BIERMANN (le Seebeck primordial)
  la batterie : dB/dt = (c k_B/e) grad(T) x grad(ln n_e) — pur CONTENU
  (memes constantes que la batterie de croute du magnetar) ; carburant
  garanti par Gamma_therm/H = 0,6 (thermalisation imparfaite).""")
for T, eps, nom, t_ep in [(62e-3, 1e-3, "au GEL (T_c = 62 MeV)", 1.6e-4)]:
    Beq = B_eq(T); a = a_ratio(T)
    Bph = eps * Beq
    Bcom = Bph * a**2
    ell_com_pc = 3e10 * t_ep / a / 3.086e18
    print(f"  époque : {nom}")
    print(f"    B_eq = {Beq:.1e} G ; eps_B ~ {eps:.0e} -> B_phys ~ {Bph:.1e} G")
    print(f"    comobile : B_com ~ {Bcom:.1e} G a l_coh ~ {ell_com_pc:.2f} pc")
    # cascade inverse helicale : B^2 lambda = const -> a 100 kpc
    lam_f = 1e5  # pc
    B_f = Bcom * np.sqrt(ell_com_pc / lam_f)
    print(f"    cascade inverse HELICALE (B^2.lambda conserve) -> a 100 kpc :"
          f" B ~ {B_f:.1e} G")
    print(f"    borne blazars (Neronov-Vovk) : B >= 3e-16 G -> "
          f"{'PASSE (x%.0e)' % (B_f/3e-16) if B_f > 3e-16 else 'TENDU'}")
print("""  SOURCE D'HELICITE [identification nommee, conditionnelle] : le MEME
  biais CP que eta_B (le lessivage) injecte une helicite magnetique de
  signe UNIFORME (lien baryogenese<->helicite, type Vachaspati) — c'est
  elle qui achete la cascade inverse ; sans helicite (decroissance
  non-helicale), la relique tombe vers ~1e-15 G : la borne blazar
  DISCRIMINE la version helicale.
  FALSIFIABLES : (i) helicite de SIGNE UNIFORME du champ intergalactique
  (correlateurs parite-impairs des gamma diffus — CTA) ; (ii) tout
  B_IGM ~ 1e-11-1e-13 G a 100 kpc est DANS la fourchette de la chaine.
  LA BOUCLE MAGNETIQUE COMPLETE DU CADRE :
  Seebeck de cascade (contenu) -> graines au gel -> helicite CP ->
  cascade inverse -> B_IGM ~ 1e-11 G -> dynamos galactiques (uG) ->
  effondrements stellaires -> dynamo de naissance -> B > B_c :
  le passage vers la nucleation du magnetar — la meme thermoelectricite
  du contenu, du premier gradient primordial au champ supercritique.""")
