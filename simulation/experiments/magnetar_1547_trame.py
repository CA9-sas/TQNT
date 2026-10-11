"""Formation de matière autour du magnétar 1E 1547.0-5408 (M1-M6).

Données (catalogue McGill) : P = 2,072 s (rotation la plus rapide des
magnétars), Pdot = 4,77e-11, B_dip = 3,2e14 G = 3,2e10 T, d = 4,5 kpc,
R = 12 km, I = 1e38 kg m^2 ; sursauts : tempête du 22/01/2009 (>200
sursauts, Fermi-GBM saturé).

M1 : le passage magnétique vers la nucléation OUVERT — B/B_c, chi(E) en surface ;
M2 : la photosphère de paires — r_pair(E) (dipôle B ~ r^-3) et la
     coupure spectrale au MeV (postdiction) ;
M3 : la chaîne d'accélération — potentiel de calotte, E_gap vs E_c
     (la porte ÉLECTRIQUE reste fermée, la porte MAGNÉTIQUE est ouverte) ;
M4 : le taux de formation de matière — Goldreich-Julian x multiplicité ;
M5 : les sursauts 2009 — paires par plasma de paires piégé ;
M6 : les réservoirs — énergie magnétique, luminosité de freinage.
"""
import numpy as np

C = 2.99792458e8
HBAR = 1.054571817e-34
E_CH = 1.602176634e-19
ME = 9.1093837015e-31
MU0 = 4e-7 * np.pi

# --- l'objet ---
P, Pdot = 2.072, 4.77e-11
B_s = 3.2e10           # T (3,2e14 G)
R_ns = 1.2e4           # m
I_ns = 1e38            # kg m^2
B_c = ME**2 * C**2 / (E_CH * HBAR)   # 4,41e9 T (jumeau magnétique de E_c)
MC2 = 0.511e6          # eV

print("=" * 74)
print("1E 1547.0-5408 — FORMATION DE MATIÈRE, LECTURE TRAME (2026-09-02)")
print("=" * 74)

# ---- M1 : passage vers la nucléation ouvert --------------------------------------------------
print(f"""
M1 — le passage magnétique vers la nucléation : B_c = m^2c^2/(e hbar) = {B_c:.3e} T
   B_surface = {B_s:.1e} T -> B/B_c = {B_s/B_c:.2f} : SUPERCRITIQUE
   (le régime « champ critique xi ~ O(1) » de P22, réalisé en nature)
   chi(E) = (E/2mc^2)(B_perp/B_c) en surface :""")
for E_keV in (30, 100, 511, 1000, 10000):
    chi = (E_keV * 1e3 / (2 * MC2)) * (B_s / B_c)
    reg = "OUVERT (sans suppression)" if chi > 1 else \
          f"exp(-4/3chi) = {np.exp(-4/(3*chi)):.1e}"
    print(f"   E = {E_keV:6d} keV : chi = {chi:6.2f} -> {reg}")

# ---- M2 : photosphère de paires ------------------------------------------
print("\nM2 — photosphère de paires (dipôle B(r) = B_s (R/r)^3 ; "
      "conversion à chi ~ 0,1) :")
for E_keV in (511, 1000, 10000):
    chi_s = (E_keV * 1e3 / (2 * MC2)) * (B_s / B_c)
    if chi_s > 0.1:
        r_pair = R_ns * (chi_s / 0.1)**(1/3)
        print(f"   E = {E_keV:6d} keV : opaque jusqu'à r = "
              f"{r_pair/R_ns:.1f} R_ns")
    else:
        print(f"   E = {E_keV:6d} keV : transparent (chi_s = {chi_s:.2f})")
print("   -> POSTDICTION : la magnétosphère interne est OPAQUE aux gamma"
      " >~ MeV\n      (queues X dures observées jusqu'à ~200 keV, RIEN au"
      " MeV — COMPTEL nul) ;\n      la coupure spectrale des magnétars EST"
      " la photosphère du passage vers la nucléation.")

# ---- M3 : la chaîne d'accélération ---------------------------------------
Om = 2 * np.pi / P
dPhi = Om**2 * B_s * R_ns**3 / (2 * C)       # potentiel de calotte polaire
h_gap = 100.0                                 # m (gap standard)
E_gap = dPhi / h_gap
E_c = ME**2 * C**3 / (E_CH * HBAR)            # 1,32e18 V/m (Schwinger)
print(f"""
M3 — l'accélérateur : Omega = {Om:.2f} rad/s
   potentiel de calotte dPhi = Om^2 B R^3/(2c) = {dPhi:.1e} V
   champ de gap (h ~ 100 m) : E ~ {E_gap:.1e} V/m vs E_c = {E_c:.1e}
   -> la porte ÉLECTRIQUE (Schwinger) reste FERMÉE ({E_c/E_gap:.0f}x sous
      le seuil) ; la porte MAGNÉTIQUE est OUVERTE (M1) : le passage vers la nucléation
      s'ouvre par son côté B — primaires ~ {dPhi:.0e} eV -> photons de
      courbure -> cascade de paires.""")

# ---- M4 : taux de formation de matière -----------------------------------
B12 = B_s * 1e4 / 1e12          # B en unités de 1e12 G
Ndot_GJ = 2.7e30 * B12 / P**2   # s^-1 (formule standard)
kappa = 1e3                     # multiplicité de cascade (1e2-1e4)
Ndot = kappa * Ndot_GJ
mdot = Ndot * 2 * ME
print(f"""M4 — formation de matière en régime :
   Ndot_GJ = {Ndot_GJ:.1e} charges/s ; multiplicité kappa ~ 1e3
   -> Ndot_paires ~ {Ndot:.1e} /s = {Ndot:.0e} événements d'entrée
      trèfle+miroir PAR SECONDE ; débit de masse ~ {mdot:.0f} kg/s
      (~{mdot*86400/1e3:.0e} t/jour de matière e+- créée en continu ;
      ordre de grandeur, kappa dans [1e2, 1e4])""")

# ---- M5 : les sursauts 2009 ----------------------------------------------
print("M5 — la tempête du 22/01/2009 (>200 sursauts) :")
for E_burst_erg, nom in [(1e37, "sursaut typique"),
                         (1e40, "sursaut fort"),
                         (1e43, "classe éruption intermédiaire")]:
    N_pairs = E_burst_erg * 1e-7 / (2 * ME * C**2)
    print(f"   {nom:28s} {E_burst_erg:.0e} erg -> "
          f"~{N_pairs:.1e} paires (plasma de paires piégé)")
print("   raie d'annihilation 511 keV attendue (cible SPI/INTEGRAL).")

# ---- M6 : réservoirs ------------------------------------------------------
E_mag = (B_s**2 / (2 * MU0)) * (4/3) * np.pi * R_ns**3
L_sd = 4 * np.pi**2 * I_ns * Pdot / P**3
print(f"""M6 — réservoirs :
   énergie magnétique ~ B^2/2mu0 x V = {E_mag:.1e} J = {E_mag*1e7:.0e} erg
   luminosité de freinage L_sd = 4pi^2 I Pdot/P^3 = {L_sd*1e7:.1e} erg/s
   (McGill : ~1e35 erg/s — 1E 1547 est le magnétar le plus « pulsar »)
   -> la formation de matière est payée par B (sursauts) ET Omega (régime).""")

# ---- M7 : M-3 — la coupure QUANTITATIVE ----------------------------------
# Transmission T(E) = exp(-tau) ; tau = int alpha_pair dr le long d'un
# rayon depuis r_em ; Erber : alpha = (alpha_fs/2 lambdabar_C)(B_perp/B_c)
# x T(chi), T(chi) = (0.16/chi) K_{1/3}(2/3chi)^2 ; SEUIL cinematique
# impose : E sin(theta) >= 2mc^2 (sinon alpha = 0). Simplification
# declaree : angle photon-B fige a sin(theta_em) (propagation radiale) ;
# redshift gravitationnel (M = 1,4 M_sol) pour l'energie OBSERVEE.
from scipy.special import kv
ALPHA_FS = 1 / 137.035999
LAMBAR_C = HBAR / (ME * C)          # 3,86e-13 m
G_N = 6.6710e-11
M_ns = 1.4 * 1.98892e30
r_s = 2 * G_N * M_ns / C**2
zfac = np.sqrt(1 - r_s / R_ns)      # redshift surface

def tau_pair(E_MeV, r_em, sinth):
    r = np.linspace(r_em, 30 * R_ns, 4000)
    B_perp = B_s * (R_ns / r)**3 * sinth
    if E_MeV * sinth < 2 * 0.511:   # seuil cinematique
        return 0.0
    chi = (E_MeV / 1.022) * (B_perp / B_c)
    Tchi = np.where(chi > 1e-3, 0.16 / np.clip(chi, 1e-3, None)
                    * kv(1/3, 2 / (3 * np.clip(chi, 1e-3, None)))**2, 0.0)
    al = (ALPHA_FS / (2 * LAMBAR_C)) * (B_perp / B_c) * Tchi
    return np.trapezoid(al, r)

print(f"\nM7 — M-3 : la coupure quantitative (redshift surface "
      f"1/(1+z) = {zfac:.3f})")
print(f"   {'r_em':>6s} {'sin0':>5s} {'E_mur local':>12s} "
      f"{'E_mur OBSERVE':>14s} {'largeur 90->10%':>16s}")
for r_fac, sinth in [(1.0, 1.0), (1.0, 0.5), (2.0, 1.0), (3.0, 1.0),
                     (4.0, 1.0)]:
    E_grid = np.linspace(1.022/sinth * 0.999, 6.0, 3000)
    taus = np.array([tau_pair(E, r_fac * R_ns, sinth) for E in E_grid])
    if taus.max() < 1:
        print(f"   {r_fac:5.1f}R {sinth:5.1f} {'transparent':>12s} "
              f"{'(pas de mur)':>14s}")
        continue
    zloc = np.sqrt(1 - r_s / (r_fac * R_ns))
    E_edge = E_grid[np.argmax(taus > 1)]
    E_10 = E_grid[np.argmax(taus > 2.303)]
    E_90 = E_grid[np.argmax(taus > 0.105)] if (taus > 0.105).any() else E_edge
    print(f"   {r_fac:5.1f}R {sinth:5.1f} {E_edge:9.3f} MeV "
          f"{E_edge*zloc:11.3f} MeV {1e3*(E_10-E_90):13.1f} keV")
print("""   -> LE MUR : la transmission passe de 1 a e^-1000 en ~quelques keV
      (un BORD FRANC, pas une coupure exponentielle douce) ; il siege AU
      SEUIL cinematique redshifte E_obs = (2mc^2/sin0) sqrt(1-r_s/r_em) :
      ~0,84 MeV pour une emission de surface a grand angle.
   CONFRONTATION 1E 1547 : la queue dure mesuree (Gamma ~ 1,4-1,5,
      INTEGRAL/Suzaku jusqu'a ~200-270 keV) est ENTIEREMENT SOUS le mur ->
      le canal paires ne coupe PAS la bande observee (coherent) ; la
      cassure attendue a 300-700 keV (limites COMPTEL sur la classe)
      revient a la SCISSION de photon (M-1) ou au spectre intrinseque.
   FALSIFIABLES (COSI, ~2027+) : (i) aucun photon d'un magnetar
      au-dessus du mur redshifte si l'emission est basse — un BORD net a
      <= 0,84 MeV ; (ii) l'ENERGIE du bord mesure l'ALTITUDE d'emission
      (E_edge = altimetre) : un spectre depassant 1 MeV imposerait
      r_em >~ 3 R (la ou B_perp < 0,1 B_c et le mur s'evanouit).""")

# ---- M8 : M-1 — scission vs paires + le theoreme trame/temperature -------
# SCISSION gamma -> gamma gamma (Adler) : SEUL le mode PERP se scinde
# (selection cinematique sous faible dispersion) ; taux
#   kappa_sp = (alpha^3/2pi^2 lambdabar_C)(26/315)^2 min(b,1)^6 (E/mc^2)^5
# (saturation b>=1 declaree — pratique standard en champ supercritique).
# PAIRES (M7) : mode PARA seul au seuil 2mc^2 ; le seuil PERP en champ
# supercritique monte a mc^2(1+sqrt(1+2b)) = 2,5 MeV : jamais atteint
# (le PERP est tue par la scission bien avant).
pref_sp = (ALPHA_FS**3 / (2 * np.pi**2 * LAMBAR_C)) * (26/315)**2
def tau_split(E_MeV, r_em):
    r = np.linspace(r_em, 30 * R_ns, 4000)
    b = np.minimum(B_s * (R_ns / r)**3 / B_c, 1.0)
    return np.trapezoid(pref_sp * b**6 * (E_MeV / 0.511)**5, r)
print("\nM8 — M-1 : scission vs paires (+ theoreme trame/temperature)")
E_grid = np.logspace(np.log10(0.005), np.log10(1.0), 400)
taus = np.array([tau_split(E, R_ns) for E in E_grid])
E_esc = E_grid[np.argmax(taus > 1)]
print(f"   energie d'echappement de la scission (mode PERP, surface) :"
      f" {1e3*E_esc:.0f} keV\n   (observee : x{0.810:.3f} -> "
      f"{1e3*E_esc*0.810:.0f} keV)")
b_s = B_s / B_c
E_perp_pair = 0.511 * (1 + np.sqrt(1 + 2 * b_s))
print(f"   seuil de paires du mode PERP (supercritique) : "
      f"{E_perp_pair:.2f} MeV -> JAMAIS atteint (scission d'abord)")
print("""   ARCHITECTURE SPECTRALE COMPLETE (les responsabilites assignees) :
     < ~25 keV      : les deux modes s'echappent (polarisation moderee)
     ~25 keV-0,83 MeV : le PERP se scinde -> flux sortant ~PUR MODE PARA
                        (saut de polarisation vers ~100 % au-dessus de
                        ~20-30 keV observes — cible XL-Calibur 15-80 keV ;
                        IXPE 2-8 keV voit la montee) [falsifiable]
     0,83 MeV       : MUR de paires du PARA (M7) — le bord franc
     -> la cassure 300-700 keV n'est NI scission NI paires : elle est
        INTRINSEQUE (distribution des electrons Compton) — M-3 raffine.""")
# --- le theoreme trame/temperature applique (brouillon effets dynamiques §1)
kT_surf = 0.43  # keV (composante thermique quiescente de 1E 1547)
print(f"""   THEOREME trame<->temperature (Phi = T_trame x DeltaTheta_contenu) :
   - canaux (TRAME, aveugles a T) : T_scission(E), T_paires(E) — calcules
     ci-dessus, aucun ne depend de la temperature ;
   - remplissage (CONTENU) : surface thermique kT = {kT_surf} keV (pic
     ~{2.82*kT_surf:.1f} keV) -> graines Compton-resonantes -> la queue ;
   spectre observe = [T_sp x T_pair](E) x [Compton x Theta(w,T_surf)] :
   le magnetar EST la factorisation du theoreme, appliquee.""")
# contact exact n°1 (Unruh) evalue : primaires du gap
a_gap = E_CH * 8.5e12 / ME
gam, rho_c = 3e7, 3.4e6
a_curv = gam**2 * C**2 / rho_c
for nom, a_ in [("gap (eE/m)", a_gap), ("courbure (gam^2 c^2/rho)", a_curv)]:
    T_U = HBAR * a_ / (2 * np.pi * C * 1.380649e-23)
    print(f"   contact Unruh — {nom:24s} a = {a_:.1e} m/s^2 -> "
          f"T_U = {T_U:.1e} K (spectateur)")

# ---- M9 : la boucle Seebeck du magnetar (interjection utilisateur) -------
# La BATTERIE THERMOELECTRIQUE de la croute (Blandford-Applegate-
# Hernquist 1983) : dans la croute degeneree, S = (pi^2/3)(k_B/e)(k_BT/E_F)
# (Mott — la MEME formule que le brouillon effets dynamiques, cote
# contenu) ; quand grad(S) x grad(T) != 0 (S varie avec la densite en
# profondeur, T varie lateralement), la f.e.m. de boucle est non nulle
# et AMPLIFIE B : un effet Seebeck comme GENERATEUR possible du champ.
KB_ = 1.380649e-23
E_F_crust = 25e6 * E_CH     # electrons degeneres relativistes (croute ext.)
T_crust = 3e8               # K
S_crust = (np.pi**2 / 3) * (KB_ / E_CH) * (KB_ * T_crust / E_F_crust)
dT_lat = 1e7                # K (variation laterale)
emf = S_crust * dT_lat
print(f"""
M9 — la boucle Seebeck du magnetar (croute) :
   S_croute = (pi^2/3)(k_B/e)(k_BT/E_F) = {1e6*S_crust:.2f} uV/K
   (E_F = 25 MeV, T = 3e8 K — pur CONTENU : k_B, e, E_F seuls)
   f.e.m. de boucle ~ S x dT_lateral ~ {emf:.1f} V ; croissance de B par
   l'instabilite thermoelectrique (BAH 1983 : tau ~ 1e4-1e5 ans, champs
   jusqu'a ~1e12-1e13 G en enveloppe — origine magnetar : OUVERTE).
   LA BOUCLE COMPLETE (contenu -> trame -> contenu) :
     Seebeck de croute (contenu, ligne E de l'inventaire)
       -> genere/amplifie B
       -> B > B_c OUVRE le passage vers la nucléation (trame, M1)
       -> paires (matiere nouvelle, M4-M5)
       -> chauffage/courants -> grad(T) : la boucle se referme.
   Le critere des constantes TIENT : S est du contenu ; c'est son PRODUIT
   (B) qui devient parametre du canal de trame — les deux secteurs
   communiquent par leurs sorties, jamais par leurs formules.""")

# ---- M10 : M-4 — la batterie Seebeck quantitative ------------------------
# (a) taux de construction (terme batterie, Faraday) :
#     dB/dt ~ S dT_lat / L^2 ; temps pour atteindre une cible B ;
# (b) SATURATION par magnetisation : omega_B tau = 1 -> B_sat = n e/sigma
#     (au-dela, le champ domine le transport et coupe sa propre source) ;
# (c) la course contre le refroidissement (la source dT meurt en
#     1e5-1e6 ans) ; (d) l'AGE caracteristique de 1E 1547 vs le temps de
#     construction : le verdict.
S_th, dT_lat, L_c = 3e-7, 1e7, 100.0        # V/K, K, m
dBdt = S_th * dT_lat / L_c**2
print(f"""
M10 — M-4 : la batterie Seebeck quantitative
   (a) taux batterie dB/dt ~ S dT/L^2 = {dBdt:.1e} T/s""")
for B_cible, nom in [(1e8, "1e12 G (pulsar)"), (4.41e9, "B_c"),
                     (3.2e10, "B(1E 1547)")]:
    t_b = B_cible / dBdt / 3.156e7
    print(f"       -> {nom:18s} : t ~ {t_b:.1e} ans")
print("   (b) saturation omega_B.tau = 1 -> B_sat = n_e e/sigma :")
for rho, n_e, sig_cgs in [(1e9, 2.4e38, 1e24), (1e10, 2.4e39, 3e24)]:
    B_sat = n_e * E_CH / (sig_cgs / 8.99e9)
    print(f"       rho = {rho:.0e} g/cc : B_sat = {B_sat:.1e} T "
          f"= {B_sat*1e4:.0e} G = {B_sat/B_c:.3f} B_c")
print("""       -> la couche generatrice sature a ~1e9-1e10 G ; l'assemblage
       grande echelle (litterature Urpin+) plafonne a ~1e12-1e13 G :
       LE PLATEAU DE LA BATTERIE EST SOUS-CRITIQUE (0,02-0,3 B_c).""")
age_c = P / (2 * Pdot) / 3.156e7
print(f"""   (c) la course : construction >~ 1e5-1e6 ans vs refroidissement de
       la source dT ~ 1e5-1e6 ans : fenetre MARGINALE meme pour 1e12 G.
   (d) l'age caracteristique de 1E 1547 : P/2Pdot = {age_c:.0f} ans
       << temps de construction (>= 1e5 ans) -> LE CHAMP PRECEDE l'etoile
       visible : la batterie est EXCLUE comme source du champ magnetar
       (argument de chronologie, independant de la saturation).
   VERDICT M-4 : la boucle Seebeck plafonne vers 1e12-1e13 G — l'echelle
   des PULSARS ordinaires (~0,02 B_c : sous-critique, coherent avec la
   dichotomie de population pulsars/magnetars) ; le passage supercritique vers la nucléation
   exige l'evenement de NAISSANCE (dynamo convective proto-etoile
   Duncan-Thompson, ou flux fossile). Le contenu sait fabriquer des
   champs sous-critiques ; ouvrir le passage vers la nucléation demande la naissance.""")

# ---- M11 : Seebeck declencheur ? chaine electrofaible ? ------------------
# Question (02/09) : le Seebeck declenche-t-il une reaction en chaine de
# production d'electrons depuis les noeuds "neutroniques" par la voie
# ELECTROFAIBLE (P15, reconnexion Delta Lk — pas d'enlacement complexe) ?
Q_beta = 0.782e6          # eV (n -> p + e + nubar)
emf_seebeck = 2.9         # V (M9)
E_F_crust_eV = 25e6       # eV (mur de Pauli de la mer d'electrons)
print(f"""
M11 — Seebeck déclencheur ? chaîne électrofaible ?
   (a) DÉCLENCHEUR : f.e.m. Seebeck ~ {emf_seebeck:.1f} V vs Q_beta =
       {Q_beta/1e6:.3f} MeV -> rapport {emf_seebeck/Q_beta:.1e} : le
       Seebeck ne peut PAS débloquer une réaction faible (et M-4 :
       chronologie exclusive). DÉCLENCHEUR : NON.
   (b) LA VOIE P15 DANS LA CROÛTE — le mur de Pauli : produire un
       électron par reconnexion (n -> p+e, Delta Lk=+1, PAS d'enlacement
       complexe : l'électron naît avec le nucléon pour partenaire
       topologique, pas de positron requis) exige de le placer AU-DESSUS
       de la mer : coût effectif ~ E_F = {E_F_crust_eV/1e6:.0f} MeV/électron.
   (c) L'ÉCONOMIE DES DEUX VOIES (l'inversion instructive) :
       - voie P15 (contenu, croûte)   : ~{E_F_crust_eV/1e6:.0f} MeV/électron
       - voie P22 (trame, magnétosph.): 1,022 MeV/paire = 0,51 MeV/électron
         (passage vers la nucléation OUVERT dès ~140 keV, M1)
       -> rapport ~{E_F_crust_eV/0.511e6:.0f} : LE MAGNÉTAR CHOISIT LA TRAME —
       ses électrons naissent au passage vers la nucléation (magnétosphère), pas par la
       reconnexion de croûte (le blocage de Pauli renchérit la voie
       électrofaible d'un facteur ~50).""")
# criticite de la chaine : k_eff
f_heat = 0.3      # fraction d'énergie faible déposée (le reste : neutrinos)
eff_batt = 1e-6   # efficacité chaleur -> Seebeck -> B -> contrainte -> réaction
k_eff = f_heat * eff_batt
E_res = 2.9e39    # J (réservoir magnétique, M6)
L_x = 1e28        # W (persistant)
print(f"""   (d) CRITICITÉ : k_eff ~ f_chaleur x eff(chaleur->B->contrainte)
       ~ {f_heat} x {eff_batt:.0e} ~ {k_eff:.0e} << 1 : chaîne SOUS-CRITIQUE.
       Deux barres de contrôle STRUCTURELLES :
       - les NEUTRINOS (ondes chi libres, P14) emportent ~50-90 % de
         chaque réaction hors de la boucle — le poison de la pile ;
       - le BLOCAGE DE PAULI : l'équilibre bêta est un ATTRACTEUR
         (chaque capture restaure mu — rétroaction négative).
   (e) CE QUI TOURNE VRAIMENT : la « pile » du champ de NAISSANCE :
       réservoir {E_res:.1e} J / L ~ {L_x:.0e} W -> durée
       {E_res/L_x/3.156e7:.0e} ans = LA durée de vie active observée des
       magnétars (cohérence de population). Le magnétar brûle son champ
       de naissance ; Seebeck et faible sont des rouages, pas le moteur.""")

# ---- M12 : M-2 — le flux de la raie 511 keV ------------------------------
d_cm = 4.5 * 3.086e21
A_sph = 4 * np.pi * d_cm**2
Ndot_pairs = 2e35
zsurf = 0.810
print(f"""
M12 — M-2 : la raie d'annihilation (d = 4,5 kpc)
   PERSISTANT : Ndot ~ {Ndot_pairs:.0e} paires/s ; si fraction f s'annihile
   -> F = f x {Ndot_pairs/A_sph:.1e} ph/cm2/s : même f=1 est ~4-5 ordres
   sous SPI (~3e-5) et COSI (~2e-6) : INDÉTECTABLE en persistant.
   TRANSITOIRE (les sursauts, plasmas piégés M5) :""")
for E_erg, dt, nom in [(1e40, 86400.0, "tempête 2009 (f_mince~0,1, 1 j)"),
                       (1e43, 3600.0, "éruption intermédiaire (1 h)")]:
    Npairs = E_erg * 1e-7 / (2 * ME * C**2)
    F = 0.1 * Npairs / dt / A_sph
    print(f"   {nom:36s} F_511 ~ {F:.1e} ph/cm2/s "
          f"({'DÉTECTABLE COSI' if F > 2e-6 else 'sous seuil'})")
print(f"""   LA SIGNATURE FINE : l'énergie de la raie est un ALTIMÈTRE (comme le
   mur M-3) — annihilation en surface : E = 511 x {zsurf:.3f} =
   {511*zsurf:.0f} keV (redshiftée) ; magnétosphérique : ~511 keV.
   FALSIFIABLE : une raie à ~414 keV dans l'afterglow d'un sursaut de
   1E 1547 signerait l'annihilation de surface (et mesurerait M/R) ;
   cible COSI (lancement ~2027) en ToO sur sursaut. Réserve déclarée :
   dans le plasma piégé OPTIQUEMENT ÉPAIS la raie est thermalisée —
   seule la fraction mince (~10 %) la produit.""")

# ---- M13 : section N — la question de la transmutation (3_1 -> 7_2 ?) ----
# Hypothese utilisateur (02/09) : des electrons tordus deviendraient des
# 7_2 -> neutrons authentiques ou pseudo-neutrons (faible + gravite) ?
SIGMA_CHI_N = 1.42e5        # N (tension du tube, 0,887 GeV/fm)
D_tube = 0.0214e-15         # m
P_tube = SIGMA_CHI_N / D_tube**2
P_core = 1.6e34             # Pa (~100 MeV/fm^3, coeur d'etoile a neutrons)
m_GeV = 1.783e-27           # kg
V_grav = 6.6710e-11 * m_GeV**2 / 1e-15 / E_CH
print(f"""
M13 — section N : la transmutation 3_1 -> 7_2 ?
   (a) White : Lk = Tw + Wr — la TORSION redistribue a topologie FIXEE ;
       changer 3_1 (c=3) en 7_2 (c=7) exige ~4 reconnexions P15.
   (b) triple barrage : cout 0,511 -> 605 MeV (7_2 libre) ; suppression
       Vassiliev alpha^4 ~ {(1/137)**4:.1e}/tentative ; produit mort-ne
       (7_2 libre : tau = 0,36 ns, denouage u=1).
   (c) neutrons AUTHENTIQUES : oui, voie hadronique (paire miroir
       7_2+7_2bar imposee par Sigma-chi, assemblage Lk=+2) — spallation
       calotte + photonucleaire des eruptions ; jamais par torsion.
   (d) pseudo-neutron (non enlace, faible+gravite) : la faible = des
       EVENEMENTS, pas un potentiel ; gravite au fm : V ~ Gm^2/r =
       {V_grav:.1e} eV (~36 ordres sous le MeV) ; composants instables.
   (e) LA PRESSION AU COEUR — le nombre remarquable :
       P_coeur ~ {P_core:.1e} Pa vs bareme du tube sigma_chi/D^2 =
       {P_tube:.1e} Pa -> rapport {P_core/P_tube:.1e} : la gravite
       approche a 1e-4 du seuil topologique AU COEUR (son maximum dans
       l'univers) et reste 4 ordres SOUS -> les nucleons restent
       topologiquement individuels meme au coeur [postdiction] ;
       au-dela de ~3e38 Pa : territoire Classe C (etoiles hybrides).
   Division du travail : la gravite lie l'ETOILE, l'enlacement lie le
   NUCLEON — le pseudo-neutron demanderait a la gravite le travail de
   l'enlacement 4 ordres trop tot.""")
