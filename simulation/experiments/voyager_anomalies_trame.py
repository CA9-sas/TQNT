"""Les « anomalies » Voyager (héliopause) face à la TQNT — trois portes fermées.

Question utilisateur (30/07) : la matière noire (réservoir diffus — les
perles n'ont aucune chance de se former localement) ou le frottement de
trame peuvent-ils expliquer le « mur de feu » (plasma VLISM à ~40 000 K)
et l'asymétrie des rayons cosmiques galactiques vus par les Voyager ?

VERDICT (chiffré ci-dessous) : NON, par dérivation — et chaque porte
fermée est une prédiction de protection VÉRIFIÉE par 47 ans de vol :
(1) GRAVITATION DE LA DM : 3e-6 M_Terre dans l'héliosphère →
    a ~ 4e-18 m/s² (l'ex-« anomalie » Pioneer, résolue thermique 2012,
    valait 8,7e-10) ; transit de perle : 1,4e-11/an (jamais) ;
(2) FROTTEMENT DM-BARYONS : l'énergie cinétique du vent DM local vaut
    ×430 le besoin du mur MAIS σ_eq ~ 1e-48 cm² (neutralisation par
    framing, dérivé) → temps de chauffage 3e31 ans : 21 ordres manquants.
    Verrou modèle-indépendant : un couplage suffisant à 0,1/cm³
    crépiterait les détecteurs souterrains (amplification ×1e23) ;
(3) FROTTEMENT DE TRAME : mouvement uniforme = friction EXACTEMENT nulle
    (théorème de protection ; route friction réfutée au critère
    cosmologique — jamais la réintroduire) ; la seule friction de trame
    est UNRUH (accélération propre) : à l'héliopause, même la giration
    des protons (a~5e3 m/s²) ne donne que T_U ~ 2e-17 K — 21 ordres
    sous le mur. Les rayons cosmiques : la DM est NEUTRE (aucune
    déviation de particules chargées hors gravité nulle) — les épisodes
    d'anisotropie de V1 sont corrélés aux chocs solaires (connectivité
    magnétique), même dossier MHD que le mur.
La chaleur du mur = budget du plasma (compression du vent, reconnexion
du champ drapé, ions de reprise) — la trame TRANSMET sans frotter ; ses
deux températures propres (Unruh, Hawking) sont honorées par l'absence.
RÉFUTATIONS OFFERTES : friction DM démontrée au mur → σ_eq tombe ;
traînée de trajectoire propre survivant aux soustractions → protection
du mouvement uniforme tombe.
"""
import numpy as np

AU, G = 1.496e13, 6.674e-8
hbar, c, kB, q, mp = 1.055e-34, 3e8, 1.381e-23, 1.602e-19, 1.673e-27

# (1) gravitation
rho = 0.4 * 1.783e-24
R = 120 * AU
M_dm = rho * 4/3*np.pi*R**3
print(f'(1) DM dans 120 UA : {M_dm:.1e} g = {M_dm/5.97e27:.1e} M_Terre ; '
      f'a = {G*M_dm/R**2*1e-2:.1e} m/s² (Pioneer résolu : 8,7e-10)')
n_p = rho*0.01/(2e-3*1.989e33)
print(f'    transit de perle (f<1 %) : {n_p*2.5e7*np.pi*R**2*3.15e7:.1e}/an')

# (2) frottement DM-baryons
n_b, dT = 0.1, 2.5e4
E_mur = 1.5*n_b*1.381e-16*dT
v = 2.5e7
E_cin = 0.5*rho*v**2
taux = 0.2*1e-48*v
t_ch = E_mur/(n_b*taux*0.5*(2*1.783e-24)*v**2)/3.15e7
print(f'(2) mur : besoin {E_mur:.1e} erg/cm³ ; E_cin DM {E_cin:.1e} (×{E_cin/E_mur:.0f}) ;')
print(f'    σ_eq=1e-48 → chauffage en {t_ch:.1e} ans ({np.log10(t_ch/1.4e10):.0f} ordres manquants)')

# (3) frottement de trame = Unruh seulement
for nom, a in [('décélération héliogaine', 4e5/(30*1.5e11/4e5)),
               ('giration proton (B drapé)', q*1e5*5e-10/mp),
               ('front 0,1 UA', 1e5/(0.1*1.5e11/1e5))]:
    print(f'(3) {nom:26s}: a={a:.1e} m/s² → T_Unruh = {hbar*a/(2*np.pi*c*kB):.1e} K')
print(f'    mur à 4e4 K → 21 ordres manquants ; mouvement uniforme : friction = 0 (théorème)')

# (2bis) le champ magnétique héliosphérique et la DM : neutre → pas de
# Lorentz ; seule prise = polarisabilité α_pol = R_K³ (anneau conducteur)
alpha_pol = 3.8e-4 * 1e-39      # cm³
B = 5e-6                        # G (0,5 nT)
U_eV = 0.5*alpha_pol*B**2/1.602e-12
print(f'(2bis) couplage magnétique du nœud : U = {U_eV:.1e} eV '
      f'(cinétique ~7e2 eV → rapport 4e-45) ; le champ chauffe les baryons')
print('       (reconnexion/turbulence/pickup — le mécanisme standard du mur)')
print('       et IGNORE la DM — d où sa froideur pendant que le plasma bout.')
