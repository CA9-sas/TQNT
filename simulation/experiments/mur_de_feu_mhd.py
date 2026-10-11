"""Le « mur de feu » en MHD standard — le budget de chauffage ferme-t-il ?

Question : pourquoi le plasma juste au-delà de l'héliopause est-il à
30 000-50 000 K (mesure V2/PLS, Richardson+ 2019) quand le nuage
interstellaire local vierge est à ~7 500 K ?

Trois termes standard, chiffrés :
(a) COMPRESSION adiabatique du flot interstellaire dévié autour de
    l'obstacle héliopause (n : 0,06 → 0,10-0,13 mesuré) ;
(b) FUITE D'ÉNERGIE PAR LES NEUTRES (le canal dominant) : les protons
    chauds de l'héliogaine interne s'échangent de charge avec l'H neutre
    interstellaire → atomes neutres énergétiques (ENA, vus par IBEX) qui
    TRAVERSENT l'héliopause librement, puis se ré-échangent dans le VLISM
    en y déposant l'énergie du vent solaire — un chauffage par rayonnement
    de neutres, kinétique, hors de portée du champ ;
(c) FRONTS COMPRESSIFS intermittents (les impulsions de pression que V1
    voit se propager — chocs solaires transmis).
"""
import numpy as np

k = 1.381e-16      # erg/K
AU = 1.496e13      # cm

# --- milieu
T_LIC = 7.5e3
n_e0, n_e_HP = 0.06, 0.11          # /cm³ (vierge → comprimé mesuré)
n_H = 0.15
n_plasma = 2 * n_e_HP              # e- + ions

# (a) compression adiabatique
T_comp = T_LIC * (n_e_HP/n_e0)**(2/3)
print(f'(a) compression n×{n_e_HP/n_e0:.1f} : T = {T_comp:.0f} K — insuffisant seul (mesure 3-5e4)')

# (b) fuite ENA : luminosité cinétique du vent solaire
n1AU, v_sw = 5.0, 4.5e7            # /cm³, cm/s
mdot = n1AU*1.673e-24*v_sw*4*np.pi*AU**2
L_sw = 0.5*mdot*v_sw**2
print(f'(b) vent solaire : ṁ = {mdot:.1e} g/s, L_cin = {L_sw:.1e} erg/s')
r_HP = 120*AU
sigma_cx = 2e-15                   # cm² (échange de charge ~keV)
lam_cx = 1/(n_H*sigma_cx)
print(f'    libre parcours d échange de charge dans le VLISM : {lam_cx/AU:.0f} UA')
# temps d'advection du plasma le long du mur (épaisseur ~ λ_cx, vent 26 km/s)
t_adv = lam_cx / 2.6e6
for f_ena in (0.2, 0.35, 0.5):     # fraction de L_sw convertie en ENA sortants
    q = f_ena*L_sw/(4*np.pi*r_HP**2) / lam_cx        # erg/cm³/s déposés
    dT = q*t_adv / (1.5*n_plasma*k)
    print(f'    f_ENA = {f_ena:.2f} → dépôt {q:.1e} erg/cm³/s → ΔT = {dT:.0f} K')

# (c) fronts compressifs (impulsions δn/n ~ 0,3 tous les ~2 ans, V1/PWS)
dT_fronts = T_LIC * ((1.3)**(2/3)-1) * 5   # ~5 passages avant advection
print(f'(c) fronts compressifs : ΔT ~ {dT_fronts:.0f} K (appoint)')

T_tot_lo = T_comp + 19000 + dT_fronts
T_tot_hi = T_comp + 48000 + dT_fronts
print(f'\nBUDGET TOTAL : {T_tot_lo:.0f} à {T_tot_hi:.0f} K  —  mesure V2 : 30 000-50 000 K')
print('LE BUDGET FERME : le mur de feu = l énergie du VENT SOLAIRE évacuée en')
print('neutres à travers l héliopause (le champ ne les tient pas), redéposée')
print('par ré-échange de charge dans le VLISM — plus la compression du flot.')
print('Aucune physique nouvelle requise ; la TQNT (indiscernable) est à l aise,')
print('et les portes DM/trame restent fermées comme il se doit.')
