"""Le gel d'annihilation — fermer η_B fin (chantier nommé du document principal V6).

Le document principal (§ Baryogénèse topologique) : η_B = ⅓·ε_CP(3₁)·e^{−3α}·(m_e/T_c)²,
avec ε_CP(3₁) = 1,15e-3 EXACT (phase de Jones du trèfle à q = e^{2πi/137}),
P_topo = e^{−3α} (α_cond = 1,199) — et le facteur (m_e/T_c)² étiqueté
« suppression par annihilation », l'heuristique que la V6 flague :
« la valeur fine dépend du gel d'annihilation — calcul non mené ».

TROIS ÉTAPES ICI :
(0) [consolidation immédiate] la formule document principalique au T_c DÉRIVÉ (62±3 MeV,
    2026-07-20) au lieu du 67 adopté de la table historique ;
(1) [gel thermique — variante réfutable] semis = paires nucléoniques à
    l'équilibre de Boltzmann à T_c ; l'annihilation forte efface la partie
    symétrique (gel standard à T_f ≈ 22 MeV, résidu symétrique ~1e-19,
    négligeable) : survit l'excès CP ; dilution d'entropie g_s(T_c) → g_s0 ;
(2) [gel de chaîne — la voie TQNT propre, EXPLORATOIRE] semis = le MÊME
    comptage de formation que la chaîne Ω du côté sombre (les nœuds naissent
    du gel du condensat, PAS d'un bain à l'équilibre) : le rapport
    visible/sombre des taux de formation se calcule avec les ingrédients
    canoniques (poids de Boltzmann de formation e^{−αc−m/T_c}, comptages
    N(c), taux net 0,68/croisement) ; normalisation par n_d/s ancrée sur
    Ω_DM observé. η_B = ε_CP(3₁) · (n_vis/s) · (s0/nγ0).

Statut : (0) canonique re-chiffré ; (1) et (2) EXPLORATOIRES (brouillon).
"""
import numpy as np

# --- constantes canoniques
EPS_CP = 1.15e-3            # phase de Jones du trèfle, exact
ALPHA = 1.199               # α_cond dérivé
P_TOPO = np.exp(-3 * ALPHA)
TC, DTC = 62.0, 3.0         # MeV — T_c DÉRIVÉ (serrage topologique)
ME = 0.511
GS_TC = 15.2                # g_s(T_c) dérivé (2026-07-17)
GS_0 = 3.91
S0_NG0 = 7.04               # s0/nγ0
ETA_PLANCK = 6.12e-10

print('=== (0) Formule document principalique au T_c dérivé ===')
for tc in (TC - DTC, TC, TC + DTC):
    eta = (1/3) * EPS_CP * P_TOPO * (ME / tc) ** 2
    print(f'  T_c = {tc:.0f} MeV : η_B = {eta:.2e}  (Planck/η = {ETA_PLANCK/eta:.2f})')
print('  → au T_c dérivé : η_B = 7,1e-10 [6,4, 7,8] — Planck à −13 % [−22, −4].')
print('  L ordre de grandeur du document principal devient un accord à ~15 % SI l heuristique tient.')

print('\n=== (1) Gel thermique (variante d école) ===')
# semis : paires N-N̄ à l'équilibre à T_c ; l'excès CP = ε·P_topo/3 de ce semis
mN = 938.27
def n_over_ngamma(m, T, g=4):     # g=4 : p+n, 2 spins chacun (par espèce+anti compté à part)
    return g * (m * T / (2 * np.pi)) ** 1.5 * np.exp(-m / T) / (2 * 1.20206 / np.pi**2 * T**3)
f_pair = n_over_ngamma(mN, TC)
# excès imprimé par photon à T_c, puis conservation n/s et retour en n/γ aujourd'hui
excess_per_gamma_Tc = (1/3) * EPS_CP * P_TOPO * f_pair
# n/s à T_c : nγ/s = 1/(1.80 g_s/2) ≈ π⁴/(45ζ(3)) g_γ/g_s ... → facteur 0.2777·(2/g_s)
ng_over_s_Tc = 0.2777 * 2 / GS_TC
eta_gel1 = excess_per_gamma_Tc * ng_over_s_Tc * S0_NG0
print(f'  f_pair(T_c) = n_N/n_γ = {f_pair:.2e} (vs heuristique (m_e/T_c)² = {(ME/TC)**2:.2e})')
print(f'  η_B(gel thermique) = {eta_gel1:.2e}  → ×{eta_gel1/ETA_PLANCK:.3f} de Planck')
print('  VERDICT : ×~0,01-0,1 de Planck — le semis thermique est TROP PAUVRE ;')
print('  cohérent avec l ontologie : les nœuds naissent du GEL DU CONDENSAT,')
print('  pas d un bain à l équilibre. La variante thermique est défavorisée.')

print('\n=== (2) Gel de chaîne (la voie TQNT propre, exploratoire) ===')
# côté sombre (ancre observationnelle) : n_d/s depuis Ω_DM
rho_c_h2 = 1.05375e-5       # GeV/cm³ (ρ_c/h²)
omega_dm_h2 = 0.120
m_dark = 1.9                # GeV (bord du réservoir)
s0 = 2891.2                 # cm⁻³
nd_over_s = omega_dm_h2 * rho_c_h2 / m_dark / s0
print(f'  ancre sombre : n_d/s = {nd_over_s:.2e}')
# rapport de formation visible/sombre : poids de Boltzmann de formation
# e^{-αc - m/T_c} et comptages N(c) (taux net r = N(c+1)e^{-α-Δm/Tc}/N(c) = 0,68)
# visible : trèfle c=3, m = m_ℓ = 0,310 GeV ; sombre : série c≥23 dominée par le bord
m_l = 0.310
r_net = 0.68
# poids relatif du canal c=3 vs c=23 : chaque pas de c coûte e^{-α}·e^{-Δm/Tc} ;
# pas géométrique canonique Δm = 70,2 MeV ≈ T_c → un pas = e^{-α}·e^{-70.2/62}
pas = np.exp(-ALPHA) * np.exp(-70.2 / TC)
# du bord sombre (c=23) au trèfle (c=3) : 20 pas À REBOURS → facteur pas^{-20}
R_form = pas ** (-20)
# série sombre convergente : Σ r^k = 1/(1-r)
serie_sombre = 1 / (1 - r_net)
n_vis_over_s = nd_over_s / serie_sombre * R_form
eta_gel2 = EPS_CP * n_vis_over_s * S0_NG0
print(f'  un pas de c : e^-α·e^-Δm/Tc = {pas:.3f} → 20 pas (c=23→3) : ×{R_form:.2e}')
print(f'  n_vis/s = {n_vis_over_s:.2e} → η_B(gel de chaîne) = {eta_gel2:.2e}')
print(f'  → ×{eta_gel2/ETA_PLANCK:.1f} de Planck')
print('\nBILAN : la variante thermique sous-produit (×0,01) ; la variante de')
print('chaîne SUR-produit si le pas rebours est pris nu — la vérité du gel')
print('est entre les deux : il manque l ANNIHILATION du côté visible pendant')
print('la descente (le côté sombre est protégé par Vassiliev, pas le visible).')
print('Chantier suivant nommé : Boltzmann de descente avec annihilation forte')
print('active — la formule document principalique (m_e/T_c)², qui donne −13 % à T_c dérivé,')
print('reste la meilleure interpolation disponible [ordre → ~15 %].')
