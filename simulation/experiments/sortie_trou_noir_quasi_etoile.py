"""Quasi-étoile = sortie de trou noir TQNT ? Expulsion au compte-gouttes, ionisation, « nouvelle trame » (16/09/2026).

Proposition : le trou noir TQNT (nœuds gelés, métrique jamais nulle, horizon = dents arrêtées) expulse de la matière
au compte-gouttes ; elle s'agglomère et s'ionise autour d'une trame modifiée par le trou noir ; l'ensemble est la
quasi-étoile (point rouge). On chiffre la seule sortie que le cadre prévoit à l'horizon — P22 en régime ξ ≫ 1
(Hawking/Schwinger unifiés : Γ = Γ_P e^{−2π/ξ}, 4π/ξ = mc²/kT_U) — et on la compare à ce qu'émet un point rouge.
"""
import math
G=6.6710e-11; c=2.998e8; hbar=1.0546e-34; kB=1.381e-23; Msun=1.989e30; me=9.109e-31
lamC = hbar/(me*c)
yr = 3.156e7
print(f"{'M (Msun)':>9} {'κ (m/s²)':>10} {'ξ horizon':>10} {'T_H (K)':>9} {'2π/ξ (e±)':>10} {'L_Hawking (W)':>13} {'L_Edd (W)':>10} {'L_Edd/L_H':>9}")
for M in (1e5, 3e5, 1e6):
    Mk = M*Msun
    kappa = c**4/(4*G*Mk)
    xi = 2*kappa*lamC/c**2
    TH = hbar*c**3/(8*math.pi*G*Mk*kB)
    LH = hbar*c**6/(15360*math.pi*G**2*Mk**2)
    LE = 1.26e31*M
    print(f"{M:9.0e} {kappa:10.1e} {xi:10.1e} {TH:9.1e} {2*math.pi/xi:10.1e} {LH:13.1e} {LE:10.1e} {LE/LH:9.0e}")
print()
print("  → à l'horizon d'un trou noir de 1e5–1e6 Msun, ξ ~ 1e-22 : la nucléation d'électrons est e^(−1e22), NULLE ;")
print("    la seule sortie est le rayonnement de Hawking des quanta sans masse, ~1e-39 W, à T ~ 1e-13 K.")
M = 3e5*Msun
LH = hbar*c**6/(15360*math.pi*G**2*M**2)
t_evap = 5120*math.pi*G**2*M**3/(hbar*c**4)
print(f"    temps d'évaporation : {t_evap/yr:.0e} ans ; masse expulsée en 1 Ga : {LH*1e9*yr/c**2:.0e} kg")
print()
print("=== Ce qu'émet un point rouge ===")
for L_erg in (1e43, 1e44, 1e45):
    L = L_erg*1e-7
    mdot = L/(0.1*c**2)*yr/Msun
    print(f"  L = {L_erg:.0e} erg/s = {L:.0e} W : {L/LH:.0e} × la sortie de Hawking ; en accrétion (η = 0,1) : {mdot:.1e} Msun/an ; Eddington atteint pour M ≥ {L/1.26e31:.0e} Msun")
print("  → la luminosité des points rouges est celle d'un trou noir de 1e5–1e7 Msun ACCRÉTANT près d'Eddington, pas d'une expulsion.")
print()
print("=== La « nouvelle trame » autour du trou noir ===")
print("  dans le cadre, la trame préexiste (h, c, M_Pl) ; un trou noir ne la produit pas, il la DÉPLÈTE : Ω² − 1 = R_S/r (déficit h/2 à l'horizon).")
for r in (1, 3, 10, 100, 1e4):
    z = 1/math.sqrt(1-1/r) - 1 if r > 1 else float('inf')
    print(f"  r = {r:>6.0f} R_S : écrasement de la trame Ω²−1 = {1/r:.1e} ; décalage gravitationnel d'une raie émise là : z_grav = {z:.2e}")
print("  → la zone où la trame est notablement modifiée (≲ 10 R_S ~ 0,06 UA pour 3e5 Msun) est bien plus petite que l'enveloppe")
print("    ionisée d'un point rouge (≳ 100–1000 UA) : l'ionisation n'est pas tenue par la trame, mais éclairée par le disque d'accrétion.")
print()
print("BILAN : l'image « trou noir entouré d'une enveloppe de gaz dense ionisée » est juste — c'est la lecture « black hole star »")
print("des points rouges — mais la source est l'ACCRÉTION (entrée), pas une expulsion (sortie) : le trou noir TQNT de 1e5 Msun")
print("ne rend rien de mesurable (Hawking 1e-39 W, nucléation P22 nulle), et il déplète la trame au lieu d'en produire une.")

print()
print("=== « Le cul du trou noir » : une sortie de type trou blanc ? ===")
print("  dans le cadre : pas de singularité, nœuds gelés, horizon = dents arrêtées (horloges figées) → rien ne ressort classiquement ;")
print("  la trame est UNIQUE (équivalence = unicité de la trame ; seul l'univers entier n'a pas d'extérieur) : pas d'« autre côté »")
print("  qui alimenterait une sortie ; un intérieur répulsif (Garriga-Vilenkin-Zhang) est exclu par le théorème de la rivière (D-6).")
for L_erg in (1e43, 1e45):
    L = L_erg*1e-7
    print(f"  une sortie devrait débiter au moins L/c² = {L/c**2*yr/Msun:.1e} Msun/an d'énergie — d'où viendrait-elle, sans autre côté ni accrétion ?")
print("  signature distinctive d'une sortie : écoulement bleui dominant et masse du moteur DÉCROISSANTE ; d'une accrétion : raies larges")
print("  symétriques/rougies par le puits, masse CROISSANTE (M ≈ 1e5–1e7 Msun à L_Edd). C'est testable sur les spectres NIRSpec.")
