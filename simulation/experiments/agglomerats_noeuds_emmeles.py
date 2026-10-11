"""Un agglomérat de nœuds premiers emmêlés (enlacés) comme cœur sombre ? (16/09/2026)

Idée : non pas empiler (tresse), mais EMMÊLER — des nœuds premiers liés entre eux par enlacement (Lk ≠ 0),
formant un amas topologiquement connexe. Questions :
 (1) combien de nœuds sont enlacés à la naissance, et quelle taille atteignent les amas connexes ?
 (2) un amas sombre enlacé est-il stable (largage P15 faible × Vassiliev) ?
 (3) un amas macroscopique (1e5–1e6 Msun) enlacé : quelle densité, et est-il sous son rayon de Schwarzschild ?
 (4) le réseau de cordes du gel est lui-même un emmêlement géant : peut-il en rester un morceau ?
"""
import math
G=6.6710e-11; c=2.998e8; Msun=1.989e30; GeV_kg=1.783e-27; hbarc=0.1973  # GeV fm
alpha=1/137.036; v=246.22  # GeV
print("=== (1) Enlacement à la naissance ===")
p = 0.0345      # fraction des nœuds du gel en paires nouées enlacées (cascade étendue à l'enlacement de Gauss, V7 §li7-cendres)
M_hor_DM = 2.1e-7*Msun; m_knot = 1.8*GeV_kg
N_hor = M_hor_DM/m_knot
print(f"  probabilité d'enlacement par nœud p ≈ {p} (mesurée) ; nœuds sombres par horizon du gel : {N_hor:.1e}")
print(f"  amas connexes : P(N) ~ p^(N−1) (sous le seuil de percolation, p ≪ 1) → plus grand amas par horizon ≈ ln N_hor/ln(1/p) = {math.log(N_hor)/math.log(1/p):.0f} nœuds")
N_univ = 2.3e-27*4/3*math.pi*(4.4e26)**3/m_knot
print(f"  dans tout l'univers observable ({N_univ:.0e} nœuds) : ≈ {math.log(N_univ)/math.log(1/p):.0f} nœuds, soit ~{math.log(N_univ)/math.log(1/p)*1.8:.0f} GeV")
print("  → les emmêlements nés au gel sont MICROSCOPIQUES (quelques dizaines de nœuds). Percoler exigerait p ≳ 0,2–0,3 par nœud.\n")
print("=== (2) Stabilité d'un amas sombre enlacé (largage d'une composante : P15 faible, Q ~ liaison) ===")
for Q in (0.1,):
    G0 = Q**5/(384*math.pi**3*v**4)/6.582e-25     # s⁻¹ (GeV → s : ħ = 6,582e-25 GeV s)
    for cc in (20, 23, 25, 28, 30):
        tau = 1/(G0*alpha**(cc-18))
        print(f"  composante c = {cc}, Q = {Q*1000:.0f} MeV : τ(largage) = {tau:.1e} s" + ("  (> âge de l'univers)" if tau > 4.4e17 else ""))
print("  → sous faible × Vassiliev, les amas de composantes c ≳ 28 survivraient ; le « théorème d'instabilité des entrelacs noirs »")
print("    (τ ≪ t_U) suppose un largage non supprimé — nuance à consigner, sans effet sur l'abondance (amas nés rares, p^N).\n")
print("=== (3) Un agglomérat macroscopique enlacé ===")
rho = 1.8*GeV_kg/(1e-15)**3
print(f"  s'enlacer exige le contact des tubes : densité ~ 1 nœud/fm³ → ρ ≈ {rho:.1e} kg/m³ (densité nucléaire ×10)")
for M in (3e5, 1e6):
    R = (3*M*Msun/(4*math.pi*rho))**(1/3); RS = 2*G*M*Msun/c**2
    print(f"  M = {M:.0e} Msun : R = {R/1e3:.0f} km, R_S = {RS/1e3:.1e} km → R/R_S = {R/RS:.1e} : TROU NOIR")
Mmax = math.sqrt(3*c**6/(32*math.pi*G**3*rho))/Msun
M=3e5*Msun; RS=2*G*M/c**2; rho_BH=3*M/(4*math.pi*RS**3)
d=(m_knot/rho_BH)**(1/3)
print(f"  pour ÉCHAPPER à l'horizon, 3e5 Msun doit rester sous ρ = {rho_BH:.0e} kg/m³ : espacement des nœuds {d*1e15:.0f} fm, boucles de ~0,6 fm → aucun enlacement possible")
print(f"  masse maximale d'un agglomérat de contact hors de son horizon : M ≈ √(3c⁶/32πG³ρ) = {Mmax:.1f} Msun")
print("  (même échelle que la masse de Chandrasekhar des perles, 0,5 Msun) → 3e5 Msun est 5 ordres au-dessus.")
print("  un amas LÂCHE ne peut pas être enlacé : l'enlacement est topologique, il exige que les boucles se traversent ;")
print("  des boucles macroscopiques pèsent σ_χ × longueur : 3e5 Msun exigerait 4e47 m de tube (D-2).\n")
print("=== (4) Le réseau de cordes du gel, emmêlement géant ===")
xi_GeV = 3.36/hbarc
rho_net = 0.1751/xi_GeV**2          # GeV⁴
rho_rad = 15.2*math.pi**2/30*(0.062)**4
print(f"  ρ_réseau = σ_χ/ξ² = {rho_net:.1e} GeV⁴ ; ρ_rad(T_c) = {rho_rad:.1e} GeV⁴ (rapport {rho_net/rho_rad:.0f})")
print("  il se fragmente en O(1) périodes de corde (~1e-23 s, dynamique de Nambu, théorème de périodicité) : c'est la cascade elle-même.")
print("  un morceau non fragmenté n'a aucune protection topologique globale (réseau Z₃ percé, parois annihilées au gel, D-7).")
print("\nBILAN : les emmêlements de nœuds premiers existent (3,45 % en paires), restent microscopiques (≤ ~60 nœuds),")
print("peuvent être stables si leurs composantes sont complexes, mais un agglomérat de 1e5 Msun serait un trou noir")
print("(contact ⇒ densité nucléaire) : pas de cœur sombre transparent, pas de surdensité primordiale.")
