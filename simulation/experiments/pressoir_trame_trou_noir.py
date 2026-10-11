"""Le pressoir : une trame nouvelle, plus déformée, sort des trous noirs ; l'écoulement a sa propre dynamique (16/09/2026).

Paramétrage minimal : un trou noir émet un flux de volume de trame F_V = ε·c·r_s² (échelle naturelle : trame sortant à c
à travers une surface ~ r_s²), éventuellement « déformée » (propriété δ différente de la trame ambiante).
Image de la rivière (Painlevé–Gullstrand) : en RG l'espace s'écoule VERS le trou noir à v_in = √(2GM/r) ; un pressoir
superpose un écoulement SORTANT v_out(r) = F_V/(4π r²). On confronte (1) aux étoiles S autour de Sgr A*, (2) à l'ombre EHT,
(3) à l'expansion cosmique, (4) à l'enveloppe d'un point rouge, (5) à une trame « déformée » portant d'autres constantes.
"""
import math
G=6.6710e-11; c=2.998e8; Msun=1.989e30; AU=1.496e11; Mpc=3.086e22; H0=2.18e-18
def rs(M): return 2*G*M/c**2
print("=== 1. Sgr A* et l'étoile S2 (péricentre 120 UA) ===")
M = 4.3e6*Msun; r = 120*AU
v_in = math.sqrt(2*G*M/r)
for eps in (1, 100, 1e4):
    F = eps*c*rs(M)**2; v_out = F/(4*math.pi*r**2)
    print(f"  ε = {eps:>6g} : F_V = {F:.1e} m³/s, v_out(120 UA) = {v_out:.1e} m/s ; v_in = {v_in:.1e} m/s → δv/v = {v_out/v_in:.0e}")
print("  GRAVITY : masse étendue ≲ 0,1 % dans l'orbite de S2, précession de Schwarzschild f = 1,10 ± 0,19 → δv/v ≲ 5e-4")
print(f"  → ε ≲ {5e-4*v_in*4*math.pi*r**2/(c*rs(M)**2):.0e} : un pressoir à l'échelle naturelle (ε ~ 1) est INVISIBLE pour S2.\n")
print("=== 2. L'ombre EHT (M87*, Sgr A*) ===")
for eps in (0.1, 1, 10):
    v_h = eps*c/(4*math.pi)
    print(f"  ε = {eps:>4g} : écoulement sortant à l'horizon v_out = ε c/4π = {v_h/c:.2f} c → déformation de l'anneau de photons ~{100*v_h/c:.0f} %")
print("  EHT : ombre conforme à la RG à ~10 % → ε ≲ 1. La borne la plus serrée est au bord du trou noir, pas au loin.\n")
print("=== 3. L'expansion cosmique ===")
rho_BH = 5e5; M_char = 1e8     # Msun/Mpc³ (Soltan), masse caractéristique des trous noirs supermassifs
sumM2 = rho_BH*M_char*Msun**2
F_tot = c*(2*G/c**2)**2*sumM2/Mpc**3
print(f"  Σ c r_s² par volume = {F_tot:.1e} s⁻¹ (ε = 1) contre 3H₀ = {3*H0:.1e} s⁻¹ → fraction {F_tot/(3*H0):.0e}")
print("  → la trame pressée par tous les trous noirs de l'univers pèse 1e-21 de l'expansion : elle ne source ni l'expansion ni l'énergie noire.")
print("  (l'hypothèse « trous noirs couplés à la cosmologie » exigerait Ω_BH ≈ 0,7 contre Ω_BH ~ 4e-6 observé, et est contestée par ailleurs)\n")
print("=== 4. L'enveloppe d'un point rouge (3e5 Msun, 100–1000 UA) ===")
M = 3e5*Msun
for R_AU in (100, 1000):
    for eps in (1, 1e4):
        v_out = eps*c*rs(M)**2/(4*math.pi*(R_AU*AU)**2)
        print(f"  {R_AU:>4} UA, ε = {eps:>5g} : v_out = {v_out:.1e} m/s (vitesses du gaz des points rouges : ~1e6 m/s)")
print("  → le jus du pressoir ne remue pas l'enveloppe : sa dynamique reste celle de l'accrétion et de la gravité.\n")
print("=== 5. Une trame « plus déformée » : d'autres constantes dans le jus ? ===")
print("  dans le cadre, α = 1/(k+2) avec k ENTIER et h = la dent sont verrouillés : une trame déformée ne peut différer que par son")
print("  amplitude (écrasement, donc G_effectif). Dilution en r⁻² depuis l'horizon : δ(r) = δ_h (r_s/r)².")
M = 4.3e6*Msun; r = 120*AU
print(f"  à S2 : (r_s/r)² = {(rs(M)/r)**2:.1e} ; décalage gravitationnel de S2 conforme à la RG à ~5 % → δ_h ≲ {0.05/(rs(M)/r)**2:.0e} : aucune borne utile")
print("  naines blanches (Δα/α ≲ 1e-4 à leur surface), étoiles S (Δα/α ≲ 1e-5) : bornes seulement si la déformation touchait α — interdit par le verrou k.")
print()
print("BILAN : un pressoir de trame à l'échelle naturelle (ε ≲ 1, sortie ≤ c) est compatible avec S2, l'EHT et la cosmologie,")
print("mais il ne change rien de mesurable loin de l'horizon : ni l'expansion (1e-21), ni l'enveloppe d'un point rouge (≤ 0,1 m/s).")
print("Ses effets sont confinés à quelques r_s — la région des rayons X. Pour qu'il compte, il faut ε ≫ 1, que l'EHT borne déjà à ~1.")
