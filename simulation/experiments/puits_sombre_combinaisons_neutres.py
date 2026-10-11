"""Nœuds agglomérés (non enlacés) dans un puits de matière noire, où se forment des combinaisons neutres (16/09/2026).

Lecture : une zone de matière noire (halo) crée un puits gravitationnel ; les éremnions y sont concentrés ;
des combinaisons neutres (entrelacs Q_phys = 0) s'y forment en libérant leur énergie de liaison Q (photons) —
une CHIMIE SOMBRE qui refroidirait le halo et compacterait son cœur, sans surdensité primordiale.
Test : le taux de combinaison n⟨σv⟩ bat-il le temps dynamique (ou de Hubble) quelque part avant l'horizon ?
Vertex canonique (T-12) : σv = α G_F² Q²/π, × Vassiliev α^((c₁−18)+(c₂−18)) pour deux composantes sombres.
"""
import math
G=6.6710e-11; c=2.998e8; Msun=1.989e30; UA=1.496e11
alpha=1/137.036; v=246220.0; GF=1/(math.sqrt(2)*v**2); hbarc_cm=197.327e-13
m = 1.8*1.783e-27
rho0 = 2.3e-27
H0 = 2.18e-18; Om = 0.315
def sv_cm3s(Q, supp): return alpha*GF**2*Q**2/math.pi*hbarc_cm**2*3e10*supp
print("=== 1. Le puits existe : halos de matière noire ordinaire ===")
print("  découplage cinétique T_kd ≈ 4–5 MeV → plus petits halos ~ 1e-4–1e-3 Msun (formule standard M_kd ∝ T_kd⁻³) ; halos de 1e5–1e6 Msun à z ≈ 20–30.")
print()
print("=== 2. Le taux de combinaison dans ces puits ===")
for z in (50, 30, 20):
    rho = 200*rho0*(1+z)**3; n = rho/m*1e-6
    H = H0*math.sqrt(Om)*(1+z)**1.5
    tdyn = 1/math.sqrt(G*rho)
    print(f"  z = {z} : halo 200×ρ̄ → ρ = {rho:.1e} kg/m³, n = {n:.1e} cm⁻³, H = {H:.1e} s⁻¹, t_dyn = {tdyn:.1e} s")
    for Q in (1.0, 100.0):
        for lab, s in (("sans Vassiliev", 1.0), ("Vassiliev c=20+20", alpha**4), ("Vassiliev c=23+23", alpha**10)):
            G_ = n*sv_cm3s(Q, s)
            print(f"      Q = {Q:5.0f} MeV, {lab:<18} : Γ = {G_:.1e} s⁻¹ ; Γ·t_dyn = {G_*tdyn:.0e}")
print()
print("=== 3. Densité à partir de laquelle la chimie sombre refroidirait (Γ ≥ 1/t_dyn ⇔ ρ ≥ m² G/(σv)²) ===")
for Q in (1.0, 100.0):
    for lab, s in (("sans Vassiliev", 1.0), ("Vassiliev c=20+20", alpha**4)):
        svm = sv_cm3s(Q, s)*1e-6
        rho_c = m**2*G/svm**2
        M = 3e5*Msun; R = (3*M/(4*math.pi*rho_c))**(1/3); RS = 2*G*M/c**2
        verdict = "TROU NOIR avant d'y arriver" if R < RS else f"R = {R/UA:.1e} UA"
        print(f"  Q = {Q:5.0f} MeV, {lab:<18} : ρ_refroidissement = {rho_c:.1e} kg/m³ → 3e5 Msun à cette densité : R = {R:.1e} m, R_S = {RS:.1e} m → {verdict}")
rhoBH = 3*3e5*Msun/(4*math.pi*(2*G*3e5*Msun/c**2)**3)
print(f"  (3e5 Msun franchit son horizon dès ρ = {rhoBH:.0e} kg/m³ ; un cœur de 10 UA : ρ = {3*3e5*Msun/(4*math.pi*(10*UA)**3):.0e} kg/m³)")
print()
print("=== 4. Ce qu'il faudrait ===")
for z in (20,):
    rho = 200*rho0*(1+z)**3; n = rho/m*1e-6; H = H0*math.sqrt(Om)*(1+z)**1.5
    need = H/n
    print(f"  refroidir un halo de z = {z} en un temps de Hubble : σv ≥ H/n = {need:.0e} cm³/s — "
          f"{need/sv_cm3s(100,1):.0e} fois la reconnexion faible non supprimée ; même ordre que l'annihilation forte d'un SIMP")
print()
print("BILAN : le puits existe (halos ordinaires), les combinaisons neutres existent, mais leur formation est une reconnexion")
print("faible : dans un halo, moins d'un événement par particule en 1e16–1e42 temps dynamiques ; la chimie sombre ne devient")
print("efficace qu'à des densités où 3e5 Msun sont déjà sous leur rayon de Schwarzschild. Pas de compaction dissipative.")
