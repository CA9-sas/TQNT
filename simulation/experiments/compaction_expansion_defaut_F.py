"""Le refroidissement par l'expansion : la compaction sans dissipation (16/09/2026).

Objection de l'utilisateur : le refroidissement se fait par extension de la trame —
(1) l'univers s'étend, (2) la trame se tend par gravité (distanciation).
On chiffre ce que cela permet pour le défaut F (Classe F, 1e4-1e6 Msun, R ~ UA) :
  A. l'expansion redshifte les impulsions (v ∝ 1/a) : Q = ρ/v³ conservé — pas de
     dissipation, mais un amas qui DÉCROCHE de l'expansion se virialise par la seule
     gravité (relaxation violente) à R_vir = R_ta/2, ρ_vir ≈ 8 ρ_ta ;
  B. la compacité du défaut F fixe donc son ÉPOQUE de décrochage : ρ_F/8 = ρ(t_ta) ;
  C. en ère de radiation, décrocher exige que la matière noire domine LOCALEMENT
     la radiation : δ_DM ≳ ρ_rad/ρ_DM (le coût réel, remplaçant la « transition ») ;
  D. la trame ne dissipe rien : Unruh nul en chute libre, g de surface → T ~ 1e-20 K.
"""
import math
G = 6.6710e-11; c = 2.998e8; hbar = 1.0546e-34; kB = 1.381e-23
Msun = 1.989e30; UA = 1.496e11; a_rad = 7.566e-16
rho_DM0 = 2.3e-27; T0 = 2.35e-4  # eV (T_CMB)
rho_c0 = 8.5e-27                  # kg/m3 (h=0,674)

def gstar(T_keV):                 # effectif énergie (photons + 3 ν après e±)
    return 3.36 if T_keV < 200 else 10.75

def rho_rad(T_keV):
    T = T_keV * 1e3 / 8.617e-5     # K
    return gstar(T_keV) / 2 * a_rad * T ** 4 / c ** 2

def rho_DM(T_keV):
    return rho_DM0 * (T_keV * 1e3 / T0) ** 3

def t_of_T(T_keV):
    rho = rho_rad(T_keV) + rho_DM(T_keV)
    H = math.sqrt(8 * math.pi * G * rho / 3); return 1 / (2 * H)

def M_DM_causal(T_keV):
    t = t_of_T(T_keV); return rho_DM(T_keV) * 4 / 3 * math.pi * (2 * c * t) ** 3 / Msun

def T_of_rho_rad(rho):            # inverse (g*=3,36)
    T = (2 * rho * c ** 2 / (3.36 * a_rad)) ** 0.25
    return T * 8.617e-5 / 1e3

print("=== A. Liouville sous l'expansion : Q = ρ/v³ ∝ a⁻³/a⁻³ = const — le refroidissement adiabatique ne change pas Q")
print("    Q_F (3e5 Msun, 10 UA) = 1e-22 ≪ Q_naissance (≥ 5e-17) : le défaut F est PERMIS ; son coût est l'énergie, payée par le décrochage.\n")

print("=== B. Compacité ⇒ époque de décrochage (ρ_vir ≈ 8 ρ_ta ; ρ_ta ≈ ρ_rad(T_ta) en ère de radiation) ===")
print(f"{'M_F':>8} {'R_F':>6} {'ρ_F (kg/m3)':>12} {'T_ta':>8} {'t_ta':>9} {'M_DM causale(T_ta)':>19} {'δ_DM requis':>12}")
for MF in (1e4, 3e4, 1e5, 3e5, 1e6):
    for RF in (1, 3, 10, 30, 100):
        rhoF = MF * Msun / (4 / 3 * math.pi * (RF * UA) ** 3)
        Tta = T_of_rho_rad(rhoF / 8)
        Mc = M_DM_causal(Tta)
        delta = rho_rad(Tta) / rho_DM(Tta)
        flag = " <- M_F ≈ M_causale" if 0.3 < MF / Mc < 3 else ""
        print(f"{MF:8.0e} {RF:4d}UA {rhoF:12.2e} {Tta:6.1f}keV {t_of_T(Tta):9.1e} {Mc:19.1e} {delta:12.1e}{flag}")
print()
print("=== C. La condition de décrochage en ère de radiation ===")
for T in (10, 7, 3, 1, 0.1):
    print(f"  T = {T:5.1f} keV : ρ_rad/ρ_DM = {rho_rad(T)/rho_DM(T):.1e}  (δ_DM local requis pour que la gravité sombre batte H)")
print("  sources connues du cadre à 3e5 Msun : Poisson 1/√N = %.0e ; sillage de corde Gμ/c⁴ = %.0e ; adiabatique inflationnaire 1e-5"
      % (1 / math.sqrt(3e5 * Msun / (1.8e9 * 1.783e-30)), G * (0.175 * 1e9 * 1.602e-19 / 0.1973e-15) / c ** 4))
print()
print("=== D. La trame ne refroidit pas : Unruh en chute libre = 0 ; à la surface du défaut ===")
for MF, RF in ((3e5, 10), (3e5, 1), (1e6, 1)):
    g = G * MF * Msun / (RF * UA) ** 2
    TU = hbar * g / (2 * math.pi * c * kB)
    print(f"  M_F = {MF:.0e}, R_F = {RF} UA : g = {g:.1e} m/s², T_Unruh = {TU:.1e} K ; v_vir = {math.sqrt(G*MF*Msun/(RF*UA))/c:.1e} c")
print("  (aucune énergie ne passe des éremnions à la trame : la relaxation violente REDISTRIBUE, elle ne dissipe pas — et n'en a pas besoin)")
