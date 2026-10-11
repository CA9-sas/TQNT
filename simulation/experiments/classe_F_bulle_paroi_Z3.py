"""Classe F = le défaut lui-même : la bulle de paroi Z₃ fermée du gel (16/09/2026).
Définition : F est une bulle FERMÉE de paroi Z₃ (le défaut π₀ du condensat χ, parois sur
les surfaces nodales, bordées par les cordes dans les rails), pincée par le réseau au
moment de son annihilation t_ann, et qui s'effondre sous sa propre tension.
Masse fixée par la trame (σ_w) et par t_ann ; pas d'assemblage gravitationnel.
"""
import math
G=6.6710e-11; c=2.998e8; hbar=1.0546e-34; kB=1.381e-23; Msun=1.989e30; UA=1.496e11; pc=3.086e16
eV=1.602e-19; hbarc_eVm=1.9733e-7
Lam=209.2e6   # eV, Λ_χ
xi=3.36e-15   # m, ξ au gel
sig_c = 0.1751e18  # eV² (σ_χ = (418,47 MeV)²)
def E_per_area(eV3): return eV3*eV/hbarc_eVm**2      # eV³ -> J/m²
def E_per_vol(eV4):  return eV4*eV/hbarc_eVm**3      # eV⁴ -> J/m³
sig_L3 = E_per_area(Lam**3)
sig_L4xi = E_per_vol(Lam**4)*xi
print("=== 1. La paroi Z₃ du gel ===")
print(f"  tension : Λ_χ³ = {sig_L3:.2e} J/m²  |  Λ_χ⁴·ξ = {sig_L4xi:.2e} J/m²  (deux estimations, facteur {sig_L4xi/sig_L3:.1f})")
sig = sig_L3
t_sig = c/(2*math.pi*G*sig); R_c = c*t_sig
t_dom = 3*c**3/(32*math.pi*G*sig)
print(f"  échelle gravitationnelle propre de la paroi : t_σ = c/(2πGσ) = {t_sig:.3f} s, R_c = {R_c:.1e} m")
print(f"  domination des parois (ρ_w = σ/ct = ρ_rad) : t_dom = 3c³/(32πGσ) = {t_dom:.1e} s = {t_dom/3.156e7:.1e} ans  [D-2 disait 1,1 an : erreur d'unités, c²/(Gσ) n'est pas un temps]")
print(f"  Zel'dovich : σ < (1 MeV)³ = {E_per_area(1e18):.1e} J/m² pour survivre à t₀ ; ici σ/σ_ZKO = {sig/E_per_area(1e18):.0e} → les parois DOIVENT s'annihiler avant t_dom")
print()
# temps <-> température (radiation, g*=3,36 après e±)
def T_of_t(t): return 1e3/math.sqrt(t/1.31)   # keV
def rho_rad(T_keV):
    T=T_keV*1e3/8.617e-5; return 3.36/2*7.566e-16*T**4/c**2
def rho_DM(T_keV): return 2.3e-27*(T_keV*1e3/2.35e-4)**3
print("=== 2. La bulle fermée à l'annihilation : M_F = 4πσ(β c t_ann)², β = R/ct ===")
print(f"{'t_ann':>9} {'T':>8} {'R=ct (m)':>10} {'M_F (Msun)':>11} {'ρ_w/ρ_rad':>10} {'ρ_w/ρ_DM':>9} {'R_S/R':>8} {'R/R_c':>8}")
for t in (1.6e-4, 1e-2, 2e-2, 1, 1e3, 1e5, 1e6, 3e6, 1e7, 2e7, 3e7, 1e8, 1e9):
    R=c*t; M=4*math.pi*sig*R**2/c**2; T=T_of_t(t)
    rw=sig/(c**3*t); RS=2*G*M/c**2
    print(f"{t:9.1e} {T:6.2e}keV {R:10.1e} {M/Msun:11.1e} {rw/rho_rad(T):10.1e} {rw/rho_DM(T):9.1e} {RS/R:8.1e} {R/R_c:8.1e}")
print()
print("=== 3. La cible Classe F : M_F = 1e5 - 1e6 Msun ===")
for M in (1e5, 3e5, 1e6):
    R=math.sqrt(M*Msun*c**2/(4*math.pi*sig)); t=R/c; T=T_of_t(t)
    dV=sig/R
    fF=sig/(c**3*t)/rho_DM(T)
    print(f"  M_F = {M:.0e} : R = {R:.1e} m = {R/pc:.2f} pc, t_ann = {t:.1e} s = {t/3.156e7:.2f} an, T = {T*1e3:.0f} eV ; "
          f"biais requis ΔV = σ/R = {dV:.1e} J/m³ = ({(dV/E_per_vol(1))**0.25:.1f} eV)⁴ ; f_F = f_fermé × {fF:.1e}")
print("  plafond des graines (16/09) : f_F ≤ 2,8e-5 ; fenêtre Classe F : f_F ∈ [1e-5, 1e-4] → f_fermé ∈ [0,3, 1] suffit.")
print()
print("=== 4. Le sort de la bulle ===")
M=3e5*Msun; R=math.sqrt(M*c**2/(4*math.pi*sig)); RS=2*G*M/c**2
print(f"  3e5 Msun : R_S = {RS:.1e} m = {RS/UA:.3f} UA ; R_bulle/R_S = {R/RS:.0e} ; R_bulle/R_c(GR) = {R/R_c:.0e}")
print("  RG : R ≫ R_c → paroi « supercritique » (gravité répulsive de la tension, Garriga-Vilenkin-Zhang : intérieur inflationnaire, trou de ver)")
print("  TQNT : la trame répond à la DENSITÉ seule (théorème de la rivière), pas à la pression → la paroi attire comme de la matière,")
print("         pas de régime répulsif : la bulle s'effondre en coquille, toute l'énergie passe sous R_S → TROU NOIR de masse 4πσR².")
print("  énergie cinétique de la coquille à R_S : γ = R²/R_S² =", f"{(R/RS)**2:.0e}", "— rien dans le cadre n'arrête la chute (horizon = dents arrêtées)")
print("  la bulle balaie-t-elle les éremnions intérieurs ? réflexion sur une paroi mobile à ~c : E → 4γ²m ≫ Λ_χ dès la première réflexion → ils traversent.")
mDM = rho_DM(T_of_t(R/c))*4/3*math.pi*R**3/Msun
print(f"  éremnions dans la bulle à t_ann : {mDM:.1e} Msun (non balayés, restent diffus) ; F n'est PAS un amas d'éremnions.")
print()
print("=== 5. Signatures ===")
t=R/c; T=T_of_t(t); z=T*1e3/2.35e-4
f_gw = (1/(2*t))/z
print(f"  GW de l'annihilation : f ~ H(t_ann)/(1+z) = {f_gw:.1e} Hz (sous les PTA de 5 ordres) ; Ω_GW ~ ε(ρ_w/ρ_rad)² ~ {(sig/(c**3*t)/rho_rad(T))**2:.0e} : invisible")
print(f"  μ-distorsion : injection ρ_w/ρ_rad = {sig/(c**3*t)/rho_rad(T):.0e} à t = {t:.0e} s (ère μ : 1e6-1e11 s) → μ ~ 1e-7 ≪ 9e-5 FIRAS ✓ (et nulle si l'énergie va en χ)")
print("  BBN : ρ_w/ρ_rad(1 s) =", f"{sig/(c**3*1)/rho_rad(T_of_t(1)):.0e}", "✓")
print("  LISA : fusions de paires F (7-22 mHz), 0,05-12 par mission (déjà chiffré) ; B1938+666 (1e6 Msun, Powell+2025) = un F")
print("  points rouges : F a un HORIZON → la lecture « étoile à matière noire transparente » ne tient plus ; F accrète comme un trou noir de 1e5-1e6 Msun")
