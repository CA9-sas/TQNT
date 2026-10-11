"""Le biais entre les trois vides Z₃ : définition et chiffrage (16/09/2026).
Un réseau de parois biaisé s'annihile quand la pression de biais ΔV dépasse la pression de
tension σ_w/R(t), R ≈ c t. Pour pincer des bulles de Classe F il faut t_ann ≈ 1 an, donc
ΔV(1 an) ≈ σ_w/(c·1 an) ≈ (4 eV)⁴ ET ΔV(t) < σ_w/(ct) à tout instant antérieur (sinon les
parois meurent plus tôt, avec des bulles plus petites). On passe en revue les sources de
biais que le cadre peut offrir, avec leur loi en T.
"""
import math
G=6.6710e-11; c=2.998e8; eV=1.602e-19; hbarc=1.9733e-7   # eV m
J_per_eV4 = eV/hbarc**3                                   # 1 eV⁴ -> J/m³
Lam=209.2e6; MPl=1.22089e28; Tc=62e6; gstar_c=10.75; EZ3=139.49e6   # eV
sig = Lam**3*eV/hbarc**2        # J/m²
tc = 1.6e-4
def T_of_t(t):   # eV ; radiation, g*=3,36 (<200 keV) sinon 10,75
    T=1e6*math.sqrt(1.31/t)
    return T if T<2e5 else 1e6*math.sqrt(0.74/t)
def P_tension(t): return sig/(c*t)                         # J/m³
def eV4(x): return (x/J_per_eV4)**0.25                     # J/m³ -> échelle en eV
t_F = 3.6e7
print("=== 1. Ce qu'il faut ===")
print(f"  σ_w = Λ_χ³ = {sig:.2e} J/m² ; à t_F = {t_F:.1e} s (1,1 an) : σ_w/(c t_F) = {P_tension(t_F):.1e} J/m³ = ({eV4(P_tension(t_F)):.1f} eV)⁴")
print(f"  à t_c = {tc:.1e} s : σ_w/(c t_c) = {P_tension(tc):.1e} J/m³ = ({eV4(P_tension(tc)):.0f} eV)⁴ = ({eV4(P_tension(tc))/1e3:.1f} keV)⁴")
print("  loi admissible : ΔV(t) ∝ t^(-m) avec m < 1 (∝ T^n, n < 2) ; tout biais en T² ou plus raide bat la tension dès t_c.\n")

def t_ann(dV_of_t):
    """premier t > t_c où ΔV(t) ≥ σ/(ct) (balayage log)."""
    t=tc
    while t<1e18:
        if dV_of_t(t) >= P_tension(t): return t
        t*=1.02
    return None

print("=== 2. Les sources thermiques et de densité (tuent les parois au gel) ===")
def rho_rad(t):   # J/m³
    return 3*c**2/(32*math.pi*G*t**2)
n_g0=4.1e8; T0=2.35e-4
def n_B(t): return 6.1e-10*n_g0*(T_of_t(t)/T0)**3
cands = [
  ("ε·ρ_rad (photons χ sensibles au caractère Z₃)", lambda t: 1e-19*rho_rad(t), "ε = 1e-19"),
  ("n_B·E_Z3 (les baryons vivent dans UN vide)", lambda t: n_B(t)*EZ3*eV, "∝ T³"),
  ("n_B·T (entropie de sélection)", lambda t: n_B(t)*T_of_t(t)*eV, "∝ T⁴"),
]
for nom, f, note in cands:
    ta=t_ann(f); r=f(tc)/P_tension(tc)
    print(f"  {nom:<48} {note:>8} : ΔV(t_c)/P_tension(t_c) = {r:.0e} → {'annihilation IMMÉDIATE au gel' if r>1 else ('jamais : parois éternelles' if ta is None else f't_ann = {ta:.1e} s')}")
print("  → pour survivre au gel il faudrait ε < %.0e ; alors ΔV ∝ T⁴ ne rattrape jamais σ/ct (t⁻² contre t⁻¹) : parois éternelles → domination." % (P_tension(tc)/rho_rad(tc)))
print()
print("=== 3. La brisure gravitationnelle (la trame qui répond ne connaît pas la trialité) ===")
for p, nom in ((1,"dimension 5 : Λ_χ⁵/M_Pl"), (2,"dimension 6 : Λ_χ⁶/M_Pl²")):
    dV = Lam**4*(Lam/MPl)**p*J_per_eV4
    ta = sig/(c*dV)
    print(f"  {nom:<26} : ΔV = ({eV4(dV):.3g} eV)⁴ = {dV:.1e} J/m³ → t_ann = σ/(cΔV) = {ta:.1e} s", end="")
    print("  = t_c à %+.0f %% !" % (100*(ta/tc-1)) if abs(math.log10(ta/tc))<0.5 else ("  > t_dom (3e14 s) : DOMINATION" if ta>3e14 else ""))
# la coïncidence dimension 5, en forme close
print("  dimension 5 ⇔ σ/(c t_c) = Λ⁵/M_Pl ; avec σ = Λ³ et t_c = 0,30·M_Pl/(√g*·T_c²) (ère de radiation) : T_c = Λ_χ·√(0,30/√g*)")
print(f"  → T_c prédit par la coïncidence = {Lam*math.sqrt(0.30/math.sqrt(gstar_c))/1e6:.1f} MeV ; T_c dérivé (serrage topologique) = 62 ± 3 MeV ; rapport T_c/Λ_χ = {Tc/Lam:.3f}")
print("  lecture : si la trame brise Z₃ au premier ordre permis, les parois meurent EXACTEMENT quand elles atteignent l'horizon au gel —")
print("            le cadre disait « les parois s'annihilent au gel » ; ce serait la raison. Coïncidence à 2 % ; règle du null : mécanisme non établi.")
p_needed = math.log(P_tension(t_F)/(Lam**4*J_per_eV4))/math.log(Lam/MPl)
print(f"\n  F à 1 an exigerait ΔV = Λ⁴(Λ/M_Pl)^p avec p = {p_needed:.2f} — entre 5 et 6, aucun opérateur ; ×{Lam**4*(Lam/MPl)*J_per_eV4/P_tension(t_F):.0e} sous la dimension 5.")
print()
print("=== 4. Ce que font les bulles du gel (dimension 5) ===")
R=c*tc; M=4*math.pi*sig*R**2/c**2; Msun=1.989e30
Rc=c**2/(2*math.pi*G*sig); RS=2*G*M/c**2
print(f"  R = {R:.1e} m (R/R_c = {R/Rc:.0e} : sous-critique), M = {M:.1e} kg = {M/Msun:.0e} Msun, R_S/R = {RS/R:.0e}")
rho_DM_c = 2.3e-27*(T_of_t(tc)/T0)**3*c**2
print(f"  énergie des parois au gel : ρ_w/ρ_rad = {P_tension(tc)/rho_rad(tc):.0e} ; ρ_w/ρ_DM = {P_tension(tc)/rho_DM_c:.0e} → f_PBH ≤ {P_tension(tc)/rho_DM_c:.0e} : inoffensif quoi qu'il arrive")
print("  (et les bulles sous-critiques non sphériques rayonnent en χ avant R_S : contraction ×1e18 requise).")
print("  t_évap(1e13 kg) ~ 5e15 ans : survivraient ; bornes γ (M < 1e14 kg) imposent f ≲ 1e-3-1e-2.")
