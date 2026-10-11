"""Des noyaux d'hydrogène qui se collent sur les tores, pression croissante, fusion (16/09/2026).
Proposition : les protons se fixent sur les (pseudo-)tores d'électrons ; la pression monte
jusqu'à la fusion ; la phase transitoire serait un point rouge du JWST.
Ce qui colle un proton à un tore dépend de la CHARGE du tore. Trois lectures :
  (N) tore neutralisé par framing (réservoir canonique, Q=0) ;
  (M) tore millichargé (réalisables : Q_phys résiduel ~ ±0,03-0,05) ;
  (C) tore chargé Q = −n (n électrons empilés, non neutralisé) — la lecture « tore d'électrons ».
Pour chacune : énergie de liaison du proton, rayon, époque de capture, fusion catalysée, bornes.
Puis la lecture collective (population de tores = halo) et le calendrier JWST.
"""
import math
alpha=1/137.036; hbarc=197.327   # MeV fm
mp=938.272; ML=310.0; slope=70.1
def M_T2(q): return ML+slope*(q-3)
M=M_T2(23); n_tr=11
mu=mp*M/(mp+M)
G=6.6710e-11; c=2.998e8; MeV=1.602e-13; fm=1e-15; GeV_kg=1.783e-27
print(f"tore T(2,23) : M = {M:.0f} MeV, {n_tr} trèfles ; masse réduite proton–tore μ = {mu:.0f} MeV ; tube L = M/σ_χ = {M/175.1*hbarc/1000:.2f} fm")
print()
print("=== (N) tore neutre (réservoir canonique) ===")
r=1.0  # fm
Eg=G*(M*GeV_kg/1000)*(mp*GeV_kg/1000)/(r*fm)/MeV
apol=3.8e-4  # fm³
Ep=alpha*hbarc*apol/(2*r**4)
print(f"  gravité à 1 fm : {Eg:.1e} MeV ; polarisabilité (α_pol = 3,8e-4 fm³) à 1 fm : {Ep:.1e} MeV contre énergie de localisation ħ²/(2μr²) = {hbarc**2/(2*mu):.0f} MeV ; σ ~ 1e-48 cm²")
print("  → aucun état lié : le proton ne « colle » pas à un tore neutre. Rien ne se fixe, rien ne se comprime.")
print()
print("=== (M) tore millichargé ===")
for eps in (0.03, 0.05):
    EB=(eps*alpha)**2*mu/2; a=hbarc/(eps*alpha*mu)
    print(f"  ε = {eps} : E_B = {EB*1e6:.0f} eV, rayon {a*1e-5:.1e} Å ({a/52918:.0f} a₀) → capture vers T ~ E_B/40 = {EB*1e6/40:.1f} eV (z ~ {EB*1e6/40/2.35e-4:.0f})")
print("  → un « atome » 60-100 fois plus petit que l'hydrogène mais 200-400 fois la portée nucléaire, UN seul proton par charge fractionnaire : aucune compression, aucune fusion.")
print("  → et exclu : millicharge de vertex +21 dex (brouillon σ_EM neutre) ; CMB : DM millichargée ε ≳ 1e-6 à 2 GeV exclue si f ≳ 0,4 %.")
print()
print("=== (C) tore chargé Q = −n (n = 11) ===")
Z=n_tr
EB=(Z*alpha)**2*mu/2; a=hbarc/(Z*alpha*mu)
print(f"  1er proton : E_B = Z²α²μ/2 = {EB:.2f} MeV, rayon de Bohr = {a:.1f} fm (tube {M/175.1*hbarc/1000:.1f} fm) : le proton est à portée nucléaire du tore")
print(f"  capture quand T ~ E_B/30 = {EB/30*1e3:.0f} keV → t ~ {1.31*(1/(EB/30))**2:.0f} s : EN PLEINE NUCLÉOSYNTHÈSE (70 keV)")
print(f"  jusqu'à {Z} protons à ~{a:.0f} fm : répulsion mutuelle écrantée par la charge −{Z} → écartement nucléaire SANS barrière coulombienne")
Q=0.42; tau_n=879.4; Qn=0.782
tau_pp=tau_n*(Qn/Q)**5
print(f"  p+p → d e⁺ ν au contact : taux faible ~ τ_n (Q_n/Q_pp)⁵ = {tau_pp:.0f} s ≈ {tau_pp/3600:.1f} h ; puis d+p → ³He (γ), ³He+³He → ⁴He : RAPIDES")
print(f"  → c'est une FUSION CATALYSÉE (l'analogue lourd de la fusion catalysée par muons) : oui, le mécanisme décrit existe sous (C).")
print()
Omega_ratio=5.36
n_ratio=Omega_ratio*mp/M
print(f"  combien de catalyseurs ? n_tores/n_p = (Ω_DM/Ω_b)(m_p/M) = {n_ratio:.1f} tores par proton (moitié +11, moitié −11 si relique symétrique)")
print(f"  → chaque proton trouve un tore −11 en {1/(n_ratio/2):.1f} : TOUT l'hydrogène est capturé et brûlé en hélium pendant la BBN")
print(f"  énergie : 26,7 MeV / ⁴He ; tout H brûlé → Y_p → 1 (observé 0,245) ; D/H → 0 ; pas d'hydrogène pour les étoiles, les raies de Balmer, le JWST.")
print("  borne : Y_p à 1,5 % → fraction chargée ≲ 1e-2 pour la seule BBN, beaucoup moins puisque la catalyse se poursuit après (1 cycle / 5 h)")
print("          (les composés neutres X⁻¹¹+11p échappent aux recherches d'hydrogène anormal ; l'hélium catalysé, non).")
print("  cadre : la neutralisation par framing (Q = Σχ + Lk = 0) est SÉLECTIONNÉE pour le réservoir ; c_min = 23 exige le DÉCOUPLAGE à la naissance.")
print()
print("=== Calendrier : le JWST ===")
for z in (4,6,8,10):
    # âge approximatif (Planck) en Ma
    ages={4:1550,6:940,8:650,10:480}
    print(f"  points rouges à z = {z} : âge {ages[z]} Ma, T_CMB = {2.725*(1+z):.0f} K")
print(f"  capture sous (C) : T ~ {EB/30*1e3:.0f} keV, t ~ 10³ s ; sous (M) : T ~ eV, z ~ 1e4 ; sous (N) : jamais")
print("  → aucune des trois lectures microscopiques ne place la phase transitoire à z = 4-10 : les captures sont fixées par E_B, pas par la structure.")
print()
print("=== Lecture collective : la population de tores comme puits ===")
print("  les tores neutres SONT la matière noire ; « les protons se collent » = le gaz tombe dans le puits collectif (halo, grumeau) ;")
print("  « la pression monte jusqu'à la fusion » = contraction de Kelvin-Helmholtz d'un nuage baryonique dans ce puits → allumage.")
print("  c'est exactement le cadre des brouillons pseudo-étoiles (z ~ 40-50, 5e2-5e4 Msun) et points rouges (polytrope à cœur sombre) ;")
print("  la nature des éremnions (tores) ne change rien à la gravité collective — sauf si le puits est COMPACT, ce qui reste la dette.")
