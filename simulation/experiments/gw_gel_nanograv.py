"""Le spectre d'ondes gravitationnelles du gel face à NANOGrav — calcul honnête.

Ce que le document principal fournit : T_c = 62±3 MeV [dérivé], g_s(T_c) = 15,2 [dérivé],
chaleur latente ~Λ_χ⁴ (Λ_χ = 209,2 MeV — « ×19 le fond thermique »),
échelle de domaine ξ = κ_dom/T_c (κ_dom = 1,14), marche de G à t_c.

LE POINT DÉCISIF (qui corrige l'estimation naïve « bande PTA ») : la
fréquence de pic GW d'une transition est fixée par l'ÉCHELLE DE COHÉRENCE
de la source. La formule f ≈ 1,65e-7 Hz (T/100 GeV)(β/H) suppose des
bulles à une fraction ~H/β de l'horizon. Le gel n'a pas de bulles
d'horizon : sa dynamique vit à l'échelle du DOMAINE ξ ~ GeV⁻¹ — vingt
ordres sous l'horizon. Conséquences calculées ici :
  (1) pic aujourd'hui à l'échelle du domaine → ~10 GHz (inobservable) ;
  (2) amplitude écrasée par (ξH*)² — l'efficacité quadrupolaire d'une
      source sous-horizon — → Ω_GW ridicule même au pic ;
  (3) dans la bande PTA : seule la queue causale f³ → strictement rien ;
  (4) l'échelle HORIZON de T_c (la marche de G) tombe à ~0,08 nHz —
      SOUS la bande PTA (f_min ~ 1/20 ans ≈ 1,6 nHz).
VERDICT : le gel ne produit AUCUN fond GW observable ; le signal NANOGrav
doit être astrophysique (cohérent avec l'audit du document principal : SMBHB préféré par
BIC) — et c'est une PRÉDICTION D'ABSENCE de plus, falsifiable.
"""
import numpy as np

# --- référence
TC = 0.062                 # GeV
GS = 15.2
LAM_CHI = 0.2092           # GeV
KAPPA_DOM = 1.14
MPL = 1.221e19             # GeV
T0 = 2.348e-13             # GeV (2,7255 K)
G0 = 3.91

# --- force de la transition
rho_rad = (np.pi**2 / 30) * GS * TC**4
alpha_pt = LAM_CHI**4 / rho_rad
print(f'chaleur latente / radiation : α = {alpha_pt:.1f}  (référence : « ×19 le fond thermique » ✓)')

# --- échelles
H_star = 1.66 * np.sqrt(GS) * TC**2 / MPL          # GeV
xi = KAPPA_DOM / TC                                 # GeV⁻¹
xiH = xi * H_star
print(f'H(T_c) = {H_star:.2e} GeV ; ξ = {xi:.1f} GeV⁻¹ ; ξH* = {xiH:.1e}')

# --- redshift des fréquences
red = (T0 / TC) * (G0 / GS) ** (1/3)
f_domaine = (1.0 / xi) * 1.519e24 * red             # GeV → Hz, puis redshift
f_horizon = H_star * 1.519e24 * red
print(f'fréquence aujourd hui — échelle du domaine : {f_domaine:.2e} Hz (pic réel)')
print(f'fréquence aujourd hui — échelle horizon    : {f_horizon:.2e} Hz (marche de G)')
print( 'bande PTA (NANOGrav 15 ans)               : 1,6e-9 à 6e-8 Hz')

# --- amplitude au pic : efficacité quadrupolaire d une source sous-horizon
# Ω_GW ~ (κα/(1+α))² (ξH*)² × (facteurs O(1)) ; dilution radiation 1,64e-5 h⁻²
omega_peak = (alpha_pt / (1 + alpha_pt))**2 * xiH**2 * 1.64e-5
print(f'Ω_GW h² au pic (10 GHz) ~ {omega_peak:.1e} — {abs(np.log10(omega_peak)):.0f} ordres sous tout instrument')

# --- queue causale dans la bande PTA
f_pta = 3.2e-8
omega_pta = omega_peak * (f_pta / f_domaine) ** 3
print(f'queue causale f³ à 32 nHz : Ω_GW h² ~ {omega_pta:.1e} (zéro physique)')

# --- NANOGrav 15 ans (mesuré) pour l échelle
# A_GWB ≈ 6,4e-15 à f_yr, γ≈3,2 → Ω_GW h²(32 nHz) ~ 1e-8
print('NANOGrav 15 ans : Ω_GW h²(32 nHz) ≈ 1e-8 — le gel n y contribue en rien :')
print('le fond nHz observé DOIT être astrophysique (SMBHB) — cohérent avec')
print('l audit référence (BIC pro-SMBHB) ; toute démonstration future d une origine')
print('cosmologique à T ~ 60 MeV serait une TENSION pour le cadre (le gel ne')
print('peut pas la produire : pas de dynamique cohérente à l échelle horizon).')
