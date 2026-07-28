import numpy as np
print("="*60); print("AUDIT (A) — REVÉRIFICATION DES NOMBRES"); print("="*60)
sig=1.096/(2*np.pi); rs=np.sqrt(sig)*1000; Lam=rs/2; EZ3=(2/3)*Lam
print(f"\n√σ_χ={rs:.2f} (doc 417.65)  Λ_χ={Lam:.2f} (208.8)  E_Z3={EZ3:.2f} (139.2)")
# leptons Koide
A=np.sqrt(2); Sk=np.sort([0.75*(1+A*np.cos(2/9+2*np.pi*k/3))**2 for k in range(3)])
M=1.5*Lam; ml=[M*s/(sum(np.sqrt(Sk))/3)**0 for s in Sk]  # careful
# Koide: √m_k=√M(1+A cos...), m_k=M(1+Acos)²... with M scale
mk=[M*(1+A*np.cos(2/9+2*np.pi*k/3))**2 for k in range(3)]
mk=np.sort(mk); exp_l=[0.511,105.66,1776.86]
print("\nLEPTONS (Koide, M=1.5Λ_χ):")
for i,(m,e) in enumerate(zip(mk,exp_l)): print(f"  m={m:.3f} exp={e} → {(m-e)/e*100:+.2f}%")
# baryons octet+decuplet
def regge(J,a0): return np.sqrt(2*np.pi*sig*1e6*(J-a0))
C=(1232-939)/1.5; r=3/5; Ds=EZ3+0.5*C*(1-r)
mN=regge(0.5,-0.30); mD=regge(1.5,0.115)
print(f"\nBARYONS: mN(Regge)={mN:.0f} (exp 939)  mΔ={mD:.0f} (1232)  Δ_s={Ds:.0f} (178)")
oct_dec={'p':(mN,0.3),'Λ':(mN+Ds,-3.6),'Ω':(mD+3*Ds-93.8,0)}
# quick check nucleon, Lambda, Omega
print(f"  p={mN:.0f}(exp938→{(mN-938.3)/938.3*100:+.1f}%)  Λ={mN+Ds:.0f}(1116→{(mN+Ds-1115.7)/1115.7*100:+.1f}%)")
# Regge slope 224
SL=2*np.pi*sig
print(f"\nPENTE σ_χ = 2πσ_χ = {SL:.3f} GeV² (doc 1.096)")
# EW
print("\nEW (ew_vev): W 80.68(+0.4%) Z 90.56(-0.7%) H 125.26(+0.0%) top 170.41(-1.3%) — importés, à vérifier source")
print("\n"+"="*60); print("COMPTAGE HONNÊTE DES PARAMÈTRES")
print("="*60)
params={
 'σ_χ (échelle, IRRÉDUCTIBLE)':'1 input dimensionnel',
 'Koide A=√2':'DÉRIVÉ (Ising k=2)',
 'Koide θ=2/9, M=1.5Λ_χ':'SUGGESTIFS (non forcés)',
 'τ_⊥ C (hyperfin)':'CALIBRÉ sur N−Δ',
 'α_0 intercepts (N,Δ,méson)':'PHYSIQUES/mesurés (Regge)',
 'E_Z3=2/3 Λ_χ':'suggestif; Δ_s dérivé de E_Z3+τ_⊥',
 'm_c,m_b,α_s (lourd/Cornell)':'CALIBRÉS (3-4)',
 'hyperfin lourd A':'CALIBRÉ (J/ψ−ηc)',
 'GMOR B, m_s/m_l':'CALIBRÉS (π,K)',
 'EW VEV':'dérivation séparée (ew_vev_derivation)',
}
for k,v in params.items(): print(f"  {k:32s} : {v}")
print("\n→ PAS 'zéro paramètre'. σ_χ + ~8-10 constantes (calibrées/physiques/suggestives).")
print("→ La FORCE réelle: (1) pente Regge=σ_χ universelle (34 états, prédictions);")
print("  (2) Koide leptons (A dérivé); (3) Cornell avec σ_χ imposé. Le RESTE = fits/calibr.")
