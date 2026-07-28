import numpy as np
from scipy.optimize import minimize_scalar, brentq
# Cornell : V(r) = -κ/r + σ·r, σ = σ_χ (MÊME corde χ). Variationnel gaussien.
# ⟨H⟩=2m_Q + 3/(2 m_Q b²) - κ·2/(√π b) + σ·2b/√π   (μ=m_Q/2, ħ=1, GeV)
sigchi=1.096/(2*np.pi)   # 0.1744 GeV²
def E_ground(mQ,kappa,sig):
    def H(b): return 3/(2*mQ*b**2) - kappa*2/(np.sqrt(np.pi)*b) + sig*2*b/np.sqrt(np.pi)
    r=minimize_scalar(H,bounds=(0.3,8),method='bounded')
    return 2*mQ+r.fun, r.x, r.fun   # masse, taille b, E_liage
# α_s courant : charm ~0.35, bottom ~0.22 → κ=4α_s/3
def kap(als): return 4*als/3
# calibrer m_c, m_b sur le 1S spin-moyenné, σ=σ_χ FIXÉ
cc_avg=(3*3.0969+2.9836)/4   # charmonium 1S spin-avg
bb_avg=(3*9.4603+9.3987)/4   # bottomonium 1S spin-avg
als_c,als_b=0.35,0.22
def solve_mQ(target,als):
    f=lambda mQ: E_ground(mQ,kap(als),sigchi)[0]-target
    return brentq(f,0.8,6.0)
mc=solve_mQ(cc_avg,als_c); mb=solve_mQ(bb_avg,als_b)
print(f"σ = σ_χ = {sigchi:.4f} GeV² (corde χ commune) ; α_s(c)={als_c}, α_s(b)={als_b}")
print(f"m_c={mc*1000:.0f}  m_b={mb*1000:.0f} MeV (calibrés 1S, σ_χ fixé)\n")
for nm,mQ,als,exp in [('cc̄ 1S',mc,als_c,cc_avg),('bb̄ 1S',mb,als_b,bb_avg)]:
    M,b,El=E_ground(mQ,kap(als),sigchi)
    print(f"{nm}: M={M*1000:.0f} (exp {exp*1000:.0f})  liage E={El*1000:+.0f} MeV  taille b={b:.2f} GeV⁻¹={0.197*b:.2f} fm")
print(f"\n--- comparaison AVANT (2m_Q pur) / APRÈS (Cornell, σ_χ) ---")
# hyperfine S-wave : Δ_hf ∝ |ψ(0)|²/m_Q² ; |ψ(0)|²=(π b²)^-3/2 pour gaussien... calibrer sur J/ψ-ηc
def psi0sq(b): return (np.pi*b**2)**-1.5
Mc,bc,_=E_ground(mc,kap(als_c),sigchi); Mb,bb_,_=E_ground(mb,kap(als_b),sigchi)
# Δ_hf = A·|ψ(0)|²/m_Q² ; A calibré sur J/ψ−ηc=113 MeV
A=(3.0969-2.9836)/(psi0sq(bc)/mc**2)
def split(b,mQ): return A*psi0sq(b)/mQ**2
hc=split(bc,mc); hb=split(bb_,mb)
res=[('η_c',Mc-0.75*hc,2.9836),('J/ψ',Mc+0.25*hc,3.0969),
     ('η_b',Mb-0.75*hb,9.3987),('Υ',Mb+0.25*hb,9.4603)]
print(f"{'état':6s} {'2m_Q pur':>9s} {'Cornell+hf':>11s} {'exp':>8s} {'Δ%(Corn)':>9s}")
for nm,M,exp in res:
    pur=2*(mc if 'c' in nm or nm=='J/ψ' else mb)
    print(f"{nm:6s} {pur*1000:>9.0f} {M*1000:>11.0f} {exp*1000:>8.0f} {(M-exp)/exp*100:>+8.2f}%")
d=[(M-exp)/exp*100 for _,M,exp in res]
print(f"\nRMS quarkonia (Cornell+hf, σ_χ) = {np.sqrt(np.mean(np.square(d))):.2f}%  (vs ~4-11% en 2m_Q pur)")
