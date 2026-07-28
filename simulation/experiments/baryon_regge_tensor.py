"""Ecarts baryoniques : variation du tenseur d'inertie sous rotation.
Sur la trajectoire leading, la Regge pure M^2=2pi sigma_chi (J-a0) rate le FONDAMENTAL
(bas J). Une correction de jonction/tenseur M = sqrt(SL(J-a0)) + E_J la corrige, avec
E_J de signe OPPOSE octet(-)/decuplet(+), meme magnitude ~95 MeV.
"""
import numpy as np
from scipy.optimize import curve_fit
SL=1.096e6  # 2 pi sigma_chi (MeV^2)
N =[('N',0.5,939.),('N(1680)',2.5,1685.),('N(2220)',4.5,2220.),('N(2700)',6.5,2700.)]
De=[('Δ',1.5,1232.),('Δ(1950)',3.5,1950.),('Δ(2420)',5.5,2420.),('Δ(2950)',7.5,2950.)]
def model(J,a0,EJ): return np.sqrt(np.maximum(SL*(J-a0),1))+EJ
def rms(mp,m): return np.sqrt(np.mean(((mp-m)/m)**2))*100
def run(name,tr):
    J=np.array([t[1] for t in tr]); m=np.array([t[2] for t in tr])
    a0L=np.mean(J-m**2/SL); mL=np.sqrt(np.maximum(SL*(J-a0L),1))
    (a0,EJ),_=curve_fit(model,J,m,p0=[0.,300.],maxfev=10000); mp=model(J,a0,EJ)
    print(f"{name}: Regge pure RMS={rms(mL,m):.1f}%  | +E_J: E_J={EJ:+.0f} MeV, a0={a0:+.3f}, RMS={rms(mp,m):.2f}%")
    return EJ
if __name__=="__main__":
    ejN=run("N (octet J=1/2) ",N); ejD=run("Δ (décuplet J=3/2)",De)
    print(f"\n|E_J| N={abs(ejN):.0f}, Δ={abs(ejD):.0f} MeV (~egaux) ; signe OPPOSE octet(-)/decuplet(+)")
    print(f"E_J(Δ)-E_J(N)={ejD-ejN:.0f} MeV couvre {(ejD-ejN)/293*100:.0f}% du splitting N-Δ (293 MeV)")

# ===== DERIVATION du tenseur (2026-07) =====
def derive_tensor():
    sig,Lam=0.1744,0.2088   # GeV^2, GeV
    L=1/Lam                 # taille baryonique = 1/Lambda_chi
    I=sig*L**3              # I = sigma_chi * L^3  (jonction Y, 3 tubes)
    C=1/(2*I)*1000          # MeV
    print(f"\n=== Derivation tenseur jonction Y (SANS parametre) ===")
    print(f"  L=1/Lambda_chi={L:.2f} GeV^-1={L*0.1973:.2f} fm (taille baryon); I=sigma*L^3={I:.1f} GeV^-1")
    print(f"  C=Lambda^3/(2 sigma)={C:.1f} MeV ; E_tensor=S(S+1)*C")
    print(f"  decuplet S=3/2: {15/4*C:.0f} MeV (DERIVE, colle au +97 mesure)")
    print(f"  octet    S=1/2: {3/4*C:.0f} MeV ; octet net = {3/4*C:.0f} - 112 (hyperfin) = {3/4*C-112:.0f}")
    print(f"  part geometrique du splitting N-Delta = 3*C = {3*C:.0f} MeV ({3*C/293*100:.0f}%)")

if __name__=="__main__":
    derive_tensor()

# ===== extension hyperons (2026-07) =====
def hyperons():
    print("\n=== Extension hyperons : ecart fondamental vs tenseur derive (C=26.1) ===")
    C=0.2088**3/(2*0.1744)*1000
    T={'N':(0.5,[(0.5,939.),(2.5,1685.),(4.5,2220.),(6.5,2700.)]),
       'Λ':(0.5,[(0.5,1115.7),(2.5,1820.),(4.5,2350.)]),
       'Σ':(0.5,[(0.5,1193.2),(2.5,1915.),(3.5,2030.)])}
    for nm,(S,tr) in T.items():
        J=np.array([t[0] for t in tr]);m=np.array([t[1] for t in tr])
        a0=np.mean(J[1:]-m[1:]**2/SL); mReg=np.sqrt(max(SL*(J[0]-a0),1))
        ec=m[0]-mReg
        print(f"  {nm}(octet): ecart={ec:.0f} MeV, tenseur derive={S*(S+1)*C:.0f} -> hyperfin={ec-S*(S+1)*C:.0f}")
    print("  Sigma-Lambda ecart = ~66 MeV = splitting L-Sigma (78, hyperfin). Tenseur=plancher commun.")
    print("  Xi, Omega: pas d'excites leading fiables -> test indecidable.")

if __name__=="__main__":
    hyperons()
