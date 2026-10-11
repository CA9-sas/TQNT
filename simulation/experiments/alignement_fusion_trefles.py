"""a_f : la probabilité de fusion (même chiralité) rapportée à l'annihilation (16/09/2026).
Trois estimations, puis le Smoluchowski REFAIT avec l'érosion : un antitrèfle sur un tore
ne détruit pas le tore, il le dévide de 3 croisements (σ₁^q σ₁^-3 = σ₁^(q-3)).
"""
import math, numpy as np
alpha=1/137.036; ML=310.0; slope=70.1
def M_T2(q): return ML+slope*(q-3)
print("=== 1. Ce qu'est a_f ===")
print("  Un couple K+K de même chiralité, s'il se LIE au contact, cascade par P15 (1e-18 s) vers son fondamental : T(2,5) = 450 MeV")
print("  (< 3₁+3₁ = 620, < 3₁#3₁ = 620). L'alignement des axes n'est donc PAS requis : a_f = P(lier | K+K) / P(lier | K+K̄).")
print("  (a) géométrique (empilement coaxial strict, deux axes à θ de la ligne des centres) : a_f = (θ²/4)² →", ", ".join(f"θ={th}°: {((math.radians(th)**2)/4)**2:.1e}" for th in (10,30,60)))
print("  (b) capture radiative ∝ E_libérée³ (dipôle χ) : K+K → T(2,5) libère 170 MeV, K+K̄ → π libère 485 MeV : a_f = (170/485)³ =", f"{(170/485)**3:.3f}")
print("  (c) capture ∝ E_libérée (phase) : a_f = 170/485 =", f"{170/485:.2f}")
print()
print("=== 2. Smoluchowski avec ÉROSION (K̄ dévide le tore) ===")
xi=3.36e-15; T_c=62.0; g_s=15.2; hbarc=197.327; c=2.998e8
n3=0.59*0.304/xi**3; s=(2*math.pi**2/45)*g_s*(T_c/hbarc)**3*1e45; Y3=n3/s
t_c=1.6e-4; K0=math.pi*(0.35e-15)**2*0.03*c
Q=list(range(3,64,2)); nq=len(Q); idx={q:i for i,q in enumerate(Q)}
def supp(q): return alpha**max(0,q-18)
Y_DM=2.3e-27/(1.8e3*1.783e-30)/2.9e9
def run(a_f):
    Y=np.zeros(nq); Y[0]=Y3; t=t_c; a=1.0; E_rel=0.0
    while t<1e3:
        dt=t*0.002; n=Y*s/a**3; dY=np.zeros(nq)
        for i,q1 in enumerate(Q):
            if Y[i]<=0: continue
            for j,q2 in enumerate(Q):
                if Y[j]<=0 or j<i: continue
                k=K0*supp(q1)*supp(q2)*n[i]*n[j]/s
                # fusion même chiralité
                r=a_f*k*(0.5 if i==j else 1.0); q3=q1+q2-1
                if q3 in idx: dY[i]-=r; dY[j]-=r; dY[idx[q3]]+=r; E_rel+=r*dt*(M_T2(q1)+M_T2(q2)-M_T2(q3))
                # rencontre chiralités opposées : érosion σ₁^q1 σ₁^-q2 -> T(2,|q1-q2|-1) ou annihilation
                ra=k*(2.0 if i!=j else 1.0)   # K1+K̄2 et K̄1+K2
                d=abs(q1-q2)-1
                dY[i]-=ra; dY[j]-=ra
                if d>=3 and d in idx: dY[idx[d]]+=ra
                E_rel+=ra*dt*(M_T2(q1)+M_T2(q2)-(M_T2(d) if d>=3 else 0))
        m=np.max(np.abs(dY)*dt/np.maximum(Y,1e-30))
        if m>0.2: dt*=0.2/m
        Y=np.maximum(Y+dY*dt,0); t+=dt; a=(t/t_c)**0.5
    Md=2*sum(Y[i]*M_T2(q) for i,q in enumerate(Q) if q>=23)
    top=sorted([(q,Y[i]) for i,q in enumerate(Q) if Y[i]>1e-3*Y.max()], key=lambda x:-x[1])[:5]
    E_fin=2*sum(Y[i]*M_T2(q) for i,q in enumerate(Q))
    return Md/(Y_DM*1.8e3), top, 1-E_fin/(2*Y3*ML)
for a_f in (1.0, 0.35, 0.1, 0.043, 0.01, 4.5e-3):
    r,top,e=run(a_f)
    print(f"  a_f = {a_f:<6} : Ω/Ω_Planck = {r:8.1e} ; fraction de la masse des trèfles rayonnée en χ = {e:.6f} ; pics : "+", ".join(f"q={q}:{y:.1e}" for q,y in top))
print()
print("  lecture : l'érosion transforme la coagulation en marche au hasard sur q (ruine du joueur, absorption à q<3, gel à c≈25) ;")
print("  la survie n'est plus ∝ a_f⁴ mais gouvernée par le biais net fusion−érosion.")
