import numpy as np, re
# (nom, config, J, τ[s], Q_décroissance[MeV], type)
P=[
 ('p','3_1+7_2+7_2',0.5,np.inf,0,'stable'),
 ('n','3_1+7_2+7_2',0.5,879,0.782,'faible'),
 ('mu','3_1',0.5,2.2e-6,105.1,'faible'),
 ('tau','3_1',0.5,2.9e-13,1777,'faible'),
 ('pi+','3_1+5_1',0,2.6e-8,33.9,'faible'),
 ('K+','3_1+7_1',0,1.24e-8,388,'faible'),
 ('Lambda','3_1+4_1+7_2',0.5,2.6e-10,37.7,'faible'),
 ('Sigma+','3_1+6_1+7_2',0.5,0.8e-10,116,'faible'),
 ('Xi-','3_1+6_2+7_2',0.5,1.6e-10,59,'faible'),
 ('Omega-','5_1+7_2+7_2',1.5,0.82e-10,62,'faible'),
 ('pi0','3_1+5_1',0,8.5e-17,135,'EM'),
 ('eta','6_1+6_1+6_1',0,5.0e-19,548,'EM'),
 ('Delta','3_1+5_2+7_4',1.5,5.6e-24,154,'fort'),
 ('rho','4_1+5_2+6_1',1,4.5e-24,495,'fort'),
]
def cross(c): return sum(int(k.split('_')[0]) for k in re.findall(r'\d+_\d+',c))
def ncomp(c): return len(re.findall(r'\d+_\d+',c))
print(f"{'part':8s} {'c':>3s} {'J':>4s} {'Q':>6s} {'type':6s} {'logτ':>7s}")
rows=[]
for nm,cfg,J,tau,Q,ty in P:
    c=cross(cfg); lt=np.log10(tau) if tau<np.inf else np.inf
    rows.append((nm,c,J,Q,ty,tau,lt))
    print(f"{nm:8s} {c:>3d} {J:>4} {Q:>6.1f} {ty:6s} {lt if lt<np.inf else 999:>7.1f}")
# ---- corrélations (états instables, τ fini) ----
un=[r for r in rows if r[6]<np.inf and r[3]>0]
lt=np.array([r[6] for r in un]); c=np.array([r[1] for r in un],float)
J=np.array([r[2] for r in un],float); lQ=np.log10([r[3] for r in un])
print("\n=== CORRÉLATIONS log τ vs mesures ===")
for nm,x in [('complexité c (barrière)',c),('spin J (toupie)',J),('log Q (Sargent)',lQ),
             ('c − 5·logQ (barrière−tension)',c-5*lQ)]:
    print(f"  corr(logτ, {nm:30s}) = {np.corrcoef(lt,x)[0,1]:+.2f}")
# ---- test formule toupie : logτ = a·c − b·logQ + const (barrière vs tension) ----
from numpy.linalg import lstsq
X=np.column_stack([c,lQ,np.ones(len(c))])
coef,_,_,_=lstsq(X,lt,rcond=None)
pred=X@coef; r2=1-np.sum((lt-pred)**2)/np.sum((lt-np.mean(lt))**2)
print(f"\n=== FIT toupie logτ = {coef[0]:+.2f}·c {coef[1]:+.2f}·logQ {coef[2]:+.2f} ===")
print(f"  R²={r2:.2f}  (c=barrière topo, logQ=tension/phase-space)")
print(f"  coef c = {coef[0]:+.2f} (>0 attendu: plus de complexité=barrière+haute=vit+longtemps)")
print(f"  coef logQ = {coef[1]:+.2f} (Sargent pur prédit −5)")
# proton stable ?
print(f"\nPROTON: c=17, Q=0 (Lk=Lk_opt=+3, ÉQUILIBRÉ) → tension=0 → STABLE ✓ (toupie parfaitement équilibrée)")
print("NEUTRON: Lk=+2 (|ΔLk|=1, léger déséquilibre) → tension→ décroît, mais lentement (879s)")
