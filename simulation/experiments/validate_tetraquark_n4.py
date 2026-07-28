r"""
VALIDATION du facteur de structure S(P)=N^α(n) avec les TÉTRAQUARKS n=4 — test
décisif de la dépendance en n (idée utilisateur).

Le brouillon Duplantier était bloqué (tous composites propres n=3). Avec l'écrantage
par appariement, on a maintenant n=2 (mésons), n=3 (baryons/EW) ET n=4 (tétraquarks
scalaires légers du nonet de Jaffe, assignés en nœuds légers 3_1/5_2). On ajuste
α(2), α(3), α(4) séparément et on regarde la TENDANCE.

RÉSULTAT DÉCISIF : α(2)=-0,18 → α(3)=+0,45 → α(4)=+1,17 — MONOTONE CROISSANTE.
Or l'exposant d'étoile de Duplantier DÉCROÎT quand on ajoute des branches (étoile
plus complexe → exposant plus négatif). La tendance observée est OPPOSÉE.
=> LE FACTEUR DE STRUCTURE N'EST PAS L'EXPOSANT DE RÉSEAU DE DUPLANTIER (tranché
avec 3 valeurs de n). Il reste PHÉNOMÉNOLOGIQUE (un exposant par n, non dérivé).

Confirme définitivement le verdict du brouillon_masse_x_duplantier (« x n'est pas
un exposant de Duplantier élémentaire »). RÉSERVES : assignations tétraquarks
spéculatives (nonet de Jaffe qqq̄q̄), masses fittent mal (RMS 19 %, σ +32 %,
a₀ -49 % — confond saveur/assignation), mais la TENDANCE CROISSANTE de α(n) est
robuste. Le facteur de structure fonctionne (fit par-n) mais n'est pas Duplantier.
"""
import sys; sys.path.insert(0,'.')
import numpy as np, itertools
from collections import Counter
import theory.omega_ts as ots
from theory.abraham import REFERENCE_N50K as GEO
def UkUt(K,f=0.5):
    g=GEO[K]; return g['int_kappa']**2/g['L'], g['int_abs_tau']**2/(f*g['L'])
NEW={k:UkUt(k) for k in GEO}
_o=ots._axiom_data
ots._axiom_data=lambda name:{**_o(name),**({'Uk':NEW[name][0],'Ut':NEW[name][1]} if name in NEW else {})}
from simulation.experiments.omega_unified import _m_naive,_T,RHO0,BETA_FLORY,P_CAS,Q_CAS
AMPHI={"4_1","6_3","8_3"}
# n=2 mésons + n=3 baryons/EW + n=4 TÉTRAQUARKS scalaires légers (nonet de Jaffe)
PART={"ρ⁰":("6_2+6_2",775.3,2),"ω":("8_1+8_1",782.7,2),
 "p":("3_1+7_2+7_2",938.3,3),"Λ":("3_1+5_2+7_2",1115.7,3),"Σ":("5_2+7_2+7_2",1193.2,3),
 "Ξ":("3_1+3_1+7_2",1318.3,3),"Δ":("5_2+6_3",1232.0,2),
 "W":("7_1+7_2+8_3",80379.,3),"Z":("7_2+8_3+9_1",91188.,3),"H":("7_1+8_3+10_1",125250.,3),
 # TÉTRAQUARKS n=4 (scalaires légers qqq̄q̄, saveur ~ nb de 5_2 étranges)
 "σ/f₀(500)":("3_1+3_1+3_1+3_1",500.,4),
 "κ(700)":("3_1+3_1+3_1+5_2",700.,4),
 "a₀(980)":("3_1+3_1+5_2+5_2",980.,4)}
def PhiF(K): d=ots._axiom_data(K); return d['Uk']**(5/3)*d['Ut']
def base(cfg,s):
    comp=[c.strip().replace('-','_') for c in cfg.split('+')]; n=len(comp); cnt=Counter(comp)
    screen=1.0
    for k,c in cnt.items():
        if c>=2: screen*=(RHO0*ots._axiom_data(k)['cr']**(-BETA_FLORY))**(c-1)
    b=[_m_naive(k)*_T(k)**P_CAS for k in comp]; holder=(sum(np.sqrt(bi) for bi in b))**2
    G=sum(PhiF(k) for k in comp)/PhiF('3_1')
    pair=1.0
    for i,j in itertools.combinations(range(n),2):
        if comp[i] not in AMPHI and comp[j] not in AMPHI: pair*=s
    N=sum(ots._axiom_data(k)['cr'] for k in comp)
    return screen*holder*G**Q_CAS*pair, N
def rms(par):
    s,a2,a3,a4=par; al={2:a2,3:a3,4:a4}; L=[]
    for nm,(cfg,mx,n) in PART.items():
        bval,N=base(cfg,s); f=bval*N**al[n]
        if not np.isfinite(f) or f<=0: return 9e9
        L.append((np.log(f),mx,n))
    off=np.mean([np.log(mx)-a for a,mx,_ in L])
    return np.sqrt(np.mean([((np.exp(a+off)-mx)/mx)**2 for a,mx,_ in L]))*100,off
from scipy.optimize import minimize
best=None
for a4 in (-2,0,1,2):
    r=minimize(lambda p:rms(p)[0],[0.53,-0.46,0.22,a4],method='Nelder-Mead',options={'maxiter':800,'fatol':1e-2})
    if best is None or r.fun<best.fun: best=r
s,a2,a3,a4=best.x; rmsv,off=rms(best.x)
print(f"Avec TÉTRAQUARKS n=4 : s={s:.3f}")
print(f"  α(2)={a2:.2f}  α(3)={a3:.2f}  α(4)={a4:.2f}   RMS={rmsv:.0f}%")
print(f"\n  Dépendance en n :  α(2)={a2:.2f} → α(3)={a3:.2f} → α(4)={a4:.2f}")
mono = "CROISSANTE" if a2<a3<a4 else ("DÉCROISSANTE" if a2>a3>a4 else "non monotone")
print(f"  Tendance : {mono}")
print(f"  Duplantier prédit DÉCROISSANTE (étoile plus complexe -> exposant plus négatif).")
print(f"\n  détail tétraquarks :")
for nm,(cfg,mx,n) in PART.items():
    if n==4:
        bval,N=base(cfg,s); f=bval*N**a4*np.exp(off)
        print(f"    {nm:11s} {cfg:20s} préd={f:6.0f} exp={mx:5.0f} ({(f-mx)/mx*100:+.0f}%)")
