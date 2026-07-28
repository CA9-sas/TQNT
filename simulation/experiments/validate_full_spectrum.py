r"""
VALIDATION SUR LE SPECTRE COMPLET de la formule de masse à 4 ingrédients — ÉCHEC.

Les 4 ingrédients (appariement chiral + Φ_F=U_κ^(5/3)U_τ + Hölder 1/2 + Steiner
S(P)=N^(c(n-2))) donnaient RMS 10-14 % sur un sous-ensemble de 10 particules. On
VALIDE sur ~24 assignations re-dérivées (excel_masses_categorisation), géométrie
Abraham correcte.

RÉSULTAT DÉCISIF : RMS GLOBAL = 93 %. Pire, les paramètres S'EFFONDRENT (multi-
départs confirment) : s_chiral → 0,998 (≈ PAS d'écrantage), c_Steiner → 0,021
(≈ PAS de Steiner). Sous-secteurs : léger 102 %, EW 72 %, saveuré lourd 68 % ;
baryons légers SEULS 37 % ; baryons+EW 59 %.

CONCLUSION : le 10 % sur 10 particules était du SURAJUSTEMENT à un sous-ensemble
favorable. Les ingrédients NE GÉNÉRALISENT PAS — sur le spectre complet, les
mécanismes physiques (appariement, Steiner) sont désactivés par l'optimiseur car
ils ne réconcilient pas la diversité. La formule de masse ABSOLUE via géométrie +
ces mécanismes NE VALIDE PAS sur le spectre complet. Confirme définitivement :
l'absolu n'est pas récupérable ; la géométrie capture la complexité (qualitatif)
mais pas les masses précises (saveur + contenu non-topologique).

CE QUI SURVIT : les prédictions STRUCTURELLES (Koide 61 ppm, Regge 1,5 %) et le
QUALITATIF complexité↔masse (r~0,94-0,98). Le spectre absolu reste non-perturbatif.
Outliers criants : η +314 %, Σ* +181 %, D +125 %, ω +113 % (assignations
douteuses + saveur non-topologique).
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
# (nom, config, masse, secteur) — assignations re-dérivées (excel_masses_categorisation)
DATA=[
 ("ρ⁰","6_2+6_2",775.3,"L"),("ω","8_1+8_1",782.7,"L"),("φ","7_5+7_5",1019.5,"L"),
 ("K*","5_2+7_5",891.7,"L"),("η","3_1+4_1+5_1",547.9,"L"),("η′","5_2+5_2+6_1",957.8,"L"),
 ("K","4_1+4_1+5_2",493.7,"L"),
 ("p","3_1+7_2+7_2",938.3,"L"),("Λ","3_1+5_2+7_2",1115.7,"L"),("Σ","5_2+7_2+7_2",1193.2,"L"),
 ("Ξ","3_1+3_1+7_2",1318.3,"L"),("Δ","5_2+6_3",1232.0,"L"),("Σ*","5_1+5_1+5_2",1384.6,"L"),
 ("Ξ*","4_1+5_2+7_2",1533.4,"L"),
 ("W","7_1+7_2+8_3",80377.,"EW"),("Z","7_2+8_3+9_1",91188.,"EW"),
 ("H","7_1+8_3+10_1",125250.,"EW"),("top","8_2+8_3+10_1",172570.,"EW"),
 ("J/ψ","7_4+7_4",3096.9,"F"),("Υ","7_6+7_6+4_1",9460.3,"F"),
 ("D","3_1+5_1+7_2",1864.8,"F"),("B","8_1+8_2",5279.3,"F"),
 ("σ","3_1+3_1+3_1+3_1",500.,"L"),("a₀","3_1+3_1+5_2+5_2",980.,"L"),
]
def PhiF(K): d=ots._axiom_data(K); return d['Uk']**(5/3)*d['Ut']
def formula(cfg,s,c):
    comp=[x.strip().replace('-','_') for x in cfg.split('+')]; n=len(comp); cnt=Counter(comp)
    screen=1.0
    for k,cc in cnt.items():
        if cc>=2: screen*=(RHO0*ots._axiom_data(k)['cr']**(-BETA_FLORY))**(cc-1)
    b=[_m_naive(k)*_T(k)**P_CAS for k in comp]; holder=(sum(np.sqrt(bi) for bi in b))**2
    G=sum(PhiF(k) for k in comp)/PhiF('3_1'); pair=1.0
    for i,j in itertools.combinations(range(n),2):
        if comp[i] not in AMPHI and comp[j] not in AMPHI: pair*=s
    N=sum(ots._axiom_data(k)['cr'] for k in comp)
    return screen*holder*G**Q_CAS*pair*N**(c*(n-2))
from scipy.optimize import minimize
def rmsall(par,subset=None):
    s,c=par; L=[]
    for nm,cfg,mx,sec in DATA:
        if subset and sec not in subset: continue
        f=formula(cfg,s,c)
        if not np.isfinite(f) or f<=0: return 9e9,None,None
        L.append((nm,np.log(f),mx,sec))
    off=np.mean([np.log(mx)-a for _,a,mx,_ in L])
    res={nm:(np.exp(a+off),mx,sec) for nm,a,mx,sec in L}
    rms=np.sqrt(np.mean([((mp-mx)/mx)**2 for mp,mx,_ in res.values()]))*100
    return rms,res,off
r=minimize(lambda p:rmsall(p)[0],[0.55,0.6],method='Nelder-Mead',options={'maxiter':800})
s,c=r.x; rms,res,off=rmsall([s,c])
print(f"=== SPECTRE COMPLET ({len(DATA)} particules) — 4 ingrédients ===")
print(f"  calibré : s_chiral={s:.3f}, c_Steiner={c:.3f}\n")
print(f"  {'part':5s} {'préd':>9s} {'exp':>9s} {'écart%':>7s} sec")
for nm,cfg,mx,sec in DATA:
    mp=res[nm][0]; print(f"  {nm:5s} {mp:9.0f} {mx:9.0f} {(mp-mx)/mx*100:+7.0f} {sec}")
print(f"\n  RMS GLOBAL = {rms:.0f}%")
for sec,lab in [("L","léger (u,d,s)"),("EW","électrofaible"),("F","saveuré lourd (c,b)")]:
    rr=np.sqrt(np.mean([((res[nm][0]-mx)/mx)**2 for nm,cfg,mx,ss in DATA if ss==sec]))*100
    print(f"  RMS {lab:22s} = {rr:.0f}%")

print("\n=== Robustesse : multi-départs + sous-secteurs ===")
best=None
for s0 in (0.3,0.5,0.7,1.0):
 for c0 in (0.0,0.3,0.6,1.0):
    rr=minimize(lambda p:rmsall(p)[0],[s0,c0],method='Nelder-Mead',options={'maxiter':1000})
    if best is None or rr.fun<best.fun: best=rr
print(f"  meilleur global : s={best.x[0]:.3f} c={best.x[1]:.3f} RMS={best.fun:.0f}%")
# baryons légers seuls (le sous-ensemble propre)
BARY={"p","Λ","Σ","Ξ","Δ","Ξ*","Σ*"}
def rms_sub(par,names):
    s,c=par; L=[]
    for nm,cfg,mx,sec in DATA:
        if nm not in names: continue
        f=formula(cfg,s,c); L.append((np.log(f),mx))
    off=np.mean([np.log(mx)-a for a,mx in L])
    return np.sqrt(np.mean([((np.exp(a+off)-mx)/mx)**2 for a,mx in L]))*100
for lab,names in [("baryons légers",BARY),
                  ("baryons+EW",BARY|{"W","Z","H","top"}),
                  ("baryons+EW+mésons diag",BARY|{"W","Z","H","top","ρ⁰","φ","K*"})]:
    rb=minimize(lambda p:rms_sub(p,names),[0.5,0.6],method='Nelder-Mead',options={'maxiter':600}).fun
    print(f"  {lab:26s} ({len(names)} part.) RMS={rb:.0f}%")
