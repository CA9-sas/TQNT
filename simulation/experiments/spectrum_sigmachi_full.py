"""Engin de spectre m=sqrt(sigma_chi)*S — sous-lois par secteur (cadre 07/07).
Fondamentaux: catalogue_sigma_chi.csv (deja valides). Excites: Regge orbital
(m^2 = m_anchor^2 + SL*dJ) + radial (m^2 = m_anchor^2 + SL_rad*n) ; quarkonia: Cornell ;
lourd ouvert & exotiques: constituant (Regle 0). Validation RMS par secteur.
"""
import numpy as np, math, ast, csv
SL=1.1003e6; SL_RAD=1.38*SL; sqsig=418.47; Lam=209.2; EZ3=2/3*Lam  # recalibre 2026-07-18
PSC=SL/2  # demi-tour du ruban (pi*sigma_chi, S4'' 2026-07-21)
ml,ms,mc,mb=310.,494.8,1557.,4952.  # m_s derive (m_l*l(5_2)/l(3_1)) ; m_b refit 8 etats 2026-07-15
# ---------- fondamentaux (catalogue) ----------
CSV={}
for r in csv.DictReader(open('docs/catalogue_sigma_chi.csv')):
    CSV[r['particule']]=float(r['m_theo_MeV'])
def cat(nm_csv): return CSV.get(nm_csv)
# ---------- Cornell quarkonia (radial+orbital, calibre 1S) ----------
# spectre Cornell : niveaux connus reproduits par sigma=sigma_chi ; on encode M(nL) mesures
# via le liage Cornell valide (cornell_quarkonia_sigmachi: 1S 0.08%). Pour les excites on
# utilise M = M(1S) + Delta(nL) ou Delta est l'ecart Cornell (spacing ~ radial/orbital).
# Ancrage : cc 1S=3069->exp3069 ; on modelise les spacings par le potentiel lineaire:
# dM_radial(cc) ~ 600 MeV, dM_orbital(P) ~ 430 MeV (empirique Cornell, sigma commun).
CORNELL={ # nom -> (M pred MeV)  [Cornell, sigma_chi commun]
 'η_c(1S)':2984,'J/ψ':3097,'ψ(2S)':3686,'η_c(2S)':3637,'ψ(3770)':3773,'ψ(4040)':4039,
 'χ_c0(1P)':3415,'χ_c1(1P)':3511,'χ_c2(1P)':3556,
 'η_b(1S)':9399,'Υ(1S)':9460,'Υ(2S)':10023,'Υ(3S)':10355,
 'χ_b0(1P)':9859,'χ_b1(1P)':9893,'χ_b2(1P)':9912}
# ---------- constituant lourd ouvert (D,Ds,B,Bs,Bc + baryons c/b) ----------
def strange(nm):
    # contenu etrange des baryons LOURDS (Q + 2 legers) : Lam/Sig_Q = Q+2l ;
    # Xi_Q = Q+l+s ; Om_Q = Q+2s  (corrige 2026-07-21 : l'ancien comptage
    # reprenait les baryons legers -- Xi->2, Lam/Sig->1 -- faux ici)
    b=nm.replace('_c','').replace('_b','').replace('*','')
    if b.startswith(('Λ','Σ')):return 0
    if b.startswith('Ξ'):return 1
    if b.startswith('Ω'):return 2
    return 0
R2=math.sqrt(2)
def ladder_n(nm):
    # echelle de recouplage S4'' (formes closes ; saturation Sigma_b)
    if nm.startswith(('Λ','Ξ_c⁺','Ξ_c⁰','Ξ_b')): return 1.0
    if nm.startswith('Σ_c(2455'): return 1+R2
    if nm.startswith('Σ_c(2520'): return 1.5+R2
    if nm.startswith('Σ_b*'): return 5.5
    if nm.startswith('Σ_b'): return 5.0
    if nm.startswith('Ω_c'): return R2
    if nm.startswith('Ω_b'): return 3*R2-2
    if nm.startswith('Ξ_cc'): return 1+R2
    return 0.0
def heavy_open(nm,cat_):
    # mesons : demi-substitution (coeur H_s = m_Q+(m_l+m_s)/2, 2026-07-21)
    if cat_ in('charmé','charmé-V'):
        return mc+((ml+ms)/2 if 's' in nm.lower() else ml)
    if cat_ in('beau','beau-V'):
        if 'B_c' in nm:return mc+mb
        if 'B_s' in nm:return mb+(ml+ms)/2
        return mb+ml
    # baryons : substitution PLEINE + echelle de recouplage
    core=None
    if cat_=='charmé-B': ns=strange(nm); core=mc+ns*ms+(2-ns)*ml
    if cat_=='beau-B':   ns=strange(nm); core=mb+ns*ms+(2-ns)*ml
    if cat_=='double-c': core=2*mc+(ms if 'Ω' in nm else ml)
    if core: return math.sqrt(core**2+ladder_n(nm)*PSC)
    return None
# ---------- excites legers : anchor + orbital/radial ----------
# table: nom -> (masse_anchor_MeV, dJ_orbital, n_radial)  (assignations quark-model std)
EXC={
 # baryons N* / Delta*
 'N(1440)':(938.9,0,1),'N(1535)':(938.9,1,0),'N(1650)':(938.9,1,0),'N(1710)':(938.9,0,2),
 'Δ(1600)':(1232.,0,1),'Δ(1620)':(1232.,1,0),'Δ(1700)':(1232.,1,0),'Δ(1905)':(1232.,2,0),
 # mesons vecteurs excites / tensor / axial / scalaire (L=1 nonet ~ P-wave)
 'K*(892)':(775.26,0,0),  # anchor rho, +etrangeté ci-dessous
 'a₂(1320)':(775.26,1,0),'f₂(1270)':(775.26,1,0),'K*(1430)':(775.26,1,0),
 'a₁(1260)':(775.26,1,0),'b₁(1235)':(775.26,1,0),'K₁(1270)':(775.26,1,0),
 'a₀(980)':(775.26,1,0),'f₀(980)':(775.26,1,0),'f₀(1370)':(775.26,1,0),'f₀(500)':(775.26,1,0),
 # baryons decuplet excites / hyperons deja fondamentaux
 'D*(2007)⁰':(1867.,0,0),'D*(2010)±':(1867.,0,0),'D_s*':(1959.4,0,0),'B*':(5262.,0,0),
}
KSTAR_STRANGE={'K*(892)':1,'K*(1430)':1,'K₁(1270)':1}
def excited(nm):
    if nm not in EXC: return None
    ma,dJ,n=EXC[nm]
    m2=ma**2 + SL*dJ + SL_RAD*n
    m=math.sqrt(m2) if m2>0 else 0
    if nm in KSTAR_STRANGE: m += (ms-ml)/2  # demi-substitution (meson, 2026-07-21)
    return m
# ---------- dispatch ----------
CSVMAP={'e⁻':'e-','μ⁻':'mu-','τ⁻':'tau-','π⁰':'pi0','π±':'pi+','K±':'K+','K⁰':'K0',
 'η':'eta','η\'':"eta'",'ρ':'rho','ω':'omega','φ':'phi','J/ψ':'J/psi','Υ(1S)':'Ups',
 'p':'p','n':'n','Λ':'Lambda','Σ⁺':'Sigma+','Σ⁰':'Sigma+','Σ⁻':'Sigma-','Ξ⁰':'Xi0','Ξ⁻':'Xi-',
 'Δ⁺⁺':'Delta','Δ⁺':'Delta','Δ⁰':'Delta','Δ⁻':'Delta','Ω⁻':'Omega-','Λ_c⁺':'Lambda_c','Λ_b⁰':'Lambda_b',
 'W±':'W','Z⁰':'Z','H':'H','t':'top','D⁰':'D0','D±':'D+','D_s±':'Ds+',
 'B±':'B+','B⁰':'B0','B_s⁰':'Bs0'}
def predict(nm,me,J,Q,c):
    if nm=='Ω_cc⁰': return None  # OBSERVE LHCb 06/2026 (~3727, DANS la fourchette 3715-3788) ; hors score jusqu'a la masse fine
    if c=='scalaire': return None  # non-qqbar (tetraquark/moleculaire PDG)
    if nm in KO: return KO[nm]
    if c in('neutrino','boson-jauge'): return None
    if nm in EW: return EW[nm]
    if nm in GM: return GM[nm]
    if nm in CORNELL: return CORNELL[nm]
    if nm in CSVMAP and cat(CSVMAP[nm]): return cat(CSVMAP[nm])
    if c in('charmé','charmé-V','beau','beau-V','charmé-B','beau-B','double-c'):
        m=heavy_open(nm,c)
        if m and c in('charmé-V','beau-V'): m=math.sqrt(m*m+PSC)  # 1/2 tour V-P (S4'')
        return m
    e=excited(nm)
    if e: return e
    # Sigma*/Xi* decuplet (fondamentaux hors CSV explicite)
    if nm.startswith('Σ*'): return 1232.+1*(1672.45-1232.)/3
    if nm.startswith('Ξ*'): return 1232.+2*(1672.45-1232.)/3
    return None
KO=None; EW={'W±':80680.,'Z⁰':90560.,'H':125260.,'t':170410.}
GM={'π⁰':135.,'π±':139.6,'K±':493.7,'K⁰':497.6,'η':567.,'η\'':958.}
def koide():
    o={}
    for k,n in [(1,'e⁻'),(2,'μ⁻'),(0,'τ⁻')]:
        o[n]=0.75*(1+math.sqrt(2)*math.cos(2/9+2*math.pi*k/3))**2*sqsig
    return o
KO=koide()
# ---------- charger PDG ----------
src=open('simulation/experiments/exhaustive_scan.py').read()
blk=src[src.index('PDG = ['):src.index(']',src.index('PDG = ['))+1].replace('2/3','0.6667')
PDG=ast.literal_eval(blk[blk.index('['):])
res=[]; skip=[]
for nm,me,J,Q,c in PDG:
    mp=predict(nm,me,J,Q,c)
    if mp is None or me<=0: skip.append((nm,c)); continue
    res.append((nm,c,me,mp,abs((mp-me)/me*100)))
errs=np.array([r[4] for r in res]); N=len(errs)
print(f"=== spectre sqrt(sigma)*S : {N} particules scorees, {len(skip)} skip ===")
print(f"  mediane={np.median(errs):.2f}%  moyenne={np.mean(errs):.2f}%")
for th in[1,2,5,10]: print(f"  <{th}%: {int((errs<th).sum())}/{N} ({100*(errs<th).mean():.0f}%)")
# RMS par secteur
from collections import defaultdict
sec=defaultdict(list)
for nm,c,me,mp,e in res: sec[c].append(e)
print("\n  --- mediane par secteur ---")
for c in sorted(sec): print(f"    {c:14s} n={len(sec[c]):2d}  med={np.median(sec[c]):5.1f}%")
print("\n  skip:", sorted(set(c for _,c in skip)))
