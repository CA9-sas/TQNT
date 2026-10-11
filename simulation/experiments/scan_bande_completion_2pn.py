"""Scan de la bande de complétion (KW, KWV) au point J0737-3039.

Charge le cache des EOM 2PN de la trame (derive_secteur_cinematique_2pn)
et mesure delta k2 aux coins de la bande résiduelle q² de la charge-source
(KW : terme scalaire V² ; KWV : habillage du vecteur), à (nu=0.25, e=0.1).

RÉSULTAT (2026-08-22) :
  KW=+0 KWV=+0 : delta k2 = -10.695   (centre : le poids exponentiel)
  KW=+1 KWV=+0 : delta k2 =  -9.201
  KW=-1 KWV=+0 : delta k2 = -12.188
  KW=+0 KWV=+1 : delta k2 =  -5.903   (coin le plus favorable)
  KW=+0 KWV=-1 : delta k2 = -15.487
=> bande [-15.5, -5.9] : l'écart sur omega_dot du pulsar double reste
entre 3,3x et 11x la précision de mesure (1,3e-6) sur TOUTE la bande —
le verdict de tension forte est robuste à la liberté de complétion.
"""
import pickle, time, numpy as np, sympy as sp
from scipy.integrate import solve_ivp
t0=time.time()
d = pickle.load(open('output/cache_eom_trame_2pn.pkl','rb'))
a1 = sp.Matrix(sp.sympify(d['a1'])); a2 = sp.Matrix(sp.sympify(d['a2']))
print(f"cache {time.time()-t0:.0f}s", flush=True)
syms = sp.symbols('x1 y1 x2 y2 vx1 vy1 vx2 vy2')
G,M1,M2,q = sp.symbols('G m1 m2 q', positive=True)
LAM,KW,KWV = sp.symbols('lambda_s KW KWV')
PARS=(G,M1,M2,q,LAM,KW,KWV)
f1 = sp.lambdify(syms+PARS, list(a1), modules='math', cse=True)
f2 = sp.lambdify(syms+PARS, list(a2), modules='math', cse=True)
print(f"lambdify {time.time()-t0:.0f}s", flush=True)

def gr_acc(rv,vv,Gn,m,qn,nu):
    r=np.hypot(*rv); n=rv/r; v2=vv@vv; rd=n@vv; gmr=Gn*m/r
    A=qn*(-1.5*rd**2*nu+v2+3*nu*v2-gmr*(4+2*nu)) \
      +qn*qn*(15*rd**4*nu/8-45*rd**4*nu**2/8-4.5*rd**2*nu*v2+6*rd**2*nu**2*v2
      +3*nu*v2**2-4*nu**2*v2**2+gmr*(-2*rd**2-25*rd**2*nu-2*rd**2*nu**2
      -6.5*nu*v2+2*nu**2*v2)+gmr**2*(9+87*nu/4))
    B=qn*(-4*rd+2*rd*nu)+qn*qn*(4.5*rd**3*nu+3*rd**3*nu**2-7.5*rd*nu*v2
      -2*rd*nu**2*v2+gmr*(2*rd+20.5*rd*nu+4*rd*nu**2))
    return -(Gn*m/r**2)*((1+A)*n+B*vv)

def run(nu,e,cl,n_orb,kw,kwv):
    Gn,m=1.0,1.0; qn=1.0/cl**2
    X1m=(1+np.sqrt(max(0,1-4*nu)))/2; X2m=1-X1m
    p=1.0; a_s=p/(1-e*e); r0=p/(1+e); v0=np.sqrt(Gn*m*(2/r0-1/a_s))
    rv0=np.array([r0,0.]); vv0=np.array([0.,v0]); u=Gn*m/(cl**2*p)
    T=2*np.pi*np.sqrt(a_s**3/Gn/m)
    Delta=X1m-X2m; P1=nu*Delta*(v0**2/2-Gn*m/(2*r0))*qn
    z1=(X2m+P1)*rv0; z2=(-X1m+P1)*rv0; w1=(X2m+P1)*vv0; w2=(-X1m+P1)*vv0
    def rT(t,s):
        a=tuple(s)+(Gn,X1m,X2m,qn,1.0,kw,kwv)
        return np.concatenate([s[4:8],f1(*a),f2(*a)])
    def rG(t,s): return np.concatenate([s[2:4],gr_acc(s[:2],s[2:4],Gn,m,qn,nu)])
    tf=n_orb*T
    sT=solve_ivp(rT,(0,tf),np.concatenate([z1,z2,w1,w2]),method='DOP853',rtol=1e-10,atol=1e-13)
    sG=solve_ivp(rG,(0,tf),np.concatenate([rv0,vv0]),method='DOP853',rtol=1e-10,atol=1e-13)
    def om(rv,vv):
        L=rv[0]*vv[1]-rv[1]*vv[0]
        ev=np.array([vv[1]*L,-vv[0]*L])/(Gn*m)-rv/np.hypot(*rv)
        return np.arctan2(ev[1],ev[0])
    rT_=sT.y[0:2,-1]-sT.y[2:4,-1]; vT_=sT.y[4:6,-1]-sT.y[6:8,-1]
    rT0=sT.y[0:2,0]-sT.y[2:4,0]; vT0=sT.y[4:6,0]-sT.y[6:8,0]
    dd=(om(rT_,vT_)-om(rT0,vT0))-(om(sG.y[0:2,-1],sG.y[2:4,-1])-om(sG.y[0:2,0],sG.y[2:4,0]))
    dd=(dd+np.pi)%(2*np.pi)-np.pi
    return dd/n_orb/(np.pi*u*u)

print("bande de complétion à (nu=0.25, e=0.1) :", flush=True)
for kw,kwv in ((0,0),(1,0),(-1,0),(0,1),(0,-1)):
    k=run(0.25,0.1,30.0,30,float(kw),float(kwv))
    print(f"  KW={kw:+d} KWV={kwv:+d} : delta k2 = {k:+.3f}", flush=True)
