"""Sonde d'épinglage de la charge-source (verrou 1PN = EIH).

Dérivation 1PN par corps de la trame avec trois coefficients-sondes :
  KD : habillage de potentiel de la charge (-KD·V_autre) ;
  KR : terme de retard scalaire ((KR/2c²) d²/dt²[m rho]) ;
  KI : couplage vectoriel (8·KI·V_i v^i dans dtau/dt²).
Réduction au CM newtonienne, comparaison symbolique à la RG 1PN
relative (A1/B1, LRR éq. 358a/359a).

RÉSULTAT (2026-08-21) : l'unique solution est (KD, KR, KI) = (0, 1, 1)
— le retard (-1/2) et la dérive du terme de Darwin (-2) saturent le
coefficient -5/2 d'EIH : AUCUN habillage de potentiel n'est permis.
Le poids covariant naïf u0/sqrt(-g) (qui donne -V) sur-tire ; la
charge épinglée est s = m·u0·(1+v²/c²)·e^{-V/c²} (écrantage
exponentiel — le facteur de rougissement de la trame). Avec KI=1 le
résidu n'a AUCUN terme de vitesse : tout le secteur cinétique/Darwin/
retard de la construction est validé d'un coup.
"""
import sympy as sp, time
t0=time.time()
q = sp.Symbol('q', positive=True)
G, M1, M2 = sp.symbols('G m1 m2', positive=True)
LAM = sp.Integer(1)
KD, KR, KI = sp.symbols('KD KR KI')   # sondes: habillage -V, retard, vecteur
x1,y1,x2,y2,vx1,vy1,vx2,vy2 = sp.symbols('x1 y1 x2 y2 vx1 vy1 vx2 vy2', real=True)
XF,YF = sp.symbols('XF YF', real=True)
Z1=sp.Matrix([x1,y1]); Z2=sp.Matrix([x2,y2]); W1=sp.Matrix([vx1,vy1]); W2=sp.Matrix([vx2,vy2])
def dot(a,b): return (a.T*b)[0,0]
def trunc(e,n):
    e=sp.expand(e); return sp.Add(*[e.coeff(q,k)*q**k for k in range(n+1)])
r12v=Z1-Z2; r12=sp.sqrt(dot(r12v,r12v)); v1sq=dot(W1,W1); v2sq=dot(W2,W2)
aN1=-G*M2*r12v/r12**3; aN2=+G*M1*r12v/r12**3
def Dt(expr):
    d=0
    for pos,vel,acc in ((Z1,W1,aN1),(Z2,W2,aN2)):
        for k in range(2):
            d+=sp.diff(expr,pos[k])*vel[k]+sp.diff(expr,vel[k])*acc[k]
    return d

# source habillee du corps 2 (1PN): s2 = m2[1 + q((1/2+lam)v2^2 - KD*V1)]
V1at2 = G*M1/r12
s2 = M2*(1 + q*((sp.Rational(1,2)+LAM)*v2sq - KD*V1at2))
rho2F = sp.sqrt((XF-x2)**2+(YF-y2)**2)
Vf = G*s2/rho2F + KR*(q*G/2)*Dt(Dt(M2*rho2F))
Vloc = trunc(Vf,1).subs({XF:x1,YF:y1})
Viloc = (G*M2*W2/rho2F).subs({XF:x1,YF:y1})

Aarg = trunc(1 - 2*q*Vloc + 2*q**2*Vloc**2 + KI*8*q**2*dot(Viloc,W1) - (1+2*q*Vloc)*q*v1sq, 2)
w = trunc(Aarg-1,2)
S = trunc(1 + w/2 - w**2/8 + w**3/16, 2)
L1 = sp.expand(-(M1/q)*S)
L1 = sp.Add(*[L1.coeff(q,k)*q**k for k in range(-1,2)])

p1 = sp.Matrix([sp.diff(L1,vx1), sp.diff(L1,vy1)])
F1 = sp.Matrix([sp.diff(L1,x1), sp.diff(L1,y1)])
H = sp.Matrix([[sp.diff(p1[i],[vx1,vy1][j]) for j in range(2)] for i in range(2)])
drift = sp.Matrix([Dt(p1[i]) for i in range(2)])   # sans a1 (H*a1 separe)
rhs = F1 - drift
hq = ((H-M1*sp.eye(2))/M1).applyfunc(lambda e: trunc(e,1))
Hinv = (sp.eye(2)-hq)/M1
a1 = sp.Matrix([trunc(sp.expand((Hinv*rhs)[i]),1) for i in range(2)])
swap = {x1:x2,x2:x1,y1:y2,y2:y1,vx1:vx2,vx2:vx1,vy1:vy2,vy2:vy1,M1:M2,M2:M1}
a2 = sp.Matrix([a1[i].subs(swap,simultaneous=True) for i in range(2)])

# reduction CM newtonienne, comparaison A1/B1
xr,yr,vxr,vyr = sp.symbols('xr yr vxr vyr', real=True)
X2s,Ms = sp.symbols('X2 M', positive=True)
sub = {x1:X2s*xr, y1:X2s*yr, x2:-(1-X2s)*xr, y2:-(1-X2s)*yr,
       vx1:X2s*vxr, vy1:X2s*vyr, vx2:-(1-X2s)*vxr, vy2:-(1-X2s)*vyr,
       M1:(1-X2s)*Ms, M2:X2s*Ms}
arel = sp.Matrix([sp.expand((a1[i]-a2[i]).coeff(q,1)).subs(sub) for i in range(2)])
r=sp.sqrt(xr**2+yr**2); v2r=vxr**2+vyr**2; rd=(xr*vxr+yr*vyr)/r
nu=(1-X2s)*X2s
A1=-sp.Rational(3,2)*rd**2*nu+v2r+3*nu*v2r-(G*Ms/r)*(4+2*nu)
B1=-4*rd+2*rd*nu
aref=-(G*Ms/r**2)*(A1*sp.Matrix([xr,yr])/r+B1*sp.Matrix([vxr,vyr]))
res=[sp.simplify(sp.expand(arel[i]-aref[i])) for i in range(2)]
print("residu x:", res[0])
sol=sp.solve([sp.expand(res[0]*r**4), sp.expand(res[1]*r**4)], [KD,KR,KI], dict=True)
print("solutions sondes:", sol)
print(f"({time.time()-t0:.0f}s)")
