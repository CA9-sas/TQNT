"""Secteur cinématique 2PN de la trame (cible n°2) — dérivation complète.

COMPLÉTION (C6), qui étend C1-C5 du brouillon EOM N-corps aux sources
MOBILES — chaque ingrédient est déjà affirmé ou utilisé par le document principal :
  - le secteur LINÉAIRE de la trame = RG linéarisée (référence : gamma=1,
    Lense-Thirring exact, GW à c) => les potentiels V (scalaire) et
    V_i (vecteur) sont les intégrales RETARDÉES linéaires standard,
    sourcées par sigma = (T^tt + T^kk)/c² et sigma_i = T^ti (sources
    covariantes du point-nœud dans le champ DES AUTRES — pas
    d'auto-source, C4) ; le paramètre lambda_s (1 = T^tt+T^kk canonique
    par le secteur linéaire RG ; 0 = variante « énergie seule ») est
    gardé explicite et testé ;
  - secteur conservatif = partie PAIRE du développement du retard
    (démi-retardé + demi-avancé ; les ordres impairs = rayonnement,
    hors périmètre) :
      V  = G[s/rho] + (G/2c²) d²/dt²[s rho] + (G/24c⁴) d⁴/dt⁴[m rho³]
      V_i = G[s v_i/rho] + (G/2c²) d²/dt²[m v_i rho]
  - formes fermées canoniques (la SEULE non-linéarité, V6 eq:BH-metric):
      -g_00 = e^{-2V/c²},  g_0i = -4V_i/c³,  g_ij = (1+2V/c²) delta_ij
  - mouvement : géodésique de chaque corps dans le champ des autres.

FAIT DE STRUCTURE : avec lambda_s=1, la métrique 1PN de la trame est
IDENTIQUE à celle de la RG en jauge harmonique (e^{-2V} et 1-2V+2V²
coïncident jusqu'à V² ; 1+2V commun ; -4V_i commun) => les dynamiques
1PN coïncident ÉQUATION PAR ÉQUATION (EIH), et les coordonnées des
deux théories coïncident à 1PN : la comparaison « mêmes conditions
initiales » est propre jusqu'à 2PN (les écarts d'étiquette sont 3PN).

MÉTHODE :
  [S] sympy : L_1 = -m_1 c² dτ/dt développé au 2PN complet (sources
      habillées, retard, réduction d'ordre it. 1PN) -> a_1, a_2 par
      corps ; contrôle symbolique : partie newtonienne exacte.
  [N] numérique : intégration par corps de la trame (CI au CM 1PN,
      éq. 354-355 Blanchet) vs RG relative 2PN harmonique (A/B,
      éq. 357-359 LRR 2014, transcrites) ; différence des avances de
      péricentre aux mêmes CI relatives -> delta k2(nu, e) ;
      delta(Delta omega) = pi u² delta k2.
  ANCRES : (i) troncature 1PN : delta k2 -> 0 (les 1PN coïncident) ;
  (ii) nu -> 0 : delta k2 -> -(8 + 3e²/2) (résultat test-masse EXACT,
  métriques exactes, brouillon précédent) ; (iii) partie newtonienne
  symbolique exacte.
SORTIE : table delta k2(nu, e), loi ajustée, verdict J0737-3039.
"""

import time
import numpy as np
import sympy as sp

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ================================================================
# [S] SYMBOLIQUE — EOM par corps de la trame au 2PN
# Moteur : SYMENGINE (cœur C++) — sympy s'effondrait sur les diff/
# expand des expressions à ~10³ termes (D⁴, dérive d'Euler-Lagrange).
# Représentation : polynômes en q par listes de coefficients.
# ================================================================
import symengine as se

q = sp.Symbol('q', positive=True)          # côté sympy (numérique)
qe = se.Symbol('q')
G, M1, M2 = se.symbols('G m1 m2')
LAM, KW, KWV = se.symbols('lambda_s KW KWV')
x1, y1, x2, y2 = se.symbols('x1 y1 x2 y2')
vx1, vy1, vx2, vy2 = se.symbols('vx1 vy1 vx2 vy2')
XF, YF = se.symbols('XF YF')

Z1 = [x1, y1]; Z2 = [x2, y2]
W1 = [vx1, vy1]; W2 = [vx2, vy2]
S0, S1 = se.Integer(0), se.Integer(1)
R12, R32, R13 = se.Rational(1, 2), se.Rational(3, 2), se.Rational(1, 3)


def dot(a, b):
    return a[0]*b[0] + a[1]*b[1]


r12v = [x1 - x2, y1 - y2]
r12 = se.sqrt(dot(r12v, r12v))
v1sq = dot(W1, W1); v2sq = dot(W2, W2)
aN1 = [-G*M2*r12v[k]/r12**3 for k in range(2)]
aN2 = [+G*M1*r12v[k]/r12**3 for k in range(2)]
rho1F = se.sqrt((XF - x1)**2 + (YF - y1)**2)
rho2F = se.sqrt((XF - x2)**2 + (YF - y2)**2)

EXPC = [S1, S1, se.Rational(1, 2), se.Rational(1, 6), se.Rational(1, 24)]
SQRTC = [S1, se.Rational(1, 2), se.Rational(-1, 8), se.Rational(1, 16),
         se.Rational(-5, 128)]
ISQRTC = [S1, se.Rational(-1, 2), se.Rational(3, 8),
          se.Rational(-5, 16), se.Rational(35, 128)]


def nterms(e):
    e = se.expand(e)
    return len(e.args) if e.func == se.Add else 1


def qp(A, n):
    A = list(A)[:n + 1]
    return A + [S0]*(n + 1 - len(A))


def qp_mul(A, B, n):
    C = [S0]*(n + 1)
    for i, a in enumerate(A[:n + 1]):
        if a == 0:
            continue
        for j, b in enumerate(B[:n + 1 - i]):
            if b == 0:
                continue
            C[i + j] = C[i + j] + se.expand(a*b)
    return C


def qp_add(n, *As):
    C = [S0]*(n + 1)
    for A in As:
        for k, a in enumerate(A[:n + 1]):
            C[k] = C[k] + a
    return C


def qp_shift(A, m, n):
    return qp([S0]*m + list(A), n)


def qp_scale(A, sc, n):
    return [se.expand(sc*a) if a != 0 else S0 for a in qp(A, n)]


def qp_series(cs, W, n):
    W = qp(W, n)
    assert W[0] == 0
    R = [S0]*(n + 1)
    R[0] = cs[0]
    P = qp([S1], n)
    for k in range(1, len(cs)):
        P = qp_mul(P, W, n)
        if all(p == 0 for p in P):
            break
        for i in range(n + 1):
            if P[i] != 0:
                R[i] = R[i] + cs[k]*P[i]
    return [se.expand(r) for r in R]


def Dt_q(A, n, accs):
    """d/dt d'une liste q-poly. accs = ((aN1|None, a1PN1|None),
    (aN2|None, a1PN2|None))."""
    R = [S0]*(n + 1)
    for k in range(n + 1):
        e = A[k] if k < len(A) else S0
        if e == 0:
            continue
        for (pos, vel), (accN, accP) in zip(((Z1, W1), (Z2, W2)), accs):
            for c_ in range(2):
                de = e.diff(pos[c_])
                if de != 0:
                    R[k] = R[k] + de*vel[c_]
                dv = e.diff(vel[c_])
                if dv != 0:
                    if accN is not None:
                        R[k] = R[k] + dv*accN[c_]
                    if accP is not None and k + 1 <= n:
                        R[k + 1] = R[k + 1] + dv*accP[c_]
    return [se.expand(r) for r in R]


def build_a1(order, a1pn1=None, a1pn2=None):
    """EOM du corps 1 à l'ordre PN donné (listes q-poly symengine)."""
    ACCF = ((aN1, a1pn1), (aN2, a1pn2))
    ACCN = ((aN1, None), (aN2, None))
    n = order

    # ---- potentiel de 1 en z2 (habillage), longueur order ----
    if order == 1:
        V1at2 = [G*M1/r12]
    else:
        ret1 = Dt_q(Dt_q([M1*rho1F], 0, ACCN), 0, ACCN)[0]
        V1at2 = [G*M1/r12,
                 se.expand(G*M1*((R12 + LAM)*v1sq - G*M2/r12)/r12
                           + (G/2)*ret1.subs({XF: x2, YF: y2}))]
    V1iat2 = [G*M1*W1[k]/r12 for k in range(2)]
    log(f'    [o{order}] V1at2 fait')

    # ---- charge source ÉPINGLÉE 1PN : s = m u0 (1+lam q v²) e^{-qV}
    # (verrou 1PN : pas d'habillage de potentiel au 1er ordre — sonde
    #  KD/KR/KI ; bande résiduelle q² : KW scalaire, KWV vecteur)
    WA = qp_shift([-2*v for v in V1at2], 1, n)
    ee = qp_series(EXPC, WA, n)
    gi = qp([S0, S0, 8*dot(V1iat2, W2)], n)
    mv = qp_mul(qp([S1] + [2*v for v in V1at2], n), qp([S0, v2sq], n), n)
    A2m = qp_add(n, ee, gi, [-t for t in mv])
    u0 = qp_series(ISQRTC, [S0] + A2m[1:], n)
    scr = qp_series(EXPC, qp_shift([-v for v in V1at2], 1, n), n)
    s2 = qp_mul(qp_mul(u0, qp([S1, LAM*v2sq], n), n), scr, n)
    s2 = qp_mul(s2, qp([S1, S0, KW*V1at2[0]**2], n), n)
    s2 = qp_scale(s2, M2, n)
    log(f'    [o{order}] s2 fait')

    # ---- potentiels du corps 2 au point de champ ----
    s2rho = qp_mul(qp(s2, order - 1), [rho2F], order - 1)
    ret2 = Dt_q(Dt_q(s2rho, order - 1, ACCF), order - 1, ACCF)
    log(f'      ret2: {[nterms(t) for t in ret2]}')
    pieces = [qp_scale(s2, G/rho2F, n),
              qp_shift(qp_scale(ret2, G/2, n - 1), 1, n)]
    if order >= 2:
        # d⁴/dt⁴[rho³] dans l'ALGÈBRE FERMÉE des produits scalaires
        # A = x - z2 (x fixe), B = z1 - z2, w = v1 - v2
        sa, sb, sc, sd, sE, sf, sg, sh, si, sj = se.symbols(
            'sa sb sc sd sE sf sg sh si sj')
        GM1, GMT = G*M1, G*(M1 + M2)
        em32 = sE**se.Rational(-3, 2)
        em12 = sE**se.Rational(-1, 2)
        rules = {sa: -2*sb, sb: -sc + GM1*sd*em32,
                 sc: 2*GM1*sf*em32, sd: -sf + sh, sE: 2*sg,
                 sf: GM1*em12 + si, sg: sj - GMT*em12,
                 sh: -si - GMT*sd*em32,
                 si: (GM1*sg - GMT*sf)*em32, sj: -2*GMT*sg*em32}

        def Dscal(e_):
            d = S0
            for sym, r_ in rules.items():
                de = e_.diff(sym)
                if de != 0:
                    d = d + de*r_
            return se.expand(d)

        AF = [XF - x2, YF - y2]
        Wrel = [vx1 - vx2, vy1 - vy2]
        comp = {sa: dot(AF, AF), sb: dot(AF, W2), sc: v2sq,
                sd: dot(AF, r12v), sE: dot(r12v, r12v),
                sf: dot(r12v, W2), sg: dot(r12v, Wrel),
                sh: dot(AF, Wrel), si: dot(W2, Wrel),
                sj: dot(Wrel, Wrel)}
        dS = sa**se.Rational(3, 2)
        for it_ in range(4):
            dS = Dscal(dS)
            if it_ == 2:
                # contrôle numérique : niveau 3 == composantes
                d3c = M2*rho2F**3
                for _ in range(3):
                    d3c = se.expand(Dt_q([d3c], 0, ACCN)[0])
                diffchk = M2*dS.subs(comp) - d3c
                import random as _rd
                _rd.seed(7)
                syms = [x1, y1, x2, y2, vx1, vy1, vx2, vy2, XF, YF,
                        G, M1, M2]
                for _t in range(3):
                    vals = {sym: se.Rational(
                        int(_rd.uniform(300, 1200)), 997)
                        for sym in syms}
                    ref = abs(float(d3c.subs(vals)))
                    err = abs(float(diffchk.subs(vals)))/max(ref, 1e-30)
                    assert err < 1e-10, f"algebre scalaire d4: {err}"
                log('      d4 : contrôle niveau 3 numérique [OK]')
        d4 = se.expand(M2*dS.subs(comp))
        log(f'      d4 fait: {nterms(d4)} termes')
        pieces.append(qp_shift([se.expand(G*d4/24)], 2, n))
    Vf = qp_add(n, *pieces)
    Vloc = [se.expand(v).subs({XF: x1, YF: y1}) for v in Vf]
    log(f'    [o{order}] Vloc fait ({[nterms(v) for v in Vloc]})')

    nv = order - 1
    if order >= 2:
        s2v = [M2, se.expand(M2*(v2sq/2 + KWV*V1at2[0]))]
    else:
        s2v = [M2]
    Viloc = []
    for k_ in range(2):
        Vi = qp_scale(qp_mul(qp(s2v, nv), [W2[k_]], nv), G/rho2F, nv)
        if order >= 2:
            retv = Dt_q(Dt_q([M2*W2[k_]*rho2F], 0, ACCN), 0, ACCN)
            Vi = qp_add(nv, Vi, qp_shift(qp_scale(retv, G/2, 0), 1, nv))
        Viloc.append([se.expand(v).subs({XF: x1, YF: y1}) for v in Vi])

    # ---- lagrangien géodésique du corps 1 ----
    npl = order + 1
    WL = qp_shift([-2*v for v in Vloc], 1, npl)
    eeL = qp_series(EXPC, WL, npl)
    giL = qp_shift([se.expand(8*(Viloc[0][k_]*vx1 + Viloc[1][k_]*vy1))
                    for k_ in range(len(Viloc[0]))], 2, npl)
    mvL = qp_mul(qp([S1] + [2*v for v in Vloc], npl),
                 qp([S0, v1sq], npl), npl)
    AargL = qp_add(npl, eeL, giL, [-t for t in mvL])
    SL = qp_series(SQRTC, [S0] + AargL[1:], npl)
    LG = [se.expand(-M1*SL[k + 1]) for k in range(order + 1)]
    log(f'    [o{order}] LG fait ({[nterms(v) for v in LG]})')

    # ---- Euler-Lagrange, résolution ordre par ordre ----
    p1 = [[LG[k].diff(vx1), LG[k].diff(vy1)]
          for k in range(order + 1)]
    F1 = [[LG[k].diff(x1), LG[k].diff(y1)]
          for k in range(order + 1)]
    Hm = [[[p1[k][i].diff([vx1, vy1][j]) for j in range(2)]
           for i in range(2)] for k in range(order + 1)]
    DRIFT_ACCS = ((None, None), (aN2, a1pn2))
    log(f'    [o{order}] avant drift')
    drift = [Dt_q([p1[k][i] for k in range(order + 1)], order,
                  DRIFT_ACCS) for i in range(2)]
    log(f'    [o{order}] drift fait')
    a1 = []
    for k in range(order + 1):
        comp_ = []
        for i in range(2):
            rhs = F1[k][i] - drift[i][k]
            for j in range(1, k + 1):
                for l_ in range(2):
                    if Hm[j][i][l_] != 0:
                        rhs = rhs - se.expand(Hm[j][i][l_]*a1[k - j][l_])
            comp_.append(se.expand(rhs/M1))
        a1.append(comp_)
        log(f'    [o{order}] a1[{k}] fait ({nterms(comp_[0])} termes)')
    return a1


import os, pickle
CACHE = 'output/cache_eom_trame_2pn.pkl'
if os.path.exists(CACHE):
    log("cache trouvé : chargement des EOM symboliques")
    with open(CACHE, 'rb') as fh:
        _d = pickle.load(fh)
    a1_full = sp.Matrix(sp.sympify(_d['a1']))
    a2_full = sp.Matrix(sp.sympify(_d['a2']))
else:
    log("passe 1 : EOM 1PN de la trame...")
    a1_l = build_a1(1)
    for i in range(2):
        assert se.expand(a1_l[0][i] - aN1[i]) == 0, "newtonien faux !"
    log("  partie newtonienne exacte  [OK]")
    swap = {x1: x2, x2: x1, y1: y2, y2: y1, vx1: vx2, vx2: vx1,
            vy1: vy2, vy2: vy1, M1: M2, M2: M1}
    a1pn1 = [a1_l[1][0], a1_l[1][1]]
    a1pn2 = [a1pn1[i].subs(swap) for i in range(2)]
    log("passe 2 : EOM 2PN de la trame...")
    a2_l = build_a1(2, a1pn1, a1pn2)
    for i in range(2):
        assert se.expand(a2_l[0][i] - aN1[i]) == 0, "newtonien faux !"
    # conversion sympy (lambdify + cache)
    a1_full = sp.Matrix([sp.sympify(
        (a2_l[0][i] + qe*a2_l[1][i] + qe**2*a2_l[2][i])._sympy_())
        for i in range(2)])
    swap_sp = {sp.Symbol(a): sp.Symbol(b) for a, b in
               [('x1', 'x2'), ('x2', 'x1'), ('y1', 'y2'), ('y2', 'y1'),
                ('vx1', 'vx2'), ('vx2', 'vx1'), ('vy1', 'vy2'),
                ('vy2', 'vy1'), ('m1', 'm2'), ('m2', 'm1')]}
    a2_full = sp.Matrix([a1_full[i].subs(swap_sp, simultaneous=True)
                         for i in range(2)])
    log(f"  a1 : {len(sp.Add.make_args(sp.expand(a1_full[0])))} termes")
    os.makedirs('output', exist_ok=True)
    with open(CACHE, 'wb') as fh:
        pickle.dump({'a1': sp.srepr(a1_full), 'a2': sp.srepr(a2_full)},
                    fh)
    log("  cache écrit")

# symboles sympy pour la partie numérique
x1, y1, x2, y2 = sp.symbols('x1 y1 x2 y2', real=True)
vx1, vy1, vx2, vy2 = sp.symbols('vx1 vy1 vx2 vy2', real=True)
G, M1, M2 = sp.symbols('G m1 m2', positive=True)
LAM, KW, KWV = sp.symbols('lambda_s KW KWV')

# ================================================================
# [N] NUMÉRIQUE
# ================================================================
from scipy.integrate import solve_ivp

log("lambdify (cse + backend math)...")
STATE = (x1, y1, x2, y2, vx1, vy1, vx2, vy2)
PARS = (G, M1, M2, q, LAM, KW, KWV)
f_a1 = sp.lambdify(STATE + PARS, list(a1_full), modules='math',
                   cse=True)
f_a2 = sp.lambdify(STATE + PARS, list(a2_full), modules='math',
                   cse=True)
# version tronquée 1PN (ancre (i)) — les entrées du cache sont déjà
# structurellement développées : coeff() direct, sans expand sympy
swap_sp = {sp.Symbol(a): sp.Symbol(b) for a, b in
           [('x1', 'x2'), ('x2', 'x1'), ('y1', 'y2'), ('y2', 'y1'),
            ('vx1', 'vx2'), ('vx2', 'vx1'), ('vy1', 'vy2'),
            ('vy2', 'vy1'), ('m1', 'm2'), ('m2', 'm1')]}
a1_1pn = sp.Matrix([a1_full[i].coeff(q, 0) + a1_full[i].coeff(q, 1)*q
                    for i in range(2)])
a2_1pn = sp.Matrix([a1_1pn[i].subs(swap_sp, simultaneous=True)
                    for i in range(2)])
f_a1_1 = sp.lambdify(STATE + PARS, list(a1_1pn), modules='math',
                     cse=True)
f_a2_1 = sp.lambdify(STATE + PARS, list(a2_1pn), modules='math',
                     cse=True)
log("lambdify fait")


def gr_rel_acc(rv, vv, Gn, m, qn, pn_max=2):
    """RG relative harmonique CM, LRR 2014 éq. 357-359 (conservatif)."""
    r = np.hypot(rv[0], rv[1])
    n = rv/r
    v2 = vv @ vv
    rd = n @ vv
    gmr = Gn*m/r
    nu = gr_rel_acc.nu
    A = 0.0
    B = 0.0
    if pn_max >= 1:
        A += qn*(-1.5*rd**2*nu + v2 + 3*nu*v2 - gmr*(4 + 2*nu))
        B += qn*(-4*rd + 2*rd*nu)
    if pn_max >= 2:
        A += qn**2*(15*rd**4*nu/8 - 45*rd**4*nu**2/8 - 4.5*rd**2*nu*v2
                    + 6*rd**2*nu**2*v2 + 3*nu*v2**2 - 4*nu**2*v2**2
                    + gmr*(-2*rd**2 - 25*rd**2*nu - 2*rd**2*nu**2
                           - 6.5*nu*v2 + 2*nu**2*v2)
                    + gmr**2*(9 + 87*nu/4))
        B += qn**2*(4.5*rd**3*nu + 3*rd**3*nu**2 - 7.5*rd*nu*v2
                    - 2*rd*nu**2*v2 + gmr*(2*rd + 20.5*rd*nu
                                           + 4*rd*nu**2))
    return -(Gn*m/r**2)*((1 + A)*n + B*vv)


def run_pair(nu, e, cl, n_orb, pn_max=2, lam=1.0, kw=0.0, kwv=0.0):
    """intègre trame (par corps) et RG (relative) aux mêmes CI
    relatives ; renvoie (dOmega_trame - dOmega_GR)/orbite et u."""
    Gn, m = 1.0, 1.0
    qn = 1.0/cl**2
    X1m = (1 + np.sqrt(max(0.0, 1 - 4*nu)))/2
    X2m = 1 - X1m
    m1n, m2n = X1m*m, X2m*m
    p = 1.0
    a_sma = p/(1 - e**2)
    r0 = p/(1 + e)                      # départ au péricentre
    v0 = np.sqrt(Gn*m*(2/r0 - 1/a_sma))
    rv0 = np.array([r0, 0.0])
    vv0 = np.array([0.0, v0])
    u = Gn*m/(cl**2*p)
    T = 2*np.pi*np.sqrt(a_sma**3/(Gn*m))

    # CI par corps au CM 1PN (Blanchet 354-355 ; rdot=0 au péricentre)
    Delta = X1m - X2m
    P1 = nu*Delta*(v0**2/2 - Gn*m/(2*r0))*qn
    z1 = (X2m + P1)*rv0
    z2 = (-X1m + P1)*rv0
    w1 = (X2m + P1)*vv0                  # Pdot = 0 à rdot = 0
    w2 = (-X1m + P1)*vv0

    fa1 = f_a1 if pn_max == 2 else f_a1_1
    fa2 = f_a2 if pn_max == 2 else f_a2_1

    def rhs_T(t, s):
        args = tuple(s) + (Gn, m1n, m2n, qn, lam, kw, kwv)
        return np.concatenate([s[4:8], np.asarray(fa1(*args)).ravel(),
                               np.asarray(fa2(*args)).ravel()])

    def rhs_G(t, s):
        return np.concatenate([s[2:4],
                               gr_rel_acc(s[:2], s[2:4], Gn, m, qn,
                                          pn_max)])

    gr_rel_acc.nu = nu
    tf = n_orb*T
    sT = solve_ivp(rhs_T, (0, tf), np.concatenate([z1, z2, w1, w2]),
                   method='DOP853', rtol=1e-10, atol=1e-13)
    sG = solve_ivp(rhs_G, (0, tf), np.concatenate([rv0, vv0]),
                   method='DOP853', rtol=1e-10, atol=1e-13)

    def omega_rel(rv, vv):
        L = rv[0]*vv[1] - rv[1]*vv[0]
        ev = np.array([vv[1]*L, -vv[0]*L])/(Gn*m) - rv/np.hypot(*rv)
        return np.arctan2(ev[1], ev[0])

    rT = sT.y[0:2, -1] - sT.y[2:4, -1]
    vT = sT.y[4:6, -1] - sT.y[6:8, -1]
    rT0 = sT.y[0:2, 0] - sT.y[2:4, 0]
    vT0 = sT.y[4:6, 0] - sT.y[6:8, 0]
    dwT = omega_rel(rT, vT) - omega_rel(rT0, vT0)
    dwG = omega_rel(sG.y[0:2, -1], sG.y[2:4, -1]) \
        - omega_rel(sG.y[0:2, 0], sG.y[2:4, 0])
    dd = (dwT - dwG + np.pi) % (2*np.pi) - np.pi
    return dd/n_orb, u


print()
log("[ancre i] troncature 1PN : delta doit tendre vers 0 en O(u³)")
for cl in (30.0, 60.0):
    d, u = run_pair(0.24, 0.4, cl, 20, pn_max=1)
    log(f"  u = {u:.2e} : delta/orbite = {d:+.3e} rad "
        f"(delta/pi u² = {d/(np.pi*u*u):+.4f})")

log("[ancre ii] nu -> 0 : delta k2 -> -(8+1.5e²)")
for e in (0.3, 0.7):
    d, u = run_pair(1e-8, e, 30.0, 30)
    k2 = d/(np.pi*u*u)
    log(f"  e = {e} : delta k2 = {k2:+.4f}  attendu {-(8+1.5*e*e):+.4f}")

log("table delta k2(nu, e) — c = 30, 30 orbites")
print(f"   {'nu':>6s} {'e':>5s} {'delta k2':>10s}")
results = []
for nu in (1e-8, 0.10, 0.20, 0.25):
    for e in (0.1, 0.4, 0.7):
        d, u = run_pair(nu, e, 30.0, 30)
        k2 = d/(np.pi*u*u)
        results.append((nu, e, k2))
        print(f"   {nu:6.2f} {e:5.2f} {k2:+10.4f}", flush=True)

# ajustement delta k2 = c00 + c01 e² + nu (c10 + c11 e²) + nu² (c20+c21 e²)
Amat = np.array([[1, e*e, nu, nu*e*e, nu*nu, nu*nu*e*e]
                 for nu, e, _ in results])
bvec = np.array([k for _, _, k in results])
coef, res, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
resid = Amat @ coef - bvec
log(f"loi ajustée : delta k2 = ({coef[0]:+.3f}) + ({coef[1]:+.3f}) e² "
    f"+ nu[({coef[2]:+.3f}) + ({coef[3]:+.3f}) e²] "
    f"+ nu²[({coef[4]:+.3f}) + ({coef[5]:+.3f}) e²]")
log(f"  résidu max de l'ajustement : {np.abs(resid).max():.2e}")

# ================================================================
# Verdict J0737-3039
# ================================================================
nu_p, e_p, u_p = 0.2497, 0.0878, 4.38e-6
k2_p = (coef[0] + coef[1]*e_p**2 + nu_p*(coef[2] + coef[3]*e_p**2)
        + nu_p**2*(coef[4] + coef[5]*e_p**2))
k2_stat = -((8 + 1.5*e_p**2) - nu_p*(23.5 + 4.5*e_p**2))
print()
log("VERDICT J0737-3039 (nu = 0.2497, e = 0.0878, u = 4.38e-6) :")
log(f"  delta k2 COMPLET (statique+cinématique) = {k2_p:+.3f}")
log(f"  (secteur statique seul, brouillon préc. : {k2_stat:+.3f})")
frac = k2_p*u_p/6
log(f"  écart relatif sur omega_dot = {frac:+.3e}"
    f"  (précision mesurée : 1.3e-6)")
log(f"  ratio |écart|/précision = {abs(frac)/1.3e-6:.2f}")
