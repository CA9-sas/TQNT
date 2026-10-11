"""EOM à N corps (2, 3, 4) de la trame — mise en place + déviation 2PN.

COMPLÉTION MINIMALE DÉCLARÉE (conditionnelle, brouillon) :
  (C1) chaque nœud suit une géodésique de la métrique construite sur le
       champ de déplétion DES AUTRES (pas d'auto-force ; équivalence V6) ;
  (C2) h superpose LINÉAIREMENT (∇²h = -κ_g|Ψ|² est linéaire, référence) :
       h_a(x) = Σ_{b≠a} 2Gm_b/(c²|x-x_b|) ;
  (C3) secteur gravitomagnétique = forme RG linéarisée (V6, section
       Lense-Thirring nodal : g_0i sourcé par le courant |Ψ|²v) ;
  (C4) le champ h n'a PAS d'auto-source (pas de t_LL) — c'est la
       linéarité du document principal, déclarée ;
  (C5) propagation standard (□, ondes à c — V6 GW).

MÉTRIQUE PAR CORPS (formes fermées canoniques) :
    ds²_a = -e^{-h_a} c²dt² + (1+h_a) dx²    (+ g_0i à 1PN)

EOM EXACTE (secteur statique), D_a = sqrt(e^{-h_a} - (1+h_a)v_a²/c²) :
    d/dt [ (1+h_a) v_a / D_a ] = -(c²/2D_a)(e^{-h_a} + v_a²/c²) ∇h_a
Limite newtonienne : dv/dt = -Σ Gm_b n̂/r².  Même forme pour N = 2, 3, 4
— seul h_a (la somme) change.

THÉORÈME A (1PN) : β_PPN = γ_PPN = 1 et secteur conservatif sans
référentiel privilégié => le lagrangien N-corps 1PN est EXACTEMENT
Einstein-Infeld-Hoffmann (le lagrangien PPN à N corps ne dépend que de
(β, γ)). Conséquences : Nordtvedt η = 4β-γ-3 = 0 (LLR passe),
éphémérides identiques RG, points de Lagrange, ω̇ 1PN du pulsar.

THÉORÈME B (déviation 2PN, secteur statique) : les écarts de forme
(δα₃ = +1/6 dans e^{-h} ; δb₂ = -3/2 dans 1+h vs Schwarzschild)
donnent la déviation lagrangienne PAR CORPS (U_a = Σ_{b≠a} Gm_b/r_ab) :

    δL = Σ_a m_a [ -(δα₃/2) U_a³ + (δb₂/2) U_a² v_a² ] / c⁴

U_a³ contient des produits m_b m_c m_d : la déviation porte des
interactions à 3 ET 4 corps irréductibles (le terme à 4 corps n'existe
que pour N ≥ 4).

À DEUX CORPS (masse réduite, ν = m₁m₂/M², v = vitesse relative) :
    δL̂ = -(δα₃/2)(GM)³(1-2ν)/(c⁴r³) + (δb₂/2)(GM)²(1-3ν) v²/(c⁴r²)
(identités m₁m₂(m₁²+m₂²)/μ = M³(1-2ν), (m₁³+m₂³)/M = M²(1-3ν),
vérifiées ici par sympy).  Moyenne séculaire de Delaunay
(Δω = -∂/∂ℓ ∮δH dt à énergie fixée, ∮dt/r³ = 2πGM/ℓ³,
∮v²dt/r² = 2π(GM)²(1+e²)/ℓ³) :

    δΔω = π u² [ -3 δα₃ (1-2ν) + (5+e²)(1-3ν) δb₂ ]
        = -π u² [ (8 + 3e²/2) - ν (47/2 + 9e²/2) ]     par orbite

Limite test-masse ν→0 : -π(8+3e²/2)u² — EXACTEMENT le résultat de
derive_coefficient_2pn_trame.py (méthode indépendante : intégrale
d'orbite exacte vs perturbation de Delaunay) : double vérification.

PÉRIMÈTRE DÉCLARÉ : le secteur CINÉMATIQUE 2PN (termes v⁴U, retard
(v/c)⁴ du champ d'un nœud mobile) n'est PAS calculé ici — même ordre,
cible nommée n°2. Le secteur statique fixe l'ordre de grandeur et la
structure en ν.

Ce script VÉRIFIE :
  [1] les identités de masses (sympy) ;
  [2] la formule δΔω par INTÉGRATION NUMÉRIQUE directe : Kepler +
      accélération de perturbation dérivée de δL̂ (Euler-Lagrange),
      mesure de la dérive du vecteur de Runge-Lenz sur ~150 orbites,
      pour (ν, e) ∈ {(0, 0.3), (0.25, 0.3), (0.25, 0.7), (0.1, 0.5)} ;
  [3] la combinatoire 3/4 corps de U_a³ (sympy) ;
puis APPLIQUE : J0737-3039 (ν = 0.2497) et S301 (ν ~ 1e-6).
"""

import numpy as np
import sympy as sp

DA3 = 1.0/6.0    # delta alpha_3 (e^{-h} vs Schwarzschild isotrope)
DB2 = -1.5       # delta b_2 (1+h vs (1+U/2)^4)


# ================================================================
# [1] Identités de masses (sympy)
# ================================================================
print("=" * 70)
print("[1] Identités de masses de la réduction à deux corps")
print("=" * 70)
m1, m2 = sp.symbols('m1 m2', positive=True)
M = m1 + m2
mu = m1*m2/M
nu = mu/M
lhs1 = sp.simplify(m1*m2*(m1**2 + m2**2)/mu - M**3*(1 - 2*nu))
lhs2 = sp.simplify((m1**3 + m2**3)/M - M**2*(1 - 3*nu))
assert lhs1 == 0 and lhs2 == 0
print("  m1m2(m1²+m2²)/μ = M³(1-2ν)  [OK]")
print("  (m1³+m2³)/M     = M²(1-3ν)  [OK]")

# ================================================================
# [3 avancé] Combinatoire U_a³ à 3 et 4 corps
# ================================================================
x, y, z = sp.symbols('x y z', positive=True)
expa = sp.expand((x + y + z)**3)
c_xyz = expa.coeff(x*y*z)
print(f"  U_a³ : coefficient du produit triple m_b m_c m_d = {c_xyz}"
      " (terme à 4 corps, N≥4)  [OK]")
assert c_xyz == 6


# ================================================================
# [2] Vérification numérique de δΔω (Kepler perturbé)
# ================================================================
# unités G = M = 1 ; c fixe u = GM/(c²p).  δL̂ = A3/r³ + B2 v²/r²
# Euler-Lagrange (1er ordre, a_N inséré dans d/dt ∂δL/∂v) :
#   δa = [-3A3/r⁴ - 2B2 v²/r³] n̂ - 2B2 [a_N/r² - 2 v (v·n̂)/r³]
def run_case(nu_v, e, c_light, n_orbits=150):
    GM = 1.0
    p = 1.0
    a_sma = p/(1 - e**2)
    A3 = -(DA3/2)*GM**3*(1 - 2*nu_v)/c_light**4
    B2 = (DB2/2)*GM**2*(1 - 3*nu_v)/c_light**4

    def rhs(t, s):
        r = s[:2]
        v = s[2:]
        rn = np.hypot(r[0], r[1])
        n = r/rn
        aN = -GM*n/rn**2
        v2 = v @ v
        da = (-3*A3/rn**4 - 2*B2*v2/rn**3)*n \
             - 2*B2*(aN/rn**2 - 2*v*(v @ n)/rn**3)
        return np.concatenate([v, aN + da])

    # départ au péricentre
    r0 = p/(1 + e)
    v0 = np.sqrt(GM*(2/r0 - 1/a_sma))
    s = np.array([r0, 0.0, 0.0, v0])
    T = 2*np.pi*np.sqrt(a_sma**3/GM)

    from scipy.integrate import solve_ivp
    t_end = n_orbits*T
    sol = solve_ivp(rhs, (0, t_end), s, method='DOP853',
                    rtol=1e-12, atol=1e-14, dense_output=True)
    # angle du vecteur de Runge-Lenz au début et à la fin
    def omega_of(sv):
        r = sv[:2]
        v = sv[2:]
        L = r[0]*v[1] - r[1]*v[0]
        ev = np.array([v[1]*L, -v[0]*L])/GM - r/np.hypot(*r)
        return np.arctan2(ev[1], ev[0])

    w0 = omega_of(sol.y[:, 0])
    w1 = omega_of(sol.y[:, -1])
    dw = (w1 - w0 + np.pi) % (2*np.pi) - np.pi
    per_orbit = dw/n_orbits
    u = GM/(c_light**2*p)
    pred = np.pi*u**2*(-3*DA3*(1 - 2*nu_v) + (5 + e**2)*(1 - 3*nu_v)*DB2)
    return per_orbit, pred, u


print("\n" + "=" * 70)
print("[2] Intégration directe : dérive du péricentre par orbite")
print("    prédiction : pi u² [-3δα₃(1-2ν) + (5+e²)(1-3ν)δb₂]")
print("=" * 70)
print(f"  {'nu':>5s} {'e':>5s} {'u':>9s} {'mesuré':>13s} {'prédit':>13s}"
      f" {'écart':>8s}")
for nu_v, e in ((0.0, 0.3), (0.25, 0.3), (0.25, 0.7), (0.1, 0.5)):
    meas, pred, u = run_case(nu_v, e, c_light=30.0)
    err = abs(meas - pred)/abs(pred)
    print(f"  {nu_v:5.2f} {e:5.2f} {u:9.2e} {meas:13.4e} {pred:13.4e}"
          f" {err:8.1e}")
    assert err < 2e-2, "VERIF ECHOUEE"
print("  [OK] l'habillage en ν et la dépendance en e sont confirmés")

# ================================================================
# Applications
# ================================================================
print("\n" + "=" * 70)
print("[4] Applications")
print("=" * 70)
G = 6.6710e-11
c = 2.99792458e8
Msun = 1.989e30


def dev_rel(u, e, nu_v):
    """écart relatif TQNT-RG sur la précession (secteur statique)."""
    coeff = (8 + 1.5*e**2) - nu_v*(23.5 + 4.5*e**2)
    return coeff*u/6.0, coeff


# ---- pulsar double J0737-3039 ----
ma, mb = 1.3381*Msun, 1.2489*Msun
Mt = ma + mb
nu_p = ma*mb/Mt**2
Pb = 2.4535*3600.0
e_p = 0.0878
a_p = (G*Mt*(Pb/(2*np.pi))**2)**(1/3)
u_p = G*Mt/c**2/(a_p*(1 - e_p**2))
fr, cf = dev_rel(u_p, e_p, nu_p)
print(f"\nJ0737-3039 : nu = {nu_p:.4f}, u = {u_p:.2e}")
print(f"  coefficient habillé = -{cf:.3f} (test-masse : -8.01 ;"
      f" suppression x{8.01/cf:.1f})")
print(f"  écart relatif sur omega_dot (secteur statique) = -{fr:.2e}")
print(f"  précision mesurée : 1,3e-6 (Kramer+ 2021)")
print("  => l'écart statique ~ 1,2x la précision brute : AU BORD de la")
print("  sensibilité — ni réfutation flagrante ni marge ; verdict")
print("  suspendu au secteur cinématique 2PN (cible n°2) et à l'analyse")
print("  PK complète (dégénérescences masses/Lense-Thirring).")

# ---- S301 (test-masse, rappel) ----
u_s, e_s = 1.769e-3, 0.9824
fr_s, cf_s = dev_rel(u_s, e_s, 1e-6)
print(f"\nS301 : nu ~ 1e-6 -> coefficient test-masse intact"
      f" (-{cf_s:.3f}) ; écart = {fr_s*100:.2f} % du 1PN (inchangé).")

print("\nStructure N-corps de la déviation (résumé) :")
print("  N=2 : termes (1-2ν), (1-3ν) ci-dessus ;")
print("  N=3 : + croisés (Gm_b/r_ab)²(Gm_c/r_ac) dans δα₃ U_a³ ;")
print("  N=4 : + produit triple 6(Gm_b/r_ab)(Gm_c/r_ac)(Gm_d/r_ad) —")
print("  force à 4 corps irréductible, propre à la déviation 2PN.")
