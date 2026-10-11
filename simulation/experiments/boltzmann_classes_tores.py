"""Boltzmann couplé classe par classe : tores T(2,c) dans le bain e± de T_c à 10 keV (16/09/2026).

Ce que T-8 n'avait pas vu, et qui structure le calcul :
  CONSERVATION DE LA CHARGE. Capturer un e⁻ ajoute un trèfle ET une charge −1 ; capturer un e⁺
  retire un trèfle (σ₁^c σ₁^-3) ET ajoute +1. Pour une graine neutre c₀ :  Q = (c₀ − c)/2.
  → le bain e± ne peut PAS peupler thermiquement des tores NEUTRES (le Y_eq du T-8 est inatteignable) ;
    la seule production neutre est la paire torus–antitorus γγ → T T̄, supprimée α^{2(c−18)} e^{−2M/T}.
  → les tores neutres ont l'abondance de leur NAISSANCE (réseau au gel, chaîne Ω) ; le bain ne fait que
    les ÉRODER (vers +1, +2…) ou les CHARGER (vers −1, −2…).

États : pour chaque graine neutre c₀, chaîne j ∈ [−J, J], c = c₀ + 2j, Q = −j.
Lien (c ↔ c+2) : descente (perte d'un trèfle, exothermique : capture e⁺ ou émission e⁻)
  D(c+2, T) = Γ₀(T) α^{max(18, c+2)−23}  (Vassiliev dès 18, calé à 23) ; montée U = D · r_eq, r_eq = n_eq(c+2)/n_eq(c) = (M₊/M)^{3/2} e^{−ΔM/T}.
Puits : c < 19 (cascade P15 rapide vers leptons).
Normalisation Γ₀(T_c)/H(T_c) du taux de reconnexion d'un tore c=23 avec le bain — le point incertain :
  A. cadre tel quel (dérivation de c_min : baryons seuls, σ_EM α^{c−18}) ............ 0,067
  B. même loi σ(c), mais bain e± (n_e±/n_b = 1,5/η, v = c au lieu de √(3T/μ)) ....... calculé
  C. contact géométrique des tubes (R_K = 0,075 fm) × α^{c−18}, bain e± ................ calculé
  D. valeur utilisée en T-8 (π(0,35 fm)², v = 0,03c) ...................................... calculé
"""
import math
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import kn

alpha = 1/137.036; hbarc = 197.327; me = 0.511; MPl = 1.221e22    # MeV
eta = 6.1e-10; Tc = 62.0
ML, slope = 310.0, 70.1
def M(c): return ML + slope*(c - 3)          # MeV, loi géométrique (toriques ACPR + T(2,23))
def gstar(T): return 15.2 if T > 30 else (10.75 if T > 0.3 else 3.36)
def H(T): return 1.66*math.sqrt(gstar(T))*T**2/MPl/6.582e-22      # s⁻¹
def n_epm(T):   # e⁺+e⁻, 4 dof, Fermi ≈ Boltzmann×correction ; MeV³
    x = me/T
    return 4*T**3/(2*math.pi**2)*(x**2*kn(2, x) if x > 1e-3 else 2.0)*0.75/ (1.0 if x < 1 else 0.75)
def v_e(T): return 1.0 if T > me else math.sqrt(8*T/(math.pi*me))
def n_b(T): return eta*0.24*T**3

# --- normalisations à T_c ---
mu23 = M(23)*938/(M(23)+938)
v_b = math.sqrt(3*Tc/mu23)
rA = 0.067
rB = rA*(n_epm(Tc)/n_b(Tc))*(1.0/v_b)
sig_geo = math.pi*(0.075)**2*alpha**5 / hbarc**2          # MeV⁻² (c=23)
rC = n_epm(Tc)*sig_geo*1.0/(H(Tc)*6.582e-22)
sig_T8 = math.pi*(0.35)**2*alpha**5/hbarc**2
rD = n_epm(Tc)*sig_T8*0.03/(H(Tc)*6.582e-22)
NORMS = [("A cadre (baryons)", rA, "b"), ("B σ(c) cadre, bain e±", rB, "e"),
         ("C contact géométrique, e±", rC, "e"), ("D valeur T-8", rD, "e")]
print("=== Normalisations : Γ(c=23, T_c)/H(T_c) ===")
for nom, r, _ in NORMS:
    print(f"  {nom:<28} {r:9.2e}   → bord naïf Γ/H = 1 à c = {23 + math.log(r)/math.log(1/alpha):.1f}")
print()

def Gamma0(T, r, bath):
    """Γ(c=23,T) en s⁻¹ avec la loi en T du bain."""
    if bath == "b":
        mu = mu23; v = math.sqrt(3*T/mu)
        return r*H(Tc)*(n_b(T)*v)/(n_b(Tc)*math.sqrt(3*Tc/mu))
    return r*H(Tc)*(n_epm(T)*v_e(T))/(n_epm(Tc)*1.0)

def run_seed(c0, r, bath, J=8, dx=2e-3, Tend=0.01):
    js = np.arange(-J, J+1); cs = c0 + 2*js
    Y = np.zeros(len(js)); Y[J] = 1.0; sink = 0.0
    x = 0.0; xend = math.log(Tc/Tend)
    while x < xend:
        T = Tc*math.exp(-x); dt = dx/H(T); G = Gamma0(T, r, bath)
        n = len(js)
        # coefficients de lien k : entre états k (c) et k+1 (c+2)
        Dl = np.array([G*alpha**(max(18, cs[k]+2)-23) for k in range(n-1)])          # descente k+1 → k
        req = np.array([((M(cs[k]+2)/M(cs[k]))**1.5)*math.exp(-(2*slope)/T) for k in range(n-1)])
        Ul = Dl*req                                                               # montée k → k+1
        # matrice A (dY/dt = A Y) ; puits : état 0 descend vers c<19 si cs[0]-2 < 19, sinon bord réfléchi
        diag = np.zeros(n); up = np.zeros(n); lo = np.zeros(n)
        for k in range(n-1):
            diag[k] -= Ul[k]; diag[k+1] -= Dl[k]
            up[k+1] += Dl[k]     # A[k, k+1]
            lo[k] += Ul[k]       # A[k+1, k]
        Dsink = G*alpha**(max(18, cs[0])-23)
        out0 = Dsink if cs[0] >= 19 else 1e30
        diag[0] -= min(out0, 1e30)
        # états sous 19 : absorbés immédiatement
        for k in range(n):
            if cs[k] < 19: Y[k] = 0.0
        ab = np.zeros((3, n))
        ab[0, 1:] = -dt*up[1:]; ab[1, :] = 1 - dt*diag; ab[2, :-1] = -dt*lo[:-1]
        Ynew = solve_banded((1, 1), ab, Y)
        Ynew = np.maximum(Ynew, 0)
        x += dx
        Y = Ynew
    neutral = Y[J]; charged_pos = Y[:J].sum(); charged_neg = Y[J+1:].sum()
    Mmean_neutral = M(c0)
    mass_neutral_like = neutral*M(c0)
    return neutral, charged_pos, charged_neg, Y, cs

P = lambda c: 0.68**(c-23)
C0 = list(range(19, 41))
norm_mass = sum(P(c)*M(c) for c in C0 if c >= 23)
Ychain_DM_per_s = 2.5e-10    # rappel : n_DM/s pour Planck (1,8 GeV)
for nom, r, bath in NORMS:
    print(f"=== Normalisation {nom} (Γ/H(23,T_c) = {r:.2e}) ===")
    print(f"  {'c₀':>3} {'M (GeV)':>7} {'survie neutre':>13} {'→ chargés +':>11} {'→ chargés −':>11} {'détruits':>9}")
    mass_surv = 0.0; pos_mass = 0.0; neg_num = 0.0; pos_num = 0.0
    for c0 in C0:
        Nn, Cp, Cm, Y, cs = run_seed(c0, r, bath)
        dead = max(0.0, 1 - Nn - Cp - Cm)
        mass_surv += P(c0)*M(c0)*Nn
        pos_num += P(c0)*Cp; neg_num += P(c0)*Cm
        if c0 <= 32 or c0 % 4 == 0:
            print(f"  {c0:3d} {M(c0)/1e3:7.2f} {Nn:13.3e} {Cp:11.2e} {Cm:11.2e} {dead:9.2e}")
    frac = mass_surv/norm_mass
    Ntot = sum(P(c) for c in C0 if c >= 23)
    Nborn = sum(P(c) for c in C0)
    print(f"  → masse neutre survivante / réservoir né (c≥23) = {frac:.3f}  ⇒  Ω = Ω_chaîne × {frac:.3f}  (0,6 → {0.6*frac:.2f} ; 3,9 à l'ancien T_c → {3.9*frac:.2f})")
    ntori_nb = 5.36*938/1850
    print(f"  → reliques chargées par nœud né (c≥19) : +{pos_num/Nborn:.2e}, −{neg_num/Nborn:.2e} ; par baryon (×{ntori_nb:.1f}) : X⁺/n_b = {pos_num/Nborn*ntori_nb:.1e}, X⁻/n_b = {neg_num/Nborn*ntori_nb:.1e}")
    print()
