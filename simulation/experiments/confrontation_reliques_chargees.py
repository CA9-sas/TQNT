"""Confrontation des reliques chargées du Boltzmann T-10 aux bornes externes (16/09/2026).
Reliques (par baryon) : X⁺/n_b ≈ 0,3–1,6e-5 ; X⁻/n_b ≈ 0,6–4e-6 ; masse 1,7–2,0 GeV.
Bornes (sources vérifiées) :
  CBBN, X⁻ à longue vie : Pospelov 2006 (hep-ph/0605215) n_X/s < 3e-17 ; Jedamzik/Kusakabe
     (τ > 5e3 s) Y_X⁻ < 2e-16, relâchable à 1e-14–1e-15 (⁶Li conservateur) — établies pour m_X ≫ m_He.
  Isotopes lourds de l'hydrogène (revue Burdin+ 2014, arXiv:1410.1374, table) :
     Muller+ 1977 (Science 196, 521) H,D, X⁺ 0,3–8 GeV : 2e-19 – 1e-13 par H ;
     Alväger & Naumann 1967, 6–16 GeV : 3e-18 par H ; Yamagata+ 1993 (PRD 47, 1231), 5–1600 u : 4e-17 par H.
"""
import math
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import kn
alpha=1/137.036; hbarc=197.327; me=0.511; mp=938.272; mHe=3727.4
nb_s = 8.6e-11; YH = 0.75
print("=== 1. X⁻ : nucléosynthèse catalysée ===")
for lo, hi, lab in ((0.6e-6, 4e-6, "X⁻/n_b (T-10)"),):
    print(f"  {lab} = {lo:.1e}–{hi:.1e}  →  Y = n/s = {lo*nb_s:.1e}–{hi*nb_s:.1e}")
print("  bornes : 3e-17 (Pospelov 2006) ; 2e-16 (τ > 5e3 s) ; 1e-14–1e-15 (⁶Li conservateur)")
print("  → au-dessus de 3e-17 et de 2e-16 pour la partie haute ; sous la borne relâchée : FRONTIÈRE, comme annoncé.")
mX = 1800.0
mu = mHe*mX/(mHe+mX); EB = 4*alpha**2*mu/2; aB = hbarc/(2*alpha*mu)
muH = mHe*1e7/(mHe+1e7); EBH = 4*alpha**2*muH/2
print(f"  réserve de masse : (⁴He X⁻) avec m_X = 1,8 GeV : μ = {mu:.0f} MeV, E_B(ponctuel) = {EB*1e3:.0f} keV, a_B = {aB:.0f} fm")
print(f"  (X⁻ lourd : E_B ponctuel {EBH*1e3:.0f} keV, ~350-400 keV avec taille finie) → formation vers T ≈ E_B/35 = {EB/35*1e3:.1f} keV au lieu de ~8 keV :")
print("  l'état lié se forme plus tard, sur un deutérium déjà gelé ; le facteur catalytique est à recalculer — la borne ne s'applique pas telle quelle.\n")
print("=== 2. Isotopes lourds terrestres ===")
for lo, hi in ((0.3e-5, 1.6e-5),):
    print(f"  X⁺ (atome ~1,8–2,0 u, voisin du deutérium) : X⁺/H cosmique = {lo/YH:.1e}–{hi/YH:.1e}")
print("  borne Muller+ 1977 (0,3–8 GeV) : 2e-19 à 1e-13 par H  →  excès de 8 à 14 ordres (même à la borne la plus faible)")
print(f"  X⁻ lié à ⁴He = (⁴He X⁻)⁺ : un « hydrogène » de {(mHe+mX)/931.5:.1f} u → Yamagata+ 1993 (5–1600 u) 4e-17 par H,")
print(f"     Alväger 1967 (6–16 GeV) 3e-18 : X⁻/H ~ 1e-6 → excès d'environ 10 ordres")
print("  réserve : suppose que les reliques suivent la chimie de l'hydrogène jusqu'à la Terre (vrai à 2 GeV : comportement de proton,")
print("  l'expulsion du disque par les chocs de supernova ne vaut que pour des masses ≳ 100 TeV).")
print("  → SI ces reliques sont STABLES, elles sont exclues de 8 à 14 ordres, quelle que soit la normalisation (A comprise).\n")
print("=== 3. Sont-elles stables ? La neutralisation P15 ===")
for R in (0.057, 0.075, 0.093):
    Ec = alpha*hbarc/R     # énergie propre coulombienne d'une charge unité sur l'anneau (ordre)
    Q = Ec - me
    G = Q**5/(384*math.pi**3*246220.0**4)/6.582e-22
    print(f"  R_K = {R} fm : énergie propre ~ αħc/R = {Ec:.0f} MeV → X^± → X⁰ + e^± + ν, Q ≈ {Q:.0f} MeV ; Fermi : τ = {1/G*1e3:.1f} ms"
          f" ; ×α^(c−18) à c=26 : τ = {1/(G*alpha**8):.1e} s")
print("  ΔLk = ±1 ne change pas le type de nœud : c'est le processus faible du cadre (neutron), g_topo = 1, SANS suppression de Vassiliev")
print("  (Vassiliev supprime la CRÉATION de croisements). Lecture canonique : neutralisation en 1–100 ms → aucune relique chargée à la BBN.")
print("  lecture supprimée : τ ≫ âge → reliques stables → exclues (§2) → il faudrait interdire la reconnexion lepton–tore elle-même.\n")

print("=== 4. Boltzmann refait avec neutralisation rapide (chaîne neutre seule) ===")
MPl=1.221e22; Tc=62.0; eta=6.1e-10
def M(c): return 310.0+70.1*(c-3)
def gstar(T): return 15.2 if T>30 else (10.75 if T>0.3 else 3.36)
def H(T): return 1.66*math.sqrt(gstar(T))*T**2/MPl/6.582e-22
def n_epm(T):
    x=me/T
    return 4*T**3/(2*math.pi**2)*(x**2*kn(2,x) if x>1e-3 else 2.0)*(0.75 if x<1 else 1.0)
def v_e(T): return 1.0 if T>me else math.sqrt(8*T/(math.pi*me))
def tau_neut(T):  # ~10 ms, gelée si > 1/H : on la prend instantanée quand τ < 1/H, sinon le lien s'arrête à l'état chargé
    return 0.01
P=lambda c: 0.68**(c-23)
C=list(range(17,45,1))
def run(r):
    # classes neutres par parité : chaîne c, c±2 ; naissance P(c) ; érosion D(c) = Γ0 α^(max(18,c)-23) ; croissance D(c+2) r_eq ×b(T)
    out_mass=0.0; norm=sum(P(c)*M(c) for c in C if c>=23)
    for par in (0,1):
        cs=[c for c in C if c%2==par]; n=len(cs)
        Y=np.array([P(c) if c>=19 else 0.0 for c in cs]); x=0.0; xend=math.log(Tc/0.01); dx=2e-3
        while x<xend:
            T=Tc*math.exp(-x); dt=dx/H(T)
            G=r*H(Tc)*(n_epm(T)*v_e(T))/(n_epm(Tc))
            b = 1.0 if tau_neut(T) < 1/H(T) else 0.0     # neutralisation plus rapide que l'expansion ?
            diag=np.zeros(n); up=np.zeros(n); lo=np.zeros(n)
            for k in range(n):
                c=cs[k]; Dk=G*alpha**(max(18,c)-23)
                diag[k]-=Dk
                if k>0 and cs[k-1]>=19: lo_idx=k-1; up[k]+=0; 
                if k>0: 
                    # érosion k → k-1 (neutralisée si b) ; sinon perdue (reste chargée, hors chaîne neutre)
                    pass
            A=np.zeros((n,n))
            for k in range(n):
                c=cs[k]; Dk=G*alpha**(max(18,c)-23)
                A[k,k]-=Dk
                if k>0 and cs[k-1]>=19: A[k-1,k]+=Dk*b
                if k<n-1:
                    Du=G*alpha**(max(18,cs[k+1])-23)*((M(cs[k+1])/M(c))**1.5)*math.exp(-140.2/T)
                    A[k,k]-=Du; A[k+1,k]+=Du*b
            Y=np.linalg.solve(np.eye(n)-dt*A, Y); Y=np.maximum(Y,0); x+=dx
        out_mass+=sum(Y[k]*M(cs[k]) for k in range(n) if cs[k]>=19)
    return out_mass/norm
for lab, r in (("A (0,067, loi en T du bain e±)", 0.067), ("B σ(c), bain e±", 2.51e8), ("C contact géométrique", 3.34e5)):
    f=run(r)
    print(f"  {lab:<46} : masse neutre finale / réservoir né = {f:.3f}  → Ω = 0,6 × {f:.3f} = {0.6*f:.2f}")
