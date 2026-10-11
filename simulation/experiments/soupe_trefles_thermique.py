"""La soupe de trèfles : abondance, énergétique réelle, équilibre thermique et survie (16/09/2026).
Objection : 52-59 % des nœuds créés sont des trèfles ; dans un petit volume de soupe ils
« collapsent à travers l'espace des phases jusqu'à fusion » — l'abondance des tores ne
devrait pas être faible. On chiffre : (1) la soupe est déjà au contact ; (2) le trèfle
LIBRE est l'électron (0,511 MeV), pas le stoikhion (310 MeV) : l'empilement d'électrons
est ENDOTHERMIQUE ; (3) la population est thermique (paires e±, porte γγ→e⁺e⁻) : l'offre
est illimitée jusqu'à T ~ 20 keV ; (4) l'abondance des tores est donc un problème de
relique thermique (équilibre + érosion), pas d'offre.
"""
import math
hbar=1.0546e-34; c=2.998e8; kB=1.381e-23; MeV=1.602e-13
alpha=1/137.036; ML=310.0; slope=70.1; me=0.511
def M_T2(q): return ML+slope*(q-3)
T_c=62.0; t_c=1.6e-4; H_c=1/(2*t_c)
hbarc=197.327  # MeV fm
print("=== 1. La soupe au gel est déjà au contact ===")
n_knot=0.304/(3.36e-15)**3; n_3=0.59*n_knot
n_e_th=0.37*(T_c/hbarc)**3*1e45           # paires e± thermiques par espèce, m^-3
d=n_3**(-1/3)
pF=hbar*(3*math.pi**2*n_3)**(1/3)*c/MeV   # MeV/c (fermions spin 1/2)
print(f"  n₃(comptage des nœuds) = {n_3:.1e} m⁻³ ; n_e±(thermique à T_c) = {n_e_th:.1e} m⁻³ par espèce — même ordre : les trèfles libres SONT le bain e±")
print(f"  espacement {d*1e15:.1f} fm (~ξ = 3,4 fm ; densité nucléaire : 1,8 fm) ; p_F = {pF:.0f} MeV/c ~ T_c : gaz dense, mi-dégénéré")
print("  → il n'y a rien à « collapser » : la soupe est au contact, le budget de collisions est 1e15 par Hubble ; la densité n'est pas la limite.\n")
print("=== 2. L'énergétique RÉELLE : le trèfle libre pèse 0,511 MeV ===")
print("  état libre du 3₁ = lepton (Koide, k=0 : électron) ; 310 MeV = masse de stoikhion DANS un composite (transmutation).")
for n in (2,11,12):
    q=2*n+1; print(f"  {n:2d} électrons ({n*me:5.1f} MeV) → T(2,{q}) = {M_T2(q):.0f} MeV : ENDOTHERMIQUE de {M_T2(q)-n*me:.0f} MeV ({(M_T2(q)-n*me)/n:.0f} MeV par électron)")
print(f"  la « fusion exothermique de 170 MeV » (T-2) supposait des trèfles à 310 MeV : elle vaut pour des stoikhions déjà liés, pas pour la soupe.")
print(f"  seuil par pas : ~{slope*2-me:.0f} MeV à prendre au bain ; à T_c, facteur de Boltzmann e^(−140/62) = {math.exp(-(2*slope-me)/T_c):.2f} — accessible, mais l'équilibre décide.\n")
print("=== 3. Équilibre thermique des tores dans le bain e± à T_c ===")
# Y_eq(q) = g (M T/2π)^{3/2} e^{-M/T} / s  (unités naturelles MeV)
g_s=15.2; s_nat=(2*math.pi**2/45)*g_s*T_c**3
def Y_eq(M,T,g=2): return g*(M*T/(2*math.pi))**1.5*math.exp(-M/T)/((2*math.pi**2/45)*g_s*T**3)
Y_DM=2.5e-10
print(f"  {'q':>3} {'M (MeV)':>8} {'Y_eq(T_c)':>10} {'Y·M/Planck':>11}")
for q in (5,11,17,21,23,25,27,31):
    Y=Y_eq(M_T2(q),T_c); print(f"  {q:3d} {M_T2(q):8.0f} {Y:10.1e} {Y*M_T2(q)/(Y_DM*1800):11.1e}")
print("  → à l'équilibre au gel, les tores c ≥ 23 pèsent déjà ~1e-8 de Planck par classe : l'offre n'est PAS le problème, la survie l'est.\n")
print("=== 4. Survie : l'érosion par les positrons jusqu'à leur disparition (T ≈ 20 keV) ===")
K0=math.pi*(0.35e-15)**2*0.03*c
def n_e(T): return 0.37*(T/hbarc)**3*1e45
def t_of_T(T): return 0.74*(1.0/T)**2 if T>0.2 else 1.31*(1.0/T)**2   # s, T en MeV
print(f"  taux d'érosion d'un tore c : Γ = n_e⁺ K₀ α^(c−18) ; à T_c pour c=23 : {n_e(T_c)*K0*alpha**5:.1e} s⁻¹ (H = {H_c:.0e}) — les tores c=23 sont en équilibre au gel")
# nombre d'événements d'érosion intégré de T_c à 0,02 MeV
def N_events(cc):
    tot=0; T=T_c
    while T>0.02:
        dT=T*0.01; t1=t_of_T(T); t2=t_of_T(T-dT)
        tot+=n_e(T)*K0*alpha**max(0,cc-18)*(t2-t1); T-=dT
    return tot
print(f"  {'c':>3} {'événements par tore (t_c → 20 keV)':>36}  verdict (il faut ≲ 10 pour survivre à l'érosion vers c<23)")
for cc in (23,25,27,29,31):
    N=N_events(cc); print(f"  {cc:3d} {N:36.1e}  {'ÉRODÉ' if N>10 else 'SURVIT'}")
cstop=18+math.log(10/N_events(18))/math.log(alpha)
print(f"  bord de survie : N(c) = 10 → c ≈ {cstop:.1f}  (le bord 23 du cadre vient de Γ/H au gel ; l'érosion intégrée le pousse à ~{math.ceil(cstop)})")
print()
print("=== 5. Ce que vaut alors Ω ===")
sv=K0*alpha**5*1e6   # cm³/s
print(f"  si l'érosion s'arrêtait au gel : relique WIMP-like, ⟨σv⟩ = K₀α⁵ = {sv:.1e} cm³/s → Ω h² ≈ 3e-27/⟨σv⟩ = {3e-27/sv:.2f} (Planck 0,12) : {3e-27/sv/0.12:.1f} × Planck")
Y27=Y_eq(M_T2(27),T_c); Y23=Y_eq(M_T2(23),T_c)
print(f"  mais les survivants sont c ≳ {math.ceil(cstop)} : Y_eq(27)/Y_eq(23) = {Y27/Y23:.1e} → Ω ≈ {3e-27/sv/0.12*Y27/Y23:.1e} × Planck (ordre, avant recouplage)")
print("  → la soupe fournit tout ; Vassiliev et l'érosion ne laissent passer qu'une queue exponentielle. Ce n'est plus a_f qui décide : c'est M(c)/T_c et α^(c−18).")
