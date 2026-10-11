"""La loi de capture des empilements, dérivée avec la reconnexion canonique (16/09/2026).

Empiler (unité + pile → pile+1) et dépiler (unité miroir + pile → pile−1) sont tous deux des reconnexions P15
exothermiques avec émission d'un photon : même vertex que l'érosion des tores (T-12),
    σv = α G_F² Q²/π   (contact 1/v², prix α du photon).
Donc a_f = (Q_empiler/Q_dépiler)² — ni E, ni E³ : E². Mais la normalisation est FAIBLE, pas géométrique
(le Smoluchowski de B-2 utilisait K₀ = π(0,35 fm)²·0,03c : incohérent avec T-12). Et les unités libres
4₁, 5₁ se dénouent (faible, Fermi) en quelques 1e-9 s. On compare.
"""
import math
alpha = 1/137.036; hbarc = 197.327; hbar = 6.582e-22
v = 246220.0; GF = 1/(math.sqrt(2)*v**2); MPl = 1.221e22
ML, slope, L31 = 310.0, 70.1, 32.7448
rope = {"4_1": 42.0928, "5_1": 47.2016}
tau_free = {"4_1": 2.9e-9, "5_1": 3.2e-9}          # table des nœuds libres (dénouage P15)
Tc = 62.0; t_c = 1.6e-4
H = 1.66*math.sqrt(15.2)*Tc**2/MPl/hbar
xi = 3.36e-15; n_knot_cm3 = 0.304/xi**3*1e-6
K0_geo = math.pi*(0.35e-13)**2*0.03*3e10           # cm³/s (B-2)
def sv(Q): return alpha*GF**2*Q**2/math.pi*(hbarc*1e-13)**2*3e10   # cm³/s
print(f"H(T_c) = {H:.1e} s⁻¹ ; n_nœuds = {n_knot_cm3:.1e} cm⁻³ ; noyau géométrique de B-2 K₀ = {K0_geo:.1e} cm³/s\n")
for K in ("4_1", "5_1"):
    M = ML*rope[K]/L31; dc = 4
    Qs, Qu = M - dc*slope, M + dc*slope
    af2 = (Qs/Qu)**2
    n_u = 0.141*0.5*n_knot_cm3
    G_s = n_u*sv(Qs)
    print(f"=== {K} (M = {M:.0f} MeV, fraction au gel ~7 %) ===")
    print(f"  Q empiler = {Qs:.0f} MeV, Q dépiler = {Qu:.0f} MeV → loi de capture DÉRIVÉE ∝ Q² : a_f = {af2:.3f}  (E : {Qs/Qu:.2f} ; E³ : {(Qs/Qu)**3:.4f})")
    print(f"  σv(empiler) = {sv(Qs):.1e} cm³/s  (B-2 utilisait {K0_geo:.1e} : ×{K0_geo/sv(Qs):.0e} trop grand)")
    print(f"  taux d'empilement par pile Γ = n_unité σv = {G_s:.1e} s⁻¹ ; Γ/H = {G_s/H:.1f}")
    print(f"  durée de vie d'une unité libre τ = {tau_free[K]:.1e} s → collisions d'empilement par unité avant dénouage : Γτ = {G_s*tau_free[K]:.1e}")
    # probabilité d'atteindre c ≥ 20 : 4 empilements successifs (4₁ : 4→20 ; 5₁ : 5→21), chacun pendant la vie de la pile (piles c<20 : dénouage aussi)
    p_step = G_s*tau_free[K]*0.5*af2/(af2+1)
    p5 = p_step**4
    frac_mass = 0.07*p5*(M+16*slope)/1800
    print(f"  probabilité par pas (orientation ½ × branchement) ≈ {p_step:.1e} ; atteindre c ≥ 20 en 4 pas ≈ {p5:.0e}")
    print(f"  → masse des piles sombres ~ {frac_mass:.0e} × (masse des nœuds du gel) : NÉGLIGEABLE\n")
print("Conclusion : la loi de capture est ∝ Q² (même vertex radiatif que T-12), a_f = 0,03 (4₁), 0,05 (5₁) ;")
print("mais la normalisation canonique est faible : les unités se dénouent 1e4 fois plus vite qu'elles ne s'empilent.")
print("Les piles Th(3,10) ne complètent PAS Ω. Le même constat corrige T-3/T-4 (coagulation « instantanée » = artefact")
print("du noyau géométrique) : la coagulation des nœuds du gel n'a pas lieu ; les abondances sont celles de la naissance.")
