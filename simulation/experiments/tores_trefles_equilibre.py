"""Tores de trèfles : 3₁ × n → T(2, q) par équilibre statique (16/09/2026).
Idée de l'utilisateur : former les nœuds sombres par empilement de n trèfles en tores.
(1) Topologie : n trèfles alignés = tresse σ₁^{3n} ; la fermeture après reconnexions P15
    est le nœud torique T(2,q), q impair — un équilibre STATIQUE (tube tendu, nœud idéal).
(2) Énergétique : loi géométrique M(T(2,q)) = m_ℓ ℓ(q)/ℓ(3) (pente 70 MeV/croisement).
(3) Cinétique au gel : coagulation trèfle–trèfle contre annihilation trèfle–antitrèfle,
    suppression de Vassiliev α^(c−18) au-delà de c = 18, dilution ; où s'arrête-t-elle ?
(4) Ce que cela donne pour le réservoir (bord c = 23-25) et pour Ω.
"""
import math
import numpy as np
alpha = 1/137.036
ML = 310.0; slope = 70.1                 # MeV, MeV/croisement (loi toriques ACPR + T(2,23))
def M_T2(q): return ML + slope*(q-3)
print("=== 1-2. Topologie et énergétique de l'empilement ===")
print(f"{'n trèfles':>9} {'q=2n+1':>7} {'M(T(2,q)) MeV':>14} {'n·m_ℓ':>8} {'liaison':>8} {'par trèfle':>10}")
for n in (1,2,3,5,8,10,11,12,15):
    q=2*n+1; M=M_T2(q); print(f"{n:9d} {q:7d} {M:14.0f} {n*ML:8.0f} {n*ML-M:8.0f} {(n*ML-M)/n:10.0f}")
print("  → chaque trèfle ajouté libère ΔE = m_ℓ − 2·pente = %.0f MeV : l'empilement est EXOTHERMIQUE, le tore est l'équilibre." % (ML-2*slope))
print("  → T(2,23) = 11 trèfles, liaison 1,7 GeV (50 %) ; charge : Q = Σχ + Lk neutralisable par framing pour tout q (réservoir).")
print("  → statique : le tore idéal (ropelength minimal) est l'équilibre du tube sous tension σ_χ ; rien à ajuster.\n")

print("=== 3. Cinétique au gel : coagulation contre annihilation ===")
# densités au gel
xi = 3.36e-15; T_c = 62.0; g_s = 15.2; hbarc = 197.327  # MeV fm
n_knot = 0.304/xi**3                       # m^-3 (0,304 nœuds par ξ le long des cordes, ~1 corde par ξ²)
n_3 = 0.59*n_knot
s = (2*math.pi**2/45)*g_s*(T_c/hbarc)**3*1e45   # m^-3
Y3 = n_3/s
t_c = 1.6e-4; H_c = 1/(2*t_c)
c = 2.998e8; sig_geo = math.pi*(0.35e-15)**2; v = 0.03*c
K0 = sig_geo*v                              # m³/s
print(f"  n_knot = {n_knot:.1e} m⁻³, trèfles n₃ = {n_3:.1e}, s = {s:.1e} → Y₃ = n₃/s = {Y3:.3f} ; H(t_c) = {H_c:.0e} s⁻¹")
print(f"  noyau géométrique K₀ = σv = {K0:.1e} m³/s ; n₃K₀/H = {n_3*K0/H_c:.0e} : la coagulation est INSTANTANÉE tant qu'elle n'est pas supprimée")
c_stop = 18 + math.log(H_c/(n_3*K0))/math.log(alpha)
print(f"  arrêt naïf (n₃K₀α^(c−18) = H) : c_stop = {c_stop:.1f}  ← le bord du réservoir (23-25) sans rien ajuster ?")
print()
# modèle Smoluchowski symétrique en chiralité, comobile, classes q impaires
Q = list(range(3, 64, 2)); nq = len(Q); idx = {q:i for i,q in enumerate(Q)}
def supp(q): return alpha**max(0, q-18)
def run(a_f, a_ann, label):
    Y = np.zeros(nq); Y[0] = Y3          # Y_q = n_q/s pour UNE chiralité (les deux symétriques)
    t = t_c; a = 1.0
    ln_t = math.log(t); ln_end = math.log(1e3)  # jusqu'à la BBN (largement gelé avant)
    while ln_t < ln_end:
        dt = t*0.002
        n = Y*s/a**3
        dY = np.zeros(nq)
        for i,q1 in enumerate(Q):
            if Y[i] <= 0: continue
            for j,q2 in enumerate(Q):
                if Y[j] <= 0 or j < i: continue
                # fusion même chiralité : q1 + q2 -> q1+q2-1
                k = a_f*K0*supp(q1)*supp(q2)
                r = k*n[i]*n[j]/s*(0.5 if i==j else 1.0)
                q3 = q1+q2-1
                if q3 in idx:
                    dY[i] -= r; dY[j] -= r; dY[idx[q3]] += r
                # annihilation chiralités opposées (K + K̄'), même noyau supprimé
                ka = a_ann*K0*supp(q1)*supp(q2)
                ra = ka*n[i]*n[j]/s*(1.0 if i!=j else 1.0)
                dY[i] -= ra; dY[j] -= ra
        # stabilité : limiter le pas
        m = np.max(np.abs(dY)*dt/np.maximum(Y,1e-30))
        if m > 0.2: dt *= 0.2/m
        Y = np.maximum(Y + dY*dt, 0); t += dt; a = (t/t_c)**0.5; ln_t = math.log(t)
    Ydark = sum(Y[i] for i,q in enumerate(Q) if q >= 23)
    Mdark = sum(Y[i]*M_T2(q) for i,q in enumerate(Q) if q >= 23)
    Y_left3 = Y[0]
    dist = {q: Y[i] for i,q in enumerate(Q) if Y[i] > 1e-3*max(Y)}
    top = sorted(dist.items(), key=lambda x: -x[1])[:5]
    print(f"  {label:<28} : Y(c≥23) par chiralité = {Ydark:.2e} ; masse sombre ΣY·M = {2*Mdark:.2e} MeV/entropie ; Y₃ restant = {Y_left3:.1e}")
    print(f"      pics : " + ", ".join(f"q={q}:{y:.1e}" for q,y in top))
    return 2*Mdark
Y_DM_needed = 2.3e-27/(1.8e3*1.783e-30)/2.9e9   # n_DM/s aujourd'hui
print(f"  cible Planck : n_DM/s = {Y_DM_needed:.1e} → masse/entropie = {Y_DM_needed*1.8e3:.1e} MeV ; baryons n_B/s = 8,6e-11")
for a_f, a_ann, lab in ((1.0, 1.0, "fusion = annihilation"), (0.1, 1.0, "fusion ×0,1"), (1e-2, 1.0, "fusion ×1e-2"), (1e-3, 1.0, "fusion ×1e-3")):
    Md = run(a_f, a_ann, lab)
    print(f"      → Ω_tores/Ω_Planck = {Md/(Y_DM_needed*1.8e3):.1e}")
print()
print("=== 4. Lecture ===")
print("  – l'empilement de trèfles est exothermique (170 MeV/trèfle) et s'arrête de lui-même par Vassiliev vers c ≈ %.0f : le BORD du réservoir" % c_stop)
print("    peut être un gel de coagulation plutôt qu'un produit direct de la cascade — à confronter au comptage n_knot/L de la chaîne Ω.")
print("  – la fraction qui échappe à l'annihilation fixe Ω : c'est le rapport fusion/annihilation (alignement des axes) qui décide,")
print("    pas la masse ; le facteur manquant ×4-10 de la chaîne Ω (0,6 contre 5,45) est dans la portée de ce canal.")
print("  – rien ici ne fait d'objet macroscopique : M ∝ q, un tore de 2e-3 Msun aurait q ~ 1e49 ; les tores restent des éremnions.")
